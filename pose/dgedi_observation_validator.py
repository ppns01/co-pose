from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from pose.alignment_scorer import (
    AlignmentScoreResult,
    AlignmentScoreWeights,
    score_alignment,
)
from pose.dgedi_runner import _diameter, _rigid


RaycastFunction = Callable[..., dict[str, np.ndarray]]


@dataclass(frozen=True)
class DGeDiObservationValidationResult:
    accepted: bool
    reasons: tuple[str, ...]
    summary_path: Path
    reference_render_path: Path
    query_render_path: Path
    metrics: dict[str, dict[str, float | int]]


def _default_raycast_function() -> RaycastFunction:
    return _raycast_mesh


def _raycast_mesh(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    camera_k: np.ndarray,
    image_height: int,
    image_width: int,
) -> dict[str, np.ndarray]:
    """Render camera-frame triangles with Open3D ray casting."""
    import open3d as o3d

    points = np.asarray(points_camera, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    intrinsic = np.asarray(camera_k, dtype=np.float64)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(
        faces.astype(np.int32, copy=False)
    )
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    pixel_u, pixel_v = np.meshgrid(
        np.arange(image_width, dtype=np.float32),
        np.arange(image_height, dtype=np.float32),
    )
    ray_direction = np.stack(
        [
            (pixel_u - cx) / fx,
            (pixel_v - cy) / fy,
            np.ones_like(pixel_u),
        ],
        axis=-1,
    )
    rays = np.concatenate(
        [np.zeros_like(ray_direction), ray_direction],
        axis=-1,
    ).astype(np.float32)
    hit = scene.cast_rays(
        o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    )["t_hit"].numpy().astype(np.float64)
    rendered_mask = np.isfinite(hit) & (hit > 0.0)
    rendered_depth = hit.astype(np.float32)
    rendered_depth[~rendered_mask] = 0.0
    return {
        "rendered_mask": rendered_mask,
        "rendered_depth": rendered_depth,
    }


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Validation mesh not found: {resolved}")

    mesh = o3d.io.read_triangle_mesh(
        str(resolved),
        enable_post_processing=True,
    )
    points = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Validation mesh has no vertices: {resolved}")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError(f"Validation mesh has no triangles: {resolved}")
    return points, triangles


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _render(
    *,
    points: np.ndarray,
    triangles: np.ndarray,
    camera_k: np.ndarray,
    image_shape: tuple[int, int],
    raycast_function: RaycastFunction,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    raycast = raycast_function(
        points_camera=points,
        triangles=triangles,
        camera_k=camera_k,
        image_height=height,
        image_width=width,
    )
    return (
        np.asarray(raycast["rendered_mask"], dtype=bool),
        np.asarray(raycast["rendered_depth"], dtype=np.float32),
    )


def _score_dict(score: AlignmentScoreResult) -> dict[str, float | int]:
    return {
        str(key): value
        for key, value in asdict(score).items()
    }


def _view_rejection_reasons(
    *,
    view_name: str,
    baseline: AlignmentScoreResult,
    cross: AlignmentScoreResult,
    minimum_depth_overlap_pixels: int,
) -> list[str]:
    reasons: list[str] = []
    minimum_iou = max(0.50, baseline.mask_iou - 0.20)
    maximum_depth_ratio = max(
        0.15,
        baseline.depth_residual_normalized + 0.10,
    )
    maximum_total_loss = max(0.45, baseline.total_loss + 0.20)

    if cross.rendered_pixel_count == 0:
        reasons.append(f"{view_name}: cross render is empty")
    if cross.valid_depth_overlap_count < minimum_depth_overlap_pixels:
        reasons.append(
            f"{view_name}: depth overlap too small "
            f"({cross.valid_depth_overlap_count} < "
            f"{minimum_depth_overlap_pixels})"
        )
    if cross.mask_iou < minimum_iou:
        reasons.append(
            f"{view_name}: mask IoU too low "
            f"({cross.mask_iou:.3f} < {minimum_iou:.3f})"
        )
    if cross.depth_residual_normalized > maximum_depth_ratio:
        reasons.append(
            f"{view_name}: normalized depth residual too large "
            f"({cross.depth_residual_normalized:.3f} > "
            f"{maximum_depth_ratio:.3f})"
        )
    if cross.total_loss > maximum_total_loss:
        reasons.append(
            f"{view_name}: alignment loss too large "
            f"({cross.total_loss:.3f} > {maximum_total_loss:.3f})"
        )
    return reasons


def validate_dgedi_against_observations(
    *,
    reference_mesh_path: Path,
    query_mesh_path: Path,
    relative_pose_query_from_reference: np.ndarray,
    reference_camera_k: np.ndarray,
    query_camera_k: np.ndarray,
    reference_mask_bool: np.ndarray,
    query_mask_bool: np.ndarray,
    reference_depth_m: np.ndarray,
    query_depth_m: np.ndarray,
    output_directory: Path,
    weights: AlignmentScoreWeights,
    depth_trim_quantile: float,
    minimum_depth_overlap_pixels: int,
    free_space_absolute_tolerance_m: float,
    free_space_relative_tolerance: float,
    raycast_function: RaycastFunction | None = None,
) -> DGeDiObservationValidationResult:
    """Validate dGeDi in both directions against real mask and depth.

    The self-aligned target mesh is rendered as a per-view baseline.  The
    opposite mesh is then transformed by dGeDi and rendered into the same
    camera.  A pose is published only when both cross renders pass.
    """
    pose = _rigid(
        relative_pose_query_from_reference,
        "dGeDi observation-validation pose",
    )
    inverse_pose = np.linalg.inv(pose)
    reference_points, reference_triangles = _load_mesh(reference_mesh_path)
    query_points, query_triangles = _load_mesh(query_mesh_path)

    reference_mask = np.asarray(reference_mask_bool, dtype=bool)
    query_mask = np.asarray(query_mask_bool, dtype=bool)
    reference_depth = np.asarray(reference_depth_m, dtype=np.float32)
    query_depth = np.asarray(query_depth_m, dtype=np.float32)
    reference_k = np.asarray(reference_camera_k, dtype=np.float64)
    query_k = np.asarray(query_camera_k, dtype=np.float64)
    raycast = raycast_function or _default_raycast_function()

    reference_baseline_mask, reference_baseline_depth = _render(
        points=reference_points,
        triangles=reference_triangles,
        camera_k=reference_k,
        image_shape=reference_mask.shape,
        raycast_function=raycast,
    )
    reference_cross_mask, reference_cross_depth = _render(
        points=_transform_points(query_points, inverse_pose),
        triangles=query_triangles,
        camera_k=reference_k,
        image_shape=reference_mask.shape,
        raycast_function=raycast,
    )
    query_baseline_mask, query_baseline_depth = _render(
        points=query_points,
        triangles=query_triangles,
        camera_k=query_k,
        image_shape=query_mask.shape,
        raycast_function=raycast,
    )
    query_cross_mask, query_cross_depth = _render(
        points=_transform_points(reference_points, pose),
        triangles=reference_triangles,
        camera_k=query_k,
        image_shape=query_mask.shape,
        raycast_function=raycast,
    )

    scoring_arguments = {
        "weights": weights,
        "depth_trim_quantile": depth_trim_quantile,
        "min_depth_overlap_pixels": minimum_depth_overlap_pixels,
        "free_space_absolute_tolerance_m": (
            free_space_absolute_tolerance_m
        ),
        "free_space_relative_tolerance": free_space_relative_tolerance,
    }
    reference_scale = _diameter(reference_points)
    query_scale = _diameter(query_points)
    reference_baseline = score_alignment(
        observed_mask=reference_mask,
        observed_depth_m=reference_depth,
        rendered_mask=reference_baseline_mask,
        rendered_depth_m=reference_baseline_depth,
        object_scale_m=reference_scale,
        **scoring_arguments,
    )
    reference_cross = score_alignment(
        observed_mask=reference_mask,
        observed_depth_m=reference_depth,
        rendered_mask=reference_cross_mask,
        rendered_depth_m=reference_cross_depth,
        object_scale_m=query_scale,
        **scoring_arguments,
    )
    query_baseline = score_alignment(
        observed_mask=query_mask,
        observed_depth_m=query_depth,
        rendered_mask=query_baseline_mask,
        rendered_depth_m=query_baseline_depth,
        object_scale_m=query_scale,
        **scoring_arguments,
    )
    query_cross = score_alignment(
        observed_mask=query_mask,
        observed_depth_m=query_depth,
        rendered_mask=query_cross_mask,
        rendered_depth_m=query_cross_depth,
        object_scale_m=reference_scale,
        **scoring_arguments,
    )

    reasons = _view_rejection_reasons(
        view_name="reference",
        baseline=reference_baseline,
        cross=reference_cross,
        minimum_depth_overlap_pixels=minimum_depth_overlap_pixels,
    )
    reasons.extend(
        _view_rejection_reasons(
            view_name="query",
            baseline=query_baseline,
            cross=query_cross,
            minimum_depth_overlap_pixels=minimum_depth_overlap_pixels,
        )
    )

    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reference_render_path = output_root / "reference_dgedi_render.npz"
    query_render_path = output_root / "query_dgedi_render.npz"
    np.savez_compressed(
        reference_render_path,
        baseline_mask=reference_baseline_mask.astype(np.uint8),
        baseline_depth_m=reference_baseline_depth,
        cross_mask=reference_cross_mask.astype(np.uint8),
        cross_depth_m=reference_cross_depth,
        observed_mask=reference_mask.astype(np.uint8),
        observed_depth_m=reference_depth,
    )
    np.savez_compressed(
        query_render_path,
        baseline_mask=query_baseline_mask.astype(np.uint8),
        baseline_depth_m=query_baseline_depth,
        cross_mask=query_cross_mask.astype(np.uint8),
        cross_depth_m=query_cross_depth,
        observed_mask=query_mask.astype(np.uint8),
        observed_depth_m=query_depth,
    )

    metrics = {
        "reference_baseline": _score_dict(reference_baseline),
        "reference_cross": _score_dict(reference_cross),
        "query_baseline": _score_dict(query_baseline),
        "query_cross": _score_dict(query_cross),
    }
    summary_path = output_root / "dgedi_observation_validation.json"
    payload = {
        "status": "CONSISTENT" if not reasons else "REJECT",
        "accepted": not reasons,
        "pose_convention": "T_query_camera_from_reference_camera",
        "relative_pose_query_from_reference": pose.tolist(),
        "policy": {
            "directions": ["query_to_reference", "reference_to_query"],
            "minimum_mask_iou": "max(0.50, self_baseline_iou - 0.20)",
            "maximum_normalized_depth_residual": (
                "max(0.15, self_baseline_depth_residual + 0.10)"
            ),
            "maximum_total_loss": (
                "max(0.45, self_baseline_total_loss + 0.20)"
            ),
            "minimum_depth_overlap_pixels": minimum_depth_overlap_pixels,
        },
        "reasons": reasons,
        "metrics": metrics,
        "artifacts": {
            "reference_render": str(reference_render_path),
            "query_render": str(query_render_path),
        },
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return DGeDiObservationValidationResult(
        accepted=not reasons,
        reasons=tuple(reasons),
        summary_path=summary_path,
        reference_render_path=reference_render_path,
        query_render_path=query_render_path,
        metrics=metrics,
    )
