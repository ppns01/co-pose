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


@dataclass(frozen=True)
class DGeDiFallbackDecision:
    selected_pose_query_from_reference: np.ndarray
    selected_method: str
    used_foundationpose_fallback: bool
    rotation_deviation_deg: float
    translation_deviation_m: float
    dgedi_mean_cross_loss: float
    foundationpose_mean_cross_loss: float
    absolute_loss_improvement: float
    relative_loss_improvement: float
    observation_improved_clearly: bool
    reason: str


def _mean_cross_loss(
    validation: DGeDiObservationValidationResult,
) -> float:
    return float(
        (
            float(validation.metrics["reference_cross"]["total_loss"])
            + float(validation.metrics["query_cross"]["total_loss"])
        )
        * 0.5
    )


def select_dgedi_with_foundationpose_fallback(
    *,
    dgedi_pose_query_from_reference: np.ndarray,
    foundationpose_pose_query_from_reference: np.ndarray,
    dgedi_validation: DGeDiObservationValidationResult,
    foundationpose_validation: DGeDiObservationValidationResult,
    maximum_rotation_deviation_deg: float = 5.0,
    maximum_translation_deviation_m: float = 0.020,
    minimum_absolute_loss_improvement: float = 0.005,
    minimum_relative_loss_improvement: float = 0.05,
) -> DGeDiFallbackDecision:
    """Fallback when a large dGeDi correction lacks clear observation gain."""
    dgedi_pose = _rigid(dgedi_pose_query_from_reference, "dGeDi pose")
    foundationpose_pose = _rigid(
        foundationpose_pose_query_from_reference,
        "FoundationPose-only pose",
    )
    rotation_delta = (
        dgedi_pose[:3, :3] @ foundationpose_pose[:3, :3].T
    )
    cosine = float(
        np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0)
    )
    rotation_deviation_deg = float(np.degrees(np.arccos(cosine)))
    translation_deviation_m = float(
        np.linalg.norm(
            dgedi_pose[:3, 3] - foundationpose_pose[:3, 3]
        )
    )
    dgedi_loss = _mean_cross_loss(dgedi_validation)
    foundationpose_loss = _mean_cross_loss(foundationpose_validation)
    absolute_improvement = foundationpose_loss - dgedi_loss
    relative_improvement = absolute_improvement / max(
        foundationpose_loss,
        1e-12,
    )
    improved_clearly = (
        dgedi_validation.accepted
        and absolute_improvement >= minimum_absolute_loss_improvement
        and relative_improvement >= minimum_relative_loss_improvement
    )
    large_deviation = (
        rotation_deviation_deg >= maximum_rotation_deviation_deg
        or translation_deviation_m >= maximum_translation_deviation_m
    )
    use_fallback = large_deviation and not improved_clearly
    if use_fallback:
        reason = (
            "dGeDi deviated beyond the 5deg/2cm trust region without "
            "an accepted, clear bidirectional observation-loss improvement"
        )
        selected_pose = foundationpose_pose
        selected_method = "foundationpose_only_fallback"
    else:
        reason = (
            "dGeDi stayed within the trust region or clearly improved "
            "bidirectional observation loss"
        )
        selected_pose = dgedi_pose
        selected_method = "dgedi_visible_surface"

    return DGeDiFallbackDecision(
        selected_pose_query_from_reference=selected_pose.copy(),
        selected_method=selected_method,
        used_foundationpose_fallback=use_fallback,
        rotation_deviation_deg=rotation_deviation_deg,
        translation_deviation_m=translation_deviation_m,
        dgedi_mean_cross_loss=dgedi_loss,
        foundationpose_mean_cross_loss=foundationpose_loss,
        absolute_loss_improvement=absolute_improvement,
        relative_loss_improvement=relative_improvement,
        observation_improved_clearly=improved_clearly,
        reason=reason,
    )


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
    minimum_mask_iou: float = 0.50,
    baseline_mask_iou_drop: float = 0.20,
    maximum_depth_residual_normalized: float = 0.15,
    baseline_depth_residual_margin: float = 0.10,
    maximum_total_loss: float = 0.45,
    baseline_total_loss_margin: float = 0.20,
) -> list[str]:
    reasons: list[str] = []
    minimum_iou = max(
        minimum_mask_iou,
        baseline.mask_iou - baseline_mask_iou_drop,
    )
    maximum_depth_ratio = max(
        maximum_depth_residual_normalized,
        (
            baseline.depth_residual_normalized
            + baseline_depth_residual_margin
        ),
    )
    allowed_total_loss = max(
        maximum_total_loss,
        baseline.total_loss + baseline_total_loss_margin,
    )

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
    if cross.total_loss > allowed_total_loss:
        reasons.append(
            f"{view_name}: alignment loss too large "
            f"({cross.total_loss:.3f} > {allowed_total_loss:.3f})"
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
    minimum_mask_iou: float = 0.50,
    baseline_mask_iou_drop: float = 0.20,
    maximum_depth_residual_normalized: float = 0.15,
    baseline_depth_residual_margin: float = 0.10,
    maximum_total_loss: float = 0.45,
    baseline_total_loss_margin: float = 0.20,
    object_scale_m: float | None = None,
    diagnostic_only: bool = False,
    raycast_function: RaycastFunction | None = None,
) -> DGeDiObservationValidationResult:
    """Validate dGeDi in both directions against real mask and depth.

    The self-aligned target mesh is rendered as a per-view baseline.  The
    opposite mesh is then transformed by dGeDi and rendered into the same
    camera. In diagnostic_only mode the legacy thresholds are reported but
    never used by the caller as a publication gate.
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
    if object_scale_m is not None and (
        not np.isfinite(object_scale_m) or object_scale_m <= 0.0
    ):
        raise ValueError("object_scale_m must be finite and positive")
    reference_scale = (
        float(object_scale_m)
        if object_scale_m is not None
        else _diameter(reference_points)
    )
    query_scale = (
        float(object_scale_m)
        if object_scale_m is not None
        else _diameter(query_points)
    )
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
        minimum_mask_iou=minimum_mask_iou,
        baseline_mask_iou_drop=baseline_mask_iou_drop,
        maximum_depth_residual_normalized=(
            maximum_depth_residual_normalized
        ),
        baseline_depth_residual_margin=(
            baseline_depth_residual_margin
        ),
        maximum_total_loss=maximum_total_loss,
        baseline_total_loss_margin=baseline_total_loss_margin,
    )
    reasons.extend(
        _view_rejection_reasons(
            view_name="query",
            baseline=query_baseline,
            cross=query_cross,
            minimum_depth_overlap_pixels=minimum_depth_overlap_pixels,
            minimum_mask_iou=minimum_mask_iou,
            baseline_mask_iou_drop=baseline_mask_iou_drop,
            maximum_depth_residual_normalized=(
                maximum_depth_residual_normalized
            ),
            baseline_depth_residual_margin=(
                baseline_depth_residual_margin
            ),
            maximum_total_loss=maximum_total_loss,
            baseline_total_loss_margin=baseline_total_loss_margin,
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
        "status": (
            "EVALUATED"
            if diagnostic_only
            else ("CONSISTENT" if not reasons else "REJECT")
        ),
        "accepted": not reasons,
        "decision_mode": (
            "diagnostic_only" if diagnostic_only else "legacy_hard_gate"
        ),
        "legacy_gate_would_accept": not reasons,
        "pose_convention": "T_query_camera_from_reference_camera",
        "relative_pose_query_from_reference": pose.tolist(),
        "policy": {
            "directions": ["query_to_reference", "reference_to_query"],
            "minimum_mask_iou": minimum_mask_iou,
            "baseline_mask_iou_drop": baseline_mask_iou_drop,
            "maximum_normalized_depth_residual": (
                maximum_depth_residual_normalized
            ),
            "baseline_depth_residual_margin": (
                baseline_depth_residual_margin
            ),
            "maximum_total_loss": maximum_total_loss,
            "baseline_total_loss_margin": baseline_total_loss_margin,
            "minimum_depth_overlap_pixels": minimum_depth_overlap_pixels,
            "fixed_object_scale_m": object_scale_m,
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
