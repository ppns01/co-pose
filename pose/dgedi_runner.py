from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DGeDiRegistrationResult:
    proxy_pose_query_from_reference: np.ndarray
    relative_pose_query_from_reference: np.ndarray
    proxy_pose_path: Path
    relative_pose_path: Path
    metadata_path: Path
    reference_self_aligned_mesh_path: Path
    query_self_aligned_mesh_path: Path
    reference_registration_cloud_path: Path
    query_registration_cloud_path: Path


class DGeDiSemanticFailure(RuntimeError):
    """Registration lacks usable geometric evidence; infrastructure is OK."""


def _worker_failure_error_type(
    failure_text: str,
) -> type[RuntimeError]:
    """Separate geometric degeneracy from operational worker failures."""
    semantic_markers = (
        "fewer than 3 correspondences",
        "Too few correspondences",
        "Too few points",
    )
    if any(marker in failure_text for marker in semantic_markers):
        return DGeDiSemanticFailure
    return RuntimeError


def _rigid(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)

    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError(
            f"Invalid {name}: shape={matrix.shape}"
        )

    if not np.allclose(
        matrix[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(
            f"Invalid homogeneous row in {name}"
        )

    rotation = matrix[:3, :3]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=5e-3,
        rtol=0.0,
    ):
        raise ValueError(
            f"Non-orthonormal rotation in {name}"
        )

    if np.linalg.det(rotation) <= 0.0:
        raise ValueError(
            f"Invalid rotation determinant in {name}"
        )

    return matrix


def _save_self_aligned_mesh(
    *,
    source_mesh_path: Path,
    pose_camera_from_proxy: Any,
    output_mesh_path: Path,
) -> Path:
    """
    FoundationPose self pose를 mesh vertex에 직접 적용하고
    camera 좌표계의 새 mesh 파일로 저장한다.

    output vertex:
        p_camera = T_camera_from_proxy @ p_proxy
    """
    import open3d as o3d

    source_mesh_path = (
        Path(source_mesh_path)
        .expanduser()
        .resolve()
    )

    output_mesh_path = (
        Path(output_mesh_path)
        .expanduser()
        .resolve()
    )

    if not source_mesh_path.is_file():
        raise FileNotFoundError(
            f"Source mesh not found: "
            f"{source_mesh_path}"
        )

    pose = _rigid(
        pose_camera_from_proxy,
        "FoundationPose self pose",
    )

    mesh = o3d.io.read_triangle_mesh(
        str(source_mesh_path),
        enable_post_processing=True,
    )

    if len(mesh.vertices) == 0:
        raise ValueError(
            f"Mesh has no vertices: "
            f"{source_mesh_path}"
        )

    # FoundationPose self pose를 실제 mesh에 bake한다.
    mesh.transform(pose)

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    output_mesh_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = o3d.io.write_triangle_mesh(
        str(output_mesh_path),
        mesh,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )

    if not saved:
        raise IOError(
            "Failed to save self-aligned mesh: "
            f"{output_mesh_path}"
        )

    if not output_mesh_path.is_file():
        raise FileNotFoundError(
            "Self-aligned mesh was not created: "
            f"{output_mesh_path}"
        )

    return output_mesh_path


def _save_depth_consistent_proxy_surface_cloud(
    *,
    local_mesh_path: Path,
    pose_camera_from_proxy: Any,
    camera_matrix: Any,
    observed_mask_bool: Any,
    observed_depth_m: Any,
    output_cloud_path: Path,
    sample_count: int,
    maximum_depth_residual_m: float = 0.010,
    minimum_consistent_pixels: int = 256,
) -> tuple[Path, Path]:
    """Save final-proxy ray hits that agree with masked observed depth.

    Visibility and depth consistency are evaluated in the camera frame, but
    the saved points remain in the final proxy-local frame so dGeDi estimates
    G=T_Pq_from_Pr and the existing H=B@G@inv(A) composition stays valid.
    """
    import open3d as o3d

    source_path = Path(local_mesh_path).expanduser().resolve()
    output_path = Path(output_cloud_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Final proxy mesh not found: {source_path}")
    if sample_count < 256:
        raise ValueError("sample_count must be at least 256")
    if maximum_depth_residual_m <= 0.0:
        raise ValueError("maximum_depth_residual_m must be positive")
    if minimum_consistent_pixels < 256:
        raise ValueError("minimum_consistent_pixels must be at least 256")

    pose = _rigid(pose_camera_from_proxy, "camera-from-proxy pose")
    intrinsic = np.asarray(camera_matrix, dtype=np.float64)
    mask = np.asarray(observed_mask_bool, dtype=bool)
    depth = np.asarray(observed_depth_m, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("camera_matrix must be a finite (3,3) matrix")
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("observed mask and depth must share shape (H,W)")
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    mesh = o3d.io.read_triangle_mesh(
        str(source_path),
        enable_post_processing=True,
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError(f"Final proxy mesh is empty: {source_path}")

    # Raycasting is done after applying the final FoundationPose self pose.
    # The source mesh itself remains untouched in its final S*/Sxyz local frame.
    camera_mesh = o3d.geometry.TriangleMesh(mesh)
    camera_vertices = (
        vertices @ pose[:3, :3].T
        + pose[:3, 3][None, :]
    )
    camera_mesh.vertices = o3d.utility.Vector3dVector(camera_vertices)

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(camera_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    valid_observation = mask & np.isfinite(depth) & (depth > 0.0)
    pixel_v, pixel_u = np.nonzero(valid_observation)
    valid_observed_pixel_count = int(len(pixel_u))
    if valid_observed_pixel_count < minimum_consistent_pixels:
        raise DGeDiSemanticFailure(
            "Too few valid masked-depth pixels before proxy raycasting: "
            f"{valid_observed_pixel_count} < {minimum_consistent_pixels}"
        )

    ray_direction = np.stack(
        (
            (pixel_u.astype(np.float64) - cx) / fx,
            (pixel_v.astype(np.float64) - cy) / fy,
            np.ones_like(pixel_u, dtype=np.float64),
        ),
        axis=-1,
    )
    rays = np.concatenate(
        (np.zeros_like(ray_direction), ray_direction),
        axis=-1,
    ).astype(np.float32)
    raycast = scene.cast_rays(
        o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    )
    rendered_depth = raycast["t_hit"].numpy().astype(np.float64)
    primitive_ids = raycast["primitive_ids"].numpy().astype(np.int64)
    valid_hit = (
        np.isfinite(rendered_depth)
        & (rendered_depth > 0.0)
        & (primitive_ids >= 0)
        & (primitive_ids < len(triangles))
    )
    observed_depth = depth[pixel_v, pixel_u]
    depth_residual = np.full(
        rendered_depth.shape,
        np.inf,
        dtype=np.float64,
    )
    depth_residual[valid_hit] = np.abs(
        rendered_depth[valid_hit] - observed_depth[valid_hit]
    )
    consistent = valid_hit & (
        depth_residual <= float(maximum_depth_residual_m)
    )
    consistent_pixel_count = int(np.count_nonzero(consistent))
    if consistent_pixel_count < minimum_consistent_pixels:
        raise DGeDiSemanticFailure(
            "Too few depth-consistent visible pixels: "
            f"{consistent_pixel_count} < {minimum_consistent_pixels}"
        )

    # With direction_z=1, t_hit is camera Z rather than Euclidean ray range.
    # Therefore these are the exact first surface hits corresponding to the
    # rendered depth comparison above, not uniformly sampled triangle points.
    camera_hit_points = (
        ray_direction[consistent]
        * rendered_depth[consistent, None]
    )

    # Row-vector inverse of p_camera = p_proxy @ R.T + t.
    proxy_hit_points = (
        camera_hit_points - pose[:3, 3][None, :]
    ) @ pose[:3, :3]

    # Ensure deterministic downsampling: Open3D's farthest_point_down_sample
    # uses random initialization, so we must seed before each call to guarantee
    # reproducible registration clouds from the same depth-consistent points.
    o3d.utility.random.seed(0)

    registration_cloud = o3d.geometry.PointCloud()
    registration_cloud.points = o3d.utility.Vector3dVector(proxy_hit_points)
    registration_cloud = registration_cloud.remove_non_finite_points()
    registration_cloud = registration_cloud.remove_duplicated_points()
    unique_point_count = int(len(registration_cloud.points))
    if unique_point_count > sample_count:
        registration_cloud = registration_cloud.farthest_point_down_sample(
            sample_count
        )
    registration_cloud = registration_cloud.remove_non_finite_points()
    registration_cloud = registration_cloud.remove_duplicated_points()
    saved_point_count = int(len(registration_cloud.points))
    if saved_point_count < minimum_consistent_pixels:
        raise DGeDiSemanticFailure(
            "Depth-consistent proxy surface cloud contains too few unique "
            f"points: {saved_point_count} < {minimum_consistent_pixels}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(
        str(output_path),
        registration_cloud,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    ):
        raise IOError(
            "Failed to save depth-consistent proxy surface cloud: "
            f"{output_path}"
        )

    finite_residuals = depth_residual[consistent]
    saved_points = np.asarray(registration_cloud.points, dtype=np.float64)
    diameter_m = _diameter(saved_points)
    if not np.isfinite(diameter_m) or diameter_m <= 0.0:
        raise ValueError(
            "Depth-consistent proxy surface cloud has invalid diameter: "
            f"{diameter_m}"
        )

    diagnostics_path = output_path.with_suffix(".json")
    diagnostics_path.write_text(
        json.dumps(
            {
                "point_source": (
                    "final_proxy_first_ray_hits_consistent_with_"
                    "observed_mask_and_depth"
                ),
                "source_final_proxy_mesh": str(source_path),
                "registration_cloud": str(output_path),
                "coordinate_frame": "proxy_local",
                "visibility_frame": "camera",
                "sample_count_requested": int(sample_count),
                "sample_count_saved": saved_point_count,
                "point_count_saved": saved_point_count,
                "point_count_unique_before_downsample": unique_point_count,
                "minimum_consistent_pixels": int(
                    minimum_consistent_pixels
                ),
                "maximum_depth_residual_m": float(
                    maximum_depth_residual_m
                ),
                "observed_mask_pixel_count": int(np.count_nonzero(mask)),
                "valid_observed_depth_pixel_count": (
                    valid_observed_pixel_count
                ),
                "rendered_first_hit_pixel_count": int(
                    np.count_nonzero(valid_hit)
                ),
                "consistent_pixel_count": consistent_pixel_count,
                "consistent_fraction_of_valid_observation": float(
                    consistent_pixel_count
                    / max(valid_observed_pixel_count, 1)
                ),
                "consistent_depth_residual_mean_m": float(
                    np.mean(finite_residuals)
                ),
                "consistent_depth_residual_median_m": float(
                    np.median(finite_residuals)
                ),
                "consistent_depth_residual_max_m": float(
                    np.max(finite_residuals)
                ),
                "proxy_local_axis_extent_m": np.ptp(
                    saved_points,
                    axis=0,
                ).tolist(),
                "diameter_m": float(diameter_m),
                "pose_camera_from_proxy": pose.tolist(),
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path, diagnostics_path


def _save_registration_cloud_pair_quality(
    *,
    reference_diagnostics_path: Path,
    query_diagnostics_path: Path,
    output_path: Path,
    minimum_point_count_ratio: float = 0.10,
    minimum_diameter_ratio: float = 0.10,
) -> tuple[Path, dict[str, Any]]:
    """Record gross input imbalance without stopping dGeDi registration.

    This gate cannot establish actual cross-view surface overlap because G is
    not known yet. It only catches degenerate point-count or physical-extent
    imbalance; the existing render-based validator remains the overlap check.
    """
    if not 0.0 < minimum_point_count_ratio <= 1.0:
        raise ValueError(
            "minimum_point_count_ratio must be in (0,1]"
        )
    if not 0.0 < minimum_diameter_ratio <= 1.0:
        raise ValueError(
            "minimum_diameter_ratio must be in (0,1]"
        )

    reference_payload = json.loads(
        Path(reference_diagnostics_path).read_text(
            encoding="utf-8"
        )
    )
    query_payload = json.loads(
        Path(query_diagnostics_path).read_text(
            encoding="utf-8"
        )
    )

    reference_point_count = int(
        reference_payload["point_count_saved"]
    )
    query_point_count = int(
        query_payload["point_count_saved"]
    )
    reference_diameter_m = float(
        reference_payload["diameter_m"]
    )
    query_diameter_m = float(
        query_payload["diameter_m"]
    )

    point_count_ratio = (
        min(reference_point_count, query_point_count)
        / max(reference_point_count, query_point_count)
    )
    diameter_ratio = (
        min(reference_diameter_m, query_diameter_m)
        / max(reference_diameter_m, query_diameter_m)
    )

    reasons: list[str] = []
    if point_count_ratio < minimum_point_count_ratio:
        reasons.append(
            "point_count_ratio_below_threshold"
        )
    if diameter_ratio < minimum_diameter_ratio:
        reasons.append(
            "diameter_ratio_below_threshold"
        )

    diagnostics = {
        "status": "EVALUATED",
        "accepted": not reasons,
        "meets_recommended_minimum": not reasons,
        "reasons": reasons,
        "policy": {
            "minimum_point_count_ratio": float(
                minimum_point_count_ratio
            ),
            "minimum_diameter_ratio": float(
                minimum_diameter_ratio
            ),
        },
        "metrics": {
            "reference_point_count": int(reference_point_count),
            "query_point_count": int(query_point_count),
            "point_count_ratio": float(point_count_ratio),
            "reference_diameter_m": float(
                reference_diameter_m
            ),
            "query_diameter_m": float(query_diameter_m),
            "diameter_ratio": float(diameter_ratio),
        },
        "reference_diagnostics_path": str(
            Path(reference_diagnostics_path).expanduser().resolve()
        ),
        "query_diagnostics_path": str(
            Path(query_diagnostics_path).expanduser().resolve()
        ),
    }

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            diagnostics,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path, diagnostics


def compose_dgedi_relative_pose(
    *,
    reference_pose_camera_from_proxy: Any,
    query_pose_camera_from_proxy: Any,
    proxy_pose_query_from_reference: Any,
) -> np.ndarray:
    reference_self = _rigid(
        reference_pose_camera_from_proxy,
        "reference self pose",
    )
    query_self = _rigid(
        query_pose_camera_from_proxy,
        "query self pose",
    )
    proxy_pose = _rigid(
        proxy_pose_query_from_reference,
        "dGeDi proxy pose",
    )

    # T_Cq_from_Cr =
    # T_Cq_from_Pq
    # @ T_Pq_from_Pr
    # @ inv(T_Cr_from_Pr)
    return _rigid(
        query_self
        @ proxy_pose
        @ np.linalg.inv(reference_self),
        "relative pose",
    )


def _mesh_to_cloud(
    path: Path,
    count: int,
    o3d: Any,
) -> Any:
    mesh = o3d.io.read_triangle_mesh(
        str(path),
        enable_post_processing=True,
    )

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    if len(vertices) == 0:
        cloud = o3d.io.read_point_cloud(
            str(path),
            remove_nan_points=True,
            remove_infinite_points=True,
        )
    else:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(vertices)

    if len(cloud.points) == 0:
        raise ValueError(f"Geometry has no points: {path}")

    # Ensure deterministic sampling operations
    o3d.utility.random.seed(0)

    if len(cloud.points) > count:
        cloud = (
            cloud.farthest_point_down_sample(
                count
            )
        )

    elif (
        len(cloud.points) < count
        and len(mesh.triangles) > 0
    ):
        cloud = mesh.sample_points_uniformly(
            number_of_points=count
        )

    cloud = cloud.remove_non_finite_points()
    cloud = cloud.remove_duplicated_points()

    if len(cloud.points) < 256:
        raise ValueError(
            f"Too few points: {len(cloud.points)}"
        )

    return cloud


def _diameter(
    points: np.ndarray,
    block: int = 256,
    maximum_points: int = 6000,
) -> float:
    """
    Point cloud diameter를 근사 계산한다.

    dGeDi descriptor와 registration에는 전체 점을 사용하지만,
    O(N^2)인 diameter 계산에는 최대 maximum_points만 사용한다.
    각 축의 최소·최대점은 항상 포함하여 물체 크기 손실을 줄인다.
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 2
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "Invalid points for diameter: "
            f"shape={points.shape}"
        )

    original_count = len(points)

    if original_count > maximum_points:
        extreme_indices = np.unique(
            np.concatenate(
                [
                    np.argmin(
                        points,
                        axis=0,
                    ),
                    np.argmax(
                        points,
                        axis=0,
                    ),
                ]
            )
        ).astype(
            np.int64,
            copy=False,
        )

        remaining_count = (
            maximum_points
            - len(extreme_indices)
        )

        all_indices = np.arange(
            original_count,
            dtype=np.int64,
        )

        available_mask = np.ones(
            original_count,
            dtype=bool,
        )

        available_mask[
            extreme_indices
        ] = False

        available_indices = (
            all_indices[available_mask]
        )

        rng = np.random.default_rng(0)

        sampled_indices = rng.choice(
            available_indices,
            size=remaining_count,
            replace=False,
        )

        selected_indices = np.concatenate(
            [
                extreme_indices,
                sampled_indices,
            ]
        )

        points = points[
            selected_indices
        ]

    maximum_squared = 0.0

    for start in range(
        0,
        len(points),
        block,
    ):
        current = points[
            start : start + block
        ]

        delta = (
            current[:, None, :]
            - points[None, :, :]
        )

        squared = np.einsum(
            "ijk,ijk->ij",
            delta,
            delta,
            optimize=True,
        )

        maximum_squared = max(
            maximum_squared,
            float(squared.max()),
        )

    diameter = float(
        np.sqrt(maximum_squared)
    )

    if (
        not np.isfinite(diameter)
        or diameter <= 0.0
    ):
        raise ValueError(
            f"Invalid diameter: {diameter}"
        )

    print(
        "[dGeDi diameter] "
        f"source_points={original_count}, "
        f"diameter_points={len(points)}, "
        f"diameter_m={diameter:.9f}"
    )

    return diameter


def _normalize(
    cloud: Any,
    diameter: float,
    o3d: Any,
) -> tuple[Any, np.ndarray]:
    points = np.asarray(
        cloud.points,
        dtype=np.float64,
    )

    center = points.mean(axis=0)

    normalized = (
        o3d.geometry.PointCloud()
    )

    normalized.points = (
        o3d.utility.Vector3dVector(
            (points - center) / diameter
        )
    )

    return normalized, center


def _restore_transform(
    transform: Any,
    source_center: np.ndarray,
    target_center: np.ndarray,
    diameter: float,
) -> np.ndarray:
    transform = np.asarray(
        transform,
        dtype=np.float64,
    )

    rotation = transform[:3, :3]

    restored = np.eye(
        4,
        dtype=np.float64,
    )

    restored[:3, :3] = rotation

    restored[:3, 3] = (
        target_center
        - rotation @ source_center
        + diameter * transform[:3, 3]
    )

    return _rigid(
        restored,
        "restored dGeDi pose",
    )


def _symmetric_chamfer(
    source_cloud: Any,
    target_cloud: Any,
    transform: Any,
    o3d: Any,
) -> float:
    """Untruncated symmetric chamfer distance, in the clouds' own units.

    ICP fitness and inlier RMSE only see correspondences closer than
    icp_threshold, which is 3% of the object diameter. Every point that a
    wrong rotation pushes beyond that contributes nothing to either number
    -- and that is exactly where the evidence separating a wrong rotation
    from the right one lives. Measured on this pipeline's own Duck output,
    ICP fitness ranked a 43-degree pose first out of 402 candidates while
    this score ranked a 6-degree pose first over the same pool.
    """

    moved = o3d.geometry.PointCloud(source_cloud)
    moved.transform(np.asarray(transform, dtype=np.float64))

    forward = np.asarray(
        moved.compute_point_cloud_distance(target_cloud),
        dtype=np.float64,
    )

    backward = np.asarray(
        target_cloud.compute_point_cloud_distance(moved),
        dtype=np.float64,
    )

    if len(forward) == 0 or len(backward) == 0:
        raise ValueError(
            "Empty point cloud during chamfer scoring."
        )

    return float(forward.mean() + backward.mean())


def _register_candidates(
    *,
    reference_norm: Any,
    reference_features: Any,
    query_norm: Any,
    query_features: Any,
    ransac_threshold: float,
    icp_threshold: float,
    candidate_count: int,
    candidate_diagnostics: bool,
    register_one: Any,
    o3d: Any,
) -> list[dict[str, Any]]:
    """Build a pose candidate pool and score every member by chamfer.

    registration_ransac_based_on_feature_matching() does not expose a seed
    argument and draws from open3d's global RNG, so re-seeding before each
    call yields a genuinely different hypothesis -- the same property that
    forced seed(0) here in the first place, used deliberately instead of
    being suppressed. The seed list is fixed, so the pool is reproducible.

    Candidates whose RANSAC or ICP stage degenerates are skipped rather
    than raising: with a pool, one degenerate draw is not a failure. The
    caller raises if nothing survives.
    """

    candidates: list[dict[str, Any]] = []

    for seed in range(candidate_count):
        o3d.utility.random.seed(seed)

        ransac, icp = register_one(
            reference_norm,
            reference_features,
            query_norm,
            query_features,
            ransac_threshold,
            icp_threshold,
        )

        if len(ransac.correspondence_set) < 3:
            continue

        if len(icp.correspondence_set) < 3:
            continue

        candidate = {
            "seed": seed,
            "ransac": ransac,
            "icp": icp,
            "chamfer": _symmetric_chamfer(
                reference_norm,
                query_norm,
                icp.transformation,
                o3d,
            ),
        }
        if candidate_diagnostics:
            candidate["ransac_chamfer"] = _symmetric_chamfer(
                reference_norm,
                query_norm,
                ransac.transformation,
                o3d,
            )
        candidates.append(candidate)

    return candidates


def _worker(
    args: argparse.Namespace,
) -> int:
    repository = (
        args.repository
        .expanduser()
        .resolve()
    )

    config_path = (
        args.config
        .expanduser()
        .resolve()
    )

    reference_mesh = (
        args.reference_mesh
        .expanduser()
        .resolve()
    )

    query_mesh = (
        args.query_mesh
        .expanduser()
        .resolve()
    )

    output = (
        args.output_directory
        .expanduser()
        .resolve()
    )

    if not repository.is_dir():
        raise FileNotFoundError(
            f"dGeDi repository: {repository}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"dGeDi config: {config_path}"
        )

    if not reference_mesh.is_file():
        raise FileNotFoundError(
            f"Reference mesh: {reference_mesh}"
        )

    if not query_mesh.is_file():
        raise FileNotFoundError(
            f"Query mesh: {query_mesh}"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    sys.path.insert(
        0,
        str(repository),
    )

    import open3d as o3d
    import torch

    # registration_ransac_based_on_feature_matching()은 seed 인자를 노출하지
    # 않고 open3d의 전역 RNG를 그대로 쓴다. 고정하지 않으면 완전히 동일한
    # mesh 쌍에도 실행마다 다른 RANSAC 결과가 나온다 -- 실측 결과 동일 mesh
    # 반복 실행에서 rotation이 5도, translation이 8.7cm까지 흔들렸다.
    # 그 변동성을 억누르는 대신 후보 생성에 쓴다: _register_candidates가
    # 매 호출 전에 고정된 seed 목록으로 다시 seeding하므로 풀 전체는
    # 재현 가능하다. 여기 seed(0)은 feature 추출 등 그 이전 단계를 위한
    # 기준점이다.
    o3d.utility.random.seed(0)

    from core.dgedi_distilled import (
        dgedi,
    )
    from utils import (
        extract_features,
        load_yaml_config,
        register_one,
    )

    config = load_yaml_config(
        str(config_path)
    )[args.mode]

    model_config = dict(
        config["model_config"]
    )

    weights = Path(
        config["weights_path"]
    )

    if not weights.is_absolute():
        weights = repository / weights

    weights = weights.resolve()

    if not weights.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights}"
        )

    model_config["weights_path"] = (
        str(weights)
    )

    device = torch.device(
        args.device
    )

    model = dgedi(
        {
            "query": model_config,
            "target": model_config,
            "device": args.device,
        }
    )

    # source = reference proxy
    # target = query proxy
    reference_metric = _mesh_to_cloud(
        reference_mesh,
        args.sample_count,
        o3d,
    )

    query_metric = _mesh_to_cloud(
        query_mesh,
        args.sample_count,
        o3d,
    )

    diameter_m = _diameter(
        np.asarray(
            reference_metric.points
        )
    )

    if (
        not np.isfinite(diameter_m)
        or diameter_m <= 0.0
    ):
        raise ValueError(
            f"Invalid diameter: {diameter_m}"
        )

    # 두 proxy mesh 모두 meter 단위이므로
    # 공식 demo.py의 target * 1000은 적용하지 않는다.
    reference_norm, reference_center = (
        _normalize(
            reference_metric,
            diameter_m,
            o3d,
        )
    )

    query_norm, query_center = (
        _normalize(
            query_metric,
            diameter_m,
            o3d,
        )
    )

    reference_features = (
        extract_features(
            reference_norm,
            model,
            device,
        )
    )

    query_features = (
        extract_features(
            query_norm,
            model,
            device,
        )
    )

    candidates = _register_candidates(
        reference_norm=reference_norm,
        reference_features=reference_features,
        query_norm=query_norm,
        query_features=query_features,
        ransac_threshold=args.ransac_threshold,
        icp_threshold=args.icp_threshold,
        candidate_count=(
            args.registration_candidate_count
        ),
        candidate_diagnostics=(
            args.candidate_diagnostics
        ),
        register_one=register_one,
        o3d=o3d,
    )

    if not candidates:
        raise RuntimeError(
            "Every dGeDi candidate found fewer "
            "than 3 correspondences."
        )

    # Selection is by chamfer, never by ICP fitness or inlier RMSE: those
    # are truncated at icp_threshold and demonstrably rank wrong rotations
    # first on this data. See _symmetric_chamfer.
    selected = min(
        candidates,
        key=lambda candidate: candidate["chamfer"],
    )

    ransac = selected["ransac"]
    icp = selected["icp"]

    print(
        "[dGeDi candidates] "
        f"count={len(candidates)}/"
        f"{args.registration_candidate_count}, "
        f"selected_seed={selected['seed']}, "
        f"chamfer_m="
        f"{selected['chamfer'] * diameter_m:.6f}"
    )

    ransac_pose = _restore_transform(
        ransac.transformation,
        reference_center,
        query_center,
        diameter_m,
    )

    final_pose = _restore_transform(
        icp.transformation,
        reference_center,
        query_center,
        diameter_m,
    )

    if args.candidate_diagnostics:
        # Diagnostic-only candidate provenance. Selection remains exactly the
        # same (minimum ICP-refined symmetric chamfer); these records let an
        # offline experiment separate search failure from objective failure.
        candidate_records: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_ransac_pose = _restore_transform(
                candidate["ransac"].transformation,
                reference_center,
                query_center,
                diameter_m,
            )
            candidate_icp_pose = _restore_transform(
                candidate["icp"].transformation,
                reference_center,
                query_center,
                diameter_m,
            )
            candidate_records.append(
                {
                    "seed": int(candidate["seed"]),
                    "ransac_chamfer_m": float(
                        candidate["ransac_chamfer"] * diameter_m
                    ),
                    "icp_chamfer_m": float(
                        candidate["chamfer"] * diameter_m
                    ),
                    # Backward-compatible alias for the selection score.
                    "chamfer_m": float(
                        candidate["chamfer"] * diameter_m
                    ),
                    "ransac_fitness": float(candidate["ransac"].fitness),
                    "ransac_inlier_rmse_m": float(
                        candidate["ransac"].inlier_rmse * diameter_m
                    ),
                    "ransac_correspondence_count": int(
                        len(candidate["ransac"].correspondence_set)
                    ),
                    "ransac_pose": candidate_ransac_pose.tolist(),
                    "icp_fitness": float(candidate["icp"].fitness),
                    "icp_inlier_rmse_m": float(
                        candidate["icp"].inlier_rmse * diameter_m
                    ),
                    "icp_correspondence_count": int(
                        len(candidate["icp"].correspondence_set)
                    ),
                    "icp_pose": candidate_icp_pose.tolist(),
                }
            )
    else:
        candidate_records = [
            {
                "seed": int(candidate["seed"]),
                "chamfer_m": float(candidate["chamfer"] * diameter_m),
                "icp_fitness": float(candidate["icp"].fitness),
                "icp_inlier_rmse_m": float(
                    candidate["icp"].inlier_rmse * diameter_m
                ),
            }
            for candidate in candidates
        ]

    ransac_path = (
        output
        / (
            "dgedi_ransac_"
            "proxy_pose_query_from_reference.npy"
        )
    )

    final_path = (
        output
        / (
            "dgedi_proxy_pose_"
            "query_from_reference.npy"
        )
    )

    metadata_path = (
        output
        / "dgedi_registration.json"
    )

    np.save(
        ransac_path,
        ransac_pose,
        allow_pickle=False,
    )

    np.save(
        final_path,
        final_pose,
        allow_pickle=False,
    )

    metadata = {
        "status": "completed",
        "backend": "dgedi",
        "pose_convention": (
            "T_target_geometry_from_source_geometry"
        ),
        "input_geometries": (
            "depth-consistent final-proxy surface point clouds in "
            "independent proxy-local frames"
        ),
        "translation_unit": "meter",
        "reference_mesh": (
            str(reference_mesh)
        ),
        "query_mesh": str(query_mesh),
        "normalization_diameter_m": (
            diameter_m
        ),
        "ransac": {
            "fitness": float(
                ransac.fitness
            ),
            "inlier_rmse_m": float(
                ransac.inlier_rmse
                * diameter_m
            ),
            "correspondence_count": (
                len(
                    ransac
                    .correspondence_set
                )
            ),
            "pose": ransac_pose.tolist(),
        },
        "icp": {
            "fitness": float(
                icp.fitness
            ),
            "inlier_rmse_m": float(
                icp.inlier_rmse
                * diameter_m
            ),
            "correspondence_count": (
                len(
                    icp
                    .correspondence_set
                )
            ),
            "pose": final_pose.tolist(),
        },
        "candidate_selection": {
            "candidate_diagnostics_enabled": bool(
                args.candidate_diagnostics
            ),
            "criterion": (
                "untruncated symmetric chamfer over an "
                "ICP-refined RANSAC candidate pool"
            ),
            "requested_count": int(
                args.registration_candidate_count
            ),
            "usable_count": len(candidates),
            "selected_seed": int(
                selected["seed"]
            ),
            "selected_chamfer_m": float(
                selected["chamfer"] * diameter_m
            ),
            "candidates": candidate_records,
        },
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[dGeDi proxy pose] "
        f"{final_path}"
    )

    print(final_pose)

    return 0


def run_dgedi_registration(
    *,
    repository_path: Path,
    python_executable: Path,
    config_path: Path,
    reference_self_alignment: Any,
    query_self_alignment: Any,
    reference_camera_matrix: Any,
    query_camera_matrix: Any,
    reference_mask_bool: Any,
    query_mask_bool: Any,
    reference_depth_m: Any,
    query_depth_m: Any,
    output_directory: Path,
    mode: str = "multi_scale",
    device: str = "cuda",
    sample_count: int = 30000,
    ransac_threshold: float = 0.03,
    icp_threshold: float = 0.03,
    registration_candidate_count: int = 32,
    maximum_surface_depth_residual_m: float = 0.010,
    minimum_visible_depth_pixels: int = 256,
    minimum_pair_point_count_ratio: float = 0.10,
    minimum_pair_diameter_ratio: float = 0.10,
) -> DGeDiRegistrationResult:
    repository = (
        Path(repository_path)
        .expanduser()
        .resolve()
    )

    python_path = (
        Path(python_executable)
        .expanduser()
        .resolve()
    )

    config_path = (
        Path(config_path)
        .expanduser()
        .resolve()
    )

    output = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    raw_reference_mesh = Path(
        reference_self_alignment
        .scaled_mesh_path
    ).expanduser().resolve()

    raw_query_mesh = Path(
        query_self_alignment
        .scaled_mesh_path
    ).expanduser().resolve()

    reference_self = _rigid(
        reference_self_alignment
        .pose_camera_from_proxy,
        "reference FoundationPose self pose",
    )

    query_self = _rigid(
        query_self_alignment
        .pose_camera_from_proxy,
        "query FoundationPose self pose",
    )

    if not repository.is_dir():
        raise FileNotFoundError(
            f"dGeDi repository: {repository}"
        )

    if not python_path.is_file():
        raise FileNotFoundError(
            f"dGeDi Python: {python_path}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"dGeDi config: {config_path}"
        )

    if not raw_reference_mesh.is_file():
        raise FileNotFoundError(
            "Reference generated mesh: "
            f"{raw_reference_mesh}"
        )

    if not raw_query_mesh.is_file():
        raise FileNotFoundError(
            "Query generated mesh: "
            f"{raw_query_mesh}"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    self_aligned_mesh_root = (
        output
        / "self_aligned_meshes"
    )

    # Reference mesh:
    # P_r -> C_r
    reference_mesh = (
        _save_self_aligned_mesh(
            source_mesh_path=(
                raw_reference_mesh
            ),
            pose_camera_from_proxy=(
                reference_self
            ),
            output_mesh_path=(
                self_aligned_mesh_root
                / (
                    "reference_self_aligned_"
                    "in_reference_camera.obj"
                )
            ),
        )
    )

    # Query mesh:
    # P_q -> C_q
    query_mesh = (
        _save_self_aligned_mesh(
            source_mesh_path=(
                raw_query_mesh
            ),
            pose_camera_from_proxy=(
                query_self
            ),
            output_mesh_path=(
                self_aligned_mesh_root
                / (
                    "query_self_aligned_"
                    "in_query_camera.obj"
                )
            ),
        )
    )

    print(
        "[Reference self-aligned mesh] "
        f"{reference_mesh}"
    )

    print(
        "[Query self-aligned mesh] "
        f"{query_mesh}"
    )

    surface_cloud_root = (
        output / "depth_consistent_proxy_surface_clouds"
    )
    (
        reference_registration_cloud,
        reference_surface_diagnostics,
    ) = _save_depth_consistent_proxy_surface_cloud(
        local_mesh_path=raw_reference_mesh,
        pose_camera_from_proxy=reference_self,
        camera_matrix=reference_camera_matrix,
        observed_mask_bool=reference_mask_bool,
        observed_depth_m=reference_depth_m,
        output_cloud_path=(
            surface_cloud_root
            / "reference_depth_consistent_proxy_surface_local.ply"
        ),
        sample_count=sample_count,
        maximum_depth_residual_m=maximum_surface_depth_residual_m,
        minimum_consistent_pixels=minimum_visible_depth_pixels,
    )
    (
        query_registration_cloud,
        query_surface_diagnostics,
    ) = _save_depth_consistent_proxy_surface_cloud(
        local_mesh_path=raw_query_mesh,
        pose_camera_from_proxy=query_self,
        camera_matrix=query_camera_matrix,
        observed_mask_bool=query_mask_bool,
        observed_depth_m=query_depth_m,
        output_cloud_path=(
            surface_cloud_root
            / "query_depth_consistent_proxy_surface_local.ply"
        ),
        sample_count=sample_count,
        maximum_depth_residual_m=maximum_surface_depth_residual_m,
        minimum_consistent_pixels=minimum_visible_depth_pixels,
    )
    (
        registration_cloud_pair_quality_path,
        registration_cloud_pair_quality,
    ) = _save_registration_cloud_pair_quality(
        reference_diagnostics_path=(
            reference_surface_diagnostics
        ),
        query_diagnostics_path=(
            query_surface_diagnostics
        ),
        output_path=(
            surface_cloud_root
            / "pair_quality_gate.json"
        ),
        minimum_point_count_ratio=minimum_pair_point_count_ratio,
        minimum_diameter_ratio=minimum_pair_diameter_ratio,
    )
    print(
        "[dGeDi depth-consistent proxy-surface input] "
        f"reference={reference_registration_cloud} "
        f"query={query_registration_cloud} "
        "pair_quality="
        f"{registration_cloud_pair_quality['status']}"
    )

    # 최종 S*/Sxyz proxy에서 mask+depth와 일치한 first ray-hit만 등록한다.
    # 점은 각 proxy-local frame에 있으므로 G=T_Pq_from_Pr이다.
    # worker 출력은 G=T_Pq_from_Pr이고, 아래에서 H=B@G@inv(A)로 합성한다.
    command = [
        str(python_path),
        str(Path(__file__).resolve()),
        "--worker",
        "--repository",
        str(repository),
        "--config",
        str(config_path),
        "--reference-mesh",
        str(reference_registration_cloud),
        "--query-mesh",
        str(query_registration_cloud),
        "--output-directory",
        str(output),
        "--mode",
        mode,
        "--device",
        device,
        "--sample-count",
        str(sample_count),
        "--ransac-threshold",
        str(ransac_threshold),
        "--icp-threshold",
        str(icp_threshold),
        "--registration-candidate-count",
        str(registration_candidate_count),
    ]

    # dGeDi 공식 checkpoint에는 argparse.Namespace 등
    # 일반 pickle 객체가 포함되어 있다.
    # 이 설정은 dGeDi 전용 subprocess에만 적용한다.
    worker_environment = os.environ.copy()
    worker_environment.pop(
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
        None,
    )
    worker_environment[
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    ] = "1"

    completed = subprocess.run(
        command,
        cwd=repository,
        env=worker_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        failure_text = f"{completed.stdout}\n{completed.stderr}"
        error_type = _worker_failure_error_type(failure_text)
        raise error_type(
            "dGeDi execution failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    proxy_pose_path = (
        output
        / (
            "dgedi_proxy_pose_"
            "query_from_reference.npy"
        )
    )

    metadata_path = (
        output
        / "dgedi_registration.json"
    )

    if not proxy_pose_path.is_file():
        raise FileNotFoundError(
            f"dGeDi pose: {proxy_pose_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"dGeDi metadata: {metadata_path}"
        )

    proxy_pose = _rigid(
        np.load(
            proxy_pose_path,
            allow_pickle=False,
        ),
        "dGeDi local proxy pose G=T_Pq_from_Pr",
    )

    relative_pose = compose_dgedi_relative_pose(
        reference_pose_camera_from_proxy=reference_self,
        query_pose_camera_from_proxy=query_self,
        proxy_pose_query_from_reference=proxy_pose,
    )

    relative_pose_path = (
        output
        / (
            "dgedi_relative_pose_"
            "query_from_reference.npy"
        )
    )

    np.save(
        relative_pose_path,
        relative_pose,
        allow_pickle=False,
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    metadata.update(
        {
            "strategy": (
                "depth_consistent_dual_proxy_surface_registration_"
                "then_pose_composition"
            ),
            "input_geometry_coordinate_frames": {
                "reference": "reference_proxy_local",
                "query": "query_proxy_local",
            },
            "raw_reference_mesh": str(
                raw_reference_mesh
            ),
            "raw_query_mesh": str(
                raw_query_mesh
            ),
            "reference_self_aligned_mesh": (
                str(reference_mesh)
            ),
            "query_self_aligned_mesh": (
                str(query_mesh)
            ),
            "reference_registration_cloud": str(
                reference_registration_cloud
            ),
            "query_registration_cloud": str(
                query_registration_cloud
            ),
            "reference_surface_diagnostics": str(
                reference_surface_diagnostics
            ),
            "query_surface_diagnostics": str(
                query_surface_diagnostics
            ),
            "registration_cloud_pair_quality": str(
                registration_cloud_pair_quality_path
            ),
            "registration_cloud_pair_quality_status": (
                registration_cloud_pair_quality["status"]
            ),
            "minimum_visible_depth_pixels": (
                minimum_visible_depth_pixels
            ),
            "maximum_surface_depth_residual_m": (
                maximum_surface_depth_residual_m
            ),
            "minimum_pair_point_count_ratio": (
                minimum_pair_point_count_ratio
            ),
            "minimum_pair_diameter_ratio": (
                minimum_pair_diameter_ratio
            ),
            "registration_candidate_count": (
                registration_candidate_count
            ),
            "reference_pose_camera_from_proxy": (
                reference_self.tolist()
            ),
            "query_pose_camera_from_proxy": (
                query_self.tolist()
            ),
            "composition": (
                "H = B @ G @ inv(A), where A=T_Cr_from_Pr, "
                "B=T_Cq_from_Pq, and G=T_Pq_from_Pr"
            ),
            "proxy_pose_convention": "G=T_query_proxy_from_reference_proxy",
            "proxy_pose_query_from_reference": proxy_pose.tolist(),
            "relative_pose_convention": (
                "T_query_camera_from_"
                "reference_camera"
            ),
            "relative_pose_query_from_reference": (
                relative_pose.tolist()
            ),
            "relative_pose_path": (
                str(relative_pose_path)
            ),
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return DGeDiRegistrationResult(
        proxy_pose_query_from_reference=(
            proxy_pose
        ),
        relative_pose_query_from_reference=(
            relative_pose
        ),
        proxy_pose_path=proxy_pose_path,
        relative_pose_path=(
            relative_pose_path
        ),
        metadata_path=metadata_path,
        reference_self_aligned_mesh_path=(
            reference_mesh
        ),
        query_self_aligned_mesh_path=(
            query_mesh
        ),
        reference_registration_cloud_path=(
            reference_registration_cloud
        ),
        query_registration_cloud_path=(
            query_registration_cloud
        ),
    )


def _parse_worker_args() -> (
    argparse.Namespace
):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker",
        action="store_true",
    )

    parser.add_argument(
        "--repository",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reference-mesh",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query-mesh",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=(
            "single_scale",
            "multi_scale",
        ),
        default="multi_scale",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--icp-threshold",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--registration-candidate-count",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--candidate-diagnostics",
        action="store_true",
        help=(
            "Store every RANSAC/ICP pose and both Chamfer scores. "
            "Diagnostic only; selection is unchanged."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_worker_args()

    if not arguments.worker:
        raise SystemExit(
            "Use --worker when executing "
            "this file directly."
        )

    raise SystemExit(
        _worker(arguments)
    )
