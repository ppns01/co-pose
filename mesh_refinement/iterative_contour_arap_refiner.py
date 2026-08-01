from __future__ import annotations

from typing import Callable

import numpy as np
import open3d as o3d
import scipy.ndimage

from mesh_refinement.depth_anchored_visible_refiner import (
    _build_uniform_laplacian,
)
from mesh_refinement.dense_strip_arap_refiner import (
    _build_adjacency,
    _expand_topology_mask,
    _local_depth_grid_handles,
    _merge_constraints,
    _quality,
    _run_arap,
    _safe_global_displacement_step,
    _sample_indices,
)
from mesh_refinement.silhouette_mesh_refiner import (
    SilhouetteMeshRefinementResult,
    _signed_distance_to_mask,
)


def _binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    intersection = int(np.count_nonzero(first_bool & second_bool))
    union = int(np.count_nonzero(first_bool | second_bool))
    return float(intersection / union) if union else 0.0


def _project_vertices(
    *,
    points_camera: np.ndarray,
    camera_k: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64)
    camera_k = np.asarray(camera_k, dtype=np.float64)

    vertex_count = int(points.shape[0])
    z = points[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-8)

    u_pixel = np.full(vertex_count, -1, dtype=np.int64)
    v_pixel = np.full(vertex_count, -1, dtype=np.int64)

    fx = float(camera_k[0, 0])
    fy = float(camera_k[1, 1])
    cx = float(camera_k[0, 2])
    cy = float(camera_k[1, 2])

    u_pixel[valid_z] = np.round(
        points[valid_z, 0] / z[valid_z] * fx + cx
    ).astype(np.int64)
    v_pixel[valid_z] = np.round(
        points[valid_z, 1] / z[valid_z] * fy + cy
    ).astype(np.int64)

    in_image = (
        valid_z
        & (u_pixel >= 0)
        & (u_pixel < image_width)
        & (v_pixel >= 0)
        & (v_pixel < image_height)
    )
    return in_image, u_pixel, v_pixel


def _raycast_mesh(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    camera_k: np.ndarray,
    image_height: int,
    image_width: int,
) -> dict[str, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    camera_k = np.asarray(camera_k, dtype=np.float64)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(
        triangles.astype(np.int32, copy=False)
    )

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    fx = float(camera_k[0, 0])
    fy = float(camera_k[1, 1])
    cx = float(camera_k[0, 2])
    cy = float(camera_k[1, 2])

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
    ray_origin = np.zeros_like(ray_direction)
    rays = np.concatenate([ray_origin, ray_direction], axis=-1).astype(
        np.float32
    )

    result = scene.cast_rays(
        o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    )
    t_hit = result["t_hit"].numpy().astype(np.float64)
    primitive_ids = result["primitive_ids"].numpy().astype(np.int64)
    primitive_uvs = result["primitive_uvs"].numpy().astype(np.float64)

    triangle_count = int(triangles.shape[0])
    valid_primitive = (primitive_ids >= 0) & (primitive_ids < triangle_count)
    rendered_mask = (
        np.isfinite(t_hit) & (t_hit > 0.0) & valid_primitive
    )
    rendered_boundary = rendered_mask & ~scipy.ndimage.binary_erosion(
        rendered_mask,
        iterations=1,
        border_value=0,
    )

    visible_vertex_mask = np.zeros(len(points), dtype=bool)
    visible_triangle_ids = np.unique(primitive_ids[rendered_mask])
    visible_triangle_ids = visible_triangle_ids[
        (visible_triangle_ids >= 0)
        & (visible_triangle_ids < triangle_count)
    ]
    if visible_triangle_ids.size:
        visible_vertex_mask[
            np.unique(triangles[visible_triangle_ids].reshape(-1))
        ] = True

    rendered_depth = t_hit.copy()
    rendered_depth[~rendered_mask] = 0.0

    return {
        "rays": rays,
        "t_hit": t_hit,
        "primitive_ids": primitive_ids,
        "primitive_uvs": primitive_uvs,
        "rendered_mask": rendered_mask,
        "rendered_boundary": rendered_boundary,
        "rendered_depth": rendered_depth,
        "visible_vertex_mask": visible_vertex_mask,
    }


def _barycentric_contour_targets(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    mask_bool: np.ndarray,
    camera_k: np.ndarray,
    raycast: dict[str, np.ndarray],
    maximum_pixel_move: float,
    minimum_pixel_move: float,
    contour_sample_stride: int,
    sdf_direction_sigma_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    points = np.asarray(points_camera, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    mask = np.asarray(mask_bool, dtype=bool)
    camera_k = np.asarray(camera_k, dtype=np.float64)

    rendered_boundary = np.asarray(
        raycast["rendered_boundary"], dtype=bool
    )
    boundary_v, boundary_u = np.nonzero(rendered_boundary)

    stride = max(1, int(contour_sample_stride))
    if stride > 1 and boundary_u.size:
        keep = np.arange(0, boundary_u.size, stride, dtype=np.int64)
        boundary_u = boundary_u[keep]
        boundary_v = boundary_v[keep]

    empty_indices = np.empty(0, dtype=np.int64)
    empty_targets = np.empty((0, 3), dtype=np.float64)
    empty_mask = np.zeros(len(points), dtype=bool)

    default_stats = {
        "target_mode": "raycast_barycentric_contour",
        "rendered_contour_sample_count": int(boundary_u.size),
        "active_contour_sample_count": 0,
        "active_boundary_vertex_count": 0,
        "sdf_abs_p50_px": None,
        "sdf_abs_p90_px": None,
        "sdf_abs_max_px": None,
        "requested_move_p50_px": None,
        "requested_move_p90_px": None,
        "requested_move_max_px": None,
        "vertex_target_move_p50_px": None,
        "vertex_target_move_p90_px": None,
        "vertex_target_move_max_px": None,
    }
    if boundary_u.size == 0:
        return empty_indices, empty_targets, empty_mask, default_stats

    primitive_ids = raycast["primitive_ids"][boundary_v, boundary_u]
    primitive_uvs = raycast["primitive_uvs"][boundary_v, boundary_u]
    t_hit = raycast["t_hit"][boundary_v, boundary_u]

    triangle_count = int(triangles.shape[0])
    valid = (
        np.isfinite(t_hit)
        & (t_hit > 0.0)
        & (primitive_ids >= 0)
        & (primitive_ids < triangle_count)
        & np.isfinite(primitive_uvs).all(axis=1)
    )
    if not valid.any():
        return empty_indices, empty_targets, empty_mask, default_stats

    boundary_u = boundary_u[valid]
    boundary_v = boundary_v[valid]
    primitive_ids = primitive_ids[valid].astype(np.int64, copy=False)
    primitive_uvs = primitive_uvs[valid]
    t_hit = t_hit[valid]

    sdf = _signed_distance_to_mask(mask)
    smoothed_sdf = scipy.ndimage.gaussian_filter(
        sdf,
        sigma=float(sdf_direction_sigma_px),
    )
    grad_v, grad_u = np.gradient(smoothed_sdf)

    residual_px = sdf[boundary_v, boundary_u]
    gu = grad_u[boundary_v, boundary_u]
    gv = grad_v[boundary_v, boundary_u]
    grad_norm = np.hypot(gu, gv)

    valid_gradient = np.isfinite(grad_norm) & (grad_norm > 1e-6)
    if not valid_gradient.any():
        return empty_indices, empty_targets, empty_mask, default_stats

    boundary_u = boundary_u[valid_gradient]
    boundary_v = boundary_v[valid_gradient]
    primitive_ids = primitive_ids[valid_gradient]
    primitive_uvs = primitive_uvs[valid_gradient]
    t_hit = t_hit[valid_gradient]
    residual_px = residual_px[valid_gradient]
    gu = gu[valid_gradient]
    gv = gv[valid_gradient]
    grad_norm = grad_norm[valid_gradient]

    du = -(gu / grad_norm) * residual_px
    dv = -(gv / grad_norm) * residual_px

    requested_norm_px = np.hypot(du, dv)
    clip_scale = np.minimum(
        1.0,
        float(maximum_pixel_move)
        / np.clip(requested_norm_px, 1e-9, None),
    )
    du *= clip_scale
    dv *= clip_scale
    clipped_norm_px = np.hypot(du, dv)

    active = (
        np.isfinite(clipped_norm_px)
        & (clipped_norm_px >= float(minimum_pixel_move))
    )
    if not active.any():
        stats = {
            **default_stats,
            "sdf_abs_p50_px": float(np.quantile(np.abs(residual_px), 0.50)),
            "sdf_abs_p90_px": float(np.quantile(np.abs(residual_px), 0.90)),
            "sdf_abs_max_px": float(np.max(np.abs(residual_px))),
        }
        return empty_indices, empty_targets, empty_mask, stats

    primitive_ids = primitive_ids[active]
    primitive_uvs = primitive_uvs[active]
    t_hit = t_hit[active]
    residual_px = residual_px[active]
    du = du[active]
    dv = dv[active]
    requested_norm_px = requested_norm_px[active]
    clipped_norm_px = clipped_norm_px[active]

    # Open3D primitive_uvs는 triangle 내부 barycentric (u, v)이다.
    # Embree/Open3D convention에 따라 vertex weights는
    # [1-u-v, u, v]로 복원한다.
    barycentric = np.stack(
        [
            1.0 - primitive_uvs[:, 0] - primitive_uvs[:, 1],
            primitive_uvs[:, 0],
            primitive_uvs[:, 1],
        ],
        axis=1,
    )
    barycentric = np.clip(barycentric, 0.0, 1.0)
    barycentric /= np.clip(
        barycentric.sum(axis=1, keepdims=True),
        1e-12,
        None,
    )

    fx = float(camera_k[0, 0])
    fy = float(camera_k[1, 1])

    delta_camera = np.zeros((len(du), 3), dtype=np.float64)
    delta_camera[:, 0] = du * t_hit / fx
    delta_camera[:, 1] = dv * t_hit / fy

    triangle_vertices = triangles[primitive_ids]
    delta_sum = np.zeros_like(points)
    support_sum = np.zeros(len(points), dtype=np.float64)

    for corner in range(3):
        vertex_indices = triangle_vertices[:, corner]
        corner_weight = barycentric[:, corner]
        np.add.at(
            delta_sum,
            vertex_indices,
            corner_weight[:, None] * delta_camera,
        )
        np.add.at(support_sum, vertex_indices, corner_weight)

    boundary_vertex_mask = support_sum > 1e-6
    boundary_indices = np.flatnonzero(boundary_vertex_mask)
    if boundary_indices.size == 0:
        return empty_indices, empty_targets, empty_mask, default_stats

    vertex_delta = delta_sum[boundary_indices] / support_sum[
        boundary_indices, None
    ]
    boundary_targets = points[boundary_indices] + vertex_delta

    z = np.clip(points[boundary_indices, 2], 1e-8, None)
    vertex_move_px = np.hypot(
        vertex_delta[:, 0] * fx / z,
        vertex_delta[:, 1] * fy / z,
    )

    stats = {
        **default_stats,
        "rendered_contour_sample_count": int(
            np.count_nonzero(rendered_boundary)
        ),
        "active_contour_sample_count": int(len(du)),
        "active_boundary_vertex_count": int(len(boundary_indices)),
        "sdf_abs_p50_px": float(np.quantile(np.abs(residual_px), 0.50)),
        "sdf_abs_p90_px": float(np.quantile(np.abs(residual_px), 0.90)),
        "sdf_abs_max_px": float(np.max(np.abs(residual_px))),
        "requested_move_p50_px": float(
            np.quantile(requested_norm_px, 0.50)
        ),
        "requested_move_p90_px": float(
            np.quantile(requested_norm_px, 0.90)
        ),
        "requested_move_max_px": float(np.max(requested_norm_px)),
        "vertex_target_move_p50_px": float(
            np.quantile(vertex_move_px, 0.50)
        ),
        "vertex_target_move_p90_px": float(
            np.quantile(vertex_move_px, 0.90)
        ),
        "vertex_target_move_max_px": float(np.max(vertex_move_px)),
    }
    return (
        boundary_indices.astype(np.int64, copy=False),
        boundary_targets,
        boundary_vertex_mask,
        stats,
    )


def _empty_depth_stats() -> dict:
    return {
        "depth_candidate_count": 0,
        "depth_handle_count": 0,
        "depth_outlier_count": 0,
        "depth_outlier_fraction": 0.0,
        "depth_outlier_threshold_m": None,
        "depth_outlier_robust_scale_m": None,
        "valid_depth_cell_count": 0,
        "depth_grid_abs_p90_m": None,
        "depth_grid_abs_max_m": None,
    }


def refine_mesh_with_iterative_contour_arap(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    mask_bool: np.ndarray,
    camera_k: np.ndarray,
    masked_depth_m: np.ndarray | None = None,
    target_scale_m: float | None = None,
    diameter_fn: Callable[[np.ndarray], float] | None = None,
    boundary_band_px: float = 3.0,
    max_silhouette_pixel_displacement: float = 2.0,
    huber_boundary_px: float = 2.0,
    huber_depth_m: float = 0.006,
    maximum_displacement_m: float = 0.008,
    laplacian_weight: float = 2.0,
    hidden_anchor_weight: float = 100.0,
    depth_outlier_neighbor_multiplier: float = 4.0,
    depth_outlier_minimum_threshold_m: float = 0.006,
    silhouette_sector_count: int = 24,
    silhouette_sector_smoothing_sigma: float = 1.25,
    silhouette_minimum_pixel_move: float = 0.15,
    silhouette_outer_iteration_count: int = 10,
    contour_sample_stride: int = 1,
    sdf_direction_sigma_px: float = 1.0,
    depth_grid_size_px: int = 16,
    depth_minimum_samples_per_cell: int = 3,
    depth_grid_smoothing_sigma_cells: float = 1.0,
    maximum_depth_move_m: float = 0.004,
    transition_ring_count: int = 4,
    maximum_anchor_count: int = 2000,
    arap_iteration_count: int = 30,
    minimum_edge_ratio: float = 0.55,
    maximum_edge_ratio: float = 1.80,
    minimum_area_ratio: float = 0.25,
    maximum_area_ratio: float = 4.00,
    minimum_step_scale: float = 0.03125,
) -> SilhouetteMeshRefinementResult:
    # Backward-compatible arguments from the previous refiner. The new
    # contour target does not use sectors or a vertex-distance boundary band.
    del (
        boundary_band_px,
        huber_boundary_px,
        huber_depth_m,
        laplacian_weight,
        hidden_anchor_weight,
        silhouette_sector_count,
        silhouette_sector_smoothing_sigma,
    )

    points = np.asarray(points_camera, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    mask = np.asarray(mask_bool, dtype=bool)
    camera_k = np.asarray(camera_k, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_camera must have shape (N,3), got {points.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"triangles must have shape (M,3), got {triangles.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask_bool must have shape (H,W), got {mask.shape}")
    if camera_k.shape != (3, 3):
        raise ValueError(f"camera_k must have shape (3,3), got {camera_k.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points_camera contains non-finite values")

    image_height, image_width = mask.shape
    vertex_count = int(points.shape[0])
    adjacency = _build_adjacency(triangles, vertex_count)

    iou_before, boundary_before = _quality(
        points_camera=points,
        mask_bool=mask,
        camera_k=camera_k,
    )
    initial_raycast = _raycast_mesh(
        points_camera=points,
        triangles=triangles,
        camera_k=camera_k,
        image_height=image_height,
        image_width=image_width,
    )
    raster_iou_before = _binary_iou(
        initial_raycast["rendered_mask"], mask
    )

    current = points.copy()
    iteration_records: list[dict] = []

    boundary_indices = np.empty(0, dtype=np.int64)
    boundary_targets = np.empty((0, 3), dtype=np.float64)
    boundary_vertex_mask = np.zeros(vertex_count, dtype=bool)
    boundary_stats: dict = {
        "target_mode": "raycast_barycentric_contour",
        "active_boundary_vertex_count": 0,
    }
    depth_indices = np.empty(0, dtype=np.int64)
    depth_targets = np.empty((0, 3), dtype=np.float64)
    depth_stats = _empty_depth_stats()
    interior_indices_all = np.empty(0, dtype=np.int64)
    anchor_indices = np.empty(0, dtype=np.int64)
    constraint_indices = np.empty(0, dtype=np.int64)
    constraint_targets = np.empty((0, 3), dtype=np.float64)

    requested_iterations = max(1, int(silhouette_outer_iteration_count))
    for outer_iteration in range(requested_iterations):
        raycast = _raycast_mesh(
            points_camera=current,
            triangles=triangles,
            camera_k=camera_k,
            image_height=image_height,
            image_width=image_width,
        )
        visible = np.asarray(
            raycast["visible_vertex_mask"], dtype=bool
        )

        (
            boundary_indices,
            boundary_targets,
            boundary_vertex_mask,
            boundary_stats,
        ) = _barycentric_contour_targets(
            points_camera=current,
            triangles=triangles,
            mask_bool=mask,
            camera_k=camera_k,
            raycast=raycast,
            maximum_pixel_move=max_silhouette_pixel_displacement,
            minimum_pixel_move=silhouette_minimum_pixel_move,
            contour_sample_stride=contour_sample_stride,
            sdf_direction_sigma_px=sdf_direction_sigma_px,
        )

        in_image, u_pixel, v_pixel = _project_vertices(
            points_camera=current,
            camera_k=camera_k,
            image_height=image_height,
            image_width=image_width,
        )
        inside_observed_mask = np.zeros(vertex_count, dtype=bool)
        visible_in_image = visible & in_image
        inside_observed_mask[visible_in_image] = mask[
            v_pixel[visible_in_image],
            u_pixel[visible_in_image],
        ]
        interior_mask = (
            visible
            & in_image
            & inside_observed_mask
            & ~boundary_vertex_mask
        )
        interior_indices_all = np.flatnonzero(interior_mask)

        depth_indices, depth_targets, depth_stats = (
            _local_depth_grid_handles(
                points_camera=current,
                triangles=triangles,
                interior_indices_all=interior_indices_all,
                u_pixel=u_pixel,
                v_pixel=v_pixel,
                masked_depth_m=masked_depth_m,
                image_height=image_height,
                image_width=image_width,
                grid_size_px=depth_grid_size_px,
                minimum_samples_per_cell=depth_minimum_samples_per_cell,
                smoothing_sigma_cells=depth_grid_smoothing_sigma_cells,
                maximum_depth_move_m=maximum_depth_move_m,
                outlier_neighbor_multiplier=(
                    depth_outlier_neighbor_multiplier
                ),
                outlier_minimum_threshold_m=(
                    depth_outlier_minimum_threshold_m
                ),
            )
        )

        transition_mask = _expand_topology_mask(
            seed_mask=visible,
            adjacency=adjacency,
            ring_count=transition_ring_count,
        )
        far_hidden_indices = np.flatnonzero(~transition_mask)
        anchor_indices = _sample_indices(
            far_hidden_indices,
            maximum_anchor_count,
        )

        if (
            len(boundary_indices)
            + len(depth_indices)
            + len(anchor_indices)
            == 0
        ):
            constraint_indices = np.empty(0, dtype=np.int64)
            constraint_targets = np.empty((0, 3), dtype=np.float64)
        else:
            constraint_indices, constraint_targets = _merge_constraints(
                boundary_indices=boundary_indices,
                boundary_targets=boundary_targets,
                depth_indices=depth_indices,
                depth_targets=depth_targets,
                anchor_indices=anchor_indices,
                points_camera=current,
            )

        raster_iou_step_before = _binary_iou(
            raycast["rendered_mask"], mask
        )
        record = {
            "iteration": int(outer_iteration),
            "raster_iou_before": float(raster_iou_step_before),
            "boundary": boundary_stats,
            "depth_handle_count": int(len(depth_indices)),
            "anchor_count": int(len(anchor_indices)),
            "constraint_count": int(len(constraint_indices)),
        }

        if len(boundary_indices) == 0:
            record["status"] = "no_active_contour_targets"
            iteration_records.append(record)
            break
        if len(constraint_indices) < 3:
            record["status"] = "insufficient_constraints"
            iteration_records.append(record)
            break

        # ARAP의 원시 결과이다. 이 결과를 바로 current로 수용하지 않는다.
        raw_proposed = _run_arap(
            points_camera=current,
            triangles=triangles,
            constraint_indices=constraint_indices,
            constraint_targets=constraint_targets,
            iteration_count=arap_iteration_count,
        )

        raw_step_displacement = raw_proposed - current
        raw_step_norm = np.linalg.norm(
            raw_step_displacement,
            axis=1,
        )

        # 현재 iteration의 current mesh를 기준으로 topology-safe
        # scalar line search를 수행한다.
        #
        # 회당 최대 3 mm로 제한한다. 2 px contour 이동이 깊이에 따라
        # 수 mm 이상이 될 수 있으므로, 한 번의 반복에서 mesh가 크게
        # 접히는 것을 방지한다.
        step_safe_proposed, step_safety_stats = (
            _safe_global_displacement_step(
                original_points=current,
                proposed_points=raw_proposed,
                triangles=triangles,
                maximum_displacement_m=min(
                    float(maximum_displacement_m),
                    0.003,
                ),
                minimum_edge_ratio=minimum_edge_ratio,
                maximum_edge_ratio=maximum_edge_ratio,
                minimum_area_ratio=minimum_area_ratio,
                maximum_area_ratio=maximum_area_ratio,
                minimum_step_scale=minimum_step_scale,
            )
        )

        topology_step_scale = float(
            step_safety_stats.get(
                "step_scale",
                1.0,
            )
        )

        topology_safe_step = (
            step_safe_proposed - current
        )

        # 누적 변위는 M0(points) 기준 maximum_displacement_m를
        # 넘지 않게 한다. 정점별 clamp가 아니라 current→candidate
        # 전체 step에 하나의 scalar를 적용하므로 새로운 국소 fold를
        # 만들지 않는다.
        cumulative_cap_scale = 1.0

        maximum_cumulative_displacement = float(
            maximum_displacement_m
        )

        if maximum_cumulative_displacement > 0.0:
            proposed_cumulative_norm = np.linalg.norm(
                step_safe_proposed - points,
                axis=1,
            )

            if (
                float(proposed_cumulative_norm.max())
                > maximum_cumulative_displacement
            ):
                lower = 0.0
                upper = 1.0

                # current는 이전 반복에서 이미 누적 제한을 만족한다.
                # 따라서 선분 위에서 허용 가능한 최대 scalar를 찾는다.
                for _ in range(40):
                    middle = 0.5 * (
                        lower + upper
                    )

                    middle_points = (
                        current
                        + middle
                        * topology_safe_step
                    )

                    middle_maximum = float(
                        np.linalg.norm(
                            middle_points - points,
                            axis=1,
                        ).max()
                    )

                    if (
                        middle_maximum
                        <= maximum_cumulative_displacement
                    ):
                        lower = middle
                    else:
                        upper = middle

                cumulative_cap_scale = lower

        accepted_proposed = (
            current
            + cumulative_cap_scale
            * topology_safe_step
        )

        accepted_step_displacement = (
            accepted_proposed - current
        )

        accepted_step_norm = np.linalg.norm(
            accepted_step_displacement,
            axis=1,
        )

        accepted_cumulative_norm = np.linalg.norm(
            accepted_proposed - points,
            axis=1,
        )

        accepted_raycast = _raycast_mesh(
            points_camera=accepted_proposed,
            triangles=triangles,
            camera_k=camera_k,
            image_height=image_height,
            image_width=image_width,
        )

        raster_iou_step_after = _binary_iou(
            accepted_raycast["rendered_mask"],
            mask,
        )

        effective_step_scale = (
            topology_step_scale
            * cumulative_cap_scale
        )

        record.update(
            {
                "raw_raster_iou_after": float(
                    _binary_iou(
                        _raycast_mesh(
                            points_camera=raw_proposed,
                            triangles=triangles,
                            camera_k=camera_k,
                            image_height=image_height,
                            image_width=image_width,
                        )["rendered_mask"],
                        mask,
                    )
                ),
                "raster_iou_after": float(
                    raster_iou_step_after
                ),
                "raw_step_displacement_p90_m": float(
                    np.quantile(
                        raw_step_norm,
                        0.90,
                    )
                ),
                "raw_step_displacement_max_m": float(
                    raw_step_norm.max()
                ),
                "step_displacement_p90_m": float(
                    np.quantile(
                        accepted_step_norm,
                        0.90,
                    )
                ),
                "step_displacement_max_m": float(
                    accepted_step_norm.max()
                ),
                "cumulative_displacement_max_m": float(
                    accepted_cumulative_norm.max()
                ),
                "topology_step_scale": float(
                    topology_step_scale
                ),
                "cumulative_cap_scale": float(
                    cumulative_cap_scale
                ),
                "effective_step_scale": float(
                    effective_step_scale
                ),
            }
        )

        # 안전장치로 인해 실질적으로 움직이지 못한 경우 종료한다.
        if float(accepted_step_norm.max()) <= 1e-8:
            record["status"] = "stalled_by_safety"
            iteration_records.append(record)
            break

        # topology는 안전해도 silhouette IoU가 감소하면 이 step은
        # 수용하지 않는다. current는 이전 상태로 유지한다.
        if (
            float(raster_iou_step_after) + 1e-6
            < float(raster_iou_step_before)
        ):
            record["status"] = "rejected_iou_regression"
            iteration_records.append(record)
            break

        record["status"] = "updated_safe"
        iteration_records.append(record)

        # 모든 검사를 통과한 mesh만 다음 반복 입력으로 사용한다.
        current = accepted_proposed

        sdf_p90 = boundary_stats.get("sdf_abs_p90_px")
        if sdf_p90 is not None and float(sdf_p90) <= float(
            silhouette_minimum_pixel_move
        ):
            iteration_records[-1]["status"] = "converged"
            break

    arap_raw = current.copy()
    raw_raycast = _raycast_mesh(
        points_camera=arap_raw,
        triangles=triangles,
        camera_k=camera_k,
        image_height=image_height,
        image_width=image_width,
    )
    raster_iou_raw = _binary_iou(raw_raycast["rendered_mask"], mask)

    # 각 outer iteration에서 이미 topology와 누적 변위를 검사했다.
    # 여기에서 M0→M1 전체 변위를 다시 global backtracking하면,
    # 이전에 확인된 것처럼 정상 영역까지 5~10%로 축소된다.
    #
    # 기존 metadata 구조를 유지하기 위해 zero-step safety audit만
    # 실행하고 실제 mesh는 변경하지 않는다.
    refined_pre_scale, safety_stats = _safe_global_displacement_step(
        original_points=arap_raw,
        proposed_points=arap_raw,
        triangles=triangles,
        maximum_displacement_m=maximum_displacement_m,
        minimum_edge_ratio=minimum_edge_ratio,
        maximum_edge_ratio=maximum_edge_ratio,
        minimum_area_ratio=minimum_area_ratio,
        maximum_area_ratio=maximum_area_ratio,
        minimum_step_scale=minimum_step_scale,
    )

    safety_stats = dict(safety_stats)
    safety_stats["mode"] = (
        "per_iteration_scalar_line_search"
    )
    safety_stats["final_global_backtracking_applied"] = False
    safety_stats["accepted_outer_iteration_count"] = int(
        sum(
            record.get("status")
            in {"updated_safe", "converged"}
            for record in iteration_records
        )
    )

    refined = refined_pre_scale.copy()

    scale_before = None
    scale_after = None
    if target_scale_m is not None and diameter_fn is not None:
        scale_before = float(diameter_fn(refined))
        centroid = refined.mean(axis=0)
        beta = float(target_scale_m) / max(scale_before, 1e-12)
        refined = centroid + beta * (refined - centroid)
        scale_after = float(diameter_fn(refined))

    displacement = refined - points
    displacement_norm = np.linalg.norm(displacement, axis=1)
    moved = displacement_norm > 1e-6

    centroid0 = points.mean(axis=0)
    radial_direction = (points - centroid0) / np.clip(
        np.linalg.norm(points - centroid0, axis=1, keepdims=True),
        1e-9,
        None,
    )
    radial_component = np.sum(displacement * radial_direction, axis=1)

    iou_after, boundary_after = _quality(
        points_camera=refined,
        mask_bool=mask,
        camera_k=camera_k,
    )
    final_raycast = _raycast_mesh(
        points_camera=refined,
        triangles=triangles,
        camera_k=camera_k,
        image_height=image_height,
        image_width=image_width,
    )
    raster_iou_after = _binary_iou(final_raycast["rendered_mask"], mask)

    laplacian = _build_uniform_laplacian(triangles, vertex_count)
    roughness_before = np.linalg.norm(laplacian @ points, axis=1)
    roughness_after = np.linalg.norm(laplacian @ refined, axis=1)

    diagnostics = {
        "status": (
            "refined"
            if safety_stats.get("step_scale", 0.0) > 0.0
            else "topology_safety_fallback"
        ),
        "refinement_mode": "iterative_contour_arap",
        "iou_before": float(iou_before),
        "iou_after": float(iou_after),
        "boundary_distance_before_px": boundary_before,
        "boundary_distance_after_px": boundary_after,
        "raster_iou_before": float(raster_iou_before),
        "raster_iou_raw": float(raster_iou_raw),
        "raster_iou_after": float(raster_iou_after),
        "outer_iteration_count_requested": int(requested_iterations),
        "outer_iteration_count_completed": int(len(iteration_records)),
        "outer_iterations": iteration_records,
        "boundary_band_vertex_count": int(len(boundary_indices)),
        "interior_vertex_count": int(len(interior_indices_all)),
        "depth_correspondence_count": int(
            depth_stats["depth_handle_count"]
        ),
        "depth_outlier_count": int(depth_stats["depth_outlier_count"]),
        "depth_outlier_fraction": float(
            depth_stats["depth_outlier_fraction"]
        ),
        "depth_outlier_threshold_m": depth_stats[
            "depth_outlier_threshold_m"
        ],
        "depth_outlier_robust_scale_m": depth_stats[
            "depth_outlier_robust_scale_m"
        ],
        "moved_vertex_count": int(np.count_nonzero(moved)),
        "inward_fraction": (
            float(np.mean(radial_component[moved] < 0.0))
            if moved.any()
            else None
        ),
        "outward_fraction": (
            float(np.mean(radial_component[moved] > 0.0))
            if moved.any()
            else None
        ),
        "displacement_p90_m": float(
            np.quantile(displacement_norm, 0.90)
        ),
        "displacement_max_m": float(displacement_norm.max()),
        "centroid_drift_m": float(
            np.linalg.norm(displacement.mean(axis=0))
        ),
        "scale_before_reprojection_m": scale_before,
        "scale_after_reprojection_m": scale_after,
        "target_scale_m": target_scale_m,
        "roughness_before_p95_m": float(
            np.quantile(roughness_before, 0.95)
        ),
        "roughness_after_p95_m": float(
            np.quantile(roughness_after, 0.95)
        ),
        "silhouette_strip": {
            "target_mode": "raycast_barycentric_contour",
            "maximum_pixel_move_per_iteration": float(
                max_silhouette_pixel_displacement
            ),
            "minimum_pixel_move": float(
                silhouette_minimum_pixel_move
            ),
            "contour_sample_stride": int(contour_sample_stride),
            "sdf_direction_sigma_px": float(sdf_direction_sigma_px),
            "last_iteration": boundary_stats,
        },
        "depth_grid": depth_stats,
        "transition_ring_count": int(transition_ring_count),
        "far_hidden_anchor_count": int(len(anchor_indices)),
        "constraint_count": int(len(constraint_indices)),
        "topology_safety": safety_stats,
    }

    intermediate_points_camera = {
        "M0_original_camera": points.copy(),
        "M1_arap_raw_camera": arap_raw.copy(),
        "M2_arap_safe_pre_scale_camera": refined_pre_scale.copy(),
        "M3_arap_safe_post_scale_camera": refined.copy(),
    }
    return SilhouetteMeshRefinementResult(
        refined_points_camera=refined,
        displacement=displacement,
        diagnostics=diagnostics,
        intermediate_points_camera=intermediate_points_camera,
    )
