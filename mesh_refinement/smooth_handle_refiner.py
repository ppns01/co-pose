from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg

from mesh_refinement.depth_anchored_visible_refiner import (
    _build_uniform_laplacian,
    _visible_vertex_mask,
)
from mesh_refinement.silhouette_mesh_refiner import (
    SilhouetteMeshRefinementResult,
    _reject_depth_outliers,
    _signed_distance_to_mask,
)


def _quality(points, mask, k):
    h, w = mask.shape

    visible, u, v = _visible_vertex_mask(
        points_camera=points,
        camera_k=k,
        image_height=h,
        image_width=w,
        mask_bool=np.ones_like(mask),
    )

    if not visible.any():
        return 0.0, float("inf")

    rendered = np.zeros(
        (h, w),
        dtype=bool,
    )

    rendered[
        v[visible],
        u[visible],
    ] = True

    union = np.count_nonzero(
        rendered | mask
    )

    iou = (
        np.count_nonzero(
            rendered & mask
        )
        / union
        if union
        else 0.0
    )

    sdf = _signed_distance_to_mask(
        mask
    )

    boundary = float(
        np.abs(
            sdf[
                v[visible],
                u[visible],
            ]
        ).mean()
    )

    return float(iou), boundary


def _fit_depth_plane(
    u,
    v,
    residual,
    width,
    height,
    huber_delta,
):
    """
    정점별 depth residual을 직접 쓰지 않고
    화면상의 저주파 평면:

        r(u,v) = a0 + a1*x + a2*y

    으로 robust fitting한다.
    """
    x = (
        u
        - 0.5 * (width - 1)
    ) / max(
        0.5 * (width - 1),
        1.0,
    )

    y = (
        v
        - 0.5 * (height - 1)
    ) / max(
        0.5 * (height - 1),
        1.0,
    )

    design = np.stack(
        [
            np.ones_like(x),
            x,
            y,
        ],
        axis=1,
    ).astype(np.float64)

    weights = np.ones(
        len(residual),
        dtype=np.float64,
    )

    coefficients = np.zeros(
        3,
        dtype=np.float64,
    )

    for _ in range(6):
        sqrt_weights = np.sqrt(
            np.clip(
                weights,
                1e-6,
                None,
            )
        )

        coefficients = np.linalg.lstsq(
            design
            * sqrt_weights[:, None],
            residual
            * sqrt_weights,
            rcond=None,
        )[0]

        error = (
            residual
            - design @ coefficients
        )

        absolute_error = np.abs(
            error
        )

        weights = np.ones_like(
            absolute_error
        )

        outside = (
            absolute_error
            > huber_delta
        )

        weights[outside] = (
            huber_delta
            / np.clip(
                absolute_error[outside],
                1e-9,
                None,
            )
        )

    return (
        design @ coefficients,
        coefficients,
    )


def _silhouette_handles(
    points,
    visible,
    u,
    v,
    mask,
    k,
    sdf,
    sector_count=24,
    vertices_per_sector=6,
    max_pixel_move=2.0,
):
    """
    화면 중심을 기준으로 silhouette를 각도 sector로 나누고,
    각 sector의 가장 바깥쪽 정점들을 한 묶음으로 잡아당긴다.

    정점마다 서로 다른 방향을 주지 않고,
    같은 sector 안에서는 동일한 대표 (du,dv)를 사용한다.
    """
    h, w = mask.shape

    fx = float(k[0, 0])
    fy = float(k[1, 1])

    visible_indices = np.flatnonzero(
        visible
    )

    if len(visible_indices) == 0:
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                (0, 3),
                dtype=np.float64,
            ),
            {
                "silhouette_handle_count": 0,
                "silhouette_active_sector_count": 0,
                "silhouette_sector_moves_px": [],
            },
        )

    visible_u = u[
        visible_indices
    ].astype(np.float64)

    visible_v = v[
        visible_indices
    ].astype(np.float64)

    center_u = float(
        np.median(visible_u)
    )

    center_v = float(
        np.median(visible_v)
    )

    angle = np.arctan2(
        visible_v - center_v,
        visible_u - center_u,
    )

    radius2 = (
        (visible_u - center_u) ** 2
        + (visible_v - center_v) ** 2
    )

    sectors = np.floor(
        (
            angle + np.pi
        )
        / (2.0 * np.pi)
        * sector_count
    ).astype(np.int64)

    sectors = np.clip(
        sectors,
        0,
        sector_count - 1,
    )

    smoothed_sdf = (
        scipy.ndimage
        .gaussian_filter(
            sdf,
            sigma=2.0,
        )
    )

    grad_v, grad_u = np.gradient(
        smoothed_sdf
    )

    indices_out = []
    targets_out = []
    sector_moves = []

    for sector_id in range(
        sector_count
    ):
        local = np.flatnonzero(
            sectors == sector_id
        )

        if len(local) < 3:
            continue

        local = local[
            np.argsort(
                radius2[local]
            )[::-1]
        ]

        chosen = visible_indices[
            local[
                :vertices_per_sector
            ]
        ]

        pixel_u = u[chosen]
        pixel_v = v[chosen]

        valid = (
            (pixel_u >= 0)
            & (pixel_u < w)
            & (pixel_v >= 0)
            & (pixel_v < h)
        )

        chosen = chosen[valid]
        pixel_u = pixel_u[valid]
        pixel_v = pixel_v[valid]

        if len(chosen) == 0:
            continue

        residual_px = sdf[
            pixel_v,
            pixel_u,
        ]

        gradient_u = grad_u[
            pixel_v,
            pixel_u,
        ]

        gradient_v = grad_v[
            pixel_v,
            pixel_u,
        ]

        gradient_norm = np.clip(
            np.sqrt(
                gradient_u**2
                + gradient_v**2
            ),
            1e-6,
            None,
        )

        raw_du = (
            -gradient_u
            / gradient_norm
            * residual_px
        )

        raw_dv = (
            -gradient_v
            / gradient_norm
            * residual_px
        )

        # 같은 외곽 구간은 대표 이동 하나를 사용한다.
        representative_du = float(
            np.median(raw_du)
        )

        representative_dv = float(
            np.median(raw_dv)
        )

        magnitude = float(
            np.hypot(
                representative_du,
                representative_dv,
            )
        )

        # 거의 일치한 구간은 handle로 만들지 않는다.
        if magnitude < 0.35:
            continue

        if magnitude > max_pixel_move:
            scale = (
                max_pixel_move
                / magnitude
            )

            representative_du *= scale
            representative_dv *= scale

        target = points[
            chosen
        ].copy()

        target[:, 0] += (
            representative_du
            * target[:, 2]
            / fx
        )

        target[:, 1] += (
            representative_dv
            * target[:, 2]
            / fy
        )

        indices_out.extend(
            chosen.tolist()
        )

        targets_out.extend(
            target.tolist()
        )

        sector_moves.append(
            [
                representative_du,
                representative_dv,
            ]
        )

    return (
        np.asarray(
            indices_out,
            dtype=np.int64,
        ),
        np.asarray(
            targets_out,
            dtype=np.float64,
        ).reshape(-1, 3),
        {
            "silhouette_handle_count":
                int(len(indices_out)),
            "silhouette_active_sector_count":
                int(len(sector_moves)),
            "silhouette_sector_moves_px":
                sector_moves,
        },
    )


def _depth_handles(
    points,
    triangles,
    visible,
    u,
    v,
    mask,
    depth,
    sdf,
    huber_depth_m,
    outlier_multiplier,
    outlier_floor_m,
    grid_size_px=16,
    max_depth_move_m=0.006,
):
    """
    Interior depth residual을 affine 저주파 field로 fitting한 뒤
    16px cell마다 대표 정점 하나만 handle로 선택한다.
    """
    h, w = mask.shape

    empty_indices = np.empty(
        0,
        dtype=np.int64,
    )

    empty_targets = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    empty_stats = {
        "depth_candidate_count": 0,
        "depth_handle_count": 0,
        "depth_outlier_count": 0,
        "depth_outlier_fraction": 0.0,
        "depth_outlier_threshold_m": None,
        "depth_outlier_robust_scale_m": None,
        "depth_plane_coefficients_m": None,
        "depth_handle_abs_p90_m": None,
    }

    if depth is None:
        return (
            empty_indices,
            empty_targets,
            empty_stats,
        )

    valid = visible.copy()

    valid[visible] &= (
        mask[
            v[visible],
            u[visible],
        ]
        & (
            sdf[
                v[visible],
                u[visible],
            ]
            < -3.0
        )
    )

    candidate = np.flatnonzero(
        valid
    )

    if len(candidate) < 3:
        return (
            empty_indices,
            empty_targets,
            empty_stats,
        )

    filtered_depth = (
        scipy.ndimage
        .median_filter(
            depth,
            size=3,
        )
    )

    observed = filtered_depth[
        v[candidate],
        u[candidate],
    ]

    has_depth = observed > 0.0

    candidate = candidate[
        has_depth
    ]

    observed = observed[
        has_depth
    ]

    if len(candidate) < 3:
        return (
            empty_indices,
            empty_targets,
            empty_stats,
        )

    residual = (
        observed
        - points[candidate, 2]
    )

    inlier_mask, outlier_stats = (
        _reject_depth_outliers(
            vertex_count=len(points),
            triangles=triangles,
            active_indices=candidate,
            residual=residual,
            neighbor_multiplier=(
                outlier_multiplier
            ),
            minimum_threshold_m=(
                outlier_floor_m
            ),
        )
    )

    candidate = candidate[
        inlier_mask
    ]

    residual = residual[
        inlier_mask
    ]

    if len(candidate) < 3:
        return (
            empty_indices,
            empty_targets,
            {
                **empty_stats,
                **outlier_stats,
                "depth_candidate_count":
                    int(len(candidate)),
            },
        )

    fitted_residual, coefficients = (
        _fit_depth_plane(
            u[candidate].astype(
                np.float64
            ),
            v[candidate].astype(
                np.float64
            ),
            residual,
            w,
            h,
            huber_depth_m,
        )
    )

    fitted_residual = np.clip(
        fitted_residual,
        -max_depth_move_m,
        max_depth_move_m,
    )

    cell_u = (
        u[candidate]
        // grid_size_px
    )

    cell_v = (
        v[candidate]
        // grid_size_px
    )

    cell_key = (
        cell_v
        * (
            w // grid_size_px
            + 2
        )
        + cell_u
    )

    selected = []

    for key in np.unique(
        cell_key
    ):
        group = np.flatnonzero(
            cell_key == key
        )

        group_u = u[
            candidate[group]
        ].astype(np.float64)

        group_v = v[
            candidate[group]
        ].astype(np.float64)

        center_u = float(
            np.median(group_u)
        )

        center_v = float(
            np.median(group_v)
        )

        representative = int(
            group[
                np.argmin(
                    (group_u - center_u) ** 2
                    + (group_v - center_v) ** 2
                )
            ]
        )

        selected.append(
            representative
        )

    selected = np.asarray(
        selected,
        dtype=np.int64,
    )

    handle_indices = candidate[
        selected
    ]

    handle_residual = (
        fitted_residual[selected]
    )

    handle_points = points[
        handle_indices
    ]

    ray_direction = (
        handle_points
        / np.clip(
            handle_points[:, 2:3],
            1e-9,
            None,
        )
    )

    handle_targets = (
        handle_points
        + ray_direction
        * handle_residual[:, None]
    )

    return (
        handle_indices,
        handle_targets,
        {
            **outlier_stats,
            "depth_candidate_count":
                int(len(candidate)),
            "depth_handle_count":
                int(len(handle_indices)),
            "depth_plane_coefficients_m":
                coefficients.tolist(),
            "depth_handle_abs_p90_m": float(
                np.quantile(
                    np.abs(handle_residual),
                    0.90,
                )
            ),
        },
    )


def _solve(
    points,
    triangles,
    indices,
    targets,
    weights,
    visible,
    max_move,
    laplacian_weight,
    anchor_weight,
):
    """
    Laplacian mesh editing:

      ||L V' - L V||²
      + handle constraints
      + hidden anchor
      + centroid constraint

    를 한 번에 푼다.
    """
    vertex_count = len(points)

    laplacian = (
        _build_uniform_laplacian(
            triangles,
            vertex_count,
        )
    )

    identity = (
        scipy.sparse.identity(
            vertex_count,
            format="csr",
        )
    )

    handle_rows = (
        scipy.sparse.diags(
            np.sqrt(weights)
        )
        @ identity[indices]
    )

    anchor_indices = np.flatnonzero(
        ~visible
    )

    anchor_rows = identity[
        anchor_indices
    ]

    mean_row = (
        scipy.sparse.csr_matrix(
            np.full(
                (1, vertex_count),
                1.0 / vertex_count,
            )
        )
    )

    centroid_weight = 100.0

    system_matrix = (
        scipy.sparse.vstack(
            [
                np.sqrt(
                    laplacian_weight
                )
                * laplacian,
                handle_rows,
                np.sqrt(
                    anchor_weight
                )
                * anchor_rows,
                np.sqrt(
                    centroid_weight
                )
                * mean_row,
            ]
        ).tocsr()
    )

    laplacian_coordinates = (
        laplacian @ points
    )

    centroid = points.mean(
        axis=0
    )

    refined = np.zeros_like(
        points
    )

    for coordinate in range(3):
        rhs = np.concatenate(
            [
                np.sqrt(
                    laplacian_weight
                )
                * laplacian_coordinates[
                    :,
                    coordinate,
                ],
                np.sqrt(weights)
                * targets[
                    :,
                    coordinate,
                ],
                np.sqrt(
                    anchor_weight
                )
                * points[
                    anchor_indices,
                    coordinate,
                ],
                [
                    np.sqrt(
                        centroid_weight
                    )
                    * centroid[
                        coordinate
                    ]
                ],
            ]
        )

        refined[
            :,
            coordinate,
        ] = scipy.sparse.linalg.lsqr(
            system_matrix,
            rhs,
            atol=1e-8,
            btol=1e-8,
            iter_lim=3000,
        )[0]

    displacement = (
        refined - points
    )

    displacement_norm = np.linalg.norm(
        displacement,
        axis=1,
    )

    global_scale = 1.0

    # 정점별 clip이 아니라 전체 변형장을 같은 비율로 축소한다.
    if (
        displacement_norm.max()
        > max_move
    ):
        global_scale = (
            max_move
            / displacement_norm.max()
        )

        displacement *= global_scale
        refined = (
            points + displacement
        )

        displacement_norm = (
            np.linalg.norm(
                displacement,
                axis=1,
            )
        )

    return (
        refined,
        {
            "global_displacement_scale":
                float(global_scale),
            "displacement_p90_m": float(
                np.quantile(
                    displacement_norm,
                    0.90,
                )
            ),
            "displacement_max_m": float(
                displacement_norm.max()
            ),
        },
    )


def refine_mesh_with_smooth_handles(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    mask_bool: np.ndarray,
    camera_k: np.ndarray,
    masked_depth_m: np.ndarray | None = None,
    target_scale_m: float | None = None,
    diameter_fn: Callable[
        [np.ndarray],
        float,
    ] | None = None,
    boundary_band_px: float = 3.0,
    max_silhouette_pixel_displacement: float = 2.0,
    huber_boundary_px: float = 2.0,
    huber_depth_m: float = 0.006,
    maximum_displacement_m: float = 0.008,
    laplacian_weight: float = 8.0,
    hidden_anchor_weight: float = 20.0,
    depth_outlier_neighbor_multiplier: float = 4.0,
    depth_outlier_minimum_threshold_m: float = 0.006,
) -> SilhouetteMeshRefinementResult:
    del huber_boundary_px

    points = np.asarray(
        points_camera,
        dtype=np.float64,
    )

    triangles = np.asarray(
        triangles,
        dtype=np.int64,
    )

    mask = np.asarray(
        mask_bool,
        dtype=bool,
    )

    camera_k = np.asarray(
        camera_k,
        dtype=np.float64,
    )

    image_height, image_width = (
        mask.shape
    )

    visible, u_pixel, v_pixel = (
        _visible_vertex_mask(
            points_camera=points,
            camera_k=camera_k,
            image_height=image_height,
            image_width=image_width,
            mask_bool=np.ones_like(mask),
        )
    )

    sdf = _signed_distance_to_mask(
        mask
    )

    (
        iou_before,
        boundary_before,
    ) = _quality(
        points,
        mask,
        camera_k,
    )

    (
        silhouette_indices,
        silhouette_targets,
        silhouette_stats,
    ) = _silhouette_handles(
        points,
        visible,
        u_pixel,
        v_pixel,
        mask,
        camera_k,
        sdf,
        max_pixel_move=(
            max_silhouette_pixel_displacement
        ),
    )

    (
        depth_indices,
        depth_targets,
        depth_stats,
    ) = _depth_handles(
        points,
        triangles,
        visible,
        u_pixel,
        v_pixel,
        mask,
        masked_depth_m,
        sdf,
        huber_depth_m,
        depth_outlier_neighbor_multiplier,
        depth_outlier_minimum_threshold_m,
        max_depth_move_m=min(
            maximum_displacement_m,
            0.006,
        ),
    )

    handle_indices = np.concatenate(
        [
            silhouette_indices,
            depth_indices,
        ]
    )

    handle_targets = np.concatenate(
        [
            silhouette_targets,
            depth_targets,
        ],
        axis=0,
    )

    handle_weights = np.concatenate(
        [
            np.full(
                len(
                    silhouette_indices
                ),
                30.0,
            ),
            np.full(
                len(depth_indices),
                12.0,
            ),
        ]
    )

    stages = {
        "M0_original_camera":
            points.copy(),
    }

    silhouette_preview = (
        points.copy()
    )

    if len(silhouette_indices):
        silhouette_preview[
            silhouette_indices
        ] = silhouette_targets

    stages[
        "M1_silhouette_handle_targets"
    ] = silhouette_preview

    depth_preview = points.copy()

    if len(depth_indices):
        depth_preview[
            depth_indices
        ] = depth_targets

    stages[
        "M2_depth_handle_targets"
    ] = depth_preview

    if len(handle_indices):
        (
            refined_pre_scale,
            solve_stats,
        ) = _solve(
            points,
            triangles,
            handle_indices,
            handle_targets,
            handle_weights,
            visible,
            maximum_displacement_m,
            laplacian_weight,
            hidden_anchor_weight,
        )
    else:
        refined_pre_scale = (
            points.copy()
        )

        solve_stats = {
            "global_displacement_scale": 1.0,
            "displacement_p90_m": 0.0,
            "displacement_max_m": 0.0,
        }

    stages[
        "M3_smooth_handle_solution_pre_scale"
    ] = refined_pre_scale.copy()

    refined = (
        refined_pre_scale.copy()
    )

    scale_before = None
    scale_after = None

    if (
        target_scale_m is not None
        and diameter_fn is not None
    ):
        scale_before = float(
            diameter_fn(refined)
        )

        centroid = refined.mean(
            axis=0
        )

        beta = (
            target_scale_m
            / max(
                scale_before,
                1e-12,
            )
        )

        refined = (
            centroid
            + beta
            * (
                refined
                - centroid
            )
        )

        scale_after = float(
            diameter_fn(refined)
        )

    stages[
        "M4_smooth_handle_solution_post_scale"
    ] = refined.copy()

    displacement = (
        refined - points
    )

    displacement_norm = np.linalg.norm(
        displacement,
        axis=1,
    )

    moved = (
        displacement_norm > 1e-6
    )

    centroid0 = points.mean(
        axis=0
    )

    radial_direction = (
        points - centroid0
    ) / np.clip(
        np.linalg.norm(
            points - centroid0,
            axis=1,
            keepdims=True,
        ),
        1e-9,
        None,
    )

    radial_component = np.sum(
        displacement
        * radial_direction,
        axis=1,
    )

    (
        iou_after,
        boundary_after,
    ) = _quality(
        refined,
        mask,
        camera_k,
    )

    laplacian = (
        _build_uniform_laplacian(
            triangles,
            len(points),
        )
    )

    roughness_before = np.linalg.norm(
        laplacian @ points,
        axis=1,
    )

    roughness_after = np.linalg.norm(
        laplacian @ refined,
        axis=1,
    )

    diagnostics = {
        "status": (
            "refined"
            if len(handle_indices)
            else "no_handles"
        ),
        "refinement_mode":
            "smooth_handles",
        "iou_before":
            float(iou_before),
        "iou_after":
            float(iou_after),
        "boundary_distance_before_px":
            boundary_before,
        "boundary_distance_after_px":
            boundary_after,
        "boundary_band_vertex_count":
            int(
                len(
                    silhouette_indices
                )
            ),
        "interior_vertex_count":
            int(
                depth_stats[
                    "depth_candidate_count"
                ]
            ),
        "depth_correspondence_count":
            int(
                depth_stats[
                    "depth_handle_count"
                ]
            ),
        "depth_outlier_count":
            int(
                depth_stats[
                    "depth_outlier_count"
                ]
            ),
        "depth_outlier_fraction":
            float(
                depth_stats[
                    "depth_outlier_fraction"
                ]
            ),
        "depth_outlier_threshold_m":
            depth_stats[
                "depth_outlier_threshold_m"
            ],
        "depth_outlier_robust_scale_m":
            depth_stats[
                "depth_outlier_robust_scale_m"
            ],
        "moved_vertex_count":
            int(moved.sum()),
        "inward_fraction": (
            float(
                (
                    radial_component[
                        moved
                    ]
                    < 0
                ).mean()
            )
            if moved.any()
            else None
        ),
        "outward_fraction": (
            float(
                (
                    radial_component[
                        moved
                    ]
                    > 0
                ).mean()
            )
            if moved.any()
            else None
        ),
        "displacement_p90_m":
            float(
                np.quantile(
                    displacement_norm,
                    0.90,
                )
            ),
        "displacement_max_m":
            float(
                displacement_norm.max()
            ),
        "centroid_drift_m":
            float(
                np.linalg.norm(
                    displacement.mean(
                        axis=0
                    )
                )
            ),
        "scale_before_reprojection_m":
            scale_before,
        "scale_after_reprojection_m":
            scale_after,
        "target_scale_m":
            target_scale_m,
        "roughness_before_p95_m":
            float(
                np.quantile(
                    roughness_before,
                    0.95,
                )
            ),
        "roughness_after_p95_m":
            float(
                np.quantile(
                    roughness_after,
                    0.95,
                )
            ),
        "silhouette_handles":
            silhouette_stats,
        "depth_handles":
            depth_stats,
        "solve":
            solve_stats,
    }

    return SilhouetteMeshRefinementResult(
        refined_points_camera=refined,
        displacement=displacement,
        diagnostics=diagnostics,
        intermediate_points_camera=stages,
    )
