from __future__ import annotations

from typing import Callable

import numpy as np
import open3d as o3d
import scipy.ndimage

from mesh_refinement.depth_anchored_visible_refiner import (
    _build_uniform_laplacian,
    _visible_vertex_mask,
)
from mesh_refinement.silhouette_mesh_refiner import (
    SilhouetteMeshRefinementResult,
    _reject_depth_outliers,
    _signed_distance_to_mask,
)


def _quality(
    *,
    points_camera: np.ndarray,
    mask_bool: np.ndarray,
    camera_k: np.ndarray,
) -> tuple[float, float]:
    image_height, image_width = mask_bool.shape

    visible, u_pixel, v_pixel = _visible_vertex_mask(
        points_camera=points_camera,
        camera_k=camera_k,
        image_height=image_height,
        image_width=image_width,
        mask_bool=np.ones_like(mask_bool),
    )

    if not visible.any():
        return 0.0, float("inf")

    rendered = np.zeros(
        (image_height, image_width),
        dtype=bool,
    )
    rendered[
        v_pixel[visible],
        u_pixel[visible],
    ] = True

    intersection = int(
        np.count_nonzero(
            rendered & mask_bool
        )
    )
    union = int(
        np.count_nonzero(
            rendered | mask_bool
        )
    )

    iou = (
        float(intersection / union)
        if union
        else 0.0
    )

    sdf = _signed_distance_to_mask(
        mask_bool
    )

    boundary_distance = float(
        np.abs(
            sdf[
                v_pixel[visible],
                u_pixel[visible],
            ]
        ).mean()
    )

    return iou, boundary_distance



def _rasterize_visible_and_boundary_vertices(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    camera_k: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    실제 triangle을 ray casting하여 다음을 계산한다.

    visible_vertex_mask:
        렌더링된 triangle에 속하는 정점

    boundary_vertex_mask:
        실제로 채워진 rendered silhouette의 외곽 pixel을
        구성하는 triangle에 속하는 정점

    rendered_mask:
        triangle 내부까지 채워진 binary silhouette

    rendered_depth:
        camera z-depth
    """
    points_camera = np.asarray(
        points_camera,
        dtype=np.float64,
    )

    triangles = np.asarray(
        triangles,
        dtype=np.int64,
    )

    camera_k = np.asarray(
        camera_k,
        dtype=np.float64,
    )

    vertex_count = int(
        points_camera.shape[0]
    )

    triangle_count = int(
        triangles.shape[0]
    )

    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = (
        o3d.utility.Vector3dVector(
            points_camera
        )
    )

    mesh.triangles = (
        o3d.utility.Vector3iVector(
            triangles.astype(
                np.int32
            )
        )
    )

    tensor_mesh = (
        o3d.t.geometry
        .TriangleMesh
        .from_legacy(mesh)
    )

    scene = (
        o3d.t.geometry
        .RaycastingScene()
    )

    scene.add_triangles(
        tensor_mesh
    )

    fx = float(
        camera_k[0, 0]
    )

    fy = float(
        camera_k[1, 1]
    )

    cx = float(
        camera_k[0, 2]
    )

    cy = float(
        camera_k[1, 2]
    )

    pixel_u, pixel_v = np.meshgrid(
        np.arange(
            image_width,
            dtype=np.float32,
        ),
        np.arange(
            image_height,
            dtype=np.float32,
        ),
    )

    ray_direction = np.stack(
        [
            (pixel_u - cx) / fx,
            (pixel_v - cy) / fy,
            np.ones_like(
                pixel_u
            ),
        ],
        axis=-1,
    )

    ray_origin = np.zeros_like(
        ray_direction
    )

    rays = np.concatenate(
        [
            ray_origin,
            ray_direction,
        ],
        axis=-1,
    ).astype(np.float32)

    result = scene.cast_rays(
        o3d.core.Tensor(
            rays,
            dtype=(
                o3d.core
                .Dtype
                .Float32
            ),
        )
    )

    rendered_depth = (
        result["t_hit"]
        .numpy()
        .astype(np.float64)
    )

    primitive_ids = (
        result["primitive_ids"]
        .numpy()
        .astype(np.int64)
    )

    rendered_mask = (
        np.isfinite(
            rendered_depth
        )
        & (
            rendered_depth
            > 0.0
        )
    )

    rendered_depth[
        ~rendered_mask
    ] = 0.0

    rendered_boundary = (
        rendered_mask
        & ~scipy.ndimage
        .binary_erosion(
            rendered_mask,
            iterations=1,
            border_value=0,
        )
    )

    visible_triangle_ids = np.unique(
        primitive_ids[
            rendered_mask
        ]
    )

    boundary_triangle_ids = np.unique(
        primitive_ids[
            rendered_boundary
        ]
    )

    visible_triangle_ids = (
        visible_triangle_ids[
            (
                visible_triangle_ids
                >= 0
            )
            & (
                visible_triangle_ids
                < triangle_count
            )
        ]
    )

    boundary_triangle_ids = (
        boundary_triangle_ids[
            (
                boundary_triangle_ids
                >= 0
            )
            & (
                boundary_triangle_ids
                < triangle_count
            )
        ]
    )

    visible_vertex_mask = np.zeros(
        vertex_count,
        dtype=bool,
    )

    boundary_vertex_mask = np.zeros(
        vertex_count,
        dtype=bool,
    )

    if (
        visible_triangle_ids.size
        > 0
    ):
        visible_vertex_mask[
            np.unique(
                triangles[
                    visible_triangle_ids
                ].reshape(-1)
            )
        ] = True

    if (
        boundary_triangle_ids.size
        > 0
    ):
        boundary_vertex_mask[
            np.unique(
                triangles[
                    boundary_triangle_ids
                ].reshape(-1)
            )
        ] = True

    # Boundary는 반드시 visible 영역 안에만 존재하게 한다.
    boundary_vertex_mask &= (
        visible_vertex_mask
    )

    return (
        visible_vertex_mask,
        boundary_vertex_mask,
        rendered_mask,
        rendered_depth,
    )


def _build_adjacency(
    triangles: np.ndarray,
    vertex_count: int,
) -> list[np.ndarray]:
    neighbor_sets: list[set[int]] = [
        set()
        for _ in range(vertex_count)
    ]

    for a, b, c in np.asarray(
        triangles,
        dtype=np.int64,
    ):
        ai = int(a)
        bi = int(b)
        ci = int(c)

        neighbor_sets[ai].update((bi, ci))
        neighbor_sets[bi].update((ai, ci))
        neighbor_sets[ci].update((ai, bi))

    return [
        np.fromiter(
            sorted(neighbors),
            dtype=np.int64,
        )
        for neighbors in neighbor_sets
    ]


def _expand_topology_mask(
    *,
    seed_mask: np.ndarray,
    adjacency: list[np.ndarray],
    ring_count: int,
) -> np.ndarray:
    expanded = np.asarray(
        seed_mask,
        dtype=bool,
    ).copy()

    frontier = np.flatnonzero(
        expanded
    ).tolist()

    for _ in range(
        max(0, int(ring_count))
    ):
        next_frontier: list[int] = []

        for vertex_index in frontier:
            for neighbor in adjacency[
                vertex_index
            ]:
                neighbor_index = int(
                    neighbor
                )

                if expanded[
                    neighbor_index
                ]:
                    continue

                expanded[
                    neighbor_index
                ] = True

                next_frontier.append(
                    neighbor_index
                )

        if not next_frontier:
            break

        frontier = next_frontier

    return expanded


def _circular_fill(
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    result = np.asarray(
        values,
        dtype=np.float64,
    ).copy()

    valid = np.asarray(
        valid,
        dtype=bool,
    )

    count = len(result)

    if count == 0 or not valid.any():
        return result

    valid_indices = np.flatnonzero(
        valid
    )

    for index in np.flatnonzero(
        ~valid
    ):
        circular_distance = np.minimum(
            np.abs(
                valid_indices - index
            ),
            count
            - np.abs(
                valid_indices - index
            ),
        )

        nearest = int(
            np.argmin(
                circular_distance
            )
        )

        result[index] = result[
            valid_indices[nearest]
        ]

    return result


def _dense_silhouette_strip_targets(
    *,
    points_camera: np.ndarray,
    band_indices: np.ndarray,
    u_pixel: np.ndarray,
    v_pixel: np.ndarray,
    mask_bool: np.ndarray,
    camera_k: np.ndarray,
    sector_count: int,
    sector_smoothing_sigma: float,
    maximum_pixel_move: float,
    minimum_pixel_move: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict,
]:
    if len(band_indices) == 0:
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
                "boundary_strip_vertex_count": 0,
                "active_sector_count": 0,
                "sector_move_p90_px": 0.0,
                "sector_move_max_px": 0.0,
            },
        )

    sdf = _signed_distance_to_mask(
        mask_bool
    )

    smoothed_sdf = (
        scipy.ndimage.gaussian_filter(
            sdf,
            sigma=2.0,
        )
    )

    grad_v, grad_u = np.gradient(
        smoothed_sdf
    )

    band_u = u_pixel[
        band_indices
    ].astype(np.float64)

    band_v = v_pixel[
        band_indices
    ].astype(np.float64)

    center_u = float(
        np.median(band_u)
    )
    center_v = float(
        np.median(band_v)
    )

    angle = np.arctan2(
        band_v - center_v,
        band_u - center_u,
    )

    sector_coordinate = (
        (
            angle + np.pi
        )
        / (2.0 * np.pi)
        * sector_count
    ) % sector_count

    sector_index = np.floor(
        sector_coordinate
    ).astype(np.int64)

    px_u = u_pixel[
        band_indices
    ]
    px_v = v_pixel[
        band_indices
    ]

    residual_px = sdf[
        px_v,
        px_u,
    ]

    gu = grad_u[
        px_v,
        px_u,
    ]
    gv = grad_v[
        px_v,
        px_u,
    ]

    grad_norm = np.clip(
        np.hypot(gu, gv),
        1e-6,
        None,
    )

    raw_du = (
        -(gu / grad_norm)
        * residual_px
    )

    raw_dv = (
        -(gv / grad_norm)
        * residual_px
    )

    sector_du = np.zeros(
        sector_count,
        dtype=np.float64,
    )

    sector_dv = np.zeros(
        sector_count,
        dtype=np.float64,
    )

    sector_valid = np.zeros(
        sector_count,
        dtype=bool,
    )

    for current_sector in range(
        sector_count
    ):
        members = (
            sector_index
            == current_sector
        )

        if not members.any():
            continue

        sector_du[
            current_sector
        ] = float(
            np.median(
                raw_du[members]
            )
        )

        sector_dv[
            current_sector
        ] = float(
            np.median(
                raw_dv[members]
            )
        )

        sector_valid[
            current_sector
        ] = True

    sector_du = _circular_fill(
        sector_du,
        sector_valid,
    )

    sector_dv = _circular_fill(
        sector_dv,
        sector_valid,
    )

    sector_du = (
        scipy.ndimage
        .gaussian_filter1d(
            sector_du,
            sigma=(
                sector_smoothing_sigma
            ),
            mode="wrap",
        )
    )

    sector_dv = (
        scipy.ndimage
        .gaussian_filter1d(
            sector_dv,
            sigma=(
                sector_smoothing_sigma
            ),
            mode="wrap",
        )
    )

    sector_norm = np.hypot(
        sector_du,
        sector_dv,
    )

    clip_scale = np.minimum(
        1.0,
        maximum_pixel_move
        / np.clip(
            sector_norm,
            1e-9,
            None,
        ),
    )

    sector_du *= clip_scale
    sector_dv *= clip_scale

    sector_norm = np.hypot(
        sector_du,
        sector_dv,
    )

    inactive = (
        sector_norm
        < minimum_pixel_move
    )

    sector_du[inactive] = 0.0
    sector_dv[inactive] = 0.0

    left = (
        np.floor(
            sector_coordinate
        ).astype(np.int64)
        % sector_count
    )

    right = (
        left + 1
    ) % sector_count

    alpha = (
        sector_coordinate
        - np.floor(
            sector_coordinate
        )
    )

    # 인접 sector 사이를 선형 보간한다.
    # sector 경계에서도 이동장이 연속적이다.
    du = (
        (1.0 - alpha)
        * sector_du[left]
        + alpha
        * sector_du[right]
    )

    dv = (
        (1.0 - alpha)
        * sector_dv[left]
        + alpha
        * sector_dv[right]
    )

    fx = float(
        camera_k[0, 0]
    )
    fy = float(
        camera_k[1, 1]
    )

    target = points_camera[
        band_indices
    ].copy()

    target[:, 0] += (
        du
        * target[:, 2]
        / fx
    )

    target[:, 1] += (
        dv
        * target[:, 2]
        / fy
    )

    return (
        np.asarray(
            band_indices,
            dtype=np.int64,
        ),
        target,
        {
            "boundary_strip_vertex_count":
                int(len(band_indices)),
            "active_sector_count":
                int(
                    np.count_nonzero(
                        sector_norm > 0.0
                    )
                ),
            "sector_move_p90_px":
                float(
                    np.quantile(
                        sector_norm,
                        0.90,
                    )
                ),
            "sector_move_max_px":
                float(
                    sector_norm.max()
                ),
        },
    )


def _build_local_depth_grid(
    *,
    candidate_indices: np.ndarray,
    u_pixel: np.ndarray,
    v_pixel: np.ndarray,
    residual: np.ndarray,
    image_height: int,
    image_width: int,
    grid_size_px: int,
    minimum_samples_per_cell: int,
    smoothing_sigma_cells: float,
    maximum_depth_move_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """
    해석 B:

    1. 16x16px cell별 residual median
    2. 유효 cell grid를 Gaussian smoothing
    3. 각 cell 대표 정점 하나를 handle로 사용

    전체 영상에 평면 하나를 fitting하지 않는다.
    """
    grid_height = int(
        np.ceil(
            image_height
            / grid_size_px
        )
    )

    grid_width = int(
        np.ceil(
            image_width
            / grid_size_px
        )
    )

    cell_u = np.clip(
        u_pixel[
            candidate_indices
        ]
        // grid_size_px,
        0,
        grid_width - 1,
    )

    cell_v = np.clip(
        v_pixel[
            candidate_indices
        ]
        // grid_size_px,
        0,
        grid_height - 1,
    )

    raw_grid = np.zeros(
        (grid_height, grid_width),
        dtype=np.float64,
    )

    valid_grid = np.zeros(
        (grid_height, grid_width),
        dtype=bool,
    )

    count_grid = np.zeros(
        (grid_height, grid_width),
        dtype=np.int64,
    )

    flat_key = (
        cell_v * grid_width
        + cell_u
    )

    for key in np.unique(
        flat_key
    ):
        members = (
            flat_key == key
        )

        row = int(
            key // grid_width
        )

        column = int(
            key % grid_width
        )

        count = int(
            np.count_nonzero(
                members
            )
        )

        count_grid[
            row,
            column,
        ] = count

        if (
            count
            < minimum_samples_per_cell
        ):
            continue

        raw_grid[
            row,
            column,
        ] = float(
            np.median(
                residual[members]
            )
        )

        valid_grid[
            row,
            column,
        ] = True

    if not valid_grid.any():
        return (
            raw_grid,
            valid_grid,
            count_grid,
            {
                "valid_depth_cell_count": 0,
                "depth_grid_abs_p90_m": None,
                "depth_grid_abs_max_m": None,
            },
        )

    weighted_values = (
        raw_grid
        * valid_grid.astype(
            np.float64
        )
    )

    numerator = (
        scipy.ndimage
        .gaussian_filter(
            weighted_values,
            sigma=(
                smoothing_sigma_cells
            ),
            mode="nearest",
        )
    )

    denominator = (
        scipy.ndimage
        .gaussian_filter(
            valid_grid.astype(
                np.float64
            ),
            sigma=(
                smoothing_sigma_cells
            ),
            mode="nearest",
        )
    )

    smooth_grid = np.zeros_like(
        raw_grid
    )

    support = (
        denominator > 1e-6
    )

    smooth_grid[support] = (
        numerator[support]
        / denominator[support]
    )

    smooth_grid = np.clip(
        smooth_grid,
        -maximum_depth_move_m,
        maximum_depth_move_m,
    )

    valid_values = smooth_grid[
        valid_grid
    ]

    return (
        smooth_grid,
        valid_grid,
        count_grid,
        {
            "valid_depth_cell_count":
                int(
                    np.count_nonzero(
                        valid_grid
                    )
                ),
            "depth_grid_abs_p90_m":
                float(
                    np.quantile(
                        np.abs(
                            valid_values
                        ),
                        0.90,
                    )
                ),
            "depth_grid_abs_max_m":
                float(
                    np.max(
                        np.abs(
                            valid_values
                        )
                    )
                ),
        },
    )


def _local_depth_grid_handles(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    interior_indices_all: np.ndarray,
    u_pixel: np.ndarray,
    v_pixel: np.ndarray,
    masked_depth_m: np.ndarray | None,
    image_height: int,
    image_width: int,
    grid_size_px: int,
    minimum_samples_per_cell: int,
    smoothing_sigma_cells: float,
    maximum_depth_move_m: float,
    outlier_neighbor_multiplier: float,
    outlier_minimum_threshold_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict,
]:
    empty_indices = np.empty(
        0,
        dtype=np.int64,
    )

    empty_targets = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    default_stats = {
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

    if (
        masked_depth_m is None
        or len(
            interior_indices_all
        ) == 0
    ):
        return (
            empty_indices,
            empty_targets,
            default_stats,
        )

    filtered_depth = (
        scipy.ndimage
        .median_filter(
            masked_depth_m,
            size=3,
        )
    )

    observed = filtered_depth[
        v_pixel[
            interior_indices_all
        ],
        u_pixel[
            interior_indices_all
        ],
    ]

    has_depth = (
        observed > 0.0
    )

    candidates = (
        interior_indices_all[
            has_depth
        ]
    )

    observed = observed[
        has_depth
    ]

    if len(candidates) == 0:
        return (
            empty_indices,
            empty_targets,
            default_stats,
        )

    residual = (
        observed
        - points_camera[
            candidates,
            2,
        ]
    )

    (
        inlier_mask,
        outlier_stats,
    ) = _reject_depth_outliers(
        vertex_count=len(
            points_camera
        ),
        triangles=triangles,
        active_indices=candidates,
        residual=residual,
        neighbor_multiplier=(
            outlier_neighbor_multiplier
        ),
        minimum_threshold_m=(
            outlier_minimum_threshold_m
        ),
    )

    candidates = candidates[
        inlier_mask
    ]

    residual = residual[
        inlier_mask
    ]

    if len(candidates) == 0:
        return (
            empty_indices,
            empty_targets,
            {
                **default_stats,
                **outlier_stats,
            },
        )

    (
        smooth_grid,
        valid_grid,
        _,
        grid_stats,
    ) = _build_local_depth_grid(
        candidate_indices=candidates,
        u_pixel=u_pixel,
        v_pixel=v_pixel,
        residual=residual,
        image_height=image_height,
        image_width=image_width,
        grid_size_px=grid_size_px,
        minimum_samples_per_cell=(
            minimum_samples_per_cell
        ),
        smoothing_sigma_cells=(
            smoothing_sigma_cells
        ),
        maximum_depth_move_m=(
            maximum_depth_move_m
        ),
    )

    if not valid_grid.any():
        return (
            empty_indices,
            empty_targets,
            {
                **default_stats,
                **outlier_stats,
                **grid_stats,
                "depth_candidate_count":
                    int(
                        len(candidates)
                    ),
            },
        )

    (
        grid_height,
        grid_width,
    ) = valid_grid.shape

    cell_u = np.clip(
        u_pixel[candidates]
        // grid_size_px,
        0,
        grid_width - 1,
    )

    cell_v = np.clip(
        v_pixel[candidates]
        // grid_size_px,
        0,
        grid_height - 1,
    )

    flat_key = (
        cell_v * grid_width
        + cell_u
    )

    handle_indices: list[int] = []
    handle_residuals: list[float] = []

    for key in np.unique(
        flat_key
    ):
        row = int(
            key // grid_width
        )

        column = int(
            key % grid_width
        )

        if not valid_grid[
            row,
            column,
        ]:
            continue

        members = np.flatnonzero(
            flat_key == key
        )

        center_u = (
            column + 0.5
        ) * grid_size_px

        center_v = (
            row + 0.5
        ) * grid_size_px

        candidate_u = u_pixel[
            candidates[members]
        ].astype(np.float64)

        candidate_v = v_pixel[
            candidates[members]
        ].astype(np.float64)

        representative_local = int(
            members[
                np.argmin(
                    (
                        candidate_u
                        - center_u
                    ) ** 2
                    + (
                        candidate_v
                        - center_v
                    ) ** 2
                )
            ]
        )

        handle_indices.append(
            int(
                candidates[
                    representative_local
                ]
            )
        )

        handle_residuals.append(
            float(
                smooth_grid[
                    row,
                    column,
                ]
            )
        )

    indices = np.asarray(
        handle_indices,
        dtype=np.int64,
    )

    depth_shift = np.asarray(
        handle_residuals,
        dtype=np.float64,
    )

    if len(indices) == 0:
        return (
            empty_indices,
            empty_targets,
            {
                **default_stats,
                **outlier_stats,
                **grid_stats,
            },
        )

    source = points_camera[
        indices
    ]

    target_z = np.clip(
        source[:, 2]
        + depth_shift,
        1e-6,
        None,
    )

    ray_scale = (
        target_z
        / np.clip(
            source[:, 2],
            1e-6,
            None,
        )
    )

    target = (
        source
        * ray_scale[:, None]
    )

    return (
        indices,
        target,
        {
            **default_stats,
            **outlier_stats,
            **grid_stats,
            "depth_candidate_count":
                int(len(candidates)),
            "depth_handle_count":
                int(len(indices)),
        },
    )


def _sample_indices(
    indices: np.ndarray,
    maximum_count: int,
) -> np.ndarray:
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    if len(indices) <= maximum_count:
        return indices

    positions = np.linspace(
        0,
        len(indices) - 1,
        maximum_count,
    ).round().astype(np.int64)

    return indices[positions]


def _merge_constraints(
    *,
    boundary_indices: np.ndarray,
    boundary_targets: np.ndarray,
    depth_indices: np.ndarray,
    depth_targets: np.ndarray,
    anchor_indices: np.ndarray,
    points_camera: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    target_by_index: dict[
        int,
        np.ndarray,
    ] = {}

    for index, target in zip(
        boundary_indices,
        boundary_targets,
    ):
        target_by_index[
            int(index)
        ] = np.asarray(
            target,
            dtype=np.float64,
        )

    # Interior depth handle은 boundary strip과
    # 겹치지 않도록 구성되어 있지만,
    # 안전하게 기존 constraint를 우선한다.
    for index, target in zip(
        depth_indices,
        depth_targets,
    ):
        index_int = int(index)

        if (
            index_int
            not in target_by_index
        ):
            target_by_index[
                index_int
            ] = np.asarray(
                target,
                dtype=np.float64,
            )

    for index in anchor_indices:
        index_int = int(index)

        if (
            index_int
            not in target_by_index
        ):
            target_by_index[
                index_int
            ] = points_camera[
                index_int
            ].copy()

    ordered_indices = np.asarray(
        sorted(
            target_by_index
        ),
        dtype=np.int64,
    )

    ordered_targets = np.stack(
        [
            target_by_index[
                int(index)
            ]
            for index
            in ordered_indices
        ],
        axis=0,
    )

    return (
        ordered_indices,
        ordered_targets,
    )


def _unique_edges(
    triangles: np.ndarray,
) -> np.ndarray:
    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        axis=0,
    )

    edges = np.sort(
        edges.astype(
            np.int64,
            copy=False,
        ),
        axis=1,
    )

    return np.unique(
        edges,
        axis=0,
    )


def _safe_global_displacement_step(
    *,
    original_points: np.ndarray,
    proposed_points: np.ndarray,
    triangles: np.ndarray,
    maximum_displacement_m: float,
    minimum_edge_ratio: float,
    maximum_edge_ratio: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
    minimum_step_scale: float,
) -> tuple[
    np.ndarray,
    dict,
]:
    """
    ARAP 결과 displacement field 전체에
    동일한 alpha를 곱한다.

        L(alpha*d) = alpha*L(d)

    정점별 독립 clip을 사용하지 않는다.

    Face reversal이나 과도한 edge/area 변화가 있으면
    alpha를 절반씩 줄이는 backtracking을 수행한다.
    """
    raw_displacement = (
        proposed_points
        - original_points
    )

    raw_norm = np.linalg.norm(
        raw_displacement,
        axis=1,
    )

    initial_scale = 1.0

    if (
        raw_norm.size
        and raw_norm.max()
        > maximum_displacement_m
    ):
        initial_scale = (
            maximum_displacement_m
            / float(
                raw_norm.max()
            )
        )

    edges = _unique_edges(
        triangles
    )

    original_edge = np.linalg.norm(
        original_points[
            edges[:, 0]
        ]
        - original_points[
            edges[:, 1]
        ],
        axis=1,
    )

    valid_edges = (
        original_edge > 1e-12
    )

    original_faces = (
        original_points[
            triangles
        ]
    )

    original_cross = np.cross(
        original_faces[:, 1]
        - original_faces[:, 0],
        original_faces[:, 2]
        - original_faces[:, 0],
    )

    original_area2 = np.linalg.norm(
        original_cross,
        axis=1,
    )

    valid_faces = (
        original_area2 > 1e-14
    )

    scale = initial_scale

    accepted = (
        original_points.copy()
    )

    safety_stats: dict[
        str,
        float | int | bool,
    ] = {}

    while (
        scale
        >= minimum_step_scale
        - 1e-12
    ):
        candidate = (
            original_points
            + scale
            * raw_displacement
        )

        candidate_edge = np.linalg.norm(
            candidate[
                edges[:, 0]
            ]
            - candidate[
                edges[:, 1]
            ],
            axis=1,
        )

        edge_ratio = (
            candidate_edge[
                valid_edges
            ]
            / original_edge[
                valid_edges
            ]
        )

        candidate_faces = candidate[
            triangles
        ]

        candidate_cross = np.cross(
            candidate_faces[:, 1]
            - candidate_faces[:, 0],
            candidate_faces[:, 2]
            - candidate_faces[:, 0],
        )

        candidate_area2 = np.linalg.norm(
            candidate_cross,
            axis=1,
        )

        area_ratio = (
            candidate_area2[
                valid_faces
            ]
            / original_area2[
                valid_faces
            ]
        )

        orientation_dot = np.einsum(
            "ij,ij->i",
            original_cross,
            candidate_cross,
        )

        reversal_count = int(
            np.count_nonzero(
                valid_faces
                & (
                    candidate_area2
                    > 1e-14
                )
                & (
                    orientation_dot
                    < 0.0
                )
            )
        )

        safe = (
            reversal_count == 0
            and (
                edge_ratio.size == 0
                or float(
                    edge_ratio.min()
                )
                >= minimum_edge_ratio
            )
            and (
                edge_ratio.size == 0
                or float(
                    edge_ratio.max()
                )
                <= maximum_edge_ratio
            )
            and (
                area_ratio.size == 0
                or float(
                    area_ratio.min()
                )
                >= minimum_area_ratio
            )
            and (
                area_ratio.size == 0
                or float(
                    area_ratio.max()
                )
                <= maximum_area_ratio
            )
        )

        safety_stats = {
            "step_scale":
                float(scale),
            "face_reversal_count":
                reversal_count,
            "edge_ratio_min": (
                float(
                    edge_ratio.min()
                )
                if edge_ratio.size
                else 1.0
            ),
            "edge_ratio_max": (
                float(
                    edge_ratio.max()
                )
                if edge_ratio.size
                else 1.0
            ),
            "area_ratio_min": (
                float(
                    area_ratio.min()
                )
                if area_ratio.size
                else 1.0
            ),
            "area_ratio_max": (
                float(
                    area_ratio.max()
                )
                if area_ratio.size
                else 1.0
            ),
            "topology_safe":
                bool(safe),
        }

        if safe:
            accepted = candidate
            break

        scale *= 0.5

    if not safety_stats.get(
        "topology_safe",
        False,
    ):
        accepted = (
            original_points.copy()
        )

        safety_stats[
            "step_scale"
        ] = 0.0

        safety_stats[
            "topology_safe"
        ] = False

    return accepted, safety_stats


def _run_arap(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    constraint_indices: np.ndarray,
    constraint_targets: np.ndarray,
    iteration_count: int,
) -> np.ndarray:
    if len(
        constraint_indices
    ) < 3:
        return points_camera.copy()

    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = (
        o3d.utility.Vector3dVector(
            points_camera.astype(
                np.float64
            )
        )
    )

    mesh.triangles = (
        o3d.utility.Vector3iVector(
            triangles.astype(
                np.int32
            )
        )
    )

    if not hasattr(
        mesh,
        "deform_as_rigid_as_possible",
    ):
        raise RuntimeError(
            "Installed Open3D does not provide "
            "TriangleMesh.deform_as_rigid_as_possible"
        )

    constraint_id_vector = (
        o3d.utility.IntVector(
            constraint_indices.astype(
                np.int32
            ).tolist()
        )
    )

    constraint_position_vector = (
        o3d.utility.Vector3dVector(
            constraint_targets.astype(
                np.float64
            )
        )
    )

    try:
        deformed = (
            mesh
            .deform_as_rigid_as_possible(
                constraint_id_vector,
                constraint_position_vector,
                int(iteration_count),
            )
        )
    except TypeError:
        # Open3D 버전에 따라 max_iter가
        # keyword-only인 경우를 지원한다.
        deformed = (
            mesh
            .deform_as_rigid_as_possible(
                constraint_id_vector,
                constraint_position_vector,
                max_iter=int(
                    iteration_count
                ),
            )
        )

    result = np.asarray(
        deformed.vertices,
        dtype=np.float64,
    )

    if (
        result.shape
        != points_camera.shape
        or not np.isfinite(
            result
        ).all()
    ):
        raise RuntimeError(
            "ARAP returned invalid vertices"
        )

    return result


def refine_mesh_with_dense_strip_arap(
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
    max_silhouette_pixel_displacement: float = 1.5,
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
    # 기존 함수와 같은 호출 계약을 유지하기 위한 인자.
    del (
        huber_boundary_px,
        huber_depth_m,
        laplacian_weight,
        hidden_anchor_weight,
    )

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

    vertex_count = len(points)

    (
        image_height,
        image_width,
    ) = mask.shape

    (
        visible,
        in_band,
        rendered_triangle_mask,
        rendered_triangle_depth,
    ) = (
        _rasterize_visible_and_boundary_vertices(
            points_camera=points,
            triangles=triangles,
            camera_k=camera_k,
            image_height=image_height,
            image_width=image_width,
        )
    )

    z_coordinate = points[:, 2]

    valid_z = (
        np.isfinite(
            z_coordinate
        )
        & (
            z_coordinate
            > 1e-8
        )
    )

    u_pixel = np.full(
        vertex_count,
        -1,
        dtype=np.int64,
    )

    v_pixel = np.full(
        vertex_count,
        -1,
        dtype=np.int64,
    )

    fx = float(
        camera_k[0, 0]
    )

    fy = float(
        camera_k[1, 1]
    )

    cx = float(
        camera_k[0, 2]
    )

    cy = float(
        camera_k[1, 2]
    )

    u_pixel[valid_z] = np.round(
        points[
            valid_z,
            0,
        ]
        / z_coordinate[
            valid_z
        ]
        * fx
        + cx
    ).astype(np.int64)

    v_pixel[valid_z] = np.round(
        points[
            valid_z,
            1,
        ]
        / z_coordinate[
            valid_z
        ]
        * fy
        + cy
    ).astype(np.int64)

    in_image = (
        valid_z
        & (
            u_pixel >= 0
        )
        & (
            u_pixel
            < image_width
        )
        & (
            v_pixel >= 0
        )
        & (
            v_pixel
            < image_height
        )
    )

    # 실제 rasterized silhouette boundary triangle에
    # 속하는 정점만 silhouette handle로 사용한다.
    band_indices = np.flatnonzero(
        in_band
        & in_image
    )

    inside_observed_mask = np.zeros(
        vertex_count,
        dtype=bool,
    )

    visible_in_image = (
        visible
        & in_image
    )

    inside_observed_mask[
        visible_in_image
    ] = mask[
        v_pixel[
            visible_in_image
        ],
        u_pixel[
            visible_in_image
        ],
    ]

    interior_mask = (
        visible
        & in_image
        & inside_observed_mask
        & ~in_band
    )

    interior_indices_all = (
        np.flatnonzero(
            interior_mask
        )
    )

    (
        iou_before,
        boundary_before,
    ) = _quality(
        points_camera=points,
        mask_bool=mask,
        camera_k=camera_k,
    )

    (
        boundary_indices,
        boundary_targets,
        boundary_stats,
    ) = (
        _dense_silhouette_strip_targets(
            points_camera=points,
            band_indices=band_indices,
            u_pixel=u_pixel,
            v_pixel=v_pixel,
            mask_bool=mask,
            camera_k=camera_k,
            sector_count=(
                silhouette_sector_count
            ),
            sector_smoothing_sigma=(
                silhouette_sector_smoothing_sigma
            ),
            maximum_pixel_move=(
                max_silhouette_pixel_displacement
            ),
            minimum_pixel_move=(
                silhouette_minimum_pixel_move
            ),
        )
    )

    (
        depth_indices,
        depth_targets,
        depth_stats,
    ) = _local_depth_grid_handles(
        points_camera=points,
        triangles=triangles,
        interior_indices_all=(
            interior_indices_all
        ),
        u_pixel=u_pixel,
        v_pixel=v_pixel,
        masked_depth_m=masked_depth_m,
        image_height=image_height,
        image_width=image_width,
        grid_size_px=(
            depth_grid_size_px
        ),
        minimum_samples_per_cell=(
            depth_minimum_samples_per_cell
        ),
        smoothing_sigma_cells=(
            depth_grid_smoothing_sigma_cells
        ),
        maximum_depth_move_m=(
            maximum_depth_move_m
        ),
        outlier_neighbor_multiplier=(
            depth_outlier_neighbor_multiplier
        ),
        outlier_minimum_threshold_m=(
            depth_outlier_minimum_threshold_m
        ),
    )

    adjacency = _build_adjacency(
        triangles,
        vertex_count,
    )

    # Visible 표면에서 transition_ring_count ring까지는
    # 자유롭게 따라오도록 두고, 그보다 먼 hidden 면만 고정한다.
    transition_mask = (
        _expand_topology_mask(
            seed_mask=visible,
            adjacency=adjacency,
            ring_count=(
                transition_ring_count
            ),
        )
    )

    far_hidden_indices = (
        np.flatnonzero(
            ~transition_mask
        )
    )

    anchor_indices = _sample_indices(
        far_hidden_indices,
        maximum_anchor_count,
    )

    (
        constraint_indices,
        constraint_targets,
    ) = _merge_constraints(
        boundary_indices=(
            boundary_indices
        ),
        boundary_targets=(
            boundary_targets
        ),
        depth_indices=depth_indices,
        depth_targets=depth_targets,
        anchor_indices=anchor_indices,
        points_camera=points,
    )

    arap_raw = _run_arap(
        points_camera=points,
        triangles=triangles,
        constraint_indices=(
            constraint_indices
        ),
        constraint_targets=(
            constraint_targets
        ),
        iteration_count=(
            arap_iteration_count
        ),
    )

    (
        refined_pre_scale,
        safety_stats,
    ) = _safe_global_displacement_step(
        original_points=points,
        proposed_points=arap_raw,
        triangles=triangles,
        maximum_displacement_m=(
            maximum_displacement_m
        ),
        minimum_edge_ratio=(
            minimum_edge_ratio
        ),
        maximum_edge_ratio=(
            maximum_edge_ratio
        ),
        minimum_area_ratio=(
            minimum_area_ratio
        ),
        maximum_area_ratio=(
            maximum_area_ratio
        ),
        minimum_step_scale=(
            minimum_step_scale
        ),
    )

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
                refined - centroid
            )
        )

        scale_after = float(
            diameter_fn(refined)
        )

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
        points_camera=refined,
        mask_bool=mask,
        camera_k=camera_k,
    )

    laplacian = (
        _build_uniform_laplacian(
            triangles,
            vertex_count,
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
            if safety_stats[
                "step_scale"
            ] > 0.0
            else (
                "topology_safety_"
                "fallback"
            )
        ),
        "refinement_mode":
            "dense_strip_arap",
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
                    boundary_indices
                )
            ),
        "interior_vertex_count":
            int(
                len(
                    interior_indices_all
                )
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
            int(
                np.count_nonzero(
                    moved
                )
            ),
        "inward_fraction": (
            float(
                np.mean(
                    radial_component[
                        moved
                    ] < 0.0
                )
            )
            if moved.any()
            else None
        ),
        "outward_fraction": (
            float(
                np.mean(
                    radial_component[
                        moved
                    ] > 0.0
                )
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
        "silhouette_strip":
            boundary_stats,
        "depth_grid":
            depth_stats,
        "transition_ring_count":
            int(
                transition_ring_count
            ),
        "far_hidden_anchor_count":
            int(
                len(
                    anchor_indices
                )
            ),
        "constraint_count":
            int(
                len(
                    constraint_indices
                )
            ),
        "topology_safety":
            safety_stats,
    }

    intermediate_points_camera = {
        "M0_original_camera":
            points.copy(),
        "M1_arap_raw_camera":
            arap_raw.copy(),
        "M2_arap_safe_pre_scale_camera":
            refined_pre_scale.copy(),
        "M3_arap_safe_post_scale_camera":
            refined.copy(),
    }

    return SilhouetteMeshRefinementResult(
        refined_points_camera=refined,
        displacement=displacement,
        diagnostics=diagnostics,
        intermediate_points_camera=(
            intermediate_points_camera
        ),
    )
