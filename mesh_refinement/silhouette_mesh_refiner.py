from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg

from mesh_refinement.depth_anchored_visible_refiner import (
    _build_uniform_laplacian,
    _visible_vertex_mask,
)


@dataclass(frozen=True)
class SilhouetteMeshRefinementResult:
    refined_points_camera: np.ndarray
    displacement: np.ndarray
    diagnostics: dict
    intermediate_points_camera: dict[str, np.ndarray]


def _unique_undirected_edges(
    triangles: np.ndarray,
) -> np.ndarray:
    triangles = np.asarray(
        triangles,
        dtype=np.int64,
    )

    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        axis=0,
    )

    edges = np.sort(
        edges,
        axis=1,
    )

    return np.unique(
        edges,
        axis=0,
    )


def _quantile_or_none(
    values: np.ndarray,
    quantile: float,
) -> float | None:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        return None

    return float(
        np.quantile(
            values,
            quantile,
        )
    )


def _geometry_stage_diagnostics(
    *,
    reference_points: np.ndarray,
    stage_points: np.ndarray,
    triangles: np.ndarray,
    laplacian: scipy.sparse.csr_matrix,
) -> dict:
    reference_points = np.asarray(
        reference_points,
        dtype=np.float64,
    )

    stage_points = np.asarray(
        stage_points,
        dtype=np.float64,
    )

    triangles = np.asarray(
        triangles,
        dtype=np.int64,
    )

    displacement = (
        stage_points
        - reference_points
    )

    laplacian_displacement = np.asarray(
        laplacian @ displacement,
        dtype=np.float64,
    )

    roughness = np.linalg.norm(
        laplacian_displacement,
        axis=1,
    )

    edges = _unique_undirected_edges(
        triangles
    )

    reference_edge_lengths = np.linalg.norm(
        reference_points[edges[:, 0]]
        - reference_points[edges[:, 1]],
        axis=1,
    )

    stage_edge_lengths = np.linalg.norm(
        stage_points[edges[:, 0]]
        - stage_points[edges[:, 1]],
        axis=1,
    )

    valid_edges = (
        reference_edge_lengths
        > 1e-12
    )

    edge_ratios = (
        stage_edge_lengths[valid_edges]
        / reference_edge_lengths[valid_edges]
    )

    reference_faces = (
        reference_points[triangles]
    )

    stage_faces = (
        stage_points[triangles]
    )

    reference_cross = np.cross(
        reference_faces[:, 1]
        - reference_faces[:, 0],
        reference_faces[:, 2]
        - reference_faces[:, 0],
    )

    stage_cross = np.cross(
        stage_faces[:, 1]
        - stage_faces[:, 0],
        stage_faces[:, 2]
        - stage_faces[:, 0],
    )

    reference_area2 = np.linalg.norm(
        reference_cross,
        axis=1,
    )

    stage_area2 = np.linalg.norm(
        stage_cross,
        axis=1,
    )

    valid_faces = (
        reference_area2
        > 1e-14
    )

    area_ratios = (
        stage_area2[valid_faces]
        / reference_area2[valid_faces]
    )

    orientation_dot = np.einsum(
        "ij,ij->i",
        reference_cross,
        stage_cross,
    )

    flipped_mask = (
        valid_faces
        & (stage_area2 > 1e-14)
        & (orientation_dot < 0.0)
    )

    degenerate_threshold = np.maximum(
        reference_area2 * 1e-3,
        1e-14,
    )

    degenerate_mask = (
        stage_area2
        <= degenerate_threshold
    )

    return {
        "laplacian_roughness_p50_m":
            _quantile_or_none(
                roughness,
                0.50,
            ),
        "laplacian_roughness_p95_m":
            _quantile_or_none(
                roughness,
                0.95,
            ),
        "laplacian_roughness_p99_m":
            _quantile_or_none(
                roughness,
                0.99,
            ),
        "laplacian_roughness_max_m": (
            float(roughness.max())
            if roughness.size
            else None
        ),
        "edge_ratio_p01":
            _quantile_or_none(
                edge_ratios,
                0.01,
            ),
        "edge_ratio_p50":
            _quantile_or_none(
                edge_ratios,
                0.50,
            ),
        "edge_ratio_p99":
            _quantile_or_none(
                edge_ratios,
                0.99,
            ),
        "edge_below_half_fraction": (
            float(
                np.mean(
                    edge_ratios < 0.5
                )
            )
            if edge_ratios.size
            else None
        ),
        "edge_over_double_fraction": (
            float(
                np.mean(
                    edge_ratios > 2.0
                )
            )
            if edge_ratios.size
            else None
        ),
        "triangle_area_ratio_p01":
            _quantile_or_none(
                area_ratios,
                0.01,
            ),
        "triangle_area_ratio_p50":
            _quantile_or_none(
                area_ratios,
                0.50,
            ),
        "triangle_area_ratio_p99":
            _quantile_or_none(
                area_ratios,
                0.99,
            ),
        "flipped_triangle_count": int(
            np.count_nonzero(
                flipped_mask
            )
        ),
        "degenerate_triangle_count": int(
            np.count_nonzero(
                degenerate_mask
            )
        ),
    }


def _signed_distance_to_mask(mask_bool: np.ndarray) -> np.ndarray:
    """Positive outside the mask, negative inside; 0 at the boundary."""
    outside = scipy.ndimage.distance_transform_edt(~mask_bool)
    inside = scipy.ndimage.distance_transform_edt(mask_bool)
    return outside - inside


def _reject_depth_outliers(
    *,
    vertex_count: int,
    triangles: np.ndarray,
    active_indices: np.ndarray,
    residual: np.ndarray,
    neighbor_multiplier: float,
    minimum_threshold_m: float,
) -> tuple[np.ndarray, dict]:
    """
    Mesh 위상적으로 이웃한 depth-active 정점끼리 residual을 비교해서,
    이웃 평균과 크게 어긋나는 정점(센서 flying-pixel, mesh 국소 디테일
    불일치 등)을 depth correspondence에서 제외한다.

    Huber damping은 개별 정점의 residual '크기'만 완화할 뿐 그 정점이
    이웃과 모순되는 이상치인지는 판단하지 못한다.

    Returns
    -------
    inlier_mask : active_indices와 같은 길이의 bool -- True면 depth
        correspondence를 그대로 사용해도 되는 정점.
    stats : 진단용 dict.
    """
    residual_full = np.zeros(vertex_count)
    in_active = np.zeros(vertex_count, dtype=bool)
    residual_full[active_indices] = residual
    in_active[active_indices] = True

    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
        axis=0,
    )
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    edges = edges[in_active[edges[:, 0]] & in_active[edges[:, 1]]]

    adjacency = scipy.sparse.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    has_neighbors = degree > 0

    neighbor_mean = np.zeros(vertex_count)
    neighbor_mean[has_neighbors] = (
        (adjacency @ residual_full)[has_neighbors] / degree[has_neighbors]
    )

    deviation = np.abs(residual_full - neighbor_mean)
    evaluable = in_active & has_neighbors

    robust_scale = (
        float(np.median(deviation[evaluable])) if evaluable.any() else 0.0
    )
    threshold = max(minimum_threshold_m, neighbor_multiplier * robust_scale)

    is_outlier_full = evaluable & (deviation > threshold)
    inlier_full = in_active & ~is_outlier_full

    stats = {
        "depth_outlier_count": int(is_outlier_full.sum()),
        "depth_outlier_fraction": (
            float(is_outlier_full.sum() / evaluable.sum())
            if evaluable.any()
            else 0.0
        ),
        "depth_outlier_threshold_m": float(threshold),
        "depth_outlier_robust_scale_m": robust_scale,
    }

    return inlier_full[active_indices], stats


def refine_mesh_for_silhouette_and_depth(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    mask_bool: np.ndarray,
    camera_k: np.ndarray,
    masked_depth_m: np.ndarray | None = None,
    target_scale_m: float | None = None,
    diameter_fn=None,
    boundary_band_px: float = 3.0,
    max_silhouette_pixel_displacement: float = 3.0,
    huber_boundary_px: float = 2.0,
    huber_depth_m: float = 0.008,
    maximum_displacement_m: float = 0.008,
    laplacian_weight: float = 2.0,
    hidden_anchor_weight: float = 100.0,
    depth_outlier_neighbor_multiplier: float = 4.0,
    depth_outlier_minimum_threshold_m: float = 0.006,
) -> SilhouetteMeshRefinementResult:
    """
    Interior visible surface는 depth로, silhouette boundary는 이미지
    평면(u,v)으로 -- 역할을 분리해서 국소 보정한다.

    설계 원칙:
      - Depth-only로는 정점이 카메라 광선(ray)을 따라서만 움직이므로
        투영 위치(u,v)가 거의 그대로다 -- silhouette/IoU가 충분히
        개선되지 않는다. 그래서 두 항을 분리해서 함께 쓴다.

      - Interior (visible, mask 내부, boundary_band_px보다 안쪽):
        depth로만 보정한다. 같은 (u,v) 광선 위에서 깊이만 바꾼다
        (target = depth(u,v)/z * p, 즉 x,y,z를 함께 스케일해서 투영
        위치는 유지) -- z만 바꾸면 투영 위치(u,v)가 같이 밀리는
        부작용이 있어 광선을 보존하는 스케일링으로 수정했다.

      - Boundary (실루엣 경계 boundary_band_px 이내): silhouette로만
        보정한다 (이미지 평면에서 mask 경계로 이동). Depth는 여기서
        의도적으로 쓰지 않는다 -- 실측 결과 mask 경계 근처 depth
        픽셀에서 극단치(최대 45mm급)가 집중적으로 나타났다. 이는
        RGB-D 센서의 전형적인 foreground/background depth 혼합
        아티팩트(flying pixel)와 일치한다. Boundary의 depth는
        Laplacian을 통해 interior의 이미 보정된 값에서 간접적으로만
        전파된다.

      - Silhouette 이동량은 raw pixel 기준으로 max_silhouette_pixel_
        displacement로 하드 클립한 뒤 Huber로 추가 감쇠한다 -- IoU=1을
        강제하지 않는다.

      - Depth correspondence는 median filter(센서 노이즈 사전 제거)와
        이웃-정점 기준 outlier rejection(사후 필터링) 두 겹으로
        보호한다.

      - 최종 정점 변위는 Laplacian-regularized 최소자승으로 풀되,
        hidden(진짜 안 보이는) 정점은 매우 강하게(hidden_anchor_weight)
        0 근처로 anchor한다.

      - 마지막에 diameter_fn 기준으로 S*에 재투영한다.

    이 함수는 결과가 "더 낫다"고 보장하지 않는다 -- 호출자가
    diagnostics(iou_before/after, depth residual 등)로 accept/reject
    게이트를 직접 적용해야 한다.
    """

    vertex_count = points_camera.shape[0]
    image_height, image_width = mask_bool.shape
    fx, fy = camera_k[0, 0], camera_k[1, 1]
    cx, cy = camera_k[0, 2], camera_k[1, 2]

    visible_mask, u_pixel, v_pixel = _visible_vertex_mask(
        points_camera=points_camera,
        camera_k=camera_k,
        image_height=image_height,
        image_width=image_width,
        mask_bool=np.ones_like(mask_bool),
    )

    silhouette = np.zeros((image_height, image_width), dtype=bool)
    silhouette[v_pixel[visible_mask], u_pixel[visible_mask]] = True

    intersection = np.count_nonzero(silhouette & mask_bool)
    union = np.count_nonzero(silhouette | mask_bool)
    iou_before = intersection / union if union else 0.0

    boundary_distance_before = float(
        np.abs(
            _signed_distance_to_mask(mask_bool)[
                v_pixel[visible_mask], u_pixel[visible_mask]
            ]
        ).mean()
    )

    # --- restrict to a thin band around the mesh's OWN silhouette edge ---
    # NOTE: distance_transform_edt(silhouette)는 silhouette=False인 곳에서
    # 항상 0이고, distance_transform_edt(~silhouette)는 silhouette=True인
    # 곳에서 항상 0이다 -- 이 둘의 min을 취하면 두 필드가 서로 반대편에서
    # 상대방을 0으로 깔아뭉개서, silhouette 내부 전체가 거리 0(경계)으로
    # 나온다. 부호 없는 거리는 min이 아니라 (한쪽이 항상 0이므로) 덧셈으로
    # 합쳐야 한다 -- 이 버그 때문에 in_band가 사실상 전체 visible mesh를
    # 덮어써서 interior/boundary 분리가 아예 작동하지 않고 있었다.
    silhouette_edge_distance = scipy.ndimage.distance_transform_edt(
        silhouette
    ) + scipy.ndimage.distance_transform_edt(~silhouette)
    in_band = np.zeros(vertex_count, dtype=bool)
    in_band[visible_mask] = (
        silhouette_edge_distance[v_pixel[visible_mask], u_pixel[visible_mask]]
        <= boundary_band_px
    )
    band_indices = np.nonzero(in_band)[0]

    inside_mask_at_own_pixel = np.zeros(vertex_count, dtype=bool)
    inside_mask_at_own_pixel[visible_mask] = mask_bool[
        v_pixel[visible_mask], u_pixel[visible_mask]
    ]

    # interior: visible, inside the mask, and NOT within the boundary band.
    interior_mask = visible_mask & inside_mask_at_own_pixel & ~in_band
    interior_indices_all = np.nonzero(interior_mask)[0]

    displacement = np.zeros_like(points_camera)
    z = points_camera[:, 2]

    # --- silhouette term (boundary only): Huber-damped + hard-capped pull
    #     toward the mask boundary, direction from the SDF gradient ---
    sdf = _signed_distance_to_mask(mask_bool)
    # 실제 mask 경계는 픽셀 단위로 계단져 있어서, raw SDF의 gradient
    # 방향이 이웃 픽셀 사이에서도 국소적으로 흔들릴 수 있다. 방향만
    # 스무딩된 필드에서 구하고, 거리 크기(residual_px)는 원본 sdf를
    # 그대로 쓴다 -- interior를 고친 뒤에도 boundary band에만 스파이크가
    # 남아있었는데, 이게 원인 중 하나로 보인다.
    smoothed_sdf_for_direction = scipy.ndimage.gaussian_filter(sdf, sigma=2.0)
    grad_v, grad_u = np.gradient(smoothed_sdf_for_direction)

    if len(band_indices) > 0:
        px_u = u_pixel[band_indices]
        px_v = v_pixel[band_indices]
        residual_px = sdf[px_v, px_u]  # >0 outside mask, <0 inside

        capped_residual_px = np.clip(
            residual_px,
            -max_silhouette_pixel_displacement,
            max_silhouette_pixel_displacement,
        )
        huber_scale = np.minimum(
            1.0,
            huber_boundary_px / np.clip(np.abs(capped_residual_px), 1e-6, None),
        )
        damped_residual_px = capped_residual_px * huber_scale

        gu = grad_u[px_v, px_u]
        gv = grad_v[px_v, px_u]
        grad_norm = np.clip(np.sqrt(gu**2 + gv**2), 1e-6, None)
        # move opposite the gradient (toward decreasing |sdf|, i.e. toward
        # the boundary) by the damped, capped residual magnitude
        du = -(gu / grad_norm) * damped_residual_px
        dv = -(gv / grad_norm) * damped_residual_px

        displacement[band_indices, 0] += du * z[band_indices] / fx
        displacement[band_indices, 1] += dv * z[band_indices] / fy

    # --- depth term (interior only): ray-preserving scale toward the real
    #     depth at the vertex's own pixel -- keeps (u,v) fixed, only moves
    #     the point along the camera ray, unlike touching z alone ---
    depth_correspondence_count = 0
    depth_outlier_stats = {
        "depth_outlier_count": 0,
        "depth_outlier_fraction": 0.0,
        "depth_outlier_threshold_m": None,
        "depth_outlier_robust_scale_m": None,
    }
    interior_indices = np.array([], dtype=np.int64)

    if masked_depth_m is not None and len(interior_indices_all) > 0:
        # 센서 flying-pixel 노이즈를 correspondence 계산 전에 미리 걷어낸다.
        filtered_depth_m = scipy.ndimage.median_filter(masked_depth_m, size=3)

        depth_at_pixel = filtered_depth_m[
            v_pixel[interior_indices_all], u_pixel[interior_indices_all]
        ]
        has_depth = depth_at_pixel > 0
        interior_with_depth = interior_indices_all[has_depth]
        depth_at_pixel = depth_at_pixel[has_depth]

        if len(interior_with_depth) > 0:
            pixel_key = (
                v_pixel[interior_with_depth].astype(np.int64) * image_width
                + u_pixel[interior_with_depth].astype(np.int64)
            )
            _, inverse, counts = np.unique(
                pixel_key, return_inverse=True, return_counts=True
            )
            aggregated_depth = np.zeros(len(counts))
            np.add.at(aggregated_depth, inverse, depth_at_pixel)
            aggregated_depth /= counts
            depth_target = aggregated_depth[inverse]

            depth_residual = depth_target - z[interior_with_depth]

            inlier_mask, depth_outlier_stats = _reject_depth_outliers(
                vertex_count=vertex_count,
                triangles=triangles,
                active_indices=interior_with_depth,
                residual=depth_residual,
                neighbor_multiplier=depth_outlier_neighbor_multiplier,
                minimum_threshold_m=depth_outlier_minimum_threshold_m,
            )
            interior_indices = interior_with_depth[inlier_mask]
            inlier_depth_target = depth_target[inlier_mask]

            ray_scale = inlier_depth_target / z[interior_indices]
            raw_displacement = (
                points_camera[interior_indices] * (ray_scale - 1.0)[:, None]
            )
            raw_norm = np.linalg.norm(raw_displacement, axis=1)
            huber_scale = np.minimum(
                1.0, huber_depth_m / np.clip(raw_norm, 1e-9, None)
            )
            displacement[interior_indices] += (
                raw_displacement * huber_scale[:, None]
            )
            depth_correspondence_count = int(len(counts))

    # --- boundary z-anchor: boundary vertices only get (u,v) correction
    #     above, so their z stays at the original value while x,y move --
    #     at a silhouette edge the surface is close to grazing angle, so
    #     that pulls them off the true surface (this was the direct cause
    #     of the boundary-band spikes even after the interior/boundary
    #     split fix). Borrow z from already-corrected interior neighbors
    #     instead of trusting raw boundary-pixel depth (flying-pixel risk).
    if len(band_indices) > 0 and len(interior_indices) > 0:
        corrected_interior_z = z[interior_indices] + displacement[interior_indices, 2]

        known_z = np.zeros(vertex_count)
        known_mask = np.zeros(vertex_count, dtype=bool)
        known_z[interior_indices] = corrected_interior_z
        known_mask[interior_indices] = True

        edges = np.concatenate(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
            axis=0,
        )
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
        edges = edges[in_band[edges[:, 0]] & known_mask[edges[:, 1]]]

        adjacency = scipy.sparse.coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(vertex_count, vertex_count),
        ).tocsr()
        neighbor_known_count = np.asarray(adjacency.sum(axis=1)).ravel()
        neighbor_z_sum = adjacency @ known_z

        has_known_neighbor = neighbor_known_count[band_indices] > 0
        borrow_indices = band_indices[has_known_neighbor]
        borrowed_z_target = (
            neighbor_z_sum[borrow_indices] / neighbor_known_count[borrow_indices]
        )
        z_residual = borrowed_z_target - z[borrow_indices]
        huber_scale_boundary_z = np.minimum(
            1.0, huber_depth_m / np.clip(np.abs(z_residual), 1e-9, None)
        )
        displacement[borrow_indices, 2] += z_residual * huber_scale_boundary_z

    active_mask = np.zeros(vertex_count, dtype=bool)
    active_mask[band_indices] = True
    active_mask[interior_indices] = True

    intermediate_points_camera: dict[
        str,
        np.ndarray,
    ] = {
        "M0_original_camera":
            points_camera.copy(),
        "M1_raw_target_camera":
            points_camera
            + displacement,
    }

    displacement_norm = np.linalg.norm(displacement, axis=1)
    clip_scale = np.minimum(
        1.0, maximum_displacement_m / np.clip(displacement_norm, 1e-12, None)
    )
    displacement = displacement * clip_scale[:, None]

    intermediate_points_camera[
        "M2_target_preclip_camera"
    ] = (
        points_camera
        + displacement
    )

    laplacian = _build_uniform_laplacian(triangles, vertex_count)
    active_indices = np.nonzero(active_mask)[0]
    hidden_indices = np.nonzero(~active_mask)[0]

    active_rows = scipy.sparse.eye(vertex_count, format="csr")[active_indices]
    hidden_rows = scipy.sparse.eye(vertex_count, format="csr")[hidden_indices]

    system_matrix = scipy.sparse.vstack(
        [
            active_rows,
            np.sqrt(hidden_anchor_weight) * hidden_rows,
            laplacian_weight * laplacian,
        ]
    ).tocsr()

    solved_displacement = np.zeros_like(points_camera)
    for coordinate in range(3):
        rhs = np.zeros(len(active_indices) + len(hidden_indices) + vertex_count)
        rhs[: len(active_indices)] = displacement[active_indices, coordinate]
        solution = scipy.sparse.linalg.lsqr(
            system_matrix, rhs, atol=1e-8, btol=1e-8, iter_lim=2000
        )[0]
        solved_displacement[:, coordinate] = solution

    intermediate_points_camera[
        "M3_post_lsqr_preclip_camera"
    ] = (
        points_camera
        + solved_displacement
    )

    solved_norm = np.linalg.norm(solved_displacement, axis=1)
    clip_scale2 = np.minimum(
        1.0, maximum_displacement_m / np.clip(solved_norm, 1e-12, None)
    )
    solved_displacement = solved_displacement * clip_scale2[:, None]

    intermediate_points_camera[
        "M4_post_vertex_clip_camera"
    ] = (
        points_camera
        + solved_displacement
    )

    refined_points = points_camera + solved_displacement

    if target_scale_m is not None and diameter_fn is not None:
        scale_before_reprojection = float(diameter_fn(refined_points))
        centroid = refined_points.mean(axis=0)
        beta = target_scale_m / scale_before_reprojection
        refined_points = centroid + beta * (refined_points - centroid)
        scale_after_reprojection = float(diameter_fn(refined_points))
    else:
        scale_before_reprojection = None
        scale_after_reprojection = None

    intermediate_points_camera[
        "M5_post_scale_camera"
    ] = refined_points.copy()

    geometry_stages = {
        stage_name:
            _geometry_stage_diagnostics(
                reference_points=points_camera,
                stage_points=stage_points,
                triangles=triangles,
                laplacian=laplacian,
            )
        for (
            stage_name,
            stage_points,
        ) in intermediate_points_camera.items()
    }

    z2 = refined_points[:, 2]
    u2 = np.round(refined_points[:, 0] / z2 * fx + cx).astype(np.int64)
    v2 = np.round(refined_points[:, 1] / z2 * fy + cy).astype(np.int64)
    inb2 = (u2 >= 0) & (u2 < image_width) & (v2 >= 0) & (v2 < image_height)
    refined_silhouette = np.zeros((image_height, image_width), dtype=bool)
    refined_silhouette[v2[inb2], u2[inb2]] = True
    inter2 = np.count_nonzero(refined_silhouette & mask_bool)
    union2 = np.count_nonzero(refined_silhouette | mask_bool)
    iou_after = inter2 / union2 if union2 else 0.0

    boundary_distance_after = float(
        np.abs(sdf[v2[inb2], u2[inb2]]).mean()
    ) if inb2.any() else None

    # --- 표면 거칠기(roughness) 진단: IoU/boundary distance는 2D 투영의
    #     전체 평균/집계 지표라, 소수의 정점이 스파이크로 튀어나와도
    #     나머지 대부분이 조금씩 개선되면 평균은 오히려 좋아질 수 있다
    #     -- 즉 국소적으로 표면이 지저분해지는 것을 전혀 못 잡는다.
    #     같은 Laplacian(이웃 평균과의 차이)을 보정 전/후 좌표에 각각
    #     적용해서, 각 정점이 자기 이웃 평균에서 얼마나 벗어났는지를
    #     비교한다 -- 매끈한 표면은 이 값이 작고, 스파이크는 크다.
    roughness_before = np.linalg.norm(laplacian @ points_camera, axis=1)
    roughness_after = np.linalg.norm(laplacian @ refined_points, axis=1)
    roughness_before_p95 = float(np.quantile(roughness_before, 0.95))
    roughness_after_p95 = float(np.quantile(roughness_after, 0.95))

    displacement_final_norm = np.linalg.norm(solved_displacement, axis=1)
    moved = displacement_final_norm > 1e-6
    centroid0 = points_camera.mean(axis=0)
    radial_dir = (points_camera - centroid0) / np.clip(
        np.linalg.norm(points_camera - centroid0, axis=1, keepdims=True), 1e-9, None
    )
    radial_component = np.sum(solved_displacement * radial_dir, axis=1)

    diagnostics = {
        "iou_before": float(iou_before),
        "iou_after": float(iou_after),
        "boundary_distance_before_px": boundary_distance_before,
        "boundary_distance_after_px": boundary_distance_after,
        "boundary_band_vertex_count": int(len(band_indices)),
        "interior_vertex_count": int(len(interior_indices_all)),
        "depth_correspondence_count": depth_correspondence_count,
        **depth_outlier_stats,
        "moved_vertex_count": int(moved.sum()),
        "inward_fraction": float((radial_component[moved] < 0).mean()) if moved.any() else None,
        "outward_fraction": float((radial_component[moved] > 0).mean()) if moved.any() else None,
        "displacement_p90_m": float(np.quantile(displacement_final_norm, 0.9)),
        "displacement_max_m": float(displacement_final_norm.max()),
        "centroid_drift_m": float(np.linalg.norm(solved_displacement.mean(axis=0))),
        "scale_before_reprojection_m": scale_before_reprojection,
        "scale_after_reprojection_m": scale_after_reprojection,
        "target_scale_m": target_scale_m,
        "roughness_before_p95_m": roughness_before_p95,
        "roughness_after_p95_m": roughness_after_p95,
        "geometry_stages": geometry_stages,
    }

    return SilhouetteMeshRefinementResult(
        refined_points_camera=refined_points,
        displacement=solved_displacement,
        diagnostics=diagnostics,
        intermediate_points_camera=(
            intermediate_points_camera
        ),
    )
