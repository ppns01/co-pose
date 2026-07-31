from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


@dataclass(frozen=True)
class DepthAnchoredRefinementResult:
    """
    Depth-anchored visible-surface local refinement 결과.

    refined_points_camera:
        S* reprojection 이전, local deformation만 적용한 camera-frame 정점.
        (호출자가 mesh_scale_projector로 이어서 S*에 재투영해야 한다)

    diagnostics:
        acceptance gate와 연구 로그에 필요한 값들.
    """

    refined_points_camera: np.ndarray
    displacement: np.ndarray
    correspondence_mask: np.ndarray
    diagnostics: dict


def _umeyama_similarity(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    source ~= scale * (rotation @ source) + translation 이 target에
    최대한 가깝도록 하는 similarity transform을 (가중) 최소자승으로 구한다.

    Returns
    -------
    rotation (3,3), scale (float), translation (3,)
    """

    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError(
            "Umeyama fit에는 3개 이상의 대응점 쌍이 필요합니다: "
            f"source={source.shape}, target={target.shape}"
        )

    if weights is None:
        weights = np.ones(source.shape[0], dtype=np.float64)

    weight_sum = float(weights.sum())
    if weight_sum <= np.finfo(np.float64).eps:
        raise ValueError("Umeyama fit weight 합이 0입니다.")

    source_mean = (weights[:, None] * source).sum(axis=0) / weight_sum
    target_mean = (weights[:, None] * target).sum(axis=0) / weight_sum

    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = (
        (weights[:, None] * target_centered).T @ source_centered
    ) / weight_sum

    u, singular_values, vt = np.linalg.svd(covariance)

    sign_correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        sign_correction[2, 2] = -1.0

    rotation = u @ sign_correction @ vt

    source_variance = float(
        (weights * np.sum(source_centered * source_centered, axis=1)).sum()
        / weight_sum
    )

    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("Umeyama fit source 분산이 부족합니다.")

    scale = float(
        np.trace(np.diag(singular_values) @ sign_correction) / source_variance
    )

    translation = target_mean - scale * (rotation @ source_mean)

    return rotation, scale, translation


def _robust_umeyama_similarity(
    source: np.ndarray,
    target: np.ndarray,
    *,
    huber_delta_m: float = 0.01,
    iterations: int = 5,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Huber IRLS로 재가중하며 반복 적합한다.

    하드 outlier cutoff로 대부분의 점을 버리면 남는 점이 공간적으로
    쏠려서 rotation 추정이 불안정해질 수 있다 (특히 입력 pose가 이미
    정확해서 진짜 rigid residual이 0에 가까운 경우, fit이 노이즈에서
    허구의 큰 회전을 만들어낸다). 대신 대부분의 점을 유지하되 residual이
    큰 점의 영향력만 점진적으로 줄인다.

    Returns
    -------
    rotation, scale, translation, final_weights
    """

    weights = np.ones(source.shape[0], dtype=np.float64)

    rotation, scale, translation = _umeyama_similarity(source, target, weights)

    for _ in range(iterations):
        predicted = scale * (source @ rotation.T) + translation
        residual_norms = np.linalg.norm(target - predicted, axis=1)

        weights = np.ones_like(residual_norms)
        outside = residual_norms > huber_delta_m
        weights[outside] = huber_delta_m / residual_norms[outside]

        rotation, scale, translation = _umeyama_similarity(
            source, target, weights
        )

    return rotation, scale, translation, weights


def _visible_vertex_mask(
    *,
    points_camera: np.ndarray,
    camera_k: np.ndarray,
    image_height: int,
    image_width: int,
    mask_bool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    각 정점을 카메라에 투영하고, 픽셀별 최소-depth 정점만 visible로 표시한다
    (다른 렌더러 의존성 없는 근사 z-buffer).

    Returns
    -------
    visible_mask (N,) bool, u_pixel (N,) int, v_pixel (N,) int
    """

    fx, fy = camera_k[0, 0], camera_k[1, 1]
    cx, cy = camera_k[0, 2], camera_k[1, 2]

    z = points_camera[:, 2]
    in_front = z > 1e-6

    u_pixel = np.full(points_camera.shape[0], -1, dtype=np.int64)
    v_pixel = np.full(points_camera.shape[0], -1, dtype=np.int64)

    u_pixel[in_front] = np.round(
        points_camera[in_front, 0] / z[in_front] * fx + cx
    ).astype(np.int64)
    v_pixel[in_front] = np.round(
        points_camera[in_front, 1] / z[in_front] * fy + cy
    ).astype(np.int64)

    in_bounds = (
        in_front
        & (u_pixel >= 0)
        & (u_pixel < image_width)
        & (v_pixel >= 0)
        & (v_pixel < image_height)
    )

    in_mask = np.zeros_like(in_bounds)
    in_mask[in_bounds] = mask_bool[
        v_pixel[in_bounds],
        u_pixel[in_bounds],
    ]

    flat_index = np.where(
        in_mask,
        v_pixel * image_width + u_pixel,
        -1,
    )

    min_z_per_pixel = np.full(image_height * image_width, np.inf)
    valid = flat_index >= 0
    np.minimum.at(min_z_per_pixel, flat_index[valid], z[valid])

    visible_mask = np.zeros(points_camera.shape[0], dtype=bool)
    # 1mm 이내로 픽셀 최소 depth와 같은 정점만 "그 픽셀에서 보이는" 정점으로 취급.
    visible_mask[valid] = (
        z[valid] - min_z_per_pixel[flat_index[valid]] < 1e-3
    )

    return visible_mask, u_pixel, v_pixel


def _build_uniform_laplacian(
    triangles: np.ndarray,
    vertex_count: int,
) -> scipy.sparse.csr_matrix:
    """(Lf)_i = f_i - mean(f_j : j in neighbors(i)) 형태의 균일 Laplacian."""

    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)

    rows = edges[:, 0]
    cols = edges[:, 1]

    adjacency = scipy.sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    adjacency.data[:] = 1.0
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0.0] = 1.0

    normalized_adjacency = scipy.sparse.diags(1.0 / degree) @ adjacency
    laplacian = scipy.sparse.identity(vertex_count) - normalized_adjacency

    return laplacian.tocsr()


def refine_visible_surface_with_depth(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    vertex_normals_camera: np.ndarray,
    masked_depth_m: np.ndarray,
    mask_bool: np.ndarray,
    camera_k: np.ndarray,
    maximum_local_displacement_m: float = 0.010,
    raw_residual_outlier_m: float = 0.040,
    huber_delta_m: float = 0.008,
    grazing_angle_cosine_floor: float = 0.15,
    correspondence_weight: float = 1.0,
    hidden_anchor_weight: float = 5.0,
    laplacian_weight: float = 1.0,
    minimum_correspondences: int = 300,
    remove_global_rigid_component: bool = True,
) -> DepthAnchoredRefinementResult:
    """
    Visible surface를 실측 depth에 국소적으로 맞춘다.

    1. 정점을 투영하고 근사 z-buffer로 visible set을 정한다.
    2. 같은 픽셀의 실측 depth로 backproject해 대응점을 만든다
       (전역 nearest-neighbor는 다른 부위로 잘못 대응될 수 있어 사용하지 않는다).
    3. 대응점 raw residual에서 outlier와 grazing-angle 점을 제거한다.
    4. remove_global_rigid_component=True면 남은 대응점으로 similarity
       transform(rotation, scale, translation)을 먼저 맞추고 제거한다 --
       pose/uniform-scale 오차가 형상으로 흡수되는 것을 막기 위함이다.
       **이 scale 성분은 버린다: S*는 호출자가 mesh_scale_projector로
       별도 복원한다.**
       입력 self-pose가 이미 신뢰할 수 있는 경우(예: GT로 baking된 경우)는
       진짜 rigid residual이 0에 가까워야 하는데, depth 쪽 오차가 순수
       회전으로 설명되지 않는 패턴(예: 국소적 shift+눌림)이면 Umeyama가
       이걸 억지로 회전으로 흡수하려다 오히려 불안정해질 수 있다.
       이런 경우 False로 끄고 raw residual을 그대로 쓰는 편이 안전하다.
    5. 남은 local residual만, Laplacian-regularized 최소자승으로
       visible 정점에 부여하고 hidden 정점은 0 근처로 anchor한다.
    """

    vertex_count = points_camera.shape[0]
    image_height, image_width = masked_depth_m.shape

    visible_mask, u_pixel, v_pixel = _visible_vertex_mask(
        points_camera=points_camera,
        camera_k=camera_k,
        image_height=image_height,
        image_width=image_width,
        mask_bool=mask_bool,
    )

    depth_observed = np.zeros(vertex_count, dtype=np.float64)
    depth_observed[visible_mask] = masked_depth_m[
        v_pixel[visible_mask],
        u_pixel[visible_mask],
    ]

    has_valid_depth = visible_mask & (depth_observed > 0.0)

    fx, fy = camera_k[0, 0], camera_k[1, 1]
    cx, cy = camera_k[0, 2], camera_k[1, 2]

    observed_points = np.zeros_like(points_camera)
    observed_points[has_valid_depth, 0] = (
        (u_pixel[has_valid_depth] - cx)
        * depth_observed[has_valid_depth]
        / fx
    )
    observed_points[has_valid_depth, 1] = (
        (v_pixel[has_valid_depth] - cy)
        * depth_observed[has_valid_depth]
        / fy
    )
    observed_points[has_valid_depth, 2] = depth_observed[has_valid_depth]

    raw_residual = observed_points - points_camera
    raw_residual_norm = np.linalg.norm(raw_residual, axis=1)

    not_outlier = raw_residual_norm < raw_residual_outlier_m

    view_direction = points_camera / np.clip(
        np.linalg.norm(points_camera, axis=1, keepdims=True),
        1e-9,
        None,
    )
    facing_cosine = -np.sum(vertex_normals_camera * view_direction, axis=1)
    not_grazing = facing_cosine > grazing_angle_cosine_floor

    correspondence_mask = has_valid_depth & not_outlier & not_grazing

    correspondence_count = int(correspondence_mask.sum())

    diagnostics: dict = {
        "visible_vertex_count": int(visible_mask.sum()),
        "valid_depth_count": int(has_valid_depth.sum()),
        "correspondence_count": correspondence_count,
        "raw_residual_median_m": (
            float(np.median(raw_residual_norm[has_valid_depth]))
            if has_valid_depth.any()
            else None
        ),
    }

    if correspondence_count < minimum_correspondences:
        diagnostics["status"] = "insufficient_correspondences"
        return DepthAnchoredRefinementResult(
            refined_points_camera=points_camera.copy(),
            displacement=np.zeros_like(points_camera),
            correspondence_mask=correspondence_mask,
            diagnostics=diagnostics,
        )

    # The mesh is denser than the depth image: many vertices round to the
    # same pixel and therefore get the IDENTICAL observed_point (it only
    # depends on pixel + depth, not which vertex). Feeding all of them into
    # the rigid/scale fit as if they were independent measurements both
    # inflates the count (fake confidence) and overweights whatever region
    # happens to be sampled densest -- this is what made the earlier rigid
    # fit unstable. Aggregate to one correspondence per unique pixel first.
    corresponded_indices = np.nonzero(correspondence_mask)[0]
    pixel_key = (
        v_pixel[corresponded_indices].astype(np.int64) * image_width
        + u_pixel[corresponded_indices].astype(np.int64)
    )
    unique_pixels, inverse, group_counts = np.unique(
        pixel_key, return_inverse=True, return_counts=True
    )

    pixel_mesh_mean = np.zeros((len(unique_pixels), 3))
    np.add.at(
        pixel_mesh_mean, inverse, points_camera[corresponded_indices]
    )
    pixel_mesh_mean /= group_counts[:, None]

    # observed_points is already identical within a group (same pixel,
    # same depth reading) -- np.add.at + divide just reads it back out.
    pixel_observed = np.zeros((len(unique_pixels), 3))
    np.add.at(
        pixel_observed, inverse, observed_points[corresponded_indices]
    )
    pixel_observed /= group_counts[:, None]

    diagnostics["unique_pixel_correspondence_count"] = int(
        len(unique_pixels)
    )
    diagnostics["mean_vertices_per_pixel"] = float(
        group_counts.mean()
    )

    if remove_global_rigid_component:
        (
            rigid_rotation,
            rigid_scale,
            rigid_translation,
            irls_weights,
        ) = _robust_umeyama_similarity(
            pixel_mesh_mean,
            pixel_observed,
            huber_delta_m=huber_delta_m,
        )
        diagnostics["irls_effective_correspondence_fraction"] = float(
            irls_weights.mean()
        )
    else:
        # Caller already trusts the input self-pose (e.g. GT baked in) --
        # skip fitting a rigid correction that has nothing genuine to
        # find. Forcing Umeyama to explain a non-rigid depth discrepancy
        # (shift + local squash, not rotation) is exactly what produced
        # the spurious 40+deg "corrections" seen with this flag on.
        rigid_rotation = np.eye(3)
        rigid_scale = 1.0
        rigid_translation = np.zeros(3)
        diagnostics["irls_effective_correspondence_fraction"] = None

    rotation_diff = rigid_rotation
    rigid_rotation_deg = float(
        np.degrees(
            np.arccos(
                np.clip((np.trace(rotation_diff) - 1.0) / 2.0, -1.0, 1.0)
            )
        )
    )

    diagnostics["rigid_residual_rotation_deg"] = rigid_rotation_deg
    diagnostics["rigid_residual_translation_m"] = float(
        np.linalg.norm(rigid_translation)
    )
    diagnostics["rigid_residual_uniform_scale"] = rigid_scale

    predicted_by_rigid = (
        rigid_scale * (points_camera @ rigid_rotation.T) + rigid_translation
    )
    local_residual = np.zeros_like(points_camera)
    local_residual[correspondence_mask] = (
        observed_points[correspondence_mask]
        - predicted_by_rigid[correspondence_mask]
    )

    local_residual_norm = np.linalg.norm(local_residual, axis=1)
    clip_scale = np.minimum(
        1.0,
        maximum_local_displacement_m
        / np.clip(local_residual_norm, 1e-12, None),
    )
    local_residual = local_residual * clip_scale[:, None]

    laplacian = _build_uniform_laplacian(triangles, vertex_count)

    correspondence_indices = np.nonzero(correspondence_mask)[0]
    hidden_indices = np.nonzero(~correspondence_mask)[0]

    correspondence_rows = scipy.sparse.eye(vertex_count, format="csr")[
        correspondence_indices
    ]
    hidden_rows = scipy.sparse.eye(vertex_count, format="csr")[hidden_indices]

    system_matrix = scipy.sparse.vstack(
        [
            np.sqrt(correspondence_weight) * correspondence_rows,
            np.sqrt(hidden_anchor_weight) * hidden_rows,
            np.sqrt(laplacian_weight) * laplacian,
        ]
    ).tocsr()

    displacement = np.zeros_like(points_camera)

    for coordinate in range(3):
        rhs = np.zeros(
            correspondence_indices.shape[0]
            + hidden_indices.shape[0]
            + vertex_count
        )
        rhs[: correspondence_indices.shape[0]] = np.sqrt(
            correspondence_weight
        ) * local_residual[correspondence_indices, coordinate]

        solution = scipy.sparse.linalg.lsqr(
            system_matrix,
            rhs,
            atol=1e-8,
            btol=1e-8,
            iter_lim=2000,
        )[0]

        displacement[:, coordinate] = solution

    displacement_norm = np.linalg.norm(displacement, axis=1)

    diagnostics["displacement_median_m"] = float(np.median(displacement_norm))
    diagnostics["displacement_p90_m"] = float(
        np.quantile(displacement_norm, 0.90)
    )
    diagnostics["displacement_max_m"] = float(displacement_norm.max())
    diagnostics["centroid_drift_m"] = float(
        np.linalg.norm(displacement.mean(axis=0))
    )
    diagnostics["status"] = "refined"

    refined_points = points_camera + displacement

    return DepthAnchoredRefinementResult(
        refined_points_camera=refined_points,
        displacement=displacement,
        correspondence_mask=correspondence_mask,
        diagnostics=diagnostics,
    )
