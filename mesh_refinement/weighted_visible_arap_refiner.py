"""Dense visible-surface refinement with robust soft observation terms.

The global least-squares step combines pixel-to-surface depth, a
rendered-to-observed contour SDF linearization, displacement Laplacian,
local/global ARAP, soft unseen anchors, centroid preservation, and scale
preservation.  Visibility and projective correspondences are refreshed after
each topology-safe outer step.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg

from mesh_refinement.depth_anchored_visible_refiner import (
    _build_uniform_laplacian,
)
from mesh_refinement.silhouette_mesh_refiner import (
    SilhouetteMeshRefinementResult,
)


RaycastFunction = Callable[..., dict[str, np.ndarray]]


def _binary_iou(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    intersection = int(
        np.count_nonzero(first_bool & second_bool)
    )
    union = int(
        np.count_nonzero(first_bool | second_bool)
    )
    return float(intersection / union) if union else 0.0


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    return mask_bool & ~scipy.ndimage.binary_erosion(
        mask_bool,
        iterations=1,
        border_value=0,
    )


def _unique_edges(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int64)
    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        axis=0,
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def _edge_incidence(
    edges: np.ndarray,
    vertex_count: int,
) -> scipy.sparse.csr_matrix:
    edge_count = int(len(edges))
    rows = np.repeat(
        np.arange(edge_count, dtype=np.int64),
        2,
    )
    columns = np.asarray(edges, dtype=np.int64).reshape(-1)
    values = np.tile(
        np.asarray([1.0, -1.0], dtype=np.float64),
        edge_count,
    )
    return scipy.sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(edge_count, vertex_count),
    ).tocsr()


def _adjacency(
    edges: np.ndarray,
    vertex_count: int,
) -> list[np.ndarray]:
    neighbors: list[list[int]] = [
        [] for _ in range(vertex_count)
    ]
    for first, second in np.asarray(
        edges,
        dtype=np.int64,
    ):
        first_int = int(first)
        second_int = int(second)
        neighbors[first_int].append(second_int)
        neighbors[second_int].append(first_int)
    return [
        np.asarray(items, dtype=np.int64)
        for items in neighbors
    ]


def _topology_distance_from_visible(
    *,
    visible_mask: np.ndarray,
    adjacency: list[np.ndarray],
) -> np.ndarray:
    visible = np.asarray(visible_mask, dtype=bool)
    distance = np.full(len(visible), -1, dtype=np.int64)
    queue: deque[int] = deque()

    for index in np.flatnonzero(visible):
        index_int = int(index)
        distance[index_int] = 0
        queue.append(index_int)

    while queue:
        current = queue.popleft()
        next_distance = int(distance[current] + 1)
        for neighbor in adjacency[current]:
            neighbor_int = int(neighbor)
            if distance[neighbor_int] >= 0:
                continue
            distance[neighbor_int] = next_distance
            queue.append(neighbor_int)

    return distance


def _soft_unseen_confidence(
    *,
    visible_mask: np.ndarray,
    adjacency: list[np.ndarray],
    transition_ring_count: int,
    visible_confidence: float,
    transition_confidence: float,
    far_hidden_confidence: float,
) -> np.ndarray:
    distance = _topology_distance_from_visible(
        visible_mask=visible_mask,
        adjacency=adjacency,
    )
    confidence = np.full(
        len(distance),
        float(far_hidden_confidence),
        dtype=np.float64,
    )
    confidence[distance == 0] = float(visible_confidence)

    ring_count = max(int(transition_ring_count), 0)
    if ring_count > 0:
        for ring in range(1, ring_count + 1):
            alpha = ring / float(ring_count + 1)
            value = (
                (1.0 - alpha) * float(transition_confidence)
                + alpha * float(far_hidden_confidence)
            )
            confidence[distance == ring] = value

    confidence[distance < 0] = float(far_hidden_confidence)
    return confidence


def _huber_irls_weights(
    residual: np.ndarray,
    delta: float | np.ndarray,
) -> np.ndarray:
    residual_abs = np.abs(
        np.asarray(residual, dtype=np.float64)
    )
    delta_array = np.asarray(delta, dtype=np.float64)
    delta_safe = np.maximum(delta_array, 1e-12)
    return np.minimum(
        1.0,
        delta_safe / np.maximum(residual_abs, 1e-12),
    )


def _stratified_sample_pixels(
    mask: np.ndarray,
    *,
    stride_px: int,
    maximum_sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask_bool = np.asarray(mask, dtype=bool)
    height, width = mask_bool.shape
    stride = max(int(stride_px), 1)
    maximum_count = max(int(maximum_sample_count), 1)

    selected_v: list[int] = []
    selected_u: list[int] = []
    for row_start in range(0, height, stride):
        row_end = min(row_start + stride, height)
        for column_start in range(0, width, stride):
            column_end = min(column_start + stride, width)
            local_v, local_u = np.nonzero(
                mask_bool[
                    row_start:row_end,
                    column_start:column_end,
                ]
            )
            if local_v.size == 0:
                continue

            center_v = 0.5 * (row_start + row_end - 1)
            center_u = 0.5 * (column_start + column_end - 1)
            absolute_v = local_v + row_start
            absolute_u = local_u + column_start
            chosen = int(
                np.argmin(
                    (absolute_v - center_v) ** 2
                    + (absolute_u - center_u) ** 2
                )
            )
            selected_v.append(int(absolute_v[chosen]))
            selected_u.append(int(absolute_u[chosen]))

    if len(selected_v) > maximum_count:
        keep = np.linspace(
            0,
            len(selected_v) - 1,
            maximum_count,
        ).round().astype(np.int64)
        selected_v = [selected_v[int(index)] for index in keep]
        selected_u = [selected_u[int(index)] for index in keep]

    return (
        np.asarray(selected_v, dtype=np.int64),
        np.asarray(selected_u, dtype=np.int64),
    )


def _backproject_depth_and_normals(
    *,
    depth_m: np.ndarray,
    camera_k: np.ndarray,
    maximum_neighbor_depth_delta_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float64)
    camera = np.asarray(camera_k, dtype=np.float64)
    height, width = depth.shape

    pixel_u, pixel_v = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    fx = float(camera[0, 0])
    fy = float(camera[1, 1])
    cx = float(camera[0, 2])
    cy = float(camera[1, 2])

    points = np.zeros((height, width, 3), dtype=np.float64)
    points[..., 0] = (pixel_u - cx) * depth / fx
    points[..., 1] = (pixel_v - cy) * depth / fy
    points[..., 2] = depth

    valid_depth = np.isfinite(depth) & (depth > 0.0)
    normals = np.zeros_like(points)
    valid_normal = np.zeros_like(valid_depth)

    if height < 3 or width < 3:
        return points, normals, valid_normal

    horizontal = points[1:-1, 2:] - points[1:-1, :-2]
    vertical = points[2:, 1:-1] - points[:-2, 1:-1]
    inner_normals = np.cross(horizontal, vertical)
    inner_norm = np.linalg.norm(inner_normals, axis=2)

    center_depth = depth[1:-1, 1:-1]
    neighbor_depth = np.stack(
        [
            depth[1:-1, :-2],
            depth[1:-1, 2:],
            depth[:-2, 1:-1],
            depth[2:, 1:-1],
        ],
        axis=-1,
    )
    depth_consistent = np.all(
        np.abs(neighbor_depth - center_depth[..., None])
        <= float(maximum_neighbor_depth_delta_m),
        axis=-1,
    )
    neighbor_valid = np.all(neighbor_depth > 0.0, axis=-1)
    inner_valid = (
        valid_depth[1:-1, 1:-1]
        & neighbor_valid
        & depth_consistent
        & np.isfinite(inner_norm)
        & (inner_norm > 1e-12)
    )

    inner_normals[inner_valid] /= inner_norm[inner_valid, None]
    inner_points = points[1:-1, 1:-1]
    points_toward_camera = np.einsum(
        "ijk,ijk->ij",
        inner_normals,
        inner_points,
    ) > 0.0
    inner_normals[points_toward_camera] *= -1.0

    normals[1:-1, 1:-1] = inner_normals
    valid_normal[1:-1, 1:-1] = inner_valid
    normals[~valid_normal] = 0.0
    return points, normals, valid_normal


def _barycentric_from_primitive_uv(
    primitive_uv: np.ndarray,
) -> np.ndarray:
    primitive_uv = np.asarray(primitive_uv, dtype=np.float64)
    barycentric = np.stack(
        [
            1.0 - primitive_uv[:, 0] - primitive_uv[:, 1],
            primitive_uv[:, 0],
            primitive_uv[:, 1],
        ],
        axis=1,
    )
    barycentric = np.clip(barycentric, 0.0, 1.0)
    barycentric /= np.clip(
        barycentric.sum(axis=1, keepdims=True),
        1e-12,
        None,
    )
    return barycentric


def _scalar_barycentric_system(
    *,
    vertex_count: int,
    triangle_vertices: np.ndarray,
    barycentric: np.ndarray,
    directions: np.ndarray,
    right_hand_side: np.ndarray,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    triangle_vertices = np.asarray(
        triangle_vertices,
        dtype=np.int64,
    )
    barycentric = np.asarray(barycentric, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    rhs = np.asarray(right_hand_side, dtype=np.float64)
    sample_count = int(len(triangle_vertices))

    if sample_count == 0:
        return (
            scipy.sparse.csr_matrix((0, 3 * vertex_count)),
            np.empty(0, dtype=np.float64),
        )

    row_indices = np.repeat(
        np.arange(sample_count, dtype=np.int64),
        9,
    )
    columns = (
        3 * triangle_vertices[:, :, None]
        + np.arange(3, dtype=np.int64)[None, None, :]
    ).reshape(-1)
    values = (
        barycentric[:, :, None]
        * directions[:, None, :]
    ).reshape(-1)
    matrix = scipy.sparse.coo_matrix(
        (values, (row_indices, columns)),
        shape=(sample_count, 3 * vertex_count),
    ).tocsr()
    return matrix, rhs


def _depth_constraint_system(
    *,
    points: np.ndarray,
    triangles: np.ndarray,
    camera_k: np.ndarray,
    raycast: dict[str, np.ndarray],
    observed_depth_m: np.ndarray,
    observed_mask: np.ndarray,
    mask_erosion_px: int,
    sample_stride_px: int,
    maximum_sample_count: int,
    maximum_neighbor_depth_delta_m: float,
    maximum_projective_residual_m: float,
    grazing_cosine_floor: float,
) -> tuple[
    scipy.sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    dict,
]:
    vertex_count = int(len(points))
    filtered_depth = scipy.ndimage.median_filter(
        np.asarray(observed_depth_m, dtype=np.float64),
        size=3,
    )
    observed_points, observed_normals, valid_normal = (
        _backproject_depth_and_normals(
            depth_m=filtered_depth,
            camera_k=camera_k,
            maximum_neighbor_depth_delta_m=(
                maximum_neighbor_depth_delta_m
            ),
        )
    )

    erosion_count = max(int(mask_erosion_px), 0)
    eroded_mask = np.asarray(observed_mask, dtype=bool).copy()
    if erosion_count > 0:
        eroded_mask = scipy.ndimage.binary_erosion(
            eroded_mask,
            iterations=erosion_count,
            border_value=0,
        )
    rendered_mask = np.asarray(raycast["rendered_mask"], dtype=bool)
    valid = (
        eroded_mask
        & rendered_mask
        & (filtered_depth > 0.0)
        & valid_normal
    )
    sample_v, sample_u = _stratified_sample_pixels(
        valid,
        stride_px=sample_stride_px,
        maximum_sample_count=maximum_sample_count,
    )

    if sample_v.size == 0:
        return (
            scipy.sparse.csr_matrix((0, 3 * vertex_count)),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            {
                "valid_depth_pixel_count": int(np.count_nonzero(valid)),
                "depth_sample_count": 0,
                "depth_rejected_projective_count": 0,
                "depth_abs_median_m": None,
                "depth_abs_p90_m": None,
            },
        )

    primitive_ids = np.asarray(
        raycast["primitive_ids"][sample_v, sample_u],
        dtype=np.int64,
    )
    primitive_uvs = np.asarray(
        raycast["primitive_uvs"][sample_v, sample_u],
        dtype=np.float64,
    )
    valid_primitive = (
        (primitive_ids >= 0)
        & (primitive_ids < len(triangles))
        & np.isfinite(primitive_uvs).all(axis=1)
    )
    sample_v = sample_v[valid_primitive]
    sample_u = sample_u[valid_primitive]
    primitive_ids = primitive_ids[valid_primitive]
    primitive_uvs = primitive_uvs[valid_primitive]

    barycentric = _barycentric_from_primitive_uv(primitive_uvs)
    triangle_vertices = np.asarray(triangles, dtype=np.int64)[primitive_ids]
    hit_points = np.einsum(
        "ij,ijk->ik",
        barycentric,
        np.asarray(points, dtype=np.float64)[triangle_vertices],
    )
    target_points = observed_points[sample_v, sample_u]
    normals = observed_normals[sample_v, sample_u]
    residual = np.einsum(
        "ij,ij->i",
        normals,
        hit_points - target_points,
    )

    view_direction = hit_points / np.clip(
        np.linalg.norm(hit_points, axis=1, keepdims=True),
        1e-12,
        None,
    )
    facing = np.abs(
        np.einsum("ij,ij->i", normals, view_direction)
    )
    keep = (
        np.isfinite(residual)
        & (np.abs(residual) <= float(maximum_projective_residual_m))
        & (facing >= float(grazing_cosine_floor))
    )
    rejected_count = int(np.count_nonzero(~keep))
    triangle_vertices = triangle_vertices[keep]
    barycentric = barycentric[keep]
    normals = normals[keep]
    target_points = target_points[keep]
    residual = residual[keep]
    facing = facing[keep]

    rhs = np.einsum("ij,ij->i", normals, target_points)
    matrix, rhs = _scalar_barycentric_system(
        vertex_count=vertex_count,
        triangle_vertices=triangle_vertices,
        barycentric=barycentric,
        directions=normals,
        right_hand_side=rhs,
    )
    confidence = np.clip(
        (facing - float(grazing_cosine_floor))
        / max(1.0 - float(grazing_cosine_floor), 1e-12),
        0.05,
        1.0,
    )

    return (
        matrix,
        rhs,
        confidence,
        {
            "valid_depth_pixel_count": int(np.count_nonzero(valid)),
            "depth_sample_count": int(len(residual)),
            "depth_rejected_projective_count": rejected_count,
            "depth_abs_median_m": (
                float(np.median(np.abs(residual)))
                if residual.size
                else None
            ),
            "depth_abs_p90_m": (
                float(np.quantile(np.abs(residual), 0.90))
                if residual.size
                else None
            ),
        },
    )


def _projection_direction(
    *,
    point: np.ndarray,
    image_gradient_u: float,
    image_gradient_v: float,
    camera_k: np.ndarray,
) -> np.ndarray:
    x_value, y_value, z_value = np.asarray(
        point,
        dtype=np.float64,
    )
    z_safe = max(float(z_value), 1e-8)
    fx = float(camera_k[0, 0])
    fy = float(camera_k[1, 1])
    return np.asarray(
        [
            image_gradient_u * fx / z_safe,
            image_gradient_v * fy / z_safe,
            -image_gradient_u * fx * x_value / (z_safe * z_safe)
            - image_gradient_v * fy * y_value / (z_safe * z_safe),
        ],
        dtype=np.float64,
    )


def _silhouette_constraint_system(
    *,
    points: np.ndarray,
    triangles: np.ndarray,
    camera_k: np.ndarray,
    raycast: dict[str, np.ndarray],
    observed_mask: np.ndarray,
    contour_sample_stride: int,
    maximum_sample_count: int,
    minimum_residual_px: float,
    maximum_residual_px: float,
) -> tuple[
    scipy.sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    vertex_count = int(len(points))
    rendered_boundary = np.asarray(
        raycast["rendered_boundary"],
        dtype=bool,
    )
    boundary_v, boundary_u = np.nonzero(rendered_boundary)
    stride = max(int(contour_sample_stride), 1)
    boundary_v = boundary_v[::stride]
    boundary_u = boundary_u[::stride]
    if boundary_v.size > int(maximum_sample_count):
        keep_indices = np.linspace(
            0,
            boundary_v.size - 1,
            int(maximum_sample_count),
        ).round().astype(np.int64)
        boundary_v = boundary_v[keep_indices]
        boundary_u = boundary_u[keep_indices]

    empty_matrix = scipy.sparse.csr_matrix((0, 3 * vertex_count))
    empty = np.empty(0, dtype=np.float64)
    if boundary_v.size == 0:
        return (
            empty_matrix,
            empty,
            empty,
            empty,
            {
                "rendered_contour_sample_count": 0,
                "active_contour_sample_count": 0,
                "contour_abs_p90_px": None,
            },
        )

    observed_mask_bool = np.asarray(observed_mask, dtype=bool)
    observed_boundary = _extract_boundary(observed_mask_bool)
    unsigned_distance = scipy.ndimage.distance_transform_edt(
        ~observed_boundary
    )
    sdf = unsigned_distance
    sdf[observed_mask_bool] *= -1.0
    sdf_direction = scipy.ndimage.gaussian_filter(sdf, sigma=1.0)
    gradient_v, gradient_u = np.gradient(sdf_direction)

    primitive_ids = np.asarray(
        raycast["primitive_ids"][boundary_v, boundary_u],
        dtype=np.int64,
    )
    primitive_uvs = np.asarray(
        raycast["primitive_uvs"][boundary_v, boundary_u],
        dtype=np.float64,
    )
    residual_px = np.asarray(
        sdf[boundary_v, boundary_u],
        dtype=np.float64,
    )
    gradient_u_values = gradient_u[boundary_v, boundary_u]
    gradient_v_values = gradient_v[boundary_v, boundary_u]
    gradient_norm = np.hypot(
        gradient_u_values,
        gradient_v_values,
    )
    valid = (
        (primitive_ids >= 0)
        & (primitive_ids < len(triangles))
        & np.isfinite(primitive_uvs).all(axis=1)
        & np.isfinite(residual_px)
        & (np.abs(residual_px) >= float(minimum_residual_px))
        & (np.abs(residual_px) <= float(maximum_residual_px))
        & (gradient_norm > 1e-6)
    )
    primitive_ids = primitive_ids[valid]
    primitive_uvs = primitive_uvs[valid]
    residual_px = residual_px[valid]
    gradient_u_values = gradient_u_values[valid] / gradient_norm[valid]
    gradient_v_values = gradient_v_values[valid] / gradient_norm[valid]

    if primitive_ids.size == 0:
        return (
            empty_matrix,
            empty,
            empty,
            empty,
            {
                "rendered_contour_sample_count": int(boundary_v.size),
                "active_contour_sample_count": 0,
                "contour_abs_p90_px": None,
            },
        )

    barycentric = _barycentric_from_primitive_uv(primitive_uvs)
    triangle_vertices = np.asarray(triangles, dtype=np.int64)[primitive_ids]
    hit_points = np.einsum(
        "ij,ijk->ik",
        barycentric,
        np.asarray(points, dtype=np.float64)[triangle_vertices],
    )
    directions = np.stack(
        [
            _projection_direction(
                point=point,
                image_gradient_u=float(gradient_u_value),
                image_gradient_v=float(gradient_v_value),
                camera_k=camera_k,
            )
            for point, gradient_u_value, gradient_v_value in zip(
                hit_points,
                gradient_u_values,
                gradient_v_values,
                strict=True,
            )
        ],
        axis=0,
    )
    rhs = np.einsum("ij,ij->i", directions, hit_points) - residual_px

    fx = float(camera_k[0, 0])
    fy = float(camera_k[1, 1])
    meters_per_pixel = hit_points[:, 2] / np.sqrt(max(fx * fy, 1e-12))
    directions *= meters_per_pixel[:, None]
    rhs *= meters_per_pixel
    matrix, rhs = _scalar_barycentric_system(
        vertex_count=vertex_count,
        triangle_vertices=triangle_vertices,
        barycentric=barycentric,
        directions=directions,
        right_hand_side=rhs,
    )
    confidence = np.ones(len(rhs), dtype=np.float64)
    huber_delta_m = meters_per_pixel

    return (
        matrix,
        rhs,
        confidence,
        huber_delta_m,
        {
            "rendered_contour_sample_count": int(boundary_v.size),
            "active_contour_sample_count": int(len(rhs)),
            "contour_abs_p90_px": float(
                np.quantile(np.abs(residual_px), 0.90)
            ),
        },
    )


def _local_rotations(
    *,
    original_points: np.ndarray,
    current_points: np.ndarray,
    adjacency: list[np.ndarray],
) -> np.ndarray:
    vertex_count = int(len(original_points))
    rotations = np.repeat(
        np.eye(3, dtype=np.float64)[None, :, :],
        vertex_count,
        axis=0,
    )
    for index in range(vertex_count):
        neighbors = adjacency[index]
        if len(neighbors) < 2:
            continue
        original_edges = (
            original_points[index]
            - original_points[neighbors]
        )
        current_edges = (
            current_points[index]
            - current_points[neighbors]
        )
        covariance = current_edges.T @ original_edges
        left, _, right_transposed = np.linalg.svd(covariance)
        rotation = left @ right_transposed
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right_transposed
        rotations[index] = rotation
    return rotations


def _arap_right_hand_side(
    *,
    original_points: np.ndarray,
    edges: np.ndarray,
    rotations: np.ndarray,
) -> np.ndarray:
    original_edges = (
        original_points[edges[:, 0]]
        - original_points[edges[:, 1]]
    )
    average_rotation = 0.5 * (
        rotations[edges[:, 0]]
        + rotations[edges[:, 1]]
    )
    rotated_edges = np.einsum(
        "eij,ej->ei",
        average_rotation,
        original_edges,
    )
    return rotated_edges.reshape(-1)


def _scale_linearization(
    *,
    current_points: np.ndarray,
    target_rms_radius_m: float,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray, float]:
    points = np.asarray(current_points, dtype=np.float64)
    vertex_count = int(len(points))
    centered = points - points.mean(axis=0, keepdims=True)
    rms_radius = float(
        np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    )
    if rms_radius <= 1e-12:
        return (
            scipy.sparse.csr_matrix((0, 3 * vertex_count)),
            np.empty(0, dtype=np.float64),
            rms_radius,
        )
    gradient = centered / (vertex_count * rms_radius)
    rhs = float(
        gradient.reshape(-1) @ points.reshape(-1)
        + target_rms_radius_m
        - rms_radius
    )
    matrix = scipy.sparse.csr_matrix(
        gradient.reshape(1, -1)
    )
    return matrix, np.asarray([rhs], dtype=np.float64), rms_radius


def _weighted_rows(
    *,
    matrix: scipy.sparse.csr_matrix,
    rhs: np.ndarray,
    row_weight: np.ndarray,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    if matrix.shape[0] == 0:
        return matrix, np.asarray(rhs, dtype=np.float64)
    square_root = np.sqrt(
        np.clip(np.asarray(row_weight, dtype=np.float64), 0.0, None)
    )
    diagonal = scipy.sparse.diags(square_root, format="csr")
    return diagonal @ matrix, square_root * np.asarray(rhs, dtype=np.float64)


def _geometry_audit(
    *,
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
    triangles: np.ndarray,
    edges: np.ndarray,
    minimum_edge_ratio: float,
    maximum_edge_ratio: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
) -> dict:
    reference_edges = np.linalg.norm(
        reference_points[edges[:, 0]]
        - reference_points[edges[:, 1]],
        axis=1,
    )
    candidate_edges = np.linalg.norm(
        candidate_points[edges[:, 0]]
        - candidate_points[edges[:, 1]],
        axis=1,
    )
    valid_edges = reference_edges > 1e-12
    edge_ratio = candidate_edges[valid_edges] / reference_edges[valid_edges]

    reference_faces = reference_points[triangles]
    candidate_faces = candidate_points[triangles]
    reference_cross = np.cross(
        reference_faces[:, 1] - reference_faces[:, 0],
        reference_faces[:, 2] - reference_faces[:, 0],
    )
    candidate_cross = np.cross(
        candidate_faces[:, 1] - candidate_faces[:, 0],
        candidate_faces[:, 2] - candidate_faces[:, 0],
    )
    reference_area2 = np.linalg.norm(reference_cross, axis=1)
    candidate_area2 = np.linalg.norm(candidate_cross, axis=1)
    valid_faces = reference_area2 > 1e-14
    area_ratio = candidate_area2[valid_faces] / reference_area2[valid_faces]
    orientation = np.einsum("ij,ij->i", reference_cross, candidate_cross)
    reversal_count = int(
        np.count_nonzero(valid_faces & (orientation <= 0.0))
    )

    edge_min = float(edge_ratio.min()) if edge_ratio.size else 1.0
    edge_max = float(edge_ratio.max()) if edge_ratio.size else 1.0
    area_min = float(area_ratio.min()) if area_ratio.size else 1.0
    area_max = float(area_ratio.max()) if area_ratio.size else 1.0
    safe = bool(
        reversal_count == 0
        and edge_min >= float(minimum_edge_ratio)
        and edge_max <= float(maximum_edge_ratio)
        and area_min >= float(minimum_area_ratio)
        and area_max <= float(maximum_area_ratio)
    )
    return {
        "topology_safe": safe,
        "face_reversal_count": reversal_count,
        "edge_ratio_min": edge_min,
        "edge_ratio_max": edge_max,
        "area_ratio_min": area_min,
        "area_ratio_max": area_max,
    }


def _safe_increment_line_search(
    *,
    original_points: np.ndarray,
    current_points: np.ndarray,
    proposed_points: np.ndarray,
    triangles: np.ndarray,
    edges: np.ndarray,
    maximum_step_displacement_m: float,
    maximum_cumulative_displacement_m: float,
    minimum_edge_ratio: float,
    maximum_edge_ratio: float,
    minimum_area_ratio: float,
    maximum_area_ratio: float,
    minimum_step_scale: float,
) -> tuple[np.ndarray, dict]:
    increment = proposed_points - current_points
    scale = 1.0
    record: dict = {
        "step_scale": 0.0,
        "topology_safe": False,
    }
    while scale >= float(minimum_step_scale) - 1e-12:
        candidate = current_points + scale * increment
        step_max = float(
            np.linalg.norm(candidate - current_points, axis=1).max()
        )
        cumulative_max = float(
            np.linalg.norm(candidate - original_points, axis=1).max()
        )
        per_step = _geometry_audit(
            reference_points=current_points,
            candidate_points=candidate,
            triangles=triangles,
            edges=edges,
            minimum_edge_ratio=minimum_edge_ratio,
            maximum_edge_ratio=maximum_edge_ratio,
            minimum_area_ratio=minimum_area_ratio,
            maximum_area_ratio=maximum_area_ratio,
        )
        cumulative = _geometry_audit(
            reference_points=original_points,
            candidate_points=candidate,
            triangles=triangles,
            edges=edges,
            minimum_edge_ratio=minimum_edge_ratio,
            maximum_edge_ratio=maximum_edge_ratio,
            minimum_area_ratio=minimum_area_ratio,
            maximum_area_ratio=maximum_area_ratio,
        )
        safe = bool(
            step_max <= float(maximum_step_displacement_m) + 1e-12
            and cumulative_max
            <= float(maximum_cumulative_displacement_m) + 1e-12
            and per_step["topology_safe"]
            and cumulative["topology_safe"]
        )
        record = {
            "step_scale": float(scale),
            "step_displacement_max_m": step_max,
            "cumulative_displacement_max_m": cumulative_max,
            "per_step_topology": per_step,
            "cumulative_topology": cumulative,
            "topology_safe": safe,
        }
        if safe:
            return candidate, record
        scale *= 0.5

    return current_points.copy(), record


def _contour_metrics(
    rendered_mask: np.ndarray,
    observed_mask: np.ndarray,
) -> dict:
    rendered_boundary = _extract_boundary(rendered_mask)
    observed_boundary = _extract_boundary(observed_mask)
    if not rendered_boundary.any() or not observed_boundary.any():
        return {
            "rendered_to_observed_boundary_p50_px": None,
            "rendered_to_observed_boundary_p90_px": None,
            "observed_to_rendered_boundary_p50_px": None,
            "observed_to_rendered_boundary_p90_px": None,
            "symmetric_boundary_distance_mean_px": None,
        }
    distance_to_observed = scipy.ndimage.distance_transform_edt(
        ~observed_boundary
    )
    distance_to_rendered = scipy.ndimage.distance_transform_edt(
        ~rendered_boundary
    )
    rendered_values = distance_to_observed[rendered_boundary]
    observed_values = distance_to_rendered[observed_boundary]
    return {
        "rendered_to_observed_boundary_p50_px": float(
            np.quantile(rendered_values, 0.50)
        ),
        "rendered_to_observed_boundary_p90_px": float(
            np.quantile(rendered_values, 0.90)
        ),
        "observed_to_rendered_boundary_p50_px": float(
            np.quantile(observed_values, 0.50)
        ),
        "observed_to_rendered_boundary_p90_px": float(
            np.quantile(observed_values, 0.90)
        ),
        "symmetric_boundary_distance_mean_px": 0.5
        * (
            float(rendered_values.mean())
            + float(observed_values.mean())
        ),
    }


def _raster_observation_metrics(
    *,
    raycast: dict[str, np.ndarray],
    observed_mask: np.ndarray,
    observed_depth_m: np.ndarray | None,
    depth_mask_erosion_px: int,
) -> dict:
    rendered_mask = np.asarray(raycast["rendered_mask"], dtype=bool)
    metrics = {
        "raster_iou": _binary_iou(rendered_mask, observed_mask),
        **_contour_metrics(rendered_mask, observed_mask),
        "depth_overlap_pixel_count": 0,
        "depth_abs_median_m": None,
        "depth_abs_p90_m": None,
    }
    if observed_depth_m is None:
        return metrics
    erosion_count = max(int(depth_mask_erosion_px), 0)
    eroded = np.asarray(observed_mask, dtype=bool).copy()
    if erosion_count > 0:
        eroded = scipy.ndimage.binary_erosion(
            eroded,
            iterations=erosion_count,
            border_value=0,
        )
    rendered_depth = np.asarray(raycast["rendered_depth"], dtype=np.float64)
    observed_depth = np.asarray(observed_depth_m, dtype=np.float64)
    valid = (
        eroded
        & rendered_mask
        & (rendered_depth > 0.0)
        & (observed_depth > 0.0)
        & np.isfinite(rendered_depth)
        & np.isfinite(observed_depth)
    )
    metrics["depth_overlap_pixel_count"] = int(np.count_nonzero(valid))
    if valid.any():
        residual = np.abs(observed_depth[valid] - rendered_depth[valid])
        metrics["depth_abs_median_m"] = float(np.median(residual))
        metrics["depth_abs_p90_m"] = float(np.quantile(residual, 0.90))
    return metrics


def _default_raycast_function() -> RaycastFunction:
    # Open3D is intentionally imported only on the actual refinement path so
    # the linear-system helpers remain unit-testable without the optional
    # renderer dependency.
    from mesh_refinement.iterative_contour_arap_refiner import (
        _raycast_mesh,
    )

    return _raycast_mesh


def _apply_solver_trust_region(
    *,
    current_points: np.ndarray,
    proposed_points: np.ndarray,
    maximum_displacement_m: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Limit one solver proposal relative to the current outer iterate.

    A single global scale is used instead of clipping vertices independently,
    because per-vertex clipping can introduce a new local topology distortion.
    """
    current = np.asarray(current_points, dtype=np.float64)
    proposed = np.asarray(proposed_points, dtype=np.float64)
    radius = float(maximum_displacement_m)

    if current.shape != proposed.shape:
        raise ValueError(
            "current_points and proposed_points must have the same shape"
        )
    if current.ndim != 2 or current.shape[1] != 3:
        raise ValueError("solver trust-region points must have shape (N,3)")
    if not np.isfinite(current).all() or not np.isfinite(proposed).all():
        raise ValueError("solver trust-region points must be finite")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("maximum_displacement_m must be a finite positive value")

    displacement = proposed - current
    unconstrained_max = float(
        np.linalg.norm(displacement, axis=1).max(initial=0.0)
    )
    trust_scale = min(1.0, radius / max(unconstrained_max, 1e-12))
    constrained = current + trust_scale * displacement
    constrained_max = float(
        np.linalg.norm(constrained - current, axis=1).max(initial=0.0)
    )
    return constrained, {
        "unconstrained_step_max_m": unconstrained_max,
        "trust_region_scale": float(trust_scale),
        "trust_region_step_max_m": constrained_max,
        "trust_region_radius_m": radius,
    }


def refine_mesh_with_weighted_visible_arap(
    *,
    points_camera: np.ndarray,
    triangles: np.ndarray,
    mask_bool: np.ndarray,
    camera_k: np.ndarray,
    masked_depth_m: np.ndarray | None = None,
    target_scale_m: float | None = None,
    diameter_fn: Callable[[np.ndarray], float] | None = None,
    depth_mask_erosion_px: int = 3,
    depth_sample_stride_px: int = 1,
    maximum_depth_sample_count: int = 2048,
    maximum_neighbor_depth_delta_m: float = 0.012,
    maximum_projective_depth_residual_m: float = 0.040,
    depth_grazing_cosine_floor: float = 0.15,
    contour_sample_stride: int = 1,
    maximum_contour_sample_count: int = 2048,
    minimum_contour_residual_px: float = 0.15,
    maximum_contour_residual_px: float = 30.0,
    huber_depth_m: float = 0.006,
    huber_contour_px: float = 2.0,
    depth_weight: float = 8.0,
    silhouette_weight: float = 2.0,
    laplacian_weight: float = 0.5,
    arap_weight: float = 2.0,
    unseen_weight: float = 4.0,
    scale_weight: float = 25.0,
    centroid_weight: float = 25.0,
    visible_anchor_confidence: float = 0.0,
    transition_anchor_confidence: float = 0.10,
    far_hidden_anchor_confidence: float = 1.0,
    transition_ring_count: int = 4,
    outer_iteration_count: int = 6,
    local_global_iteration_count: int = 3,
    linear_solver_iteration_count: int = 500,
    step_damping_weight: float = 8.0,
    maximum_step_displacement_m: float = 0.003,
    maximum_cumulative_displacement_m: float = 0.008,
    minimum_edge_ratio: float = 0.55,
    maximum_edge_ratio: float = 1.80,
    minimum_area_ratio: float = 0.25,
    maximum_area_ratio: float = 4.00,
    minimum_step_scale: float = 1.0 / 4096.0,
    maximum_final_scale_correction_ratio: float = 0.005,
    raycast_function: RaycastFunction | None = None,
    **_legacy_arguments: object,
) -> SilhouetteMeshRefinementResult:
    points = np.asarray(points_camera, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    observed_mask = np.asarray(mask_bool, dtype=bool)
    camera = np.asarray(camera_k, dtype=np.float64)
    observed_depth = (
        None
        if masked_depth_m is None
        else np.asarray(masked_depth_m, dtype=np.float64)
    )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points_camera must have shape (N,3), got {points.shape}"
        )
    if len(points) == 0:
        raise ValueError("points_camera must contain at least one vertex")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"triangles must have shape (M,3), got {faces.shape}"
        )
    if len(faces) == 0:
        raise ValueError("triangles must contain at least one face")
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("triangles contains an out-of-range vertex index")
    if observed_mask.ndim != 2:
        raise ValueError(
            f"mask_bool must have shape (H,W), got {observed_mask.shape}"
        )
    if not observed_mask.any():
        raise ValueError("mask_bool must contain at least one foreground pixel")
    if camera.shape != (3, 3):
        raise ValueError(f"camera_k must have shape (3,3), got {camera.shape}")
    if observed_depth is not None and observed_depth.shape != observed_mask.shape:
        raise ValueError(
            "masked_depth_m and mask_bool must have the same shape"
        )
    if not np.isfinite(points).all():
        raise ValueError("points_camera contains non-finite values")
    if not np.isfinite(camera).all():
        raise ValueError("camera_k contains non-finite values")
    if float(camera[0, 0]) <= 0.0 or float(camera[1, 1]) <= 0.0:
        raise ValueError("camera_k focal lengths must be positive")
    if not np.isfinite(step_damping_weight) or step_damping_weight < 0.0:
        raise ValueError("step_damping_weight must be finite and non-negative")
    if (
        not np.isfinite(maximum_step_displacement_m)
        or maximum_step_displacement_m <= 0.0
    ):
        raise ValueError(
            "maximum_step_displacement_m must be finite and positive"
        )

    raycast_fn = raycast_function or _default_raycast_function()
    image_height, image_width = observed_mask.shape
    vertex_count = int(len(points))
    edges = _unique_edges(faces)
    adjacency = _adjacency(edges, vertex_count)
    edge_incidence = _edge_incidence(edges, vertex_count)
    identity_three = scipy.sparse.identity(3, format="csr")
    arap_matrix = scipy.sparse.kron(
        edge_incidence,
        identity_three,
        format="csr",
    )
    laplacian = _build_uniform_laplacian(faces, vertex_count)
    laplacian_xyz = scipy.sparse.kron(
        laplacian,
        identity_three,
        format="csr",
    )
    laplacian_rhs = laplacian_xyz @ points.reshape(-1)
    centroid_columns = (
        3 * np.arange(vertex_count, dtype=np.int64)[:, None]
        + np.arange(3, dtype=np.int64)[None, :]
    )
    centroid_matrix = scipy.sparse.coo_matrix(
        (
            np.full(3 * vertex_count, 1.0 / max(vertex_count, 1)),
            (
                np.tile(np.arange(3, dtype=np.int64), vertex_count),
                centroid_columns.reshape(-1),
            ),
        ),
        shape=(3, 3 * vertex_count),
    ).tocsr()
    centroid_rhs = points.mean(axis=0)
    step_damping_matrix = scipy.sparse.identity(
        3 * vertex_count,
        format="csr",
    )

    original_rms_radius = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (points - points.mean(axis=0, keepdims=True)) ** 2,
                    axis=1,
                )
            )
        )
    )
    original_diameter = (
        float(diameter_fn(points))
        if diameter_fn is not None
        else None
    )
    target_rms_radius = original_rms_radius
    if (
        target_scale_m is not None
        and original_diameter is not None
        and original_diameter > 1e-12
    ):
        target_rms_radius *= float(target_scale_m) / original_diameter

    initial_raycast = raycast_fn(
        points_camera=points,
        triangles=faces,
        camera_k=camera,
        image_height=image_height,
        image_width=image_width,
    )
    initial_metrics = _raster_observation_metrics(
        raycast=initial_raycast,
        observed_mask=observed_mask,
        observed_depth_m=observed_depth,
        depth_mask_erosion_px=depth_mask_erosion_px,
    )

    current = points.copy()
    last_raw = points.copy()
    outer_records: list[dict] = []
    last_depth_stats: dict = {}
    last_silhouette_stats: dict = {}
    last_anchor_confidence = np.zeros(vertex_count, dtype=np.float64)

    for outer_iteration in range(max(int(outer_iteration_count), 1)):
        raycast = raycast_fn(
            points_camera=current,
            triangles=faces,
            camera_k=camera,
            image_height=image_height,
            image_width=image_width,
        )
        visible = np.asarray(
            raycast["visible_vertex_mask"],
            dtype=bool,
        )
        (
            depth_matrix,
            depth_rhs,
            depth_confidence,
            depth_stats,
        ) = _depth_constraint_system(
            points=current,
            triangles=faces,
            camera_k=camera,
            raycast=raycast,
            observed_depth_m=(
                np.zeros_like(observed_mask, dtype=np.float64)
                if observed_depth is None
                else observed_depth
            ),
            observed_mask=observed_mask,
            mask_erosion_px=depth_mask_erosion_px,
            sample_stride_px=depth_sample_stride_px,
            maximum_sample_count=maximum_depth_sample_count,
            maximum_neighbor_depth_delta_m=maximum_neighbor_depth_delta_m,
            maximum_projective_residual_m=(
                maximum_projective_depth_residual_m
            ),
            grazing_cosine_floor=depth_grazing_cosine_floor,
        )
        (
            silhouette_matrix,
            silhouette_rhs,
            silhouette_confidence,
            silhouette_pixel_scale_m,
            silhouette_stats,
        ) = _silhouette_constraint_system(
            points=current,
            triangles=faces,
            camera_k=camera,
            raycast=raycast,
            observed_mask=observed_mask,
            contour_sample_stride=contour_sample_stride,
            maximum_sample_count=maximum_contour_sample_count,
            minimum_residual_px=minimum_contour_residual_px,
            maximum_residual_px=maximum_contour_residual_px,
        )
        anchor_confidence = _soft_unseen_confidence(
            visible_mask=visible,
            adjacency=adjacency,
            transition_ring_count=transition_ring_count,
            visible_confidence=visible_anchor_confidence,
            transition_confidence=transition_anchor_confidence,
            far_hidden_confidence=far_hidden_anchor_confidence,
        )
        anchor_indices = np.flatnonzero(anchor_confidence > 0.0)
        anchor_rows = scipy.sparse.identity(
            3 * vertex_count,
            format="csr",
        )[
            np.concatenate(
                [
                    3 * anchor_indices,
                    3 * anchor_indices + 1,
                    3 * anchor_indices + 2,
                ]
            )
        ]
        anchor_rhs = points.reshape(-1)[
            np.concatenate(
                [
                    3 * anchor_indices,
                    3 * anchor_indices + 1,
                    3 * anchor_indices + 2,
                ]
            )
        ]
        anchor_row_confidence = np.concatenate(
            [
                anchor_confidence[anchor_indices],
                anchor_confidence[anchor_indices],
                anchor_confidence[anchor_indices],
            ]
        )

        proposed = current.copy()
        inner_records: list[dict] = []
        for inner_iteration in range(max(int(local_global_iteration_count), 1)):
            rotations = _local_rotations(
                original_points=points,
                current_points=proposed,
                adjacency=adjacency,
            )
            arap_rhs = _arap_right_hand_side(
                original_points=points,
                edges=edges,
                rotations=rotations,
            )
            scale_matrix, scale_rhs, rms_radius = _scale_linearization(
                current_points=proposed,
                target_rms_radius_m=target_rms_radius,
            )

            matrices: list[scipy.sparse.csr_matrix] = []
            right_hand_sides: list[np.ndarray] = []

            if depth_matrix.shape[0] > 0:
                depth_residual = depth_matrix @ proposed.reshape(-1) - depth_rhs
                depth_robust = _huber_irls_weights(
                    depth_residual,
                    huber_depth_m,
                )
                depth_rows, depth_values = _weighted_rows(
                    matrix=depth_matrix,
                    rhs=depth_rhs,
                    row_weight=(
                        float(depth_weight)
                        * depth_confidence
                        * depth_robust
                        / max(depth_matrix.shape[0], 1)
                    ),
                )
                matrices.append(depth_rows)
                right_hand_sides.append(depth_values)

            if silhouette_matrix.shape[0] > 0:
                silhouette_residual = (
                    silhouette_matrix @ proposed.reshape(-1)
                    - silhouette_rhs
                )
                silhouette_delta = (
                    float(huber_contour_px)
                    * silhouette_pixel_scale_m
                )
                silhouette_robust = _huber_irls_weights(
                    silhouette_residual,
                    silhouette_delta,
                )
                silhouette_rows, silhouette_values = _weighted_rows(
                    matrix=silhouette_matrix,
                    rhs=silhouette_rhs,
                    row_weight=(
                        float(silhouette_weight)
                        * silhouette_confidence
                        * silhouette_robust
                        / max(silhouette_matrix.shape[0], 1)
                    ),
                )
                matrices.append(silhouette_rows)
                right_hand_sides.append(silhouette_values)

            arap_rows, arap_values = _weighted_rows(
                matrix=arap_matrix,
                rhs=arap_rhs,
                row_weight=np.full(
                    arap_matrix.shape[0],
                    float(arap_weight) / max(len(edges), 1),
                    dtype=np.float64,
                ),
            )
            matrices.append(arap_rows)
            right_hand_sides.append(arap_values)

            laplacian_rows, laplacian_values = _weighted_rows(
                matrix=laplacian_xyz,
                rhs=laplacian_rhs,
                row_weight=np.full(
                    laplacian_xyz.shape[0],
                    float(laplacian_weight) / max(vertex_count, 1),
                    dtype=np.float64,
                ),
            )
            matrices.append(laplacian_rows)
            right_hand_sides.append(laplacian_values)

            if anchor_rows.shape[0] > 0:
                weighted_anchor_rows, weighted_anchor_rhs = _weighted_rows(
                    matrix=anchor_rows,
                    rhs=anchor_rhs,
                    row_weight=(
                        float(unseen_weight)
                        * anchor_row_confidence
                        / max(float(anchor_row_confidence.sum()), 1.0)
                    ),
                )
                matrices.append(weighted_anchor_rows)
                right_hand_sides.append(weighted_anchor_rhs)

            if scale_matrix.shape[0] > 0 and scale_weight > 0.0:
                weighted_scale_rows, weighted_scale_rhs = _weighted_rows(
                    matrix=scale_matrix,
                    rhs=scale_rhs,
                    row_weight=np.asarray([float(scale_weight)]),
                )
                matrices.append(weighted_scale_rows)
                right_hand_sides.append(weighted_scale_rhs)

            if centroid_weight > 0.0:
                weighted_centroid_rows, weighted_centroid_rhs = _weighted_rows(
                    matrix=centroid_matrix,
                    rhs=centroid_rhs,
                    row_weight=np.full(
                        3,
                        float(centroid_weight),
                        dtype=np.float64,
                    ),
                )
                matrices.append(weighted_centroid_rows)
                right_hand_sides.append(weighted_centroid_rhs)

            if step_damping_weight > 0.0:
                weighted_step_rows, weighted_step_rhs = _weighted_rows(
                    matrix=step_damping_matrix,
                    rhs=current.reshape(-1),
                    row_weight=np.full(
                        3 * vertex_count,
                        float(step_damping_weight)
                        / max(3 * vertex_count, 1),
                        dtype=np.float64,
                    ),
                )
                matrices.append(weighted_step_rows)
                right_hand_sides.append(weighted_step_rhs)

            system_matrix = scipy.sparse.vstack(matrices, format="csr")
            system_rhs = np.concatenate(right_hand_sides)
            solved_flat = scipy.sparse.linalg.lsmr(
                system_matrix,
                system_rhs,
                atol=1e-7,
                btol=1e-7,
                maxiter=max(int(linear_solver_iteration_count), 1),
            )[0]
            unconstrained_solved = solved_flat.reshape((-1, 3))
            solved, trust_region = _apply_solver_trust_region(
                current_points=current,
                proposed_points=unconstrained_solved,
                maximum_displacement_m=maximum_step_displacement_m,
            )
            solve_step_max = float(
                np.linalg.norm(solved - proposed, axis=1).max()
            )
            inner_records.append(
                {
                    "iteration": int(inner_iteration),
                    "linear_row_count": int(system_matrix.shape[0]),
                    "solve_step_max_m": solve_step_max,
                    "rms_radius_m": float(rms_radius),
                    **trust_region,
                }
            )
            proposed = solved
            if solve_step_max <= 1e-6:
                break

        last_raw = proposed.copy()
        accepted, safety = _safe_increment_line_search(
            original_points=points,
            current_points=current,
            proposed_points=proposed,
            triangles=faces,
            edges=edges,
            maximum_step_displacement_m=maximum_step_displacement_m,
            maximum_cumulative_displacement_m=(
                maximum_cumulative_displacement_m
            ),
            minimum_edge_ratio=minimum_edge_ratio,
            maximum_edge_ratio=maximum_edge_ratio,
            minimum_area_ratio=minimum_area_ratio,
            maximum_area_ratio=maximum_area_ratio,
            minimum_step_scale=minimum_step_scale,
        )
        record = {
            "iteration": int(outer_iteration),
            "depth": depth_stats,
            "silhouette": silhouette_stats,
            "visible_vertex_count": int(np.count_nonzero(visible)),
            "soft_anchor_vertex_count": int(len(anchor_indices)),
            "inner_iterations": inner_records,
            "safety": safety,
        }
        outer_records.append(record)
        last_depth_stats = depth_stats
        last_silhouette_stats = silhouette_stats
        last_anchor_confidence = anchor_confidence

        accepted_step_max = float(
            np.linalg.norm(accepted - current, axis=1).max()
        )
        if safety.get("topology_safe", False):
            current = accepted
        if not safety.get("topology_safe", False):
            record["status"] = "stalled_by_safety"
            break
        if accepted_step_max <= 1e-5:
            record["status"] = "converged"
            break
        record["status"] = "updated_safe"

    refined_pre_scale = current.copy()
    scale_before = (
        float(diameter_fn(refined_pre_scale))
        if diameter_fn is not None
        else None
    )
    refined = refined_pre_scale.copy()
    scale_correction_applied = False
    scale_correction_attempted = False
    scale_correction_rejected_reason = None
    scale_correction_topology = None
    scale_correction_ratio = None
    if (
        target_scale_m is not None
        and scale_before is not None
        and scale_before > 1e-12
    ):
        scale_correction_ratio = float(target_scale_m) / scale_before
        if (
            abs(scale_correction_ratio - 1.0)
            <= float(maximum_final_scale_correction_ratio)
        ):
            scale_correction_attempted = True
            centroid = refined.mean(axis=0)
            scale_candidate = (
                centroid
                + scale_correction_ratio * (refined - centroid)
            )
            scale_correction_topology = _geometry_audit(
                reference_points=points,
                candidate_points=scale_candidate,
                triangles=faces,
                edges=edges,
                minimum_edge_ratio=minimum_edge_ratio,
                maximum_edge_ratio=maximum_edge_ratio,
                minimum_area_ratio=minimum_area_ratio,
                maximum_area_ratio=maximum_area_ratio,
            )
            if scale_correction_topology["topology_safe"]:
                refined = scale_candidate
                scale_correction_applied = True
            else:
                scale_correction_rejected_reason = (
                    "scale correction would make cumulative topology unsafe"
                )
    scale_after = (
        float(diameter_fn(refined))
        if diameter_fn is not None
        else None
    )

    final_raycast = raycast_fn(
        points_camera=refined,
        triangles=faces,
        camera_k=camera,
        image_height=image_height,
        image_width=image_width,
    )
    final_metrics = _raster_observation_metrics(
        raycast=final_raycast,
        observed_mask=observed_mask,
        observed_depth_m=observed_depth,
        depth_mask_erosion_px=depth_mask_erosion_px,
    )
    pre_scale_raycast = raycast_fn(
        points_camera=refined_pre_scale,
        triangles=faces,
        camera_k=camera,
        image_height=image_height,
        image_width=image_width,
    )
    pre_scale_metrics = _raster_observation_metrics(
        raycast=pre_scale_raycast,
        observed_mask=observed_mask,
        observed_depth_m=observed_depth,
        depth_mask_erosion_px=depth_mask_erosion_px,
    )

    displacement = refined - points
    displacement_norm = np.linalg.norm(displacement, axis=1)
    roughness_before = np.linalg.norm(laplacian @ points, axis=1)
    roughness_after = np.linalg.norm(laplacian @ refined, axis=1)
    final_topology = _geometry_audit(
        reference_points=points,
        candidate_points=refined,
        triangles=faces,
        edges=edges,
        minimum_edge_ratio=minimum_edge_ratio,
        maximum_edge_ratio=maximum_edge_ratio,
        minimum_area_ratio=minimum_area_ratio,
        maximum_area_ratio=maximum_area_ratio,
    )
    boundary_distance_before = initial_metrics[
        "symmetric_boundary_distance_mean_px"
    ]
    boundary_distance_after = final_metrics[
        "symmetric_boundary_distance_mean_px"
    ]

    diagnostics = {
        "status": (
            "refined" if final_topology["topology_safe"] else "unsafe"
        ),
        "refinement_mode": "weighted_visible_arap",
        "iou_before": float(initial_metrics["raster_iou"]),
        "iou_after": float(final_metrics["raster_iou"]),
        "raster_iou_before": float(initial_metrics["raster_iou"]),
        "raster_iou_raw": float(pre_scale_metrics["raster_iou"]),
        "raster_iou_after": float(final_metrics["raster_iou"]),
        "boundary_distance_before_px": boundary_distance_before,
        "boundary_distance_after_px": boundary_distance_after,
        "contour_metrics_before": initial_metrics,
        "contour_metrics_pre_scale": pre_scale_metrics,
        "contour_metrics_after": final_metrics,
        "boundary_band_vertex_count": int(
            last_silhouette_stats.get("active_contour_sample_count", 0)
        ),
        "interior_vertex_count": int(
            last_depth_stats.get("valid_depth_pixel_count", 0)
        ),
        "depth_correspondence_count": int(
            last_depth_stats.get("depth_sample_count", 0)
        ),
        "depth_outlier_count": int(
            last_depth_stats.get("depth_rejected_projective_count", 0)
        ),
        "depth_outlier_fraction": (
            float(
                last_depth_stats.get("depth_rejected_projective_count", 0)
                / max(
                    last_depth_stats.get("depth_sample_count", 0)
                    + last_depth_stats.get(
                        "depth_rejected_projective_count", 0
                    ),
                    1,
                )
            )
        ),
        "depth_outlier_threshold_m": float(
            maximum_projective_depth_residual_m
        ),
        "depth_outlier_robust_scale_m": float(huber_depth_m),
        "moved_vertex_count": int(np.count_nonzero(displacement_norm > 1e-6)),
        "inward_fraction": None,
        "outward_fraction": None,
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
        "scale_correction_ratio": scale_correction_ratio,
        "scale_correction_attempted": bool(scale_correction_attempted),
        "scale_correction_applied": bool(scale_correction_applied),
        "scale_correction_rejected_reason": scale_correction_rejected_reason,
        "scale_correction_topology": scale_correction_topology,
        "scale_raster_iou_delta": float(
            pre_scale_metrics["raster_iou"] - final_metrics["raster_iou"]
        ),
        "roughness_before_p95_m": float(
            np.quantile(roughness_before, 0.95)
        ),
        "roughness_after_p95_m": float(
            np.quantile(roughness_after, 0.95)
        ),
        "silhouette_strip": last_silhouette_stats,
        "depth_grid": last_depth_stats,
        "transition_ring_count": int(transition_ring_count),
        "far_hidden_anchor_count": int(
            np.count_nonzero(
                last_anchor_confidence >= far_hidden_anchor_confidence
            )
        ),
        "constraint_count": int(
            last_depth_stats.get("depth_sample_count", 0)
            + last_silhouette_stats.get("active_contour_sample_count", 0)
        ),
        "outer_iteration_count_requested": int(outer_iteration_count),
        "outer_iteration_count_completed": int(len(outer_records)),
        "depth_sample_stride_px": int(depth_sample_stride_px),
        "step_damping_weight": float(step_damping_weight),
        "solver_trust_region_radius_m": float(
            maximum_step_displacement_m
        ),
        "outer_iterations": outer_records,
        "topology_safety": final_topology,
        "final_cumulative_topology": final_topology,
    }
    intermediate_points_camera = {
        "M0_original_camera": points.copy(),
        "M1_weighted_raw_camera": last_raw.copy(),
        "M2_weighted_safe_pre_scale_camera": refined_pre_scale.copy(),
        "M3_weighted_final_camera": refined.copy(),
    }
    return SilhouetteMeshRefinementResult(
        refined_points_camera=refined,
        displacement=displacement,
        diagnostics=diagnostics,
        intermediate_points_camera=intermediate_points_camera,
    )
