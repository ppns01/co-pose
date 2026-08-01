"""Fuse two aligned masked-depth observations into one partial mesh.

The output intentionally remains an open visible-surface mesh.  It does not
close unseen regions or use an independently generated single-view proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TriangleMeshArrays:
    vertices_m: NDArray[np.float64]
    triangles: NDArray[np.int64]
    vertex_colors_rgb: NDArray[np.uint8]
    vertex_normals: NDArray[np.float64]

    def __post_init__(self) -> None:
        vertex_count = int(len(self.vertices_m))
        if self.vertices_m.shape != (vertex_count, 3):
            raise ValueError(
                "vertices_m must have shape (N, 3): "
                f"{self.vertices_m.shape}"
            )
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(
                "triangles must have shape (M, 3): "
                f"{self.triangles.shape}"
            )
        if self.vertex_colors_rgb.shape != (vertex_count, 3):
            raise ValueError(
                "vertex_colors_rgb must have shape (N, 3): "
                f"{self.vertex_colors_rgb.shape}"
            )
        if self.vertex_normals.shape != (vertex_count, 3):
            raise ValueError(
                "vertex_normals must have shape (N, 3): "
                f"{self.vertex_normals.shape}"
            )
        if self.vertices_m.dtype != np.float64:
            raise TypeError("vertices_m must use float64")
        if self.triangles.dtype != np.int64:
            raise TypeError("triangles must use int64")
        if self.vertex_colors_rgb.dtype != np.uint8:
            raise TypeError("vertex_colors_rgb must use uint8")
        if self.vertex_normals.dtype != np.float64:
            raise TypeError("vertex_normals must use float64")
        if vertex_count == 0 or len(self.triangles) == 0:
            raise ValueError("surface mesh must contain vertices and triangles")
        if self.triangles.min() < 0 or self.triangles.max() >= vertex_count:
            raise ValueError("triangles contain an out-of-range vertex index")
        if not np.isfinite(self.vertices_m).all():
            raise ValueError("vertices_m contain non-finite values")
        if not np.isfinite(self.vertex_normals).all():
            raise ValueError("vertex_normals contain non-finite values")


@dataclass(frozen=True)
class SurfaceFusionResult:
    mesh: TriangleMeshArrays
    reference_observation_count: NDArray[np.int64]
    query_observation_count: NDArray[np.int64]
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        vertex_count = len(self.mesh.vertices_m)
        for name, values in (
            (
                "reference_observation_count",
                self.reference_observation_count,
            ),
            (
                "query_observation_count",
                self.query_observation_count,
            ),
        ):
            if values.shape != (vertex_count,):
                raise ValueError(
                    f"{name} must have shape (N,): {values.shape}"
                )
            if values.dtype != np.int64:
                raise TypeError(f"{name} must use int64")


@dataclass(frozen=True)
class TSDFFusionResult:
    mesh: TriangleMeshArrays
    diagnostics: dict[str, Any]


def _as_rigid_transform(transform: Any, *, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4): {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(
        matrix[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=5e-3,
        rtol=0.0,
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=5e-3):
        raise ValueError(f"{name} does not contain a rigid rotation")
    return matrix


def _compute_vertex_normals(
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> NDArray[np.float64]:
    triangle_vertices = vertices[triangles]
    face_normals = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    normal_lengths = np.linalg.norm(face_normals, axis=1)
    valid_faces = normal_lengths > 1e-12
    face_normals[valid_faces] /= normal_lengths[valid_faces, None]
    face_normals[~valid_faces] = 0.0

    vertex_normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(
            vertex_normals,
            triangles[:, corner],
            face_normals,
        )

    vertex_lengths = np.linalg.norm(vertex_normals, axis=1)
    valid_vertices = vertex_lengths > 1e-12
    vertex_normals[valid_vertices] /= vertex_lengths[valid_vertices, None]
    vertex_normals[~valid_vertices] = np.asarray([0.0, 0.0, -1.0])
    return np.ascontiguousarray(vertex_normals, dtype=np.float64)


def _compact_mesh_arrays(
    *,
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
    colors: NDArray[np.uint8],
    reference_count: NDArray[np.int64] | None = None,
    query_count: NDArray[np.int64] | None = None,
) -> tuple[
    TriangleMeshArrays,
    NDArray[np.int64] | None,
    NDArray[np.int64] | None,
]:
    nondegenerate = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 0] != triangles[:, 2])
    )
    triangles = triangles[nondegenerate]
    if len(triangles) == 0:
        raise ValueError("all triangles became degenerate")

    canonical = np.sort(triangles, axis=1)
    _, unique_indices = np.unique(canonical, axis=0, return_index=True)
    triangles = triangles[np.sort(unique_indices)]

    used_vertices = np.unique(triangles.reshape(-1))
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    compact_triangles = old_to_new[triangles]
    compact_vertices = np.ascontiguousarray(
        vertices[used_vertices],
        dtype=np.float64,
    )
    compact_colors = np.ascontiguousarray(
        colors[used_vertices],
        dtype=np.uint8,
    )
    compact_normals = _compute_vertex_normals(
        compact_vertices,
        compact_triangles,
    )

    mesh = TriangleMeshArrays(
        vertices_m=compact_vertices,
        triangles=np.ascontiguousarray(compact_triangles, dtype=np.int64),
        vertex_colors_rgb=compact_colors,
        vertex_normals=compact_normals,
    )
    compact_reference_count = (
        None
        if reference_count is None
        else np.ascontiguousarray(reference_count[used_vertices], dtype=np.int64)
    )
    compact_query_count = (
        None
        if query_count is None
        else np.ascontiguousarray(query_count[used_vertices], dtype=np.int64)
    )
    return mesh, compact_reference_count, compact_query_count


def build_depth_surface_mesh(
    *,
    masked_depth_m: NDArray[np.floating[Any]],
    mask_bool: NDArray[np.bool_],
    camera_k: NDArray[np.floating[Any]],
    rgb: NDArray[np.uint8],
    pixel_stride: int = 1,
    maximum_triangle_depth_delta_m: float = 0.012,
    maximum_triangle_edge_length_m: float = 0.020,
) -> TriangleMeshArrays:
    """Triangulate one masked depth image in its camera coordinate frame."""

    depth = np.asarray(masked_depth_m, dtype=np.float64)
    mask = np.asarray(mask_bool, dtype=bool)
    camera = np.asarray(camera_k, dtype=np.float64)
    color_image = np.asarray(rgb, dtype=np.uint8)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("masked_depth_m and mask_bool must share shape (H, W)")
    if color_image.shape != (*depth.shape, 3):
        raise ValueError("rgb must have shape (H, W, 3)")
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise ValueError("camera_k must be a finite (3, 3) matrix")
    if float(camera[0, 0]) <= 0.0 or float(camera[1, 1]) <= 0.0:
        raise ValueError("camera_k focal lengths must be positive")
    if isinstance(pixel_stride, bool) or int(pixel_stride) < 1:
        raise ValueError("pixel_stride must be a positive integer")
    if maximum_triangle_depth_delta_m <= 0.0:
        raise ValueError("maximum_triangle_depth_delta_m must be positive")
    if maximum_triangle_edge_length_m <= 0.0:
        raise ValueError("maximum_triangle_edge_length_m must be positive")

    stride = int(pixel_stride)
    ys = np.arange(0, depth.shape[0], stride, dtype=np.int64)
    xs = np.arange(0, depth.shape[1], stride, dtype=np.int64)
    sampled_depth = depth[np.ix_(ys, xs)]
    sampled_mask = mask[np.ix_(ys, xs)]
    valid = sampled_mask & np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    if np.count_nonzero(valid) < 3:
        raise ValueError("masked depth contains fewer than three valid samples")

    grid_vertex_index = np.full(valid.shape, -1, dtype=np.int64)
    valid_y_grid, valid_x_grid = np.nonzero(valid)
    grid_vertex_index[valid] = np.arange(len(valid_y_grid), dtype=np.int64)
    pixel_y = ys[valid_y_grid]
    pixel_x = xs[valid_x_grid]
    z = depth[pixel_y, pixel_x]
    fx, fy = float(camera[0, 0]), float(camera[1, 1])
    cx, cy = float(camera[0, 2]), float(camera[1, 2])
    vertices = np.stack(
        (
            (pixel_x.astype(np.float64) - cx) * z / fx,
            (pixel_y.astype(np.float64) - cy) * z / fy,
            z,
        ),
        axis=1,
    )
    colors = color_image[pixel_y, pixel_x]

    top_left = grid_vertex_index[:-1, :-1].reshape(-1)
    top_right = grid_vertex_index[:-1, 1:].reshape(-1)
    bottom_left = grid_vertex_index[1:, :-1].reshape(-1)
    bottom_right = grid_vertex_index[1:, 1:].reshape(-1)
    candidate_triangles = np.concatenate(
        (
            np.stack((top_left, bottom_left, top_right), axis=1),
            np.stack((top_right, bottom_left, bottom_right), axis=1),
        ),
        axis=0,
    )
    candidate_triangles = candidate_triangles[
        np.all(candidate_triangles >= 0, axis=1)
    ]
    if len(candidate_triangles) == 0:
        raise ValueError("valid depth samples do not form any triangle")

    triangle_vertices = vertices[candidate_triangles]
    triangle_depths = triangle_vertices[:, :, 2]
    depth_ok = (
        np.max(triangle_depths, axis=1)
        - np.min(triangle_depths, axis=1)
        <= float(maximum_triangle_depth_delta_m)
    )
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangle_vertices[:, 0] - triangle_vertices[:, 1], axis=1),
            np.linalg.norm(triangle_vertices[:, 1] - triangle_vertices[:, 2], axis=1),
            np.linalg.norm(triangle_vertices[:, 2] - triangle_vertices[:, 0], axis=1),
        ),
        axis=1,
    )
    edge_ok = np.max(edge_lengths, axis=1) <= float(
        maximum_triangle_edge_length_m
    )
    triangles = candidate_triangles[depth_ok & edge_ok]
    if len(triangles) == 0:
        raise ValueError("all triangles were rejected as depth discontinuities")

    mesh, _, _ = _compact_mesh_arrays(
        vertices=np.ascontiguousarray(vertices, dtype=np.float64),
        triangles=np.ascontiguousarray(triangles, dtype=np.int64),
        colors=np.ascontiguousarray(colors, dtype=np.uint8),
    )
    return mesh


def transform_surface_mesh(
    mesh: TriangleMeshArrays,
    transform_target_from_source: Any,
) -> TriangleMeshArrays:
    transform = _as_rigid_transform(
        transform_target_from_source,
        name="transform_target_from_source",
    )
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    vertices = mesh.vertices_m @ rotation.T + translation
    normals = mesh.vertex_normals @ rotation.T
    normal_lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(normal_lengths[:, None], 1e-12)
    return TriangleMeshArrays(
        vertices_m=np.ascontiguousarray(vertices, dtype=np.float64),
        triangles=mesh.triangles.copy(),
        vertex_colors_rgb=mesh.vertex_colors_rgb.copy(),
        vertex_normals=np.ascontiguousarray(normals, dtype=np.float64),
    )


def fuse_aligned_surface_meshes(
    *,
    reference_mesh: TriangleMeshArrays,
    query_mesh_in_reference: TriangleMeshArrays,
    merge_distance_m: float = 0.003,
    minimum_normal_cosine: float = 0.30,
) -> SurfaceFusionResult:
    """Merge geometrically consistent cross-view vertices and retain all faces."""

    if merge_distance_m <= 0.0:
        raise ValueError("merge_distance_m must be positive")
    if not -1.0 <= minimum_normal_cosine <= 1.0:
        raise ValueError("minimum_normal_cosine must be in [-1, 1]")

    tree = cKDTree(reference_mesh.vertices_m)
    nearest_distances, nearest_reference = tree.query(
        query_mesh_in_reference.vertices_m,
        k=1,
    )
    normal_cosines = np.einsum(
        "ij,ij->i",
        query_mesh_in_reference.vertex_normals,
        reference_mesh.vertex_normals[nearest_reference],
    )
    matched_query = (
        (nearest_distances <= float(merge_distance_m))
        & (normal_cosines >= float(minimum_normal_cosine))
    )

    reference_count = len(reference_mesh.vertices_m)
    query_count = len(query_mesh_in_reference.vertices_m)
    query_group = np.arange(query_count, dtype=np.int64) + reference_count
    query_group[matched_query] = nearest_reference[matched_query]
    group_index = np.concatenate(
        (
            np.arange(reference_count, dtype=np.int64),
            query_group,
        )
    )
    unique_groups, inverse_group = np.unique(group_index, return_inverse=True)
    fused_vertex_count = len(unique_groups)

    all_vertices = np.concatenate(
        (reference_mesh.vertices_m, query_mesh_in_reference.vertices_m),
        axis=0,
    )
    all_colors = np.concatenate(
        (
            reference_mesh.vertex_colors_rgb.astype(np.float64),
            query_mesh_in_reference.vertex_colors_rgb.astype(np.float64),
        ),
        axis=0,
    )
    observation_count = np.bincount(
        inverse_group,
        minlength=fused_vertex_count,
    ).astype(np.float64)
    fused_vertices = np.zeros((fused_vertex_count, 3), dtype=np.float64)
    fused_colors_float = np.zeros((fused_vertex_count, 3), dtype=np.float64)
    for dimension in range(3):
        np.add.at(
            fused_vertices[:, dimension],
            inverse_group,
            all_vertices[:, dimension],
        )
        np.add.at(
            fused_colors_float[:, dimension],
            inverse_group,
            all_colors[:, dimension],
        )
    fused_vertices /= observation_count[:, None]
    fused_colors = np.clip(
        np.rint(fused_colors_float / observation_count[:, None]),
        0,
        255,
    ).astype(np.uint8)

    reference_observation_count = np.bincount(
        inverse_group[:reference_count],
        minlength=fused_vertex_count,
    ).astype(np.int64)
    query_observation_count = np.bincount(
        inverse_group[reference_count:],
        minlength=fused_vertex_count,
    ).astype(np.int64)

    reference_triangles = inverse_group[reference_mesh.triangles]
    query_triangles = inverse_group[
        reference_count + query_mesh_in_reference.triangles
    ]
    combined_triangles = np.concatenate(
        (reference_triangles, query_triangles),
        axis=0,
    )
    fused_mesh, compact_reference_count, compact_query_count = (
        _compact_mesh_arrays(
            vertices=fused_vertices,
            triangles=combined_triangles,
            colors=fused_colors,
            reference_count=reference_observation_count,
            query_count=query_observation_count,
        )
    )
    assert compact_reference_count is not None
    assert compact_query_count is not None

    distance_quantiles = np.quantile(
        nearest_distances,
        [0.5, 0.9, 0.95],
    )
    diagnostics: dict[str, Any] = {
        "coordinate_frame": "reference_camera",
        "reference_input_vertex_count": reference_count,
        "query_input_vertex_count": query_count,
        "matched_query_vertex_count": int(np.count_nonzero(matched_query)),
        "matched_query_vertex_fraction": float(np.mean(matched_query)),
        "matched_reference_vertex_count": int(
            len(np.unique(nearest_reference[matched_query]))
        ),
        "nearest_distance_median_m": float(distance_quantiles[0]),
        "nearest_distance_p90_m": float(distance_quantiles[1]),
        "nearest_distance_p95_m": float(distance_quantiles[2]),
        "merge_distance_m": float(merge_distance_m),
        "minimum_normal_cosine": float(minimum_normal_cosine),
        "fused_vertex_count": int(len(fused_mesh.vertices_m)),
        "fused_triangle_count": int(len(fused_mesh.triangles)),
        "shared_fused_vertex_count": int(
            np.count_nonzero(
                (compact_reference_count > 0) & (compact_query_count > 0)
            )
        ),
    }
    return SurfaceFusionResult(
        mesh=fused_mesh,
        reference_observation_count=compact_reference_count,
        query_observation_count=compact_query_count,
        diagnostics=diagnostics,
    )


def fuse_masked_rgbd_tsdf(
    *,
    reference_depth_m: NDArray[np.floating[Any]],
    reference_rgb: NDArray[np.uint8],
    reference_camera_k: NDArray[np.floating[Any]],
    query_depth_m: NDArray[np.floating[Any]],
    query_rgb: NDArray[np.uint8],
    query_camera_k: NDArray[np.floating[Any]],
    transform_query_from_reference: Any,
    voxel_length_m: float = 0.0015,
    sdf_truncation_m: float = 0.006,
    depth_truncation_m: float | None = None,
    minimum_component_triangle_count: int = 20,
) -> TSDFFusionResult:
    """Integrate two masked RGB-D frames in the reference-camera frame.

    Open3D is imported lazily because the repository's CPU-only test Python
    does not necessarily include it.  The normal pipeline environment already
    uses Open3D for mesh refinement and rendering utilities.
    """

    if voxel_length_m <= 0.0:
        raise ValueError("voxel_length_m must be positive")
    if sdf_truncation_m <= 0.0:
        raise ValueError("sdf_truncation_m must be positive")
    if sdf_truncation_m < 2.0 * voxel_length_m:
        raise ValueError(
            "sdf_truncation_m must be at least twice voxel_length_m"
        )
    if (
        isinstance(minimum_component_triangle_count, bool)
        or int(minimum_component_triangle_count) < 1
    ):
        raise ValueError(
            "minimum_component_triangle_count must be a positive integer"
        )
    transform = _as_rigid_transform(
        transform_query_from_reference,
        name="transform_query_from_reference",
    )

    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "TSDF fusion requires Open3D. Run this script with the "
            "project's sam3_ros Python environment."
        ) from error

    observations = (
        (
            "reference",
            np.asarray(reference_depth_m, dtype=np.float32),
            np.asarray(reference_rgb, dtype=np.uint8),
            np.asarray(reference_camera_k, dtype=np.float64),
            np.eye(4, dtype=np.float64),
        ),
        (
            "query",
            np.asarray(query_depth_m, dtype=np.float32),
            np.asarray(query_rgb, dtype=np.uint8),
            np.asarray(query_camera_k, dtype=np.float64),
            transform,
        ),
    )
    maximum_valid_depth = 0.0
    for name, depth, rgb, camera, extrinsic in observations:
        if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
            raise ValueError(
                f"{name} depth/rgb shapes are incompatible: "
                f"depth={depth.shape}, rgb={rgb.shape}"
            )
        if camera.shape != (3, 3) or not np.isfinite(camera).all():
            raise ValueError(f"{name} camera_k must be a finite (3, 3) matrix")
        if not np.isfinite(depth).all() or np.any(depth < 0.0):
            raise ValueError(f"{name} depth must be finite and non-negative")
        if not np.any(depth > 0.0):
            raise ValueError(f"{name} depth has no valid sample")
        if extrinsic.shape != (4, 4):
            raise ValueError(f"{name} extrinsic must have shape (4, 4)")
        maximum_valid_depth = max(
            maximum_valid_depth,
            float(np.max(depth)),
        )

    if depth_truncation_m is None:
        effective_depth_truncation_m = maximum_valid_depth + max(
            0.10,
            10.0 * float(voxel_length_m),
        )
    else:
        effective_depth_truncation_m = float(depth_truncation_m)
        if effective_depth_truncation_m <= maximum_valid_depth:
            raise ValueError(
                "depth_truncation_m must exceed the maximum valid depth: "
                f"truncation={effective_depth_truncation_m}, "
                f"maximum={maximum_valid_depth}"
            )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length_m),
        sdf_trunc=float(sdf_truncation_m),
        color_type=(
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        ),
    )
    valid_pixel_counts: dict[str, int] = {}
    for name, depth, rgb, camera, extrinsic in observations:
        height, width = depth.shape
        valid_pixel_counts[name] = int(np.count_nonzero(depth > 0.0))
        color_image = o3d.geometry.Image(np.ascontiguousarray(rgb))
        depth_image = o3d.geometry.Image(np.ascontiguousarray(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_image,
            depth=depth_image,
            depth_scale=1.0,
            depth_trunc=effective_depth_truncation_m,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(width),
            int(height),
            float(camera[0, 0]),
            float(camera[1, 1]),
            float(camera[0, 2]),
            float(camera[1, 2]),
        )
        # Open3D expects world-to-camera extrinsics.  Reference camera is the
        # world frame and T_query_from_reference is therefore query extrinsic.
        volume.integrate(rgbd, intrinsic, extrinsic)

    open3d_mesh = volume.extract_triangle_mesh()
    open3d_mesh.remove_duplicated_vertices()
    open3d_mesh.remove_duplicated_triangles()
    open3d_mesh.remove_degenerate_triangles()
    open3d_mesh.remove_unreferenced_vertices()
    component_labels, component_triangle_counts, _ = (
        open3d_mesh.cluster_connected_triangles()
    )
    component_labels_array = np.asarray(component_labels, dtype=np.int64)
    component_counts_array = np.asarray(
        component_triangle_counts,
        dtype=np.int64,
    )
    component_count_before_filter = int(len(component_counts_array))
    remove_triangle_mask = (
        component_counts_array[component_labels_array]
        < int(minimum_component_triangle_count)
    )
    removed_triangle_count = int(np.count_nonzero(remove_triangle_mask))
    if removed_triangle_count > 0:
        open3d_mesh.remove_triangles_by_mask(remove_triangle_mask)
        open3d_mesh.remove_unreferenced_vertices()
    _, filtered_counts, _ = (
        open3d_mesh.cluster_connected_triangles()
    )
    open3d_mesh.compute_vertex_normals()
    vertices = np.asarray(open3d_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(open3d_mesh.triangles, dtype=np.int64)
    normals = np.asarray(open3d_mesh.vertex_normals, dtype=np.float64)
    colors_float = np.asarray(open3d_mesh.vertex_colors, dtype=np.float64)
    if len(vertices) == 0 or len(triangles) == 0:
        raise RuntimeError(
            "TSDF integration produced an empty mesh; verify pose and depth units"
        )
    if colors_float.shape == (len(vertices), 3):
        colors = np.clip(np.rint(colors_float * 255.0), 0, 255).astype(np.uint8)
    else:
        colors = np.full((len(vertices), 3), 127, dtype=np.uint8)

    mesh = TriangleMeshArrays(
        vertices_m=np.ascontiguousarray(vertices, dtype=np.float64),
        triangles=np.ascontiguousarray(triangles, dtype=np.int64),
        vertex_colors_rgb=np.ascontiguousarray(colors, dtype=np.uint8),
        vertex_normals=np.ascontiguousarray(normals, dtype=np.float64),
    )
    diagnostics: dict[str, Any] = {
        "fusion_method": "scalable_tsdf",
        "coordinate_frame": "reference_camera",
        "reference_valid_depth_pixel_count": valid_pixel_counts["reference"],
        "query_valid_depth_pixel_count": valid_pixel_counts["query"],
        "voxel_length_m": float(voxel_length_m),
        "sdf_truncation_m": float(sdf_truncation_m),
        "depth_truncation_m": float(effective_depth_truncation_m),
        "minimum_component_triangle_count": int(
            minimum_component_triangle_count
        ),
        "component_count_before_filter": component_count_before_filter,
        "component_count_after_filter": int(len(filtered_counts)),
        "removed_small_component_triangle_count": removed_triangle_count,
        "fused_vertex_count": int(len(vertices)),
        "fused_triangle_count": int(len(triangles)),
        "edge_manifold_with_boundary": bool(
            open3d_mesh.is_edge_manifold(allow_boundary_edges=True)
        ),
        "vertex_manifold": bool(open3d_mesh.is_vertex_manifold()),
        "watertight": bool(open3d_mesh.is_watertight()),
    }
    return TSDFFusionResult(mesh=mesh, diagnostics=diagnostics)


def write_triangle_mesh_ply(
    path: Path,
    mesh: TriangleMeshArrays,
) -> Path:
    """Write an ASCII PLY with meter vertices, normals, colors, and faces."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        (
            "ply",
            "format ascii 1.0",
            "comment coordinates are meters",
            f"element vertex {len(mesh.vertices_m)}",
            "property double x",
            "property double y",
            "property double z",
            "property double nx",
            "property double ny",
            "property double nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"element face {len(mesh.triangles)}",
            "property list uchar int vertex_indices",
            "end_header",
        )
    )
    with output_path.open("w", encoding="ascii", newline="\n") as file:
        file.write(header)
        file.write("\n")
        for vertex, normal, color in zip(
            mesh.vertices_m,
            mesh.vertex_normals,
            mesh.vertex_colors_rgb,
            strict=True,
        ):
            file.write(
                f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for triangle in mesh.triangles:
            file.write(
                f"3 {int(triangle[0])} {int(triangle[1])} "
                f"{int(triangle[2])}\n"
            )
    return output_path
