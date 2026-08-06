from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def _project(points_camera: np.ndarray, camera_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx, fy = camera_k[0, 0], camera_k[1, 1]
    cx, cy = camera_k[0, 2], camera_k[1, 2]
    z = points_camera[:, 2]
    u = points_camera[:, 0] / z * fx + cx
    v = points_camera[:, 1] / z * fy + cy
    return u, v


def render_mesh_on_photo(
    *,
    reference_mesh_path: Path,
    query_mesh_path: Path,
    reference_camera_k: np.ndarray,
    query_camera_k: np.ndarray,
    reference_rgb: np.ndarray,
    query_rgb: np.ndarray,
    reference_mask_bool: np.ndarray,
    query_mask_bool: np.ndarray,
    output_path: Path,
    title: str,
) -> Path:
    """
    Reference와 Query의 FoundationPose self-alignment 결과를 RGB 위에 투영한다.

    저장된 mesh는 dgedi_runner.py의 _save_self_aligned_mesh()에서
    이미 A1 (reference) 또는 B1 (query) pose가 적용된 camera frame 상태이다.

    Args:
        reference_mesh_path: reference self-aligned mesh (이미 A1이 적용된 camera frame)
        query_mesh_path: query self-aligned mesh (이미 B1이 적용된 camera frame)
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Reference: A1 추정 결과 (이미 camera frame)
    reference_mesh = o3d.io.read_triangle_mesh(str(reference_mesh_path))
    reference_vertices_camera = np.asarray(reference_mesh.vertices, dtype=np.float64)

    # z > 0 필터링 및 이미지 경계 체크
    valid_z = reference_vertices_camera[:, 2] > 1e-6
    reference_vertices_camera = reference_vertices_camera[valid_z]

    reference_u, reference_v = _project(reference_vertices_camera, reference_camera_k)

    # 이미지 경계 내부만 선택
    h, w = reference_rgb.shape[:2]
    inside = (reference_u >= 0) & (reference_u < w) & (reference_v >= 0) & (reference_v < h)
    reference_u = reference_u[inside]
    reference_v = reference_v[inside]

    # Query: B1 추정 결과 (이미 camera frame)
    query_mesh = o3d.io.read_triangle_mesh(str(query_mesh_path))
    query_vertices_camera = np.asarray(query_mesh.vertices, dtype=np.float64)

    # z > 0 필터링 및 이미지 경계 체크
    valid_z = query_vertices_camera[:, 2] > 1e-6
    query_vertices_camera = query_vertices_camera[valid_z]

    query_u, query_v = _project(query_vertices_camera, query_camera_k)

    # 이미지 경계 내부만 선택
    h, w = query_rgb.shape[:2]
    inside = (query_u >= 0) & (query_u < w) & (query_v >= 0) & (query_v < h)
    query_u = query_u[inside]
    query_v = query_v[inside]

    # Plot reference
    reference_mask_ys, reference_mask_xs = np.nonzero(reference_mask_bool)
    axes[0].imshow(reference_rgb)
    if len(reference_mask_xs) > 0:
        axes[0].scatter(
            reference_mask_xs[::13], reference_mask_ys[::13], s=1.0, c="white", alpha=0.5,
            label="observed mask",
        )
    axes[0].scatter(reference_u, reference_v, s=2.0, c="cyan", alpha=0.5, label="A1 estimate")
    if len(reference_mask_xs) > 0:
        axes[0].set_xlim(reference_mask_xs.min() - 40, reference_mask_xs.max() + 40)
        axes[0].set_ylim(reference_mask_ys.max() + 40, reference_mask_ys.min() - 40)
    axes[0].set_title("Reference A1 estimate", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=8, markerscale=6)
    axes[0].axis("off")

    # Plot query
    query_mask_ys, query_mask_xs = np.nonzero(query_mask_bool)
    axes[1].imshow(query_rgb)
    if len(query_mask_xs) > 0:
        axes[1].scatter(
            query_mask_xs[::13], query_mask_ys[::13], s=1.0, c="white", alpha=0.5,
            label="observed mask",
        )
    axes[1].scatter(query_u, query_v, s=2.0, c="red", alpha=0.5, label="B1 estimate")
    if len(query_mask_xs) > 0:
        axes[1].set_xlim(query_mask_xs.min() - 40, query_mask_xs.max() + 40)
        axes[1].set_ylim(query_mask_ys.max() + 40, query_mask_ys.min() - 40)
    axes[1].set_title("Query B1 estimate", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=8, markerscale=6)
    axes[1].axis("off")

    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)

    return output_path


def render_mesh_with_gt_pose(
    *,
    cad_mesh_path: Path,
    reference_gt_pose: np.ndarray,
    query_gt_pose: np.ndarray,
    reference_camera_k: np.ndarray,
    query_camera_k: np.ndarray,
    reference_rgb: np.ndarray,
    query_rgb: np.ndarray,
    reference_mask_bool: np.ndarray,
    query_mask_bool: np.ndarray,
    output_path: Path,
    title: str,
) -> Path:
    """
    CAD mesh를 GT pose로 변환해서 RGB 위에 투영한다.

    Reference와 query 모두 GT pose 위치에 mesh가 렌더링되어야 한다.
    이것이 올바른 위치이고, 예측 pose는 이 위치와 비교된다.

    Args:
        cad_mesh_path: 원본 CAD mesh (object frame)
        reference_gt_pose: T_camera_from_object (4x4)
        query_gt_pose: T_camera_from_object (4x4)
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Load CAD mesh once
    cad_mesh = o3d.io.read_triangle_mesh(str(cad_mesh_path))
    cad_vertices = np.asarray(cad_mesh.vertices, dtype=np.float64)

    # BOP CAD models are in mm, convert to meters
    cad_vertices_m = cad_vertices * 0.001

    views = [
        ("reference (GT)", reference_gt_pose, reference_camera_k, reference_rgb, reference_mask_bool),
        ("query (GT)", query_gt_pose, query_camera_k, query_rgb, query_mask_bool),
    ]

    for axis, (view_name, gt_pose, camera_k, rgb, mask_bool) in zip(axes, views):
        # Transform CAD vertices to camera frame using GT pose
        # GT pose already expects meter-scale input
        vertices_camera = (
            cad_vertices_m @ gt_pose[:3, :3].T
            + gt_pose[:3, 3][None, :]
        )

        # Filter z > 0 and within image bounds
        valid_z = vertices_camera[:, 2] > 1e-6
        vertices_camera = vertices_camera[valid_z]

        u, v = _project(vertices_camera, camera_k)

        # Filter within image bounds
        h, w = rgb.shape[:2]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[inside]
        v = v[inside]

        mask_ys, mask_xs = np.nonzero(mask_bool)

        axis.imshow(rgb)
        if len(mask_xs) > 0:
            axis.scatter(
                mask_xs[::13], mask_ys[::13], s=1.0, c="white", alpha=0.5,
                label="real mask",
            )
        axis.scatter(u, v, s=2.0, c="lime", alpha=0.5, label="GT mesh")

        if len(mask_xs) > 0:
            axis.set_xlim(mask_xs.min() - 40, mask_xs.max() + 40)
            axis.set_ylim(mask_ys.max() + 40, mask_ys.min() - 40)

        axis.set_title(view_name, fontsize=11)
        axis.legend(loc="upper right", fontsize=8, markerscale=6)
        axis.axis("off")

    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)

    return output_path
