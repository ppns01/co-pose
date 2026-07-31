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
    최종 self-aligned mesh(camera frame, pose baked-in)를 실제 RGB 위에
    투영해서 실제 mask와 나란히 비교한다.

    Rotation/translation error 숫자만으로는 self-pose 자체가 정확한지,
    아니면 dGeDi나 2차 pose refine 단계에서 문제가 생겼는지 구분할 수
    없다 -- 이 시각화가 그 둘을 직접 눈으로 구분하게 해준다.
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))

    views = [
        ("reference", reference_mesh_path, reference_camera_k, reference_rgb, reference_mask_bool),
        ("query", query_mesh_path, query_camera_k, query_rgb, query_mask_bool),
    ]

    for axis, (view_name, mesh_path, camera_k, rgb, mask_bool) in zip(axes, views):
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        u, v = _project(vertices, camera_k)

        mask_ys, mask_xs = np.nonzero(mask_bool)

        axis.imshow(rgb)
        if len(mask_xs) > 0:
            axis.scatter(
                mask_xs[::13], mask_ys[::13], s=1.0, c="white", alpha=0.5,
                label="real mask",
            )
        axis.scatter(u[::3], v[::3], s=1.0, c="red", alpha=0.4, label="mesh")

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
