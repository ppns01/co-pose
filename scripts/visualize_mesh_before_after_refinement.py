from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.triangles, dtype=np.int64),
    )


def project(points: np.ndarray, camera_k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy = camera_k[0, 0], camera_k[1, 1]
    cx, cy = camera_k[0, 2], camera_k[1, 2]
    z = points[:, 2]
    u = points[:, 0] / z * fx + cx
    v = points[:, 1] / z * fy + cy
    return u, v, z


def render_shaded(axis, vertices: np.ndarray, triangles: np.ndarray, title: str, elev: float, azim: float) -> None:
    face_vertices = vertices[triangles]
    face_normals = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    face_normals = face_normals / norms

    light_direction = np.array([0.3, -0.5, -0.8])
    light_direction = light_direction / np.linalg.norm(light_direction)
    intensity = np.clip(face_normals @ light_direction, 0.15, 1.0)

    face_colors = np.zeros((len(triangles), 4))
    face_colors[:, 0] = 0.95 * intensity
    face_colors[:, 1] = 0.78 * intensity
    face_colors[:, 2] = 0.25 * intensity
    face_colors[:, 3] = 1.0

    collection = Poly3DCollection(face_vertices, facecolor=face_colors, edgecolor="none")
    axis.add_collection3d(collection)

    center = vertices.mean(axis=0)
    radius = np.linalg.norm(vertices - center, axis=1).max()
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=elev, azim=azim)
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-mesh", type=Path, required=True)
    parser.add_argument("--refined-mesh", type=Path, required=True)
    parser.add_argument("--rgb-image", type=Path, required=True)
    parser.add_argument("--mask-bool", type=Path, required=True)
    parser.add_argument("--camera-k-json", type=Path, required=True)
    parser.add_argument("--camera-k-key", type=str, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v_orig, t_orig = load_mesh(args.original_mesh)
    v_refined, t_refined = load_mesh(args.refined_mesh)

    with open(args.camera_k_json) as f:
        cams = json.load(f)
    camera_k = np.asarray(cams[args.camera_k_key]["cam_K"]).reshape(3, 3)

    rgb = plt.imread(str(args.rgb_image))
    mask_bool = np.load(args.mask_bool)

    figure = plt.figure(figsize=(20, 10))

    # --- row 1: shaded 3D mesh renders, original vs refined, 2 angles each ---
    for column, (elev, azim) in enumerate([(10, 60), (10, 150)]):
        axis = figure.add_subplot(2, 4, column + 1, projection="3d")
        render_shaded(axis, v_orig, t_orig, f"ORIGINAL (elev={elev},az={azim})", elev, azim)

    for column, (elev, azim) in enumerate([(10, 60), (10, 150)]):
        axis = figure.add_subplot(2, 4, column + 3, projection="3d")
        render_shaded(axis, v_refined, t_refined, f"REFINED (elev={elev},az={azim})", elev, azim)

    # --- row 2: silhouette overlay on the real photo ---
    axis = figure.add_subplot(2, 4, 5)
    axis.imshow(rgb)
    mask_ys, mask_xs = np.nonzero(mask_bool)
    if len(mask_xs) > 0:
        axis.scatter(mask_xs[::13], mask_ys[::13], s=0.6, c="white", alpha=0.5, label="real mask")
    u, v, z = project(v_orig, camera_k)
    axis.scatter(u[::5], v[::5], s=0.6, c="red", alpha=0.5, label="original mesh")
    axis.set_xlim(mask_xs.min() - 30, mask_xs.max() + 30)
    axis.set_ylim(mask_ys.max() + 30, mask_ys.min() - 30)
    axis.set_title("ORIGINAL vs real mask (on photo)", fontsize=9)
    axis.legend(loc="upper right", fontsize=6, markerscale=6)
    axis.axis("off")

    axis = figure.add_subplot(2, 4, 6)
    axis.imshow(rgb)
    if len(mask_xs) > 0:
        axis.scatter(mask_xs[::13], mask_ys[::13], s=0.6, c="white", alpha=0.5, label="real mask")
    u, v, z = project(v_refined, camera_k)
    axis.scatter(u[::5], v[::5], s=0.6, c="orange", alpha=0.5, label="refined mesh")
    axis.set_xlim(mask_xs.min() - 30, mask_xs.max() + 30)
    axis.set_ylim(mask_ys.max() + 30, mask_ys.min() - 30)
    axis.set_title("REFINED vs real mask (on photo)", fontsize=9)
    axis.legend(loc="upper right", fontsize=6, markerscale=6)
    axis.axis("off")

    axis = figure.add_subplot(2, 4, 7)
    axis.imshow(rgb)
    if len(mask_xs) > 0:
        axis.scatter(mask_xs[::13], mask_ys[::13], s=0.6, c="white", alpha=0.6, label="real mask")
    u_o, v_o, _ = project(v_orig, camera_k)
    u_r, v_r, _ = project(v_refined, camera_k)
    axis.scatter(u_o[::5], v_o[::5], s=0.6, c="red", alpha=0.4, label="original")
    axis.scatter(u_r[::5], v_r[::5], s=0.6, c="orange", alpha=0.4, label="refined")
    axis.set_xlim(mask_xs.min() - 30, mask_xs.max() + 30)
    axis.set_ylim(mask_ys.max() + 30, mask_ys.min() - 30)
    axis.set_title("BOTH overlaid", fontsize=9)
    axis.legend(loc="upper right", fontsize=6, markerscale=6)
    axis.axis("off")

    axis = figure.add_subplot(2, 4, 8)
    axis.imshow(rgb)
    axis.set_xlim(mask_xs.min() - 30, mask_xs.max() + 30)
    axis.set_ylim(mask_ys.max() + 30, mask_ys.min() - 30)
    axis.set_title("real photo (reference)", fontsize=9)
    axis.axis("off")

    figure.suptitle(args.title, fontsize=13)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
