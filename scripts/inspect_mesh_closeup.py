from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def render_shaded(axis, vertices, triangles, title, elev, azim, wireframe=False):
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

    edgecolor = (0, 0, 0, 0.15) if wireframe else "none"
    collection = Poly3DCollection(
        face_vertices, facecolor=face_colors, edgecolor=edgecolor, linewidth=0.1
    )
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
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wireframe", action="store_true")
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)

    angles = [
        (10, 0), (10, 45), (10, 90), (10, 135),
        (10, 180), (10, 225), (10, 270), (10, 315),
        (60, 45), (-60, 45), (10, 315),
    ][:8]

    figure = plt.figure(figsize=(24, 12))
    for index, (elev, azim) in enumerate(angles):
        axis = figure.add_subplot(2, 4, index + 1, projection="3d")
        render_shaded(
            axis, vertices, triangles,
            f"elev={elev},az={azim}", elev, azim, wireframe=args.wireframe,
        )

    figure.suptitle(args.title, fontsize=13)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
