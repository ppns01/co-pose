from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def sample_triangle_mesh(
    path: Path,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64,
    )

    if len(vertices) == 0:
        raise RuntimeError(
            f"Mesh has no vertices: {path}"
        )

    if len(triangles) == 0:
        if len(vertices) <= sample_count:
            return vertices

        rng = np.random.default_rng(seed)

        indices = rng.choice(
            len(vertices),
            size=sample_count,
            replace=False,
        )

        return vertices[indices]

    triangle_vertices = vertices[triangles]

    edge_1 = (
        triangle_vertices[:, 1]
        - triangle_vertices[:, 0]
    )

    edge_2 = (
        triangle_vertices[:, 2]
        - triangle_vertices[:, 0]
    )

    areas = (
        np.linalg.norm(
            np.cross(edge_1, edge_2),
            axis=1,
        )
        * 0.5
    )

    valid = np.isfinite(areas) & (areas > 0.0)

    triangle_vertices = triangle_vertices[valid]
    areas = areas[valid]

    if len(areas) == 0:
        raise RuntimeError(
            f"Mesh has no valid triangles: {path}"
        )

    probabilities = areas / areas.sum()

    rng = np.random.default_rng(seed)

    triangle_indices = rng.choice(
        len(triangle_vertices),
        size=sample_count,
        replace=True,
        p=probabilities,
    )

    selected = triangle_vertices[triangle_indices]

    random_values = rng.random(
        (sample_count, 2),
    )

    square_root = np.sqrt(
        random_values[:, 0]
    )

    barycentric_0 = 1.0 - square_root

    barycentric_1 = (
        square_root
        * (1.0 - random_values[:, 1])
    )

    barycentric_2 = (
        square_root
        * random_values[:, 1]
    )

    points = (
        selected[:, 0] * barycentric_0[:, None]
        + selected[:, 1] * barycentric_1[:, None]
        + selected[:, 2] * barycentric_2[:, None]
    )

    return points


def robust_pca_statistics(
    points: np.ndarray,
) -> dict[str, Any]:
    center = np.median(
        points,
        axis=0,
    )

    centered = points - center

    covariance = np.cov(
        centered,
        rowvar=False,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    projected = centered @ eigenvectors

    lower = np.quantile(
        projected,
        0.05,
        axis=0,
    )

    upper = np.quantile(
        projected,
        0.95,
        axis=0,
    )

    extents = upper - lower

    geometric_mean_extent = float(
        np.cbrt(
            np.prod(
                np.maximum(extents, 1e-12)
            )
        )
    )

    normalized_extents = (
        extents
        / geometric_mean_extent
    )

    eigenvalue_separation = [
        float(
            eigenvalues[0]
            / max(eigenvalues[1], 1e-12)
        ),
        float(
            eigenvalues[1]
            / max(eigenvalues[2], 1e-12)
        ),
    ]

    return {
        "center": center.tolist(),
        "eigenvalues": (
            eigenvalues.tolist()
        ),
        "robust_pca_extents_m": (
            extents.tolist()
        ),
        "normalized_pca_extents": (
            normalized_extents.tolist()
        ),
        "eigenvalue_separation": (
            eigenvalue_separation
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--scene-id",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--query-image-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=100000,
    )

    args = parser.parse_args()

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    query_label = (
        f"q{args.scene_id:06d}_"
        f"{args.query_image_id:06d}_i00"
    )

    batch_query_root = (
        output_root
        / "queries"
        / query_label
    )

    if batch_query_root.is_dir():
        query_root = batch_query_root
    else:
        query_root = output_root

    mesh_root = (
        query_root
        / "mesh_registration"
        / "dgedi"
        / "self_aligned_meshes"
    )

    reference_path = (
        mesh_root
        / (
            "reference_self_aligned_"
            "in_reference_camera.obj"
        )
    )

    query_path = (
        mesh_root
        / (
            "query_self_aligned_"
            "in_query_camera.obj"
        )
    )

    for path in (
        reference_path,
        query_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    reference_points = sample_triangle_mesh(
        reference_path,
        sample_count=args.sample_count,
        seed=0,
    )

    query_points = sample_triangle_mesh(
        query_path,
        sample_count=args.sample_count,
        seed=1,
    )

    reference = robust_pca_statistics(
        reference_points
    )

    query = robust_pca_statistics(
        query_points
    )

    reference_extents = np.asarray(
        reference["robust_pca_extents_m"],
        dtype=np.float64,
    )

    query_extents = np.asarray(
        query["robust_pca_extents_m"],
        dtype=np.float64,
    )

    reference_normalized = np.asarray(
        reference["normalized_pca_extents"],
        dtype=np.float64,
    )

    query_normalized = np.asarray(
        query["normalized_pca_extents"],
        dtype=np.float64,
    )

    extent_ratio = (
        query_extents
        / np.maximum(reference_extents, 1e-12)
    )

    normalized_extent_ratio = (
        query_normalized
        / np.maximum(reference_normalized, 1e-12)
    )

    maximum_anisotropic_deviation = float(
        np.max(
            np.abs(
                normalized_extent_ratio - 1.0
            )
        )
    )

    if maximum_anisotropic_deviation <= 0.05:
        diagnosis = "low_anisotropic_difference"
    elif maximum_anisotropic_deviation <= 0.10:
        diagnosis = "moderate_anisotropic_difference"
    else:
        diagnosis = "high_anisotropic_difference"

    axis_order_warning = bool(
        min(
            reference[
                "eigenvalue_separation"
            ]
            + query[
                "eigenvalue_separation"
            ]
        )
        < 1.15
    )

    result = {
        "query_image_id": (
            args.query_image_id
        ),
        "reference_mesh_path": str(
            reference_path
        ),
        "query_mesh_path": str(
            query_path
        ),
        "reference": reference,
        "query": query,
        "query_reference_extent_ratio": (
            extent_ratio.tolist()
        ),
        "query_reference_normalized_extent_ratio": (
            normalized_extent_ratio.tolist()
        ),
        "maximum_anisotropic_deviation": (
            maximum_anisotropic_deviation
        ),
        "diagnosis": diagnosis,
        "axis_order_warning": (
            axis_order_warning
        ),
        "notes": {
            "low": (
                "모든 normalized axis ratio가 "
                "대략 ±5% 이내"
            ),
            "moderate": (
                "최대 normalized axis deviation이 "
                "약 5~10%"
            ),
            "high": (
                "최대 normalized axis deviation이 "
                "약 10% 초과"
            ),
            "axis_order_warning": (
                "PCA eigenvalue가 가까우면 축 순서가 "
                "불안정할 수 있음"
            ),
        },
    }

    diagnosis_root = (
        query_root
        / "mesh_registration"
        / "dgedi"
        / "axis_scale_diagnosis"
    )

    diagnosis_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        diagnosis_root
        / "axis_scale_diagnosis.json"
    )

    json_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    axis_labels = [
        "PCA-1",
        "PCA-2",
        "PCA-3",
    ]

    x_positions = np.arange(3)
    width = 0.35

    figure, axis = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    axis.bar(
        x_positions - width / 2.0,
        reference_extents,
        width,
        label="Reference",
    )

    axis.bar(
        x_positions + width / 2.0,
        query_extents,
        width,
        label="Query",
    )

    axis.set_xticks(
        x_positions,
        axis_labels,
    )

    axis.set_ylabel(
        "Robust PCA extent [m]"
    )

    axis.set_title(
        (
            f"Query {args.query_image_id}: "
            "principal-axis mesh extents"
        )
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    for index, ratio in enumerate(
        extent_ratio
    ):
        height = max(
            reference_extents[index],
            query_extents[index],
        )

        axis.text(
            index,
            height,
            f"Q/R={ratio:.3f}",
            ha="center",
            va="bottom",
        )

    image_path = (
        diagnosis_root
        / "axis_scale_diagnosis.png"
    )

    figure.savefig(
        image_path,
        dpi=200,
    )

    plt.close(figure)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nSaved:")
    print(json_path)
    print(image_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
