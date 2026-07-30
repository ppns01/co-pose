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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from evaluation.relative_pose_evaluator import (
    BOPFrameGTSpec,
    build_ground_truth_relative_pose,
    load_bop_linemod_absolute_gt_pose,
)


def load_vertices(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(
            f"Invalid mesh vertices: {path}"
        )

    if len(vertices) == 0:
        raise RuntimeError(
            f"Mesh contains no vertices: {path}"
        )

    return vertices


def sample_points(
    points: np.ndarray,
    *,
    maximum_count: int = 15000,
    seed: int = 0,
) -> np.ndarray:
    if len(points) <= maximum_count:
        return points

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(points),
        size=maximum_count,
        replace=False,
    )

    return points[indices]


def transform_points(
    points: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    return (
        points @ pose[:3, :3].T
        + pose[:3, 3]
    )


def pose_error(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, float]:
    rotation_difference = (
        ground_truth[:3, :3].T
        @ estimate[:3, :3]
    )

    cosine = float(
        np.clip(
            (
                np.trace(rotation_difference)
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )
    )

    rotation_error_deg = float(
        np.degrees(
            np.arccos(cosine)
        )
    )

    translation_error_cm = float(
        np.linalg.norm(
            estimate[:3, 3]
            - ground_truth[:3, 3]
        )
        * 100.0
    )

    return (
        rotation_error_deg,
        translation_error_cm,
    )


def diagonal(points: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            points.max(axis=0)
            - points.min(axis=0)
        )
    )


def set_projection_limits(
    axis: plt.Axes,
    points: np.ndarray,
    first_dimension: int,
    second_dimension: int,
) -> None:
    values = points[
        :,
        [first_dimension, second_dimension],
    ]

    low = np.quantile(
        values,
        0.01,
        axis=0,
    )

    high = np.quantile(
        values,
        0.99,
        axis=0,
    )

    center = (low + high) / 2.0
    span = float(np.max(high - low))

    if span <= 1e-9:
        span = 1.0

    half = span * 0.58

    axis.set_xlim(
        center[0] - half,
        center[0] + half,
    )

    axis.set_ylim(
        center[1] - half,
        center[1] + half,
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--scene-id",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--reference-image-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--query-image-id",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--object-id",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    dgedi_root = (
        output_root
        / "mesh_registration"
        / "dgedi"
    )

    reference_mesh_path = (
        dgedi_root
        / "self_aligned_meshes"
        / (
            "reference_self_aligned_"
            "in_reference_camera.obj"
        )
    )

    query_mesh_path = (
        dgedi_root
        / "self_aligned_meshes"
        / (
            "query_self_aligned_"
            "in_query_camera.obj"
        )
    )

    estimate_path = (
        output_root
        / "method_results"
        / "self_mesh"
        / "final_relative_pose.npy"
    )

    for path in (
        reference_mesh_path,
        query_mesh_path,
        estimate_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    reference_points = sample_points(
        load_vertices(reference_mesh_path),
        seed=0,
    )

    query_points = sample_points(
        load_vertices(query_mesh_path),
        seed=1,
    )

    estimate = np.load(
        estimate_path,
        allow_pickle=False,
    ).astype(np.float64)

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    reference_gt = (
        load_bop_linemod_absolute_gt_pose(
            BOPFrameGTSpec(
                dataset_root=dataset_root,
                split="test",
                scene_id=args.scene_id,
                image_id=args.reference_image_id,
                object_id=args.object_id,
                instance_index=0,
            )
        )
    )

    query_gt = (
        load_bop_linemod_absolute_gt_pose(
            BOPFrameGTSpec(
                dataset_root=dataset_root,
                split="test",
                scene_id=args.scene_id,
                image_id=args.query_image_id,
                object_id=args.object_id,
                instance_index=0,
            )
        )
    )

    ground_truth = (
        build_ground_truth_relative_pose(
            reference_absolute_pose=reference_gt,
            query_absolute_pose=query_gt,
        )
    ).astype(np.float64)

    reference_estimated = transform_points(
        reference_points,
        estimate,
    )

    reference_ground_truth = transform_points(
        reference_points,
        ground_truth,
    )

    rotation_error_deg, translation_error_cm = (
        pose_error(
            estimate,
            ground_truth,
        )
    )

    reference_diagonal_m = diagonal(
        reference_points
    )

    query_diagonal_m = diagonal(
        query_points
    )

    scale_ratio = (
        query_diagonal_m
        / reference_diagonal_m
    )

    projections = (
        (0, 1, "X", "Y"),
        (0, 2, "X", "Z"),
        (1, 2, "Y", "Z"),
    )

    combined = np.vstack(
        (
            query_points,
            reference_estimated,
            reference_ground_truth,
        )
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10),
        constrained_layout=True,
    )

    for column, (
        first,
        second,
        first_label,
        second_label,
    ) in enumerate(projections):
        estimated_axis = axes[0, column]

        estimated_axis.scatter(
            query_points[:, first],
            query_points[:, second],
            s=1.0,
            alpha=0.35,
            label="Query mesh",
        )

        estimated_axis.scatter(
            reference_estimated[:, first],
            reference_estimated[:, second],
            s=1.0,
            alpha=0.35,
            label="Reference transformed by estimate",
        )

        estimated_axis.set_title(
            f"Estimated alignment: "
            f"{first_label}{second_label}"
        )

        estimated_axis.set_xlabel(
            f"{first_label} [m]"
        )

        estimated_axis.set_ylabel(
            f"{second_label} [m]"
        )

        estimated_axis.grid(
            True,
            alpha=0.25,
        )

        set_projection_limits(
            estimated_axis,
            combined,
            first,
            second,
        )

        ground_truth_axis = axes[1, column]

        ground_truth_axis.scatter(
            query_points[:, first],
            query_points[:, second],
            s=1.0,
            alpha=0.35,
            label="Query mesh",
        )

        ground_truth_axis.scatter(
            reference_ground_truth[:, first],
            reference_ground_truth[:, second],
            s=1.0,
            alpha=0.35,
            label="Reference transformed by GT",
        )

        ground_truth_axis.set_title(
            f"Ground-truth alignment: "
            f"{first_label}{second_label}"
        )

        ground_truth_axis.set_xlabel(
            f"{first_label} [m]"
        )

        ground_truth_axis.set_ylabel(
            f"{second_label} [m]"
        )

        ground_truth_axis.grid(
            True,
            alpha=0.25,
        )

        set_projection_limits(
            ground_truth_axis,
            combined,
            first,
            second,
        )

    axes[0, 0].legend(
        loc="best",
        markerscale=5,
    )

    axes[1, 0].legend(
        loc="best",
        markerscale=5,
    )

    figure.suptitle(
        (
            "dGeDi self-mesh registration\n"
            f"Rotation error: "
            f"{rotation_error_deg:.4f} deg | "
            f"Translation error: "
            f"{translation_error_cm:.4f} cm | "
            f"Query/Reference mesh diagonal: "
            f"{scale_ratio:.4f}"
        )
    )

    visualization_root = (
        output_root
        / "visualizations"
    )

    visualization_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path = (
        visualization_root
        / (
            "dgedi_registration_"
            "estimate_vs_gt.png"
        )
    )

    figure.savefig(
        image_path,
        dpi=200,
    )

    plt.close(figure)

    metrics = {
        "rotation_error_deg": (
            rotation_error_deg
        ),
        "translation_error_cm": (
            translation_error_cm
        ),
        "reference_mesh_diagonal_m": (
            reference_diagonal_m
        ),
        "query_mesh_diagonal_m": (
            query_diagonal_m
        ),
        "query_reference_diagonal_ratio": (
            scale_ratio
        ),
        "image_path": str(image_path),
    }

    metrics_path = (
        visualization_root
        / "visualization_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
