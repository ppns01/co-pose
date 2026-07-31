from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.relative_pose_evaluator import (
    BOPFrameGTSpec,
    build_ground_truth_relative_pose,
    load_bop_linemod_absolute_gt_pose,
)
from scripts.visualize_dgedi_alignment import (
    diagonal,
    load_vertices,
    pose_error,
    sample_points,
    set_projection_limits,
    transform_points,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the estimated relative pose against the true BOP "
            "CAD model, transformed by each image's own GT absolute pose. "
            "No reconstructed (InstantMesh) mesh is used on either side, "
            "so any mismatch is pure pose error, not reconstruction noise."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, default=8)
    parser.add_argument("--reference-image-id", type=int, default=0)
    parser.add_argument("--query-image-id", type=int, default=4)
    parser.add_argument("--object-id", type=int, default=8)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/models.",
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    models_dir = (
        args.models_dir.expanduser().resolve()
        if args.models_dir is not None
        else dataset_root / "models"
    )

    model_path = models_dir / f"obj_{args.object_id:06d}.ply"

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    estimate_path = (
        output_root
        / "method_results"
        / "self_mesh"
        / "final_relative_pose.npy"
    )

    if not estimate_path.is_file():
        raise FileNotFoundError(estimate_path)

    # BOP models are stored in millimeters; the rest of this pipeline is
    # in meters (load_bop_linemod_absolute_gt_pose already converts the
    # translation component of scene_gt.json the same way).
    model_points_mm = load_vertices(model_path)
    model_points_m = sample_points(model_points_mm, seed=0) * 0.001

    estimate = np.load(estimate_path, allow_pickle=False).astype(np.float64)

    reference_gt = load_bop_linemod_absolute_gt_pose(
        BOPFrameGTSpec(
            dataset_root=dataset_root,
            split="test",
            scene_id=args.scene_id,
            image_id=args.reference_image_id,
            object_id=args.object_id,
            instance_index=0,
        )
    )

    query_gt = load_bop_linemod_absolute_gt_pose(
        BOPFrameGTSpec(
            dataset_root=dataset_root,
            split="test",
            scene_id=args.scene_id,
            image_id=args.query_image_id,
            object_id=args.object_id,
            instance_index=0,
        )
    )

    ground_truth = build_ground_truth_relative_pose(
        reference_absolute_pose=reference_gt,
        query_absolute_pose=query_gt,
    ).astype(np.float64)

    # True object surface as it actually sits in each camera's frame,
    # per the dataset's own GT absolute pose -- no reconstruction involved.
    reference_true = transform_points(model_points_m, reference_gt)
    query_true = transform_points(model_points_m, query_gt)

    reference_estimated = transform_points(reference_true, estimate)
    reference_ground_truth = transform_points(reference_true, ground_truth)

    rotation_error_deg, translation_error_cm = pose_error(estimate, ground_truth)

    model_diagonal_m = diagonal(model_points_m)

    projections = (
        (0, 1, "X", "Y"),
        (0, 2, "X", "Z"),
        (1, 2, "Y", "Z"),
    )

    combined = np.vstack((query_true, reference_estimated, reference_ground_truth))

    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    for column, (first, second, first_label, second_label) in enumerate(
        projections
    ):
        estimated_axis = axes[0, column]

        estimated_axis.scatter(
            query_true[:, first],
            query_true[:, second],
            s=1.0,
            alpha=0.35,
            label="True model @ query GT pose",
        )

        estimated_axis.scatter(
            reference_estimated[:, first],
            reference_estimated[:, second],
            s=1.0,
            alpha=0.35,
            label="True model @ reference GT pose, then our estimate",
        )

        estimated_axis.set_title(f"Estimated alignment: {first_label}{second_label}")
        estimated_axis.set_xlabel(f"{first_label} [m]")
        estimated_axis.set_ylabel(f"{second_label} [m]")
        estimated_axis.grid(True, alpha=0.25)

        set_projection_limits(estimated_axis, combined, first, second)

        gt_axis = axes[1, column]

        gt_axis.scatter(
            query_true[:, first],
            query_true[:, second],
            s=1.0,
            alpha=0.35,
            label="True model @ query GT pose",
        )

        gt_axis.scatter(
            reference_ground_truth[:, first],
            reference_ground_truth[:, second],
            s=1.0,
            alpha=0.35,
            label="True model @ reference GT pose, then GT relative pose",
        )

        gt_axis.set_title(f"Ground-truth alignment (sanity check): {first_label}{second_label}")
        gt_axis.set_xlabel(f"{first_label} [m]")
        gt_axis.set_ylabel(f"{second_label} [m]")
        gt_axis.grid(True, alpha=0.25)

        set_projection_limits(gt_axis, combined, first, second)

    axes[0, 0].legend(loc="best", markerscale=5, fontsize=8)
    axes[1, 0].legend(loc="best", markerscale=5, fontsize=8)

    figure.suptitle(
        "True BOP CAD model, our estimated pose vs GT "
        "(zero reconstruction noise on either side)\n"
        f"Rotation error: {rotation_error_deg:.4f} deg | "
        f"Translation error: {translation_error_cm:.4f} cm | "
        f"Model diagonal: {model_diagonal_m:.4f} m"
    )

    visualization_root = output_root / "visualizations"
    visualization_root.mkdir(parents=True, exist_ok=True)

    image_path = visualization_root / "gt_model_estimate_vs_gt.png"

    figure.savefig(image_path, dpi=200)
    plt.close(figure)

    metrics = {
        "rotation_error_deg": rotation_error_deg,
        "translation_error_cm": translation_error_cm,
        "model_diagonal_m": model_diagonal_m,
        "image_path": str(image_path),
    }

    metrics_path = visualization_root / "gt_model_alignment_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
