from __future__ import annotations

import argparse
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
            "Same mesh, two poses (estimated vs GT). "
            "Isolates pose error from cross-mesh "
            "reconstruction discrepancy."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, default=8)
    parser.add_argument("--reference-image-id", type=int, default=0)
    parser.add_argument("--query-image-id", type=int, default=4)
    parser.add_argument("--object-id", type=int, default=8)
    args = parser.parse_args()

    # estimate/ground_truth are both T_query_from_reference, so only the
    # reference mesh (which they're defined to act on) gives a meaningful
    # "same shape, two poses" comparison; the query mesh already lives in
    # the query camera frame and isn't a valid input to this transform.
    mesh_name = "reference"

    output_root = args.output_root.expanduser().resolve()
    dgedi_root = output_root / "mesh_registration" / "dgedi"

    mesh_path = (
        dgedi_root
        / "self_aligned_meshes"
        / (
            f"{mesh_name}_self_aligned_"
            f"in_{mesh_name}_camera.obj"
        )
    )

    estimate_path = (
        output_root
        / "method_results"
        / "self_mesh"
        / "final_relative_pose.npy"
    )

    for path in (mesh_path, estimate_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    points = sample_points(load_vertices(mesh_path), seed=0)

    estimate = np.load(estimate_path, allow_pickle=False).astype(np.float64)

    dataset_root = args.dataset_root.expanduser().resolve()

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

    points_estimated = transform_points(points, estimate)
    points_ground_truth = transform_points(points, ground_truth)

    rotation_error_deg, translation_error_cm = pose_error(estimate, ground_truth)
    mesh_diagonal_m = diagonal(points)

    projections = (
        (0, 1, "X", "Y"),
        (0, 2, "X", "Z"),
        (1, 2, "Y", "Z"),
    )

    combined = np.vstack((points_estimated, points_ground_truth))

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    for column, (first, second, first_label, second_label) in enumerate(
        projections
    ):
        axis = axes[column]

        axis.scatter(
            points_ground_truth[:, first],
            points_ground_truth[:, second],
            s=1.0,
            alpha=0.35,
            label=f"{mesh_name} mesh @ GT pose",
        )

        axis.scatter(
            points_estimated[:, first],
            points_estimated[:, second],
            s=1.0,
            alpha=0.35,
            label=f"{mesh_name} mesh @ estimated pose",
        )

        axis.set_title(f"{first_label}{second_label}")
        axis.set_xlabel(f"{first_label} [m]")
        axis.set_ylabel(f"{second_label} [m]")
        axis.grid(True, alpha=0.25)

        set_projection_limits(axis, combined, first, second)

    axes[0].legend(loc="best", markerscale=5)

    figure.suptitle(
        "Pose-only error (single mesh, two poses — no cross-mesh shape noise)\n"
        f"Rotation error: {rotation_error_deg:.4f} deg | "
        f"Translation error: {translation_error_cm:.4f} cm | "
        f"Mesh diagonal: {mesh_diagonal_m:.4f} m ({mesh_name})"
    )

    visualization_root = output_root / "visualizations"
    visualization_root.mkdir(parents=True, exist_ok=True)

    image_path = visualization_root / f"pose_only_error_{mesh_name}.png"

    figure.savefig(image_path, dpi=200)
    plt.close(figure)

    print(f"rotation_error_deg={rotation_error_deg:.4f}")
    print(f"translation_error_cm={translation_error_cm:.4f}")
    print(f"image_path={image_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
