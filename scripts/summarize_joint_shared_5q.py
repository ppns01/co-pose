from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

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


def pose_error(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, float, list[float]]:
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
        np.degrees(np.arccos(cosine))
    )

    translation_xyz_cm = (
        estimate[:3, 3]
        - ground_truth[:3, 3]
    ) * 100.0

    translation_error_cm = float(
        np.linalg.norm(translation_xyz_cm)
    )

    return (
        rotation_error_deg,
        translation_error_cm,
        translation_xyz_cm.tolist(),
    )


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "population_std": float(
            statistics.pstdev(values)
        ),
    }


def save_bar_chart(
    *,
    query_ids: list[int],
    values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    axis.bar(
        [str(value) for value in query_ids],
        values,
    )

    axis.set_xlabel("Query image ID")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for index, value in enumerate(values):
        axis.text(
            index,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


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
        "--query-image-ids",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
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

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

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

    records: list[dict[str, Any]] = []

    for query_id in args.query_image_ids:
        query_label = (
            f"q{args.scene_id:06d}_"
            f"{query_id:06d}_i00"
        )

        query_root = (
            output_root
            / "queries"
            / query_label
        )

        final_pose_path = (
            query_root
            / "method_results"
            / "self_mesh"
            / "final_relative_pose.npy"
        )

        selection_path = (
            query_root
            / "visible_scale_refinement"
            / "joint_shared"
            / "selection.json"
        )

        if not final_pose_path.is_file():
            raise FileNotFoundError(
                final_pose_path
            )

        if not selection_path.is_file():
            raise FileNotFoundError(
                selection_path
            )

        estimate = np.load(
            final_pose_path,
            allow_pickle=False,
        ).astype(np.float64)

        query_gt = (
            load_bop_linemod_absolute_gt_pose(
                BOPFrameGTSpec(
                    dataset_root=dataset_root,
                    split="test",
                    scene_id=args.scene_id,
                    image_id=query_id,
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

        (
            rotation_error_deg,
            translation_error_cm,
            translation_xyz_cm,
        ) = pose_error(
            estimate,
            ground_truth,
        )

        selection = json.loads(
            selection_path.read_text(
                encoding="utf-8"
            )
        )

        selected_index = int(
            selection[
                "selected_candidate_index"
            ]
        )

        candidate_count = len(
            selection["scale_candidates_m"]
        )

        boundary_selected = (
            selected_index == 0
            or selected_index
            == candidate_count - 1
        )

        records.append(
            {
                "query_image_id": query_id,
                "rotation_error_deg": (
                    rotation_error_deg
                ),
                "translation_error_cm": (
                    translation_error_cm
                ),
                "translation_x_cm": (
                    translation_xyz_cm[0]
                ),
                "translation_y_cm": (
                    translation_xyz_cm[1]
                ),
                "translation_z_cm": (
                    translation_xyz_cm[2]
                ),
                "selected_scale_m": float(
                    selection[
                        "selected_scale_m"
                    ]
                ),
                "selected_candidate_index": (
                    selected_index
                ),
                "candidate_count": (
                    candidate_count
                ),
                "boundary_selected": (
                    boundary_selected
                ),
                "joint_worst_normalized_loss": float(
                    selection[
                        "selected_record"
                    ][
                        "joint_worst_normalized_loss"
                    ]
                ),
                "joint_mean_normalized_loss": float(
                    selection[
                        "selected_record"
                    ][
                        "joint_mean_normalized_loss"
                    ]
                ),
                "reference_visible_estimate_m": float(
                    selection[
                        "reference_visible_estimate_m"
                    ]
                ),
                "query_visible_estimate_m": float(
                    selection[
                        "query_visible_estimate_m"
                    ]
                ),
                "acc_5deg": (
                    rotation_error_deg <= 5.0
                ),
                "acc_10deg": (
                    rotation_error_deg <= 10.0
                ),
                "acc_15deg": (
                    rotation_error_deg <= 15.0
                ),
                "acc_30deg": (
                    rotation_error_deg <= 30.0
                ),
                "success_5deg_5cm": (
                    rotation_error_deg <= 5.0
                    and translation_error_cm <= 5.0
                ),
                "success_10deg_5cm": (
                    rotation_error_deg <= 10.0
                    and translation_error_cm <= 5.0
                ),
                "success_10deg_10cm": (
                    rotation_error_deg <= 10.0
                    and translation_error_cm <= 10.0
                ),
            }
        )

    result_root = (
        output_root
        / "joint_shared_5q_evaluation"
    )

    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        result_root / "per_query_results.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )

        writer.writeheader()
        writer.writerows(records)

    rotations = [
        float(row["rotation_error_deg"])
        for row in records
    ]

    translations = [
        float(row["translation_error_cm"])
        for row in records
    ]

    scales = [
        float(row["selected_scale_m"])
        for row in records
    ]

    count = len(records)

    aggregate = {
        "query_count": count,
        "rotation_error_deg": (
            summary(rotations)
        ),
        "translation_error_cm": (
            summary(translations)
        ),
        "selected_scale_m": (
            summary(scales)
        ),
        "success_rates": {
            "acc_5deg": sum(
                row["acc_5deg"]
                for row in records
            ) / count,
            "acc_10deg": sum(
                row["acc_10deg"]
                for row in records
            ) / count,
            "acc_15deg": sum(
                row["acc_15deg"]
                for row in records
            ) / count,
            "acc_30deg": sum(
                row["acc_30deg"]
                for row in records
            ) / count,
            "success_5deg_5cm": sum(
                row["success_5deg_5cm"]
                for row in records
            ) / count,
            "success_10deg_5cm": sum(
                row["success_10deg_5cm"]
                for row in records
            ) / count,
            "success_10deg_10cm": sum(
                row["success_10deg_10cm"]
                for row in records
            ) / count,
        },
        "boundary_selection": {
            "count": sum(
                row["boundary_selected"]
                for row in records
            ),
            "rate": sum(
                row["boundary_selected"]
                for row in records
            ) / count,
        },
        "records": records,
    }

    summary_path = (
        result_root / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            aggregate,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    query_ids = [
        int(row["query_image_id"])
        for row in records
    ]

    save_bar_chart(
        query_ids=query_ids,
        values=rotations,
        ylabel="Rotation error [deg]",
        title=(
            "Joint shared scale: "
            "rotation error by query"
        ),
        output_path=(
            result_root
            / "rotation_error_by_query.png"
        ),
    )

    save_bar_chart(
        query_ids=query_ids,
        values=translations,
        ylabel="Translation error [cm]",
        title=(
            "Joint shared scale: "
            "translation error by query"
        ),
        output_path=(
            result_root
            / "translation_error_by_query.png"
        ),
    )

    save_bar_chart(
        query_ids=query_ids,
        values=scales,
        ylabel="Selected S* [m]",
        title=(
            "Joint shared scale: "
            "selected S* by query"
        ),
        output_path=(
            result_root
            / "selected_scale_by_query.png"
        ),
    )

    print(
        "query | rotation | translation | "
        "S* | candidate | boundary"
    )

    for row in records:
        print(
            f"{row['query_image_id']:5d} | "
            f"{row['rotation_error_deg']:8.4f}° | "
            f"{row['translation_error_cm']:8.4f} cm | "
            f"{row['selected_scale_m']:.6f} m | "
            f"{row['selected_candidate_index']}/"
            f"{row['candidate_count'] - 1} | "
            f"{row['boundary_selected']}"
        )

    print("\nMean:")
    print(
        f"{aggregate['rotation_error_deg']['mean']:.6f}° / "
        f"{aggregate['translation_error_cm']['mean']:.6f} cm"
    )

    print("\nMedian:")
    print(
        f"{aggregate['rotation_error_deg']['median']:.6f}° / "
        f"{aggregate['translation_error_cm']['median']:.6f} cm"
    )

    print(
        "\nBoundary selections:",
        aggregate["boundary_selection"],
    )

    print("\nSaved:")
    print(csv_path)
    print(summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
