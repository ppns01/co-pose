from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.relative_pose_evaluator import (
    BOPFrameGTSpec,
    build_ground_truth_relative_pose,
    load_bop_linemod_absolute_gt_pose,
)


def _load_pose(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    pose = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=np.float64,
    )

    if pose.shape != (4, 4):
        raise ValueError(
            f"Invalid pose shape: {path}, {pose.shape}"
        )

    if not np.all(np.isfinite(pose)):
        raise ValueError(
            f"Pose contains non-finite values: {path}"
        )

    return pose


def _find_pose_path(
    *,
    output_root: Path,
    scene_id: int,
    image_id: int,
    instance_index: int,
) -> Path:
    summary_path = output_root / "batch_summary.json"

    if summary_path.is_file():
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            summary = json.load(file)

        for record in summary.get("queries", []):
            if (
                int(record.get("scene_id", -1))
                == scene_id
                and int(record.get("image_id", -1))
                == image_id
                and int(
                    record.get(
                        "instance_index",
                        -1,
                    )
                )
                == instance_index
            ):
                if record.get("status") != "completed":
                    raise RuntimeError(
                        f"Query failed: image={image_id}, "
                        f"record={record}"
                    )

                pose_value = record.get("pose_path")

                if not pose_value:
                    raise RuntimeError(
                        f"pose_path missing: image={image_id}"
                    )

                pose_path = Path(pose_value)

                if not pose_path.is_absolute():
                    pose_path = output_root / pose_path

                if pose_path.is_file():
                    return pose_path.resolve()

    query_label = (
        f"q{scene_id:06d}_"
        f"{image_id:06d}_"
        f"i{instance_index:02d}"
    )

    expected = (
        output_root
        / "queries"
        / query_label
        / "method_results"
        / "self_mesh"
        / "final_relative_pose.npy"
    )

    if expected.is_file():
        return expected.resolve()

    candidates = [
        path
        for path in output_root.rglob(
            "final_relative_pose.npy"
        )
        if query_label in str(path)
    ]

    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected exactly one final pose: "
            f"root={output_root}, "
            f"query={query_label}, "
            f"found={len(candidates)}"
        )

    return candidates[0].resolve()


def _rotation_error_deg(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    difference = (
        predicted[:3, :3]
        @ ground_truth[:3, :3].T
    )

    cosine = float(
        np.clip(
            (np.trace(difference) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(np.arccos(cosine))
    )


def _translation_error_cm(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, np.ndarray]:
    difference_cm = (
        predicted[:3, 3]
        - ground_truth[:3, 3]
    ) * 100.0

    return (
        float(np.linalg.norm(difference_cm)),
        difference_cm,
    )


def _method_summary(
    rows: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    rotation = np.asarray(
        [
            row[f"{prefix}_rotation_error_deg"]
            for row in rows
        ],
        dtype=np.float64,
    )

    translation = np.asarray(
        [
            row[f"{prefix}_translation_error_cm"]
            for row in rows
        ],
        dtype=np.float64,
    )

    return {
        "count": int(len(rows)),
        "rotation_error_deg": {
            "mean": float(rotation.mean()),
            "median": float(np.median(rotation)),
            "std": float(rotation.std(ddof=0)),
            "minimum": float(rotation.min()),
            "maximum": float(rotation.max()),
        },
        "translation_error_cm": {
            "mean": float(translation.mean()),
            "median": float(np.median(translation)),
            "std": float(translation.std(ddof=0)),
            "minimum": float(translation.min()),
            "maximum": float(translation.max()),
        },
        "success_count_5deg_5cm": int(
            np.sum(
                (rotation <= 5.0)
                & (translation <= 5.0)
            )
        ),
        "success_count_2deg_2cm": int(
            np.sum(
                (rotation <= 2.0)
                & (translation <= 2.0)
            )
        ),
        "success_count_1deg_1cm": int(
            np.sum(
                (rotation <= 1.0)
                & (translation <= 1.0)
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bufferx-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dgedi-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query-ids",
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

    parser.add_argument(
        "--instance-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    args = parser.parse_args()

    bufferx_root = (
        args.bufferx_root.expanduser().resolve()
    )

    dgedi_root = (
        args.dgedi_root.expanduser().resolve()
    )

    dataset_root = (
        args.dataset_root.expanduser().resolve()
    )

    output_directory = (
        args.output_directory
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_spec = BOPFrameGTSpec(
        dataset_root=dataset_root,
        split=args.split,
        scene_id=args.scene_id,
        image_id=args.reference_image_id,
        object_id=args.object_id,
        instance_index=args.instance_index,
    )

    reference_absolute = (
        load_bop_linemod_absolute_gt_pose(
            reference_spec
        )
    )

    rows: list[dict[str, Any]] = []

    for query_id in args.query_ids:
        query_spec = BOPFrameGTSpec(
            dataset_root=dataset_root,
            split=args.split,
            scene_id=args.scene_id,
            image_id=query_id,
            object_id=args.object_id,
            instance_index=args.instance_index,
        )

        query_absolute = (
            load_bop_linemod_absolute_gt_pose(
                query_spec
            )
        )

        ground_truth = (
            build_ground_truth_relative_pose(
                reference_absolute_pose=(
                    reference_absolute
                ),
                query_absolute_pose=query_absolute,
            )
        ).astype(np.float64)

        bufferx_path = _find_pose_path(
            output_root=bufferx_root,
            scene_id=args.scene_id,
            image_id=query_id,
            instance_index=args.instance_index,
        )

        dgedi_path = _find_pose_path(
            output_root=dgedi_root,
            scene_id=args.scene_id,
            image_id=query_id,
            instance_index=args.instance_index,
        )

        bufferx_pose = _load_pose(bufferx_path)
        dgedi_pose = _load_pose(dgedi_path)

        bufferx_rotation = _rotation_error_deg(
            bufferx_pose,
            ground_truth,
        )

        dgedi_rotation = _rotation_error_deg(
            dgedi_pose,
            ground_truth,
        )

        (
            bufferx_translation,
            bufferx_xyz,
        ) = _translation_error_cm(
            bufferx_pose,
            ground_truth,
        )

        (
            dgedi_translation,
            dgedi_xyz,
        ) = _translation_error_cm(
            dgedi_pose,
            ground_truth,
        )

        row = {
            "query_image_id": query_id,
            "bufferx_rotation_error_deg": (
                bufferx_rotation
            ),
            "bufferx_translation_error_cm": (
                bufferx_translation
            ),
            "bufferx_translation_x_cm": float(
                bufferx_xyz[0]
            ),
            "bufferx_translation_y_cm": float(
                bufferx_xyz[1]
            ),
            "bufferx_translation_z_cm": float(
                bufferx_xyz[2]
            ),
            "dgedi_rotation_error_deg": (
                dgedi_rotation
            ),
            "dgedi_translation_error_cm": (
                dgedi_translation
            ),
            "dgedi_translation_x_cm": float(
                dgedi_xyz[0]
            ),
            "dgedi_translation_y_cm": float(
                dgedi_xyz[1]
            ),
            "dgedi_translation_z_cm": float(
                dgedi_xyz[2]
            ),
            "rotation_delta_dgedi_minus_bufferx": (
                dgedi_rotation
                - bufferx_rotation
            ),
            (
                "translation_delta_cm_"
                "dgedi_minus_bufferx"
            ): (
                dgedi_translation
                - bufferx_translation
            ),
            "rotation_winner": (
                "bufferx"
                if bufferx_rotation
                < dgedi_rotation
                else "dgedi"
                if dgedi_rotation
                < bufferx_rotation
                else "tie"
            ),
            "translation_winner": (
                "bufferx"
                if bufferx_translation
                < dgedi_translation
                else "dgedi"
                if dgedi_translation
                < bufferx_translation
                else "tie"
            ),
            "bufferx_pose_path": str(
                bufferx_path
            ),
            "dgedi_pose_path": str(
                dgedi_path
            ),
            "bufferx_pose": (
                bufferx_pose.tolist()
            ),
            "dgedi_pose": dgedi_pose.tolist(),
            "ground_truth_pose": (
                ground_truth.tolist()
            ),
        }

        rows.append(row)

    csv_path = (
        output_directory / "comparison.csv"
    )

    csv_columns = [
        key
        for key in rows[0]
        if key not in {
            "bufferx_pose",
            "dgedi_pose",
            "ground_truth_pose",
        }
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=csv_columns,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row[key]
                    for key in csv_columns
                }
            )

    bufferx_summary = _method_summary(
        rows,
        "bufferx",
    )

    dgedi_summary = _method_summary(
        rows,
        "dgedi",
    )

    summary = {
        "experiment": {
            "scene_id": args.scene_id,
            "object_id": args.object_id,
            "reference_image_id": (
                args.reference_image_id
            ),
            "query_image_ids": args.query_ids,
            "bufferx_root": str(bufferx_root),
            "dgedi_root": str(dgedi_root),
        },
        "bufferx": bufferx_summary,
        "dgedi": dgedi_summary,
        "pairwise_wins": {
            "rotation": {
                "bufferx": sum(
                    row["rotation_winner"]
                    == "bufferx"
                    for row in rows
                ),
                "dgedi": sum(
                    row["rotation_winner"]
                    == "dgedi"
                    for row in rows
                ),
            },
            "translation": {
                "bufferx": sum(
                    row["translation_winner"]
                    == "bufferx"
                    for row in rows
                ),
                "dgedi": sum(
                    row["translation_winner"]
                    == "dgedi"
                    for row in rows
                ),
            },
            "both_metrics": {
                "bufferx": sum(
                    row["rotation_winner"]
                    == "bufferx"
                    and row["translation_winner"]
                    == "bufferx"
                    for row in rows
                ),
                "dgedi": sum(
                    row["rotation_winner"]
                    == "dgedi"
                    and row["translation_winner"]
                    == "dgedi"
                    for row in rows
                ),
            },
        },
        "rows": rows,
    }

    summary_path = (
        output_directory / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nPer-query results")
    print(
        "query | BUFFER-X R / t | "
        "dGeDi R / t"
    )

    for row in rows:
        print(
            f"{row['query_image_id']:5d} | "
            f"{row['bufferx_rotation_error_deg']:8.4f}° / "
            f"{row['bufferx_translation_error_cm']:8.4f} cm | "
            f"{row['dgedi_rotation_error_deg']:8.4f}° / "
            f"{row['dgedi_translation_error_cm']:8.4f} cm"
        )

    print("\nMean results")

    print(
        "BUFFER-X:",
        f"{bufferx_summary['rotation_error_deg']['mean']:.6f}° /",
        f"{bufferx_summary['translation_error_cm']['mean']:.6f} cm",
    )

    print(
        "dGeDi:",
        f"{dgedi_summary['rotation_error_deg']['mean']:.6f}° /",
        f"{dgedi_summary['translation_error_cm']['mean']:.6f} cm",
    )

    print("\nSaved:")
    print(csv_path)
    print(summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
