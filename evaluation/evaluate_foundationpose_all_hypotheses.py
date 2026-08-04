#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pose.alignment_scorer import (
    AlignmentScoreWeights,
    score_alignment,
)
from pose.mesh_renderer import (
    FoundationPoseMeshRenderer,
)


def rotation_delta_deg(
    first_rotation: np.ndarray,
    second_rotation: np.ndarray,
) -> float:
    relative = (
        first_rotation
        @ second_rotation.T
    )

    cosine = np.clip(
        (
            np.trace(relative)
            - 1.0
        )
        / 2.0,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def erode_mask(
    mask: np.ndarray,
    erosion_px: int,
) -> np.ndarray:
    if erosion_px <= 0:
        return np.ascontiguousarray(
            mask,
            dtype=np.bool_,
        )

    kernel_size = (
        2 * erosion_px + 1
    )

    eroded = cv2.erode(
        mask.astype(np.uint8),
        np.ones(
            (
                kernel_size,
                kernel_size,
            ),
            dtype=np.uint8,
        ),
        iterations=1,
    )

    return np.ascontiguousarray(
        eroded > 0,
        dtype=np.bool_,
    )


def compute_depth_statistics(
    *,
    observed_mask: np.ndarray,
    observed_depth_m: np.ndarray,
    rendered_mask: np.ndarray,
    rendered_depth_m: np.ndarray,
) -> dict[str, float | int | None]:
    valid = (
        observed_mask
        & rendered_mask
        & (observed_depth_m > 0.0)
        & (rendered_depth_m > 0.0)
    )

    valid_count = int(
        np.count_nonzero(valid)
    )

    if valid_count == 0:
        return {
            "depth_valid_pixel_count": 0,
            "signed_depth_median_m": None,
            "absolute_depth_median_m": None,
            "debiased_depth_median_m": None,
            "absolute_depth_p90_m": None,
        }

    residual = (
        observed_depth_m[valid].astype(
            np.float64,
            copy=False,
        )
        - rendered_depth_m[valid].astype(
            np.float64,
            copy=False,
        )
    )

    signed_median = float(
        np.median(residual)
    )

    absolute = np.abs(residual)

    return {
        "depth_valid_pixel_count":
            valid_count,
        "signed_depth_median_m":
            signed_median,
        "absolute_depth_median_m":
            float(
                np.median(absolute)
            ),
        "debiased_depth_median_m":
            float(
                np.median(
                    np.abs(
                        residual
                        - signed_median
                    )
                )
            ),
        "absolute_depth_p90_m":
            float(
                np.quantile(
                    absolute,
                    0.90,
                )
            ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--foundationpose-repository",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--render-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--evaluation-chunk-size",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--maximum-texture-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--mask-erosion-px",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save-top-count",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    snapshot_path = (
        args.snapshot
        .expanduser()
        .resolve()
    )

    metadata_path = (
        snapshot_path.with_suffix(
            ".json"
        )
    )

    if not snapshot_path.is_file():
        raise FileNotFoundError(
            snapshot_path
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            metadata_path
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    with np.load(
        snapshot_path,
        allow_pickle=False,
    ) as snapshot:
        poses = np.asarray(
            snapshot[
                "poses_external"
            ],
            dtype=np.float32,
        )

        scores = np.asarray(
            snapshot[
                "foundationpose_scores"
            ],
            dtype=np.float64,
        ).reshape(-1)

        valid_pose_mask = np.asarray(
            snapshot[
                "valid_pose_mask"
            ],
            dtype=np.bool_,
        )

        camera_matrix = np.asarray(
            snapshot[
                "camera_matrix"
            ],
            dtype=np.float32,
        )

        observed_depth_m = np.asarray(
            snapshot[
                "observed_depth_m"
            ],
            dtype=np.float32,
        )

        observed_mask = np.asarray(
            snapshot[
                "observed_mask"
            ],
            dtype=np.bool_,
        )

    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
    ):
        raise ValueError(
            f"Invalid pose shape: {poses.shape}"
        )

    if (
        scores.shape[0] != poses.shape[0]
        or valid_pose_mask.shape[0]
        != poses.shape[0]
    ):
        raise ValueError(
            "Pose/score/valid mask count mismatch"
        )

    selected_rank = int(
        metadata.get(
            "selected_sorted_rank",
            0,
        )
    )

    if not valid_pose_mask[
        selected_rank
    ]:
        raise RuntimeError(
            "Selected rank is invalid"
        )

    selected_pose = poses[
        selected_rank
    ].astype(
        np.float64,
        copy=False,
    )

    evaluation_mask = erode_mask(
        observed_mask,
        args.mask_erosion_px,
    )

    valid_indices = np.flatnonzero(
        valid_pose_mask
    )

    output_directory = (
        args.output_directory
        if args.output_directory
        is not None
        else snapshot_path.parent
        / "all_hypothesis_evaluation"
    )

    output_directory = (
        output_directory
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh_path = Path(
        metadata["scaled_mesh_path"]
    ).expanduser().resolve()

    object_scale_m = float(
        metadata["scale_m"]
    )

    rows: list[dict[str, Any]] = []

    with FoundationPoseMeshRenderer(
        foundationpose_repository_path=(
            args.foundationpose_repository
        ),
        device=args.device,
        render_batch_size=(
            args.render_batch_size
        ),
        maximum_texture_size=(
            args.maximum_texture_size
        ),
    ) as renderer:
        for start in range(
            0,
            len(valid_indices),
            args.evaluation_chunk_size,
        ):
            chunk_indices = valid_indices[
                start:
                start
                + args.evaluation_chunk_size
            ]

            rendered = renderer.render(
                mesh_path=mesh_path,
                poses_camera_from_proxy=(
                    poses[chunk_indices]
                ),
                camera_matrix=camera_matrix,
                image_height=(
                    observed_mask.shape[0]
                ),
                image_width=(
                    observed_mask.shape[1]
                ),
                output_directory=None,
            )

            for local_index, rank in enumerate(
                chunk_indices
            ):
                alignment = score_alignment(
                    observed_mask=(
                        evaluation_mask
                    ),
                    observed_depth_m=(
                        observed_depth_m
                    ),
                    rendered_mask=(
                        rendered.rendered_masks[
                            local_index
                        ]
                    ),
                    rendered_depth_m=(
                        rendered.rendered_depth_m[
                            local_index
                        ]
                    ),
                    object_scale_m=(
                        object_scale_m
                    ),
                    weights=(
                        AlignmentScoreWeights()
                    ),
                )

                depth_statistics = (
                    compute_depth_statistics(
                        observed_mask=(
                            evaluation_mask
                        ),
                        observed_depth_m=(
                            observed_depth_m
                        ),
                        rendered_mask=(
                            rendered
                            .rendered_masks[
                                local_index
                            ]
                        ),
                        rendered_depth_m=(
                            rendered
                            .rendered_depth_m[
                                local_index
                            ]
                        ),
                    )
                )

                pose = poses[
                    rank
                ].astype(
                    np.float64,
                    copy=False,
                )

                rows.append(
                    {
                        "sorted_rank": int(rank),
                        "foundationpose_score":
                            float(scores[rank]),
                        "is_selected_rank":
                            bool(
                                rank
                                == selected_rank
                            ),
                        "mask_iou":
                            alignment.mask_iou,
                        "alignment_total_loss":
                            alignment.total_loss,
                        "alignment_depth_residual_m":
                            alignment
                            .depth_residual_m,
                        "alignment_depth_residual_normalized":
                            alignment
                            .depth_residual_normalized,
                        "free_space_loss":
                            alignment
                            .free_space_loss,
                        "boundary_loss":
                            alignment
                            .boundary_loss,
                        "rotation_delta_from_selected_deg":
                            rotation_delta_deg(
                                pose[:3, :3],
                                selected_pose[
                                    :3,
                                    :3,
                                ],
                            ),
                        "translation_delta_from_selected_m":
                            float(
                                np.linalg.norm(
                                    pose[:3, 3]
                                    - selected_pose[
                                        :3,
                                        3,
                                    ]
                                )
                            ),
                        **depth_statistics,
                    }
                )

        rows_by_rank = sorted(
            rows,
            key=lambda item: int(
                item["sorted_rank"]
            ),
        )

        finite_depth_rows = [
            item
            for item in rows
            if item[
                "debiased_depth_median_m"
            ] is not None
        ]

        if not finite_depth_rows:
            raise RuntimeError(
                "No hypothesis has valid depth overlap"
            )

        best_by_depth = min(
            finite_depth_rows,
            key=lambda item: float(
                item[
                    "debiased_depth_median_m"
                ]
            ),
        )

        best_by_alignment = min(
            rows,
            key=lambda item: float(
                item[
                    "alignment_total_loss"
                ]
            ),
        )

        depth_top = sorted(
            finite_depth_rows,
            key=lambda item: float(
                item[
                    "debiased_depth_median_m"
                ]
            ),
        )[
            : args.save_top_count
        ]

        score_top = rows_by_rank[
            : args.save_top_count
        ]

        render_ranks = np.asarray(
            sorted(
                {
                    selected_rank,
                    *(
                        int(item["sorted_rank"])
                        for item in depth_top
                    ),
                    *(
                        int(item["sorted_rank"])
                        for item in score_top
                    ),
                }
            ),
            dtype=np.int64,
        )

        render_directory = (
            output_directory
            / "selected_renders"
        )

        renderer.render(
            mesh_path=mesh_path,
            poses_camera_from_proxy=(
                poses[render_ranks]
            ),
            camera_matrix=camera_matrix,
            image_height=(
                observed_mask.shape[0]
            ),
            image_width=(
                observed_mask.shape[1]
            ),
            output_directory=(
                render_directory
            ),
            filename_prefix=(
                "evaluated_hypothesis"
            ),
        )

    csv_path = (
        output_directory
        / "all_hypothesis_metrics.csv"
    )

    with csv_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows_by_rank[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows_by_rank)

    selected_row = next(
        item
        for item in rows_by_rank
        if int(item["sorted_rank"])
        == selected_rank
    )

    summary = {
        "view_name": metadata["view_name"],
        "candidate_index": (
            metadata["candidate_index"]
        ),
        "scaled_mesh_path": str(
            mesh_path
        ),
        "candidate_count": len(rows),
        "selected_rank": selected_rank,
        "selected": selected_row,
        "best_by_debiased_depth": (
            best_by_depth
        ),
        "best_by_alignment_total_loss": (
            best_by_alignment
        ),
        "render_array_index_to_sorted_rank": {
            str(array_index): int(rank)
            for array_index, rank
            in enumerate(
                render_ranks.tolist()
            )
        },
        "parameters": {
            "mask_erosion_px":
                args.mask_erosion_px,
            "render_batch_size":
                args.render_batch_size,
            "evaluation_chunk_size":
                args.evaluation_chunk_size,
        },
    }

    summary_path = (
        output_directory
        / "all_hypothesis_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"[metrics] {csv_path}"
    )

    print(
        f"[summary] {summary_path}"
    )

    print(
        "selected: "
        f"rank={selected_rank}, "
        "depth="
        f"{selected_row['debiased_depth_median_m']}, "
        "iou="
        f"{selected_row['mask_iou']}"
    )

    print(
        "best depth: "
        f"rank={best_by_depth['sorted_rank']}, "
        "depth="
        f"{best_by_depth['debiased_depth_median_m']}, "
        "iou="
        f"{best_by_depth['mask_iou']}, "
        "rotation_delta="
        f"{best_by_depth['rotation_delta_from_selected_deg']} deg"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
