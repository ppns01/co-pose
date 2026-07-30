from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation.research_result_logger import (
    ResearchRunContext,
    _cuda_version_from_torch_version,
    save_pair_research_results,
)


TEST_DIRECTORY = Path(__file__).resolve().parent


def _pose(
    x: float,
    y: float,
    z: float,
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = (x, y, z)
    return pose


def _alignment(
    *,
    mask_iou: float,
    depth_loss: float,
    depth_residual_m: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        total_loss=0.2,
        mask_loss=1.0 - mask_iou,
        depth_loss=depth_loss,
        free_space_loss=0.1,
        boundary_loss=0.1,
        mask_iou=mask_iou,
        depth_residual_m=depth_residual_m,
        depth_residual_normalized=0.1,
        overlap_pixel_count=80,
        valid_depth_overlap_count=70,
        rendered_pixel_count=100,
        free_space_violation_count=5,
    )


class ResearchResultLoggerTests(unittest.TestCase):
    def test_extracts_cuda_version_from_torch_build(
        self,
    ) -> None:
        self.assertEqual(
            _cuda_version_from_torch_version(
                "2.11.0+cu128"
            ),
            "12.8",
        )

    def test_writes_long_format_csv_and_all_required_poses(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "datasets"
            scene_root = (
                dataset_root / "test" / "000008"
            )
            scene_root.mkdir(parents=True)
            scene_gt = {
                "0": [
                    {
                        "obj_id": 8,
                        "cam_R_m2c": [
                            1,
                            0,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            1,
                        ],
                        "cam_t_m2c": [0, 0, 0],
                    }
                ],
                "1": [
                    {
                        "obj_id": 8,
                        "cam_R_m2c": [
                            1,
                            0,
                            0,
                            0,
                            1,
                            0,
                            0,
                            0,
                            1,
                        ],
                        "cam_t_m2c": [10, 20, 30],
                    }
                ],
            }
            scene_gt["2"] = scene_gt["1"]
            (
                scene_root / "scene_gt.json"
            ).write_text(
                json.dumps(scene_gt),
                encoding="utf-8",
            )

            mesh_path = root / "proxy.obj"
            mesh_path.write_text(
                "\n".join(
                    (
                        "v 0 0 0",
                        "v 1 0 0",
                        "v 0 1 0",
                        "f 1 2 3",
                    )
                ),
                encoding="utf-8",
            )
            mask_path = root / "mask.npy"
            np.save(
                mask_path,
                np.ones((4, 5), dtype=np.bool_),
                allow_pickle=False,
            )
            config_path = root / "pipeline_config.json"
            config_path.write_text(
                "{}",
                encoding="utf-8",
            )
            research_root = root / "research"
            research_root.mkdir()
            context = ResearchRunContext(
                run_id="run_test",
                started_at="2026-07-27T00:00:00+09:00",
                output_directory=research_root,
                config_path=config_path,
                config_hash="abc123",
                git_commit="",
                instantmesh_commit="instant",
                foundationpose_commit="foundation",
                python_version="3.10",
                torch_version="2.11.0+cu128",
                cuda_version="12.8",
                gpu_name="test-gpu",
                random_seed=42,
            )

            reference_pose = _pose(
                0.01,
                0.02,
                0.03,
            )
            query_pose = _pose(
                0.02,
                0.02,
                0.03,
            )
            reference_candidate = SimpleNamespace(
                path_name="reference_proxy",
                scale_m=0.2,
                scaled_mesh_path=mesh_path,
                self_candidate_index=0,
                self_hypothesis_rank=0,
                self_foundationpose_score=0.9,
                self_alignment_loss=0.2,
                cross_candidate_index=0,
                cross_hypothesis_rank=0,
                cross_foundationpose_score=0.8,
                relative_pose_query_from_reference=(
                    reference_pose
                ),
            )
            query_candidate = SimpleNamespace(
                path_name="query_proxy",
                scale_m=0.3,
                scaled_mesh_path=mesh_path,
                self_candidate_index=0,
                self_hypothesis_rank=0,
                self_foundationpose_score=0.85,
                self_alignment_loss=0.2,
                cross_candidate_index=0,
                cross_hypothesis_rank=0,
                cross_foundationpose_score=0.75,
                relative_pose_query_from_reference=(
                    query_pose
                ),
            )
            consistency_pair = SimpleNamespace(
                reference_candidate_index=0,
                query_candidate_index=0,
                reference_candidate=(
                    reference_candidate
                ),
                query_candidate=query_candidate,
                rotation_difference_deg=0.0,
                translation_difference_m=0.01,
                consistency_loss=0.1,
                passes_rotation_gate=True,
                passes_translation_gate=True,
                passes_scale_gate=True,
                passes_hard_gate=True,
            )
            reference_score = SimpleNamespace(
                dino_available=False,
                dino_loss=None,
                path_loss=0.15,
            )
            query_score = SimpleNamespace(
                dino_available=False,
                dino_loss=None,
                path_loss=0.25,
            )
            pair_score = SimpleNamespace(
                consistency_pair=consistency_pair,
                reference_score=reference_score,
                query_score=query_score,
                normalized_consistency_loss=0.1,
            )
            final_result = SimpleNamespace(
                status="CONSISTENT",
                selected_path_name="reference_proxy",
                selected_relative_pose_query_from_reference=(
                    reference_pose
                ),
                final_loss=0.2,
                confidence=0.8,
                score_margin=0.1,
                best_pair_score=pair_score,
                evaluated_pair_scores=(pair_score,),
            )

            reference_cross = _alignment(
                mask_iou=0.8,
                depth_loss=0.2,
                depth_residual_m=0.01,
            )
            query_cross = _alignment(
                mask_iou=0.7,
                depth_loss=0.3,
                depth_residual_m=0.02,
            )
            cross_evidence = SimpleNamespace(
                reference_proxy=SimpleNamespace(
                    path_name="reference_proxy",
                    evidences=(
                        SimpleNamespace(
                            candidate_index=0,
                            cross_alignment=(
                                reference_cross
                            ),
                        ),
                    ),
                ),
                query_proxy=SimpleNamespace(
                    path_name="query_proxy",
                    evidences=(
                        SimpleNamespace(
                            candidate_index=0,
                            cross_alignment=(
                                query_cross
                            ),
                        ),
                    ),
                ),
            )

            source_alignment = _alignment(
                mask_iou=0.9,
                depth_loss=0.1,
                depth_residual_m=0.005,
            )
            self_alignment = SimpleNamespace(
                candidate_index=0,
                hypothesis_rank=0,
                scale_m=0.2,
                scaled_mesh_path=mesh_path,
                foundationpose_score=0.9,
            )
            self_evaluation = SimpleNamespace(
                evaluations=(
                    SimpleNamespace(
                        candidate_result=(
                            SimpleNamespace(
                                candidate_index=0
                            )
                        ),
                        hypothesis=SimpleNamespace(
                            rank=0
                        ),
                        alignment_score=(
                            source_alignment
                        ),
                    ),
                )
            )
            segmentation = SimpleNamespace(
                score=None,
                mask_bool=np.ones(
                    (4, 5),
                    dtype=np.bool_,
                ),
                mask_bool_path=mask_path,
            )
            prepared_view = SimpleNamespace(
                segmentation=segmentation
            )
            mesh_result = SimpleNamespace(
                generator_name="instantmesh",
                primary_output_path=mesh_path,
            )
            reference_frame = SimpleNamespace(
                scene_id=8,
                image_id=0,
                instance_index=0,
            )
            query_frame = SimpleNamespace(
                scene_id=8,
                image_id=1,
                instance_index=0,
            )
            pair_output_root = root / "pair"

            result = save_pair_research_results(
                context=context,
                pair_output_root=pair_output_root,
                dataset_root=dataset_root,
                split="test",
                object_id=8,
                object_name="driller",
                reference_frame=reference_frame,
                query_frame=query_frame,
                mask_type="sam3",
                segmentation_mode="text_prompt",
                segmentation_model="sam3",
                reference_prepared_view=prepared_view,
                query_prepared_view=prepared_view,
                reference_mesh_result=mesh_result,
                query_mesh_result=mesh_result,
                reference_self_evaluation=(
                    self_evaluation
                ),
                query_self_evaluation=(
                    self_evaluation
                ),
                reference_self_alignment=(
                    self_alignment
                ),
                query_self_alignment=(
                    self_alignment
                ),
                cross_evidence=cross_evidence,
                final_result=final_result,
                timings={
                    "total_time_sec": 12.5,
                },
            )

            with result.pair_results_path.open(
                encoding="utf-8",
                newline="",
            ) as file:
                pair_rows = list(
                    csv.DictReader(file)
                )
            with result.path_results_path.open(
                encoding="utf-8",
                newline="",
            ) as file:
                path_rows = list(
                    csv.DictReader(file)
                )
            with result.proxy_results_path.open(
                encoding="utf-8",
                newline="",
            ) as file:
                proxy_rows = list(
                    csv.DictReader(file)
                )

            self.assertEqual(len(pair_rows), 3)
            self.assertEqual(len(path_rows), 2)
            self.assertEqual(len(proxy_rows), 2)
            self.assertEqual(
                [row["method"] for row in pair_rows],
                [
                    "ref_only",
                    "query_only",
                    "dual_validated",
                ],
            )
            self.assertEqual(
                pair_rows[-1]["selected_path"],
                "reference_proxy",
            )
            self.assertEqual(
                pair_rows[-1]["segmentation_mode"],
                "text_prompt",
            )
            self.assertEqual(
                pair_rows[-1]["segmentation_model"],
                "sam3",
            )
            self.assertEqual(
                pair_rows[-1]["mask_type"],
                "sam3",
            )
            self.assertEqual(
                pair_rows[-1]["selector_correct"],
                "True",
            )
            self.assertAlmostEqual(
                float(
                    pair_rows[1][
                        "translation_error_x_cm"
                    ]
                ),
                1.0,
                places=6,
            )
            self.assertEqual(
                pair_rows[-1][
                    "success_1deg_1cm"
                ],
                "True",
            )
            self.assertEqual(
                pair_rows[-1]["add"],
                "",
            )

            for pose_path in (
                result.reference_pose_path,
                result.query_pose_path,
                result.final_pose_path,
                result.ground_truth_pose_path,
            ):
                self.assertIsNotNone(pose_path)
                array = np.load(
                    pose_path,
                    allow_pickle=False,
                )
                self.assertEqual(array.shape, (4, 4))
                self.assertEqual(
                    array.dtype,
                    np.float64,
                )

            second_query_frame = SimpleNamespace(
                scene_id=8,
                image_id=2,
                instance_index=0,
            )
            save_pair_research_results(
                context=context,
                pair_output_root=(
                    root / "pair_second"
                ),
                dataset_root=dataset_root,
                split="test",
                object_id=8,
                object_name="driller",
                reference_frame=reference_frame,
                query_frame=second_query_frame,
                mask_type="mask_visib",
                reference_prepared_view=prepared_view,
                query_prepared_view=prepared_view,
                reference_mesh_result=mesh_result,
                query_mesh_result=mesh_result,
                reference_self_evaluation=(
                    self_evaluation
                ),
                query_self_evaluation=(
                    self_evaluation
                ),
                reference_self_alignment=(
                    self_alignment
                ),
                query_self_alignment=(
                    self_alignment
                ),
                cross_evidence=cross_evidence,
                final_result=final_result,
            )

            with result.pair_results_path.open(
                encoding="utf-8",
                newline="",
            ) as file:
                appended_pair_rows = list(
                    csv.DictReader(file)
                )
            with result.path_results_path.open(
                encoding="utf-8",
                newline="",
            ) as file:
                appended_path_rows = list(
                    csv.DictReader(file)
                )
            with result.proxy_results_path.open(
                encoding="utf-8",
                newline="",
            ) as file:
                appended_proxy_rows = list(
                    csv.DictReader(file)
                )

            self.assertEqual(
                len(appended_pair_rows),
                6,
            )
            self.assertEqual(
                len(appended_path_rows),
                4,
            )
            self.assertEqual(
                len(appended_proxy_rows),
                4,
            )
            self.assertEqual(
                len(
                    {
                        row["pair_id"]
                        for row in appended_pair_rows
                    }
                ),
                2,
            )

            rejected_root = root / "research_reject"
            rejected_root.mkdir()
            rejected_context = ResearchRunContext(
                **{
                    **context.__dict__,
                    "run_id": "run_reject",
                    "output_directory": (
                        rejected_root
                    ),
                }
            )
            rejected_result = (
                save_pair_research_results(
                    context=rejected_context,
                    pair_output_root=(
                        root / "pair_reject"
                    ),
                    dataset_root=dataset_root,
                    split="test",
                    object_id=8,
                    object_name="driller",
                    reference_frame=reference_frame,
                    query_frame=query_frame,
                    mask_type="mask_visib",
                    reference_prepared_view=(
                        prepared_view
                    ),
                    query_prepared_view=(
                        prepared_view
                    ),
                    reference_mesh_result=(
                        mesh_result
                    ),
                    query_mesh_result=mesh_result,
                    reference_self_evaluation=(
                        self_evaluation
                    ),
                    query_self_evaluation=(
                        self_evaluation
                    ),
                    reference_self_alignment=(
                        self_alignment
                    ),
                    query_self_alignment=(
                        self_alignment
                    ),
                    cross_evidence=cross_evidence,
                    final_result=SimpleNamespace(
                        status="REJECT",
                        selected_path_name=None,
                        selected_relative_pose_query_from_reference=None,
                        final_loss=0.6,
                        confidence=0.0,
                        score_margin=None,
                        best_pair_score=None,
                        evaluated_pair_scores=(
                            pair_score,
                        ),
                    ),
                )
            )
            self.assertIsNone(
                rejected_result.final_pose_path
            )

            with (
                rejected_result
                .pair_results_path
                .open(
                    encoding="utf-8",
                    newline="",
                )
            ) as file:
                rejected_rows = list(
                    csv.DictReader(file)
                )

            self.assertEqual(
                rejected_rows[-1]["method"],
                "dual_validated_reject",
            )
            self.assertEqual(
                rejected_rows[-1]["rejected"],
                "True",
            )
            self.assertEqual(
                rejected_rows[-1][
                    "estimated_pose_path"
                ],
                "",
            )
            self.assertNotEqual(
                rejected_rows[-1][
                    "gt_relative_translation_cm"
                ],
                "",
            )
            self.assertEqual(
                rejected_rows[-1][
                    "rotation_error_deg"
                ],
                "",
            )
            self.assertEqual(
                rejected_rows[-1][
                    "confidence_raw"
                ],
                "0.0",
            )
            self.assertEqual(
                rejected_rows[-1][
                    "rejection_reason"
                ],
                (
                    "no_pair_passed_"
                    "bidirectional_hard_gate"
                ),
            )


if __name__ == "__main__":
    unittest.main()
