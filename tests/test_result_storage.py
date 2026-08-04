from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from result_storage import (
    remove_pipeline_intermediate_directory,
    save_compact_pair_result,
)


class ResultStorageTests(unittest.TestCase):
    def test_saves_pose_and_numeric_gt_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "datasets"
            scene_root = dataset_root / "test" / "000009"
            scene_root.mkdir(parents=True)
            scene_gt = {
                "0": [
                    {
                        "obj_id": 9,
                        "cam_R_m2c": np.eye(3).reshape(-1).tolist(),
                        "cam_t_m2c": [0.0, 0.0, 1000.0],
                    }
                ],
                "1": [
                    {
                        "obj_id": 9,
                        "cam_R_m2c": np.eye(3).reshape(-1).tolist(),
                        "cam_t_m2c": [100.0, 0.0, 1000.0],
                    }
                ],
            }
            (scene_root / "scene_gt.json").write_text(
                json.dumps(scene_gt),
                encoding="utf-8",
            )
            model_root = dataset_root / "models_eval"
            model_root.mkdir()
            (model_root / "models_info.json").write_text(
                json.dumps({"9": {"diameter": 100.0}}),
                encoding="utf-8",
            )
            (model_root / "obj_000009.ply").write_text(
                "\n".join(
                    (
                        "ply",
                        "format ascii 1.0",
                        "element vertex 3",
                        "property float x",
                        "property float y",
                        "property float z",
                        "element face 1",
                        "property list uchar int vertex_indices",
                        "end_header",
                        "0 0 0",
                        "10 0 0",
                        "0 10 0",
                        "3 0 1 2",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            work_root = root / "work"
            work_root.mkdir()
            pose = np.eye(4, dtype=np.float64)
            pose[0, 3] = 0.1
            pose_path = work_root / "pose.npy"
            np.save(pose_path, pose, allow_pickle=False)
            summary_path = work_root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "status": "COMPLETED",
                        "method": "self_mesh",
                        "sources": {
                            "large_artifact": "/deleted/mesh.obj",
                            "axis_scale_refinement": {
                                "method": "independent_local_axis_scale",
                                "shared_axes": False,
                                "views": {
                                    "reference": {
                                        "accepted": False,
                                        "requested_scale_factors_xyz": [
                                            1.0,
                                            0.9,
                                            1.1,
                                        ],
                                        "applied_scale_factors_xyz": [
                                            1.0,
                                            1.0,
                                            1.0,
                                        ],
                                        "scale_factors_xyz": [1.0, 1.0, 1.0],
                                    }
                                },
                            },
                            "selected_candidate": "H1_confidence_shared_D",
                            "selected_dgedi_stage": "g1",
                            "final_selection_confidence": 0.72,
                            "h0_score": {"total_score": 0.21},
                            "h1_score": {"total_score": 0.18},
                            "confidence_shared_dimensions": {
                                "g0_confidence": 0.64,
                                "raw_shared_dimensions_m": [0.1, 0.2, 0.3],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            paths = save_compact_pair_result(
                output_root=root / "output",
                dataset_root=dataset_root,
                split="test",
                object_id=9,
                object_name="duck",
                reference_scene_id=9,
                reference_image_id=0,
                reference_instance_index=0,
                query_scene_id=9,
                query_image_id=1,
                query_instance_index=0,
                pipeline_status="completed",
                final_status="CONSISTENT",
                pose_accepted=True,
                reference_mask_source="sam3",
                query_mask_source="gt_fallback",
                summary_path=summary_path,
                pose_path=pose_path,
            )

            result = json.loads(paths.result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["selected_method"], "self_mesh")
            self.assertTrue(result["segmentation"]["gt_assisted"])
            self.assertAlmostEqual(
                result["evaluation"]["rotation_error_deg"],
                0.0,
            )
            self.assertAlmostEqual(
                result["evaluation"]["translation_error_cm"],
                0.0,
            )
            self.assertEqual(
                result["evaluation"]["add_metric_used"],
                "ADD",
            )
            self.assertLess(result["evaluation"]["add_m"], 1e-7)
            self.assertTrue(result["evaluation"]["add_or_adds_0_1d_passed"])
            self.assertIsNone(result["add_evaluation_error"])
            self.assertNotIn(
                "sources",
                result["final_summary"],
            )
            self.assertEqual(
                result["final_summary"]["axis_scale_refinement"],
                {
                    "method": "independent_local_axis_scale",
                    "shared_axes": False,
                    "views": {
                        "reference": {
                            "accepted": False,
                            "requested_scale_factors_xyz": [
                                1.0,
                                0.9,
                                1.1,
                            ],
                            "applied_scale_factors_xyz": [
                                1.0,
                                1.0,
                                1.0,
                            ],
                            "scale_factors_xyz": [
                                1.0,
                                1.0,
                                1.0,
                            ],
                        }
                    },
                },
            )
            self.assertEqual(
                result["final_summary"]["continuous_pose_selection"],
                {
                    "selected_candidate": "H1_confidence_shared_D",
                    "selected_dgedi_stage": "g1",
                    "final_selection_confidence": 0.72,
                    "h0_score": {"total_score": 0.21},
                    "h1_score": {"total_score": 0.18},
                    "confidence_shared_dimensions": {
                        "g0_confidence": 0.64,
                        "raw_shared_dimensions_m": [0.1, 0.2, 0.3],
                    },
                },
            )
            self.assertTrue(paths.pose_path.is_file())
            self.assertTrue(paths.evaluation_path.is_file())

            rejected_paths = save_compact_pair_result(
                output_root=root / "rejected_output",
                dataset_root=dataset_root,
                split="test",
                object_id=9,
                object_name="duck",
                reference_scene_id=9,
                reference_image_id=0,
                reference_instance_index=0,
                query_scene_id=9,
                query_image_id=1,
                query_instance_index=0,
                pipeline_status="completed",
                final_status="REJECT",
                pose_accepted=False,
                summary_path=summary_path,
                pose_path=pose_path,
            )
            rejected_result = json.loads(
                rejected_paths.result_path.read_text(encoding="utf-8")
            )
            self.assertFalse(rejected_result["pose_accepted"])
            self.assertEqual(rejected_result["pose_status"], "rejected")
            self.assertEqual(
                rejected_result["artifacts"]["pose"],
                "rejected_relative_pose.npy",
            )
            self.assertIsNotNone(rejected_result["evaluation"])
            self.assertTrue(rejected_paths.pose_path.is_file())

    def test_removes_only_known_intermediate_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "output"
            query_root = output_root / "queries" / "q000009_000001_i00"
            query_root.mkdir(parents=True)
            (query_root / "mesh.obj").write_text("mesh", encoding="utf-8")

            self.assertTrue(
                remove_pipeline_intermediate_directory(
                    path=query_root,
                    output_root=output_root,
                )
            )
            self.assertFalse(query_root.exists())

            results_root = output_root / "results"
            results_root.mkdir()
            with self.assertRaisesRegex(
                ValueError,
                "non-intermediate",
            ):
                remove_pipeline_intermediate_directory(
                    path=results_root,
                    output_root=output_root,
                )

            with self.assertRaisesRegex(
                ValueError,
                "outside output_root",
            ):
                remove_pipeline_intermediate_directory(
                    path=Path(temporary_directory) / "outside",
                    output_root=output_root,
                )


if __name__ == "__main__":
    unittest.main()
