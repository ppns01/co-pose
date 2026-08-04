from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.linemod_add_evaluator import (
    calculate_add_auc,
    evaluate_linemod_add_metrics,
)
from evaluation.relative_pose_evaluator import BOPFrameGTSpec
from result_storage import backfill_compact_result_add_metrics
from scripts.backfill_linemod_add_metrics import _result_paths


PLY_TEMPLATE = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
-10 -20 0
10 -20 0
10 20 0
-10 20 0
3 0 1 2
3 0 2 3
"""


def _pose(
    rotation: np.ndarray | None = None,
    translation_m: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = rotation
    result[:3, 3] = translation_m
    return result


def _annotation(object_id: int, pose: np.ndarray) -> dict:
    return {
        "obj_id": object_id,
        "cam_R_m2c": pose[:3, :3].reshape(-1).tolist(),
        "cam_t_m2c": (pose[:3, 3] * 1000.0).tolist(),
    }


def _write_fixture(
    root: Path,
    *,
    object_id: int,
    reference_pose: np.ndarray,
    query_pose: np.ndarray,
    symmetric: bool,
) -> tuple[BOPFrameGTSpec, BOPFrameGTSpec]:
    scene_root = root / "test" / f"{object_id:06d}"
    scene_root.mkdir(parents=True)
    (scene_root / "scene_gt.json").write_text(
        json.dumps(
            {
                "0": [_annotation(object_id, reference_pose)],
                "1": [_annotation(object_id, query_pose)],
            }
        ),
        encoding="utf-8",
    )
    model_root = root / "models_eval"
    model_root.mkdir()
    model_info = {"diameter": 100.0}
    if symmetric:
        model_info["symmetries_discrete"] = [
            [
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ]
    (model_root / "models_info.json").write_text(
        json.dumps({str(object_id): model_info}),
        encoding="utf-8",
    )
    (model_root / f"obj_{object_id:06d}.ply").write_text(
        PLY_TEMPLATE,
        encoding="utf-8",
    )
    return (
        BOPFrameGTSpec(
            dataset_root=root,
            split="test",
            scene_id=object_id,
            image_id=0,
            object_id=object_id,
        ),
        BOPFrameGTSpec(
            dataset_root=root,
            split="test",
            scene_id=object_id,
            image_id=1,
            object_id=object_id,
        ),
    )


class LinemodADDEvaluatorTests(unittest.TestCase):
    def test_auc_integrates_recall_and_counts_missing_targets(self) -> None:
        auc = calculate_add_auc(
            [0.0, 0.05, 0.1, float("inf")],
            maximum_threshold=0.1,
            target_count=5,
        )

        self.assertAlmostEqual(auc, 0.3)

    def test_exact_relative_pose_has_zero_add(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            angle = np.deg2rad(20.0)
            query_rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            reference_pose = _pose(
                translation_m=(0.1, -0.2, 1.0)
            )
            query_pose = _pose(
                query_rotation,
                (0.15, -0.18, 0.9),
            )
            reference, query = _write_fixture(
                root,
                object_id=1,
                reference_pose=reference_pose,
                query_pose=query_pose,
                symmetric=False,
            )
            relative_pose = query_pose @ np.linalg.inv(reference_pose)

            result = evaluate_linemod_add_metrics(
                predicted_relative_pose=relative_pose,
                reference_frame=reference,
                query_frame=query,
            )

            self.assertEqual(result.add_metric_used, "ADD")
            self.assertLess(result.add_m, 1e-7)
            self.assertLess(result.adds_m, 1e-7)
            self.assertTrue(result.add_or_adds_0_1d_passed)

    def test_symmetric_object_uses_adds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_pose = _pose()
            query_pose = _pose()
            reference, query = _write_fixture(
                root,
                object_id=10,
                reference_pose=reference_pose,
                query_pose=query_pose,
                symmetric=True,
            )
            predicted_relative = np.eye(4, dtype=np.float64)
            predicted_relative[:3, :3] = np.diag([-1.0, -1.0, 1.0])

            result = evaluate_linemod_add_metrics(
                predicted_relative_pose=predicted_relative,
                reference_frame=reference,
                query_frame=query,
            )

            self.assertEqual(result.add_metric_used, "ADD-S")
            self.assertTrue(result.symmetry_aware)
            self.assertGreater(result.add_m, 0.01)
            self.assertLess(result.adds_m, 1e-9)
            self.assertTrue(result.add_or_adds_0_1d_passed)

    def test_metric_units_and_0_1d_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference, query = _write_fixture(
                root,
                object_id=1,
                reference_pose=_pose(),
                query_pose=_pose(),
                symmetric=False,
            )
            predicted_relative = np.eye(4, dtype=np.float64)
            predicted_relative[0, 3] = 0.005

            result = evaluate_linemod_add_metrics(
                predicted_relative_pose=predicted_relative,
                reference_frame=reference,
                query_frame=query,
            )

            self.assertAlmostEqual(result.object_diameter_m, 0.1)
            self.assertAlmostEqual(result.add_m, 0.005)
            self.assertAlmostEqual(result.add_normalized, 0.05)
            self.assertAlmostEqual(
                result.add_or_adds_0_1d_threshold_m,
                0.01,
            )
            self.assertTrue(result.add_or_adds_0_1d_passed)

    def test_backfills_existing_compact_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture(
                root,
                object_id=1,
                reference_pose=_pose(),
                query_pose=_pose(),
                symmetric=False,
            )
            result_path = root / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "object_id": 1,
                        "reference": {
                            "scene_id": 1,
                            "image_id": 0,
                            "instance_index": 0,
                        },
                        "query": {
                            "scene_id": 1,
                            "image_id": 1,
                            "instance_index": 0,
                        },
                        "relative_pose_query_from_reference": (
                            np.eye(4).tolist()
                        ),
                        "evaluation": {"rotation_error_deg": 0.0},
                    }
                ),
                encoding="utf-8",
            )

            backfill_compact_result_add_metrics(
                result_path=result_path,
                dataset_root=root,
            )

            updated = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(
                updated["evaluation"]["add_metric_used"],
                "ADD",
            )
            self.assertTrue(
                updated["evaluation"]["add_or_adds_0_1d_passed"]
            )
            self.assertIsNone(updated["add_evaluation_error"])

    def test_backfill_root_uses_runner_summary_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            object_root_a = root / "object_01_ape"
            result_a = (
                object_root_a
                / "results"
                / "q000001_000001_i00"
                / "result.json"
            )
            result_a.parent.mkdir(parents=True)
            result_a.write_text("{}", encoding="utf-8")

            object_root_b = root / "object_02_benchvise"
            result_b = (
                object_root_b
                / "results"
                / "q000002_000001_i00"
                / "result.json"
            )
            result_b.parent.mkdir(parents=True)
            result_b.write_text("{}", encoding="utf-8")

            (root / "linemod_all_summary.json").write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "object_id": 1,
                                "query_image_ids": [1],
                                "output_root": str(object_root_a),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                _result_paths(root),
                (result_a.resolve(),),
            )


if __name__ == "__main__":
    unittest.main()
