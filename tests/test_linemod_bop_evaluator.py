from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import imageio.v3 as iio
import numpy as np

from evaluation.linemod_bop_evaluator import (
    create_bop_linemod_dataset_view,
    relative_pose_to_query_absolute_mm,
    run_official_bop19_evaluation,
)
from evaluation.relative_pose_evaluator import BOPFrameGTSpec
from scripts.evaluate_linemod_pose_metrics import (
    _export_bop19_csv,
    _target_entries,
    find_compact_result_paths,
)


def _annotation(object_id: int, pose_m: np.ndarray) -> dict:
    return {
        "obj_id": object_id,
        "cam_R_m2c": pose_m[:3, :3].reshape(-1).tolist(),
        "cam_t_m2c": (pose_m[:3, 3] * 1000.0).tolist(),
    }


def _write_dataset_fixture(root: Path) -> BOPFrameGTSpec:
    object_id = 1
    reference_pose = np.eye(4, dtype=np.float64)
    reference_pose[:3, 3] = (0.02, -0.01, 1.0)
    query_pose = reference_pose.copy()
    scene_root = root / "test" / f"{object_id:06d}"
    (scene_root / "depth").mkdir(parents=True)
    (scene_root / "scene_gt.json").write_text(
        json.dumps(
            {
                "0": [_annotation(object_id, reference_pose)],
                "1": [_annotation(object_id, query_pose)],
            }
        ),
        encoding="utf-8",
    )
    camera = {
        "cam_K": [
            500.0,
            0.0,
            3.5,
            0.0,
            500.0,
            3.5,
            0.0,
            0.0,
            1.0,
        ],
        "depth_scale": 1.0,
    }
    (scene_root / "scene_camera.json").write_text(
        json.dumps({"0": camera, "1": camera}),
        encoding="utf-8",
    )
    depth_mm = np.full((8, 8), 1000, dtype=np.uint16)
    iio.imwrite(scene_root / "depth" / "000001.png", depth_mm)

    model_root = root / "models_eval"
    model_root.mkdir()
    (model_root / "models_info.json").write_text(
        json.dumps({"1": {"diameter": 100.0}}),
        encoding="utf-8",
    )
    return BOPFrameGTSpec(
        dataset_root=root,
        split="test",
        scene_id=1,
        image_id=0,
        object_id=1,
    )


class LinemodBOP19EvaluatorTests(unittest.TestCase):
    def test_bop_export_pose_uses_millimetres(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = _write_dataset_fixture(root)

            pose_mm = relative_pose_to_query_absolute_mm(
                predicted_relative_pose=np.eye(4),
                reference_frame=reference,
            )

            np.testing.assert_allclose(
                pose_mm[:3, 3],
                np.array([20.0, -10.0, 1000.0]),
            )

    def test_dataset_view_links_data_and_writes_camera_and_targets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "datasets"
            _write_dataset_fixture(dataset_root)
            targets = [
                {
                    "scene_id": 1,
                    "im_id": 1,
                    "obj_id": 1,
                    "inst_count": 1,
                }
            ]

            view_root, targets_path = create_bop_linemod_dataset_view(
                dataset_root=dataset_root,
                split="test",
                output_directory=root / "evaluation",
                targets=targets,
            )

            linemod_root = view_root / "lm"
            self.assertTrue((linemod_root / "test").is_symlink())
            self.assertEqual(
                (linemod_root / "test").resolve(),
                (dataset_root / "test").resolve(),
            )
            self.assertTrue((linemod_root / "models_eval").is_symlink())
            self.assertEqual(
                json.loads(targets_path.read_text(encoding="utf-8")),
                targets,
            )
            camera = json.loads(
                (linemod_root / "camera.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(camera["width"], 8)
            self.assertEqual(camera["height"], 8)
            self.assertEqual(camera["fx"], 500.0)
            self.assertEqual(camera["depth_scale"], 1.0)

    def test_official_runner_uses_cli_and_loads_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            toolkit_root = root / "bop_toolkit"
            script_path = toolkit_root / "scripts" / "eval_bop19_pose.py"
            script_path.parent.mkdir(parents=True)
            script_path.write_text("# fixture\n", encoding="utf-8")
            result_csv_path = root / "results" / "selfmesh_lm-test.csv"
            result_csv_path.parent.mkdir()
            result_csv_path.write_text(
                "scene_id,im_id,obj_id,score,R,t,time\n",
                encoding="utf-8",
            )
            targets_path = root / "targets.json"
            targets_path.write_text("[]\n", encoding="utf-8")
            dataset_view_root = root / "dataset_view"
            dataset_view_root.mkdir()
            evaluation_directory = root / "official_eval"
            scores_path = (
                evaluation_directory
                / "selfmesh_lm-test"
                / "scores_bop19.json"
            )

            def _fake_run(command, *, check, env):
                self.assertFalse(check)
                self.assertIn("--renderer_type=vispy", command)
                self.assertIn("--num_workers=2", command)
                self.assertEqual(env["BOP_PATH"], str(dataset_view_root))
                scores_path.parent.mkdir(parents=True)
                scores_path.write_text(
                    json.dumps(
                        {
                            "bop19_average_recall_vsd": 0.5,
                            "bop19_average_recall_mssd": 0.6,
                            "bop19_average_recall_mspd": 0.7,
                            "bop19_average_recall": 0.6,
                            "bop19_average_time_per_image": float("nan"),
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with patch(
                "evaluation.linemod_bop_evaluator.subprocess.run",
                side_effect=_fake_run,
            ) as run:
                result = run_official_bop19_evaluation(
                    toolkit_root=toolkit_root,
                    dataset_view_root=dataset_view_root,
                    result_csv_path=result_csv_path,
                    targets_path=targets_path,
                    evaluation_directory=evaluation_directory,
                    renderer_type="vispy",
                    num_workers=2,
                )

            run.assert_called_once()
            self.assertEqual(result.scores_path, scores_path)
            self.assertEqual(result.scores["bop19_average_recall"], 0.6)
            self.assertNotIn(
                "bop19_average_time_per_image",
                result.scores,
            )

    def test_rejected_pose_is_exported_and_missing_pose_stays_absent(
        self,
    ) -> None:
        from bop_toolkit_lib import inout

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "datasets"
            _write_dataset_fixture(dataset_root)
            output_root = root / "outputs"
            object_root = output_root / "object_01_ape"
            summary = {
                "records": [
                    {
                        "object_id": 1,
                        "output_root": str(object_root),
                        "query_image_ids": [1, 2],
                    }
                ]
            }
            (output_root / "linemod_all_summary.json").parent.mkdir(
                parents=True
            )
            (output_root / "linemod_all_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            for image_id, pose in ((1, np.eye(4).tolist()), (2, None)):
                result_root = (
                    object_root
                    / "results"
                    / f"q000001_{image_id:06d}_i00"
                )
                result_root.mkdir(parents=True)
                (result_root / "result.json").write_text(
                    json.dumps(
                        {
                            "object_id": 1,
                            "reference": {
                                "scene_id": 1,
                                "image_id": 0,
                            },
                            "query": {
                                "scene_id": 1,
                                "image_id": image_id,
                            },
                            "pose_accepted": False,
                            "pose_status": "rejected",
                            "relative_pose_query_from_reference": pose,
                        }
                    ),
                    encoding="utf-8",
                )

            result_paths = find_compact_result_paths(output_root)
            targets = _target_entries(
                output_root=output_root,
                result_paths=result_paths,
            )
            csv_path, exported_count, missing_pose_count = (
                _export_bop19_csv(
                    result_paths=result_paths,
                    dataset_root=dataset_root,
                    split="test",
                    output_directory=output_root / "evaluation",
                )
            )

            estimates = inout.load_bop_results(csv_path, version="bop19")
            self.assertEqual(len(targets), 2)
            self.assertEqual(exported_count, 1)
            self.assertEqual(missing_pose_count, 1)
            self.assertEqual(len(estimates), 1)
            self.assertEqual(estimates[0]["im_id"], 1)


if __name__ == "__main__":
    unittest.main()
