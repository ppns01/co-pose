from __future__ import annotations

import json
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from linemod_all_runner import (
    _run_object_process,
    _run_pose_metric_postprocessing,
    _save_aggregate_results,
    _terminate_process_group,
    _remove_object_ids_option,
    _remove_single_value_option,
    build_object_command,
    load_object_image_ids,
    run_all_linemod_sequence,
)


class LinemodAllRunnerTest(unittest.TestCase):
    def test_postprocessing_runs_when_all_target_poses_are_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_path = root / "evaluation_report.json"
            with patch(
                "scripts.evaluate_linemod_pose_metrics.evaluate_output_root",
                return_value={
                    "add_failure_count": 0,
                    "report_path": str(report_path),
                },
            ) as evaluate:
                result = _run_pose_metric_postprocessing(
                    output_root=root,
                    dataset_root=root / "datasets",
                    split="test",
                    records=(
                        {
                            "object_id": 1,
                            "query_image_ids": [1],
                        },
                    ),
                )

            self.assertEqual(result, report_path)
            evaluate.assert_called_once()

    def test_loads_only_requested_object_frames(
        self,
    ) -> None:
        dataset_root = (
            Path(__file__).parent
            / "_linemod_dataset"
        )
        scene_directory = (
            dataset_root / "test" / "000008"
        )
        scene_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        try:
            payload = {
                "9": [{"obj_id": 8}],
                "2": [{"obj_id": 3}],
                "5": [
                    {"obj_id": 3},
                    {"obj_id": 8},
                ],
            }
            with (
                scene_directory / "scene_gt.json"
            ).open(
                mode="w",
                encoding="utf-8",
            ) as file:
                json.dump(payload, file)

            image_ids = load_object_image_ids(
                dataset_root=dataset_root,
                split="test",
                object_id=8,
            )
        finally:
            (
                scene_directory / "scene_gt.json"
            ).unlink()
            scene_directory.rmdir()
            scene_directory.parent.rmdir()
            dataset_root.rmdir()

        self.assertEqual(image_ids, (5, 9))

    def test_removes_output_root_forms(
        self,
    ) -> None:
        self.assertEqual(
            _remove_single_value_option(
                [
                    "--config",
                    "override.yaml",
                    "--output-root",
                    "outputs/run",
                    "--device",
                    "cuda:0",
                ],
                "--output-root",
            ),
            [
                "--config",
                "override.yaml",
                "--device",
                "cuda:0",
            ],
        )
        self.assertEqual(
            _remove_single_value_option(
                [
                    "--output-root=outputs/run",
                    "--top-k",
                    "2",
                ],
                "--output-root",
            ),
            ["--top-k", "2"],
        )

    def test_removes_object_filter_before_child_launch(self) -> None:
        self.assertEqual(
            _remove_object_ids_option(
                [
                    "--object-ids",
                    "1",
                    "9",
                    "--top-k",
                    "2",
                ]
            ),
            ["--top-k", "2"],
        )

    def test_builds_sequential_object_command(
        self,
    ) -> None:
        command = build_object_command(
            python_executable="/env/bin/python",
            entrypoint_path=Path(
                "/project/main_self_mesh.py"
            ),
            base_argv=["--top-k", "2"],
            object_id=8,
            object_name="driller",
            sam3_prompt="power drill",
            reference_image_id=0,
            query_image_ids=(1, 2, 7),
            output_root=Path(
                "/outputs/object_08_driller"
            ),
        )

        self.assertEqual(
            command[:4],
            [
                "/env/bin/python",
                str(
                    Path(
                        "/project/main_self_mesh.py"
                    ).resolve()
                ),
                "--top-k",
                "2",
            ],
        )
        query_option_index = command.index(
            "--query-image-ids"
        )
        output_option_index = command.index(
            "--output-root"
        )
        self.assertEqual(
            command[
                query_option_index
                + 1 : output_option_index
            ],
            ["1", "2", "7"],
        )
        self.assertEqual(
            command[
                command.index("--object-id") + 1
            ],
            "8",
        )

    def test_all_runner_uses_configured_objects_stride_and_limit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "all"
            parsed = SimpleNamespace(output_root=output_root)
            config = SimpleNamespace(
                output_root=output_root,
                dataset_root=Path(temporary_directory) / "dataset",
                split="test",
                reference=SimpleNamespace(image_id=0),
                linemod_all_object_ids=(8,),
                linemod_all_query_stride=2,
                linemod_all_maximum_queries_per_object=2,
                linemod_all_continue_on_error=True,
            )
            commands: list[list[str]] = []

            def fake_run(command: list[str]) -> int:
                commands.append(command)
                return 0

            with (
                patch("main.parse_args", return_value=parsed),
                patch("main.build_config", return_value=config),
                patch(
                    "linemod_all_runner.load_object_image_ids",
                    return_value=(0, 1, 2, 3, 4, 5),
                ),
                patch(
                    "linemod_all_runner._run_object_process",
                    side_effect=fake_run,
                ),
            ):
                exit_code = run_all_linemod_sequence(
                    pose_path="self_mesh",
                    entrypoint_path=Path("main_self_mesh.py"),
                    argv=(),
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(commands), 1)
            command = commands[0]
            query_index = command.index("--query-image-ids")
            output_index = command.index("--output-root")
            self.assertEqual(
                command[query_index + 1 : output_index],
                ["1", "3"],
            )

    def test_all_runner_accepts_one_object_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "one"
            parsed = SimpleNamespace(
                output_root=output_root,
                object_ids=(9,),
            )
            config = SimpleNamespace(
                output_root=output_root,
                dataset_root=Path(temporary_directory) / "dataset",
                split="test",
                reference=SimpleNamespace(image_id=0),
                linemod_all_object_ids=(8, 9),
                linemod_all_query_stride=1,
                linemod_all_maximum_queries_per_object=1,
                linemod_all_continue_on_error=True,
            )
            commands: list[list[str]] = []

            with (
                patch("main.parse_args", return_value=parsed),
                patch("main.build_config", return_value=config),
                patch(
                    "linemod_all_runner.load_object_image_ids",
                    return_value=(0, 1),
                ),
                patch(
                    "linemod_all_runner._run_object_process",
                    side_effect=lambda command: commands.append(command) or 0,
                ),
            ):
                exit_code = run_all_linemod_sequence(
                    pose_path="self_mesh",
                    entrypoint_path=Path("main_self_mesh.py"),
                    argv=("--object-ids", "9"),
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(commands), 1)
            self.assertEqual(
                commands[0][commands[0].index("--object-id") + 1],
                "9",
            )
            self.assertNotIn("--object-ids", commands[0])

    def test_all_runner_continues_after_abnormal_object_exit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "all"
            parsed = SimpleNamespace(output_root=output_root)
            config = SimpleNamespace(
                output_root=output_root,
                dataset_root=Path(temporary_directory) / "dataset",
                split="test",
                reference=SimpleNamespace(image_id=0),
                linemod_all_object_ids=(8, 9),
                linemod_all_query_stride=1,
                linemod_all_maximum_queries_per_object=1,
                linemod_all_continue_on_error=True,
            )

            with (
                patch("main.parse_args", return_value=parsed),
                patch("main.build_config", return_value=config),
                patch(
                    "linemod_all_runner.load_object_image_ids",
                    return_value=(0, 1),
                ),
                patch(
                    "linemod_all_runner._run_object_process",
                    side_effect=(-9, 0),
                ) as run_object_process,
            ):
                exit_code = run_all_linemod_sequence(
                    pose_path="self_mesh",
                    entrypoint_path=Path("main_self_mesh.py"),
                    argv=(),
                )

            summary = json.loads(
                (
                    output_root / "linemod_all_summary.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(run_object_process.call_count, 2)
        self.assertEqual(
            [record["status"] for record in summary["records"]],
            ["failed", "completed"],
        )

    def test_object_process_uses_own_session_and_cleans_failure(
        self,
    ) -> None:
        process = MagicMock()
        process.pid = 43210
        process.wait.return_value = -9

        with (
            patch(
                "linemod_all_runner.subprocess.Popen",
                return_value=process,
            ) as popen,
            patch(
                "linemod_all_runner._terminate_process_group"
            ) as terminate_process_group,
        ):
            return_code = _run_object_process(
                ["/env/bin/python", "worker.py"]
            )

        self.assertEqual(return_code, -9)
        popen.assert_called_once_with(
            ["/env/bin/python", "worker.py"],
            start_new_session=True,
        )
        terminate_process_group.assert_called_once_with(
            process
        )

    def test_object_process_does_not_clean_success(
        self,
    ) -> None:
        process = MagicMock()
        process.pid = 43210
        process.wait.return_value = 0

        with (
            patch(
                "linemod_all_runner.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "linemod_all_runner._terminate_process_group"
            ) as terminate_process_group,
        ):
            return_code = _run_object_process(
                ["/env/bin/python", "worker.py"]
            )

        self.assertEqual(return_code, 0)
        terminate_process_group.assert_not_called()

    def test_object_process_cleans_parent_interruption(
        self,
    ) -> None:
        process = MagicMock()
        process.pid = 43210
        process.wait.side_effect = KeyboardInterrupt

        with (
            patch(
                "linemod_all_runner.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "linemod_all_runner._terminate_process_group"
            ) as terminate_process_group,
            self.assertRaises(KeyboardInterrupt),
        ):
            _run_object_process(
                ["/env/bin/python", "worker.py"]
            )

        terminate_process_group.assert_called_once_with(
            process
        )

    def test_process_group_cleanup_escalates_to_sigkill(
        self,
    ) -> None:
        process = MagicMock()
        process.pid = 43210
        process.poll.return_value = 7

        with (
            patch(
                "linemod_all_runner._send_process_group_signal",
                return_value=True,
            ) as send_signal,
            patch(
                "linemod_all_runner._process_group_exists",
                return_value=True,
            ),
            patch(
                "linemod_all_runner.time.monotonic",
                side_effect=(0.0, 3.0),
            ),
        ):
            _terminate_process_group(process)

        self.assertEqual(
            send_signal.call_args_list[0].args,
            (43210, signal.SIGTERM),
        )
        self.assertEqual(
            send_signal.call_args_list[1].args,
            (43210, signal.SIGKILL),
        )

    def test_aggregates_only_current_record_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "all"
            object_root = output_root / "object_09_duck"
            result_root = (
                object_root
                / "results"
                / "q000009_000003_i00"
            )
            result_root.mkdir(parents=True)
            (result_root / "result.json").write_text(
                json.dumps(
                    {
                        "object_id": 9,
                        "object_name": "duck",
                        "reference": {"scene_id": 9, "image_id": 0},
                        "query": {"scene_id": 9, "image_id": 3},
                        "pipeline_status": "completed",
                        "final_status": "CONSISTENT",
                        "pose_accepted": True,
                        "pose_status": "accepted",
                        "selected_method": "self_mesh",
                        "relative_pose_query_from_reference": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "evaluation": {
                            "rotation_error_deg": 1.25,
                            "translation_error_cm": 2.5,
                            "add_metric_used": "ADD",
                            "add_m": 0.005,
                            "adds_m": 0.004,
                            "add_normalized": 0.05,
                            "adds_normalized": 0.04,
                            "add_or_adds_m": 0.005,
                            "add_or_adds_normalized": 0.05,
                            "add_or_adds_0_1d_threshold_m": 0.01,
                            "add_or_adds_0_1d_passed": True,
                            "object_diameter_m": 0.1,
                            "symmetry_aware": False,
                            "symmetry_type": "none",
                            "model_point_count": 100,
                        },
                    }
                ),
                encoding="utf-8",
            )
            official_scores_path = (
                output_root
                / "evaluation"
                / "bop19"
                / "official_eval"
                / "selfmesh_lm-test"
                / "scores_bop19.json"
            )
            official_scores_path.parent.mkdir(parents=True)
            official_scores_path.write_text(
                json.dumps(
                    {
                        "bop19_average_recall_vsd": 0.7,
                        "bop19_average_recall_mssd": 0.9,
                        "bop19_average_recall_mspd": 0.8,
                        "bop19_average_recall": 0.8,
                        "bop19_average_time_per_image": float("nan"),
                    }
                ),
                encoding="utf-8",
            )
            json_path, csv_path = _save_aggregate_results(
                output_root=output_root,
                records=(
                    {
                        "object_id": 9,
                        "output_root": str(object_root),
                        "query_image_ids": [3],
                    },
                ),
            )

            aggregate = json.loads(
                json_path.read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate["result_count"], 1)
            metric_summary = aggregate[
                "add_or_adds_0_1d_summary"
            ]["overall"]
            self.assertEqual(metric_summary["target_count"], 1)
            self.assertEqual(metric_summary["passed_count"], 1)
            self.assertEqual(
                metric_summary["recall_over_all_targets"],
                1.0,
            )
            self.assertAlmostEqual(
                aggregate["add_auc_summary"]["overall"][
                    "add_or_adds_auc_0_1m"
                ],
                0.95,
            )
            self.assertAlmostEqual(
                aggregate["bop19_official"]["scores"][
                    "bop19_average_recall"
                ],
                0.8,
            )
            self.assertIsNone(
                aggregate["bop19_official"]["scores"][
                    "bop19_average_time_per_image"
                ]
            )
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("pose_accepted", csv_text)
            self.assertIn("add_or_adds_0_1d_passed", csv_text)
            self.assertIn("accepted", csv_text)
            self.assertIn("1.25", csv_text)
            self.assertNotIn("bop19_average_recall", csv_text)


if __name__ == "__main__":
    unittest.main()
