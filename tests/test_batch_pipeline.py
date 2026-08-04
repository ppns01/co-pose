from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import main
from generators import instantmesh_generator


TEST_DIRECTORY = Path(__file__).resolve().parent


class BatchPipelineTests(unittest.TestCase):
    def test_results_only_keeps_compact_values_and_removes_workdirs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "batch"
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
                        "cam_t_m2c": [0.0, 0.0, 1000.0],
                    }
                ],
            }
            (scene_root / "scene_gt.json").write_text(
                json.dumps(scene_gt),
                encoding="utf-8",
            )
            config = replace(
                main.build_config(
                    main.parse_args(
                        [
                            "--object-id",
                            "9",
                            "--reference-scene-id",
                            "9",
                            "--reference-image-id",
                            "0",
                            "--query-scene-id",
                            "9",
                            "--query-image-ids",
                            "1",
                        ]
                    )
                ),
                output_root=output_root,
                dataset_root=dataset_root,
                storage_results_only=True,
                storage_keep_failed_intermediates=False,
                visible_scale_policy="independent",
                enable_shared_axis_scale_refinement=False,
            )

            def fake_prepare(
                *,
                view_name: str,
                frame: main.FrameSpec,
                output_root: Path,
                **_: object,
            ) -> SimpleNamespace:
                output_root.mkdir(parents=True, exist_ok=True)
                (output_root / "intermediate.bin").write_bytes(b"large")
                return SimpleNamespace(view_name=view_name, frame=frame)

            def fake_generate(
                *,
                view_name: str,
                frame: main.FrameSpec,
                **_: object,
            ) -> SimpleNamespace:
                return SimpleNamespace(view_name=view_name, frame=frame)

            def fake_align(
                *,
                generated_states: object,
                output_root: Path,
                **_: object,
            ) -> tuple[SimpleNamespace, ...]:
                return tuple(
                    SimpleNamespace(
                        generated=state,
                        self_alignment=SimpleNamespace(candidate_index=0),
                        self_evaluation=SimpleNamespace(
                            summary_path=output_root / "self_summary.json"
                        ),
                    )
                    for state in generated_states
                )

            def fake_pair(
                *,
                output_root: Path,
                **_: object,
            ) -> main.PairPipelineOutcome:
                final_root = output_root / "final"
                final_root.mkdir(parents=True)
                pose_path = final_root / "final_relative_pose.npy"
                np.save(pose_path, np.eye(4), allow_pickle=False)
                summary_path = final_root / "final_selection.json"
                summary_path.write_text(
                    json.dumps({"status": "COMPLETED", "method": "self_mesh"}),
                    encoding="utf-8",
                )
                return main.PairPipelineOutcome(
                    final_status="CONSISTENT",
                    summary_path=summary_path,
                    pose_path=pose_path,
                    pose_accepted=True,
                    visualization_path=None,
                )

            generator_context = MagicMock()
            generator_context.__enter__.return_value = object()
            generator_context.__exit__.return_value = False
            with (
                patch.object(
                    instantmesh_generator,
                    "InstantMeshGenerator",
                    return_value=generator_context,
                ),
                patch.object(main, "_prepare_linemod_view", side_effect=fake_prepare),
                patch.object(main, "_build_generated_state", side_effect=fake_generate),
                patch.object(
                    main, "_self_align_generated_states", side_effect=fake_align
                ),
                patch.object(
                    main,
                    "_refine_aligned_states_independent_axis_scale",
                    side_effect=lambda *, aligned_states, **_: tuple(aligned_states),
                ),
                patch.object(
                    main,
                    "_extract_dino_inputs",
                    side_effect=lambda prepared_views, **_: tuple(
                        (None, None) for _view in prepared_views
                    ),
                ),
                patch.object(main, "_run_aligned_pair", side_effect=fake_pair),
                patch.object(main, "validate_config", return_value=None),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                exit_code = main.run_batch_pipeline(config)

            result_root = output_root / "results" / "q000009_000001_i00"
            self.assertEqual(exit_code, 0)
            self.assertTrue((result_root / "result.json").is_file())
            self.assertTrue((result_root / "final_relative_pose.npy").is_file())
            self.assertFalse((output_root / "queries").exists())
            self.assertFalse((output_root / "shared_reference").exists())

    def test_reuses_reference_and_namespaces_queries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            output_root = Path(temp_dir) / "batch"
            config = replace(
                main.build_config(
                    main.parse_args(
                        [
                            "--query-image-ids",
                            "1",
                            "2",
                            "--foundationpose-workers",
                            "2",
                        ]
                    )
                ),
                output_root=output_root,
                storage_results_only=False,
                visible_scale_policy="independent",
                enable_shared_axis_scale_refinement=False,
            )

            prepared_frames: list[tuple[str, int]] = []
            generated_frames: list[tuple[str, int]] = []
            pair_roots: list[Path] = []
            pair_dino_inputs: list[tuple[object, object, object, object]] = []

            def fake_prepare(
                *,
                view_name: str,
                frame: main.FrameSpec,
                **_: object,
            ) -> SimpleNamespace:
                prepared_frames.append((view_name, frame.image_id))
                return SimpleNamespace(
                    view_name=view_name,
                    frame=frame,
                )

            def fake_generate(
                *,
                view_name: str,
                frame: main.FrameSpec,
                **_: object,
            ) -> SimpleNamespace:
                generated_frames.append((view_name, frame.image_id))
                return SimpleNamespace(
                    view_name=view_name,
                    frame=frame,
                )

            def fake_extract_dino(
                *,
                prepared_views: object,
                **_: object,
            ) -> tuple[
                tuple[object, object],
                ...,
            ]:
                return tuple(
                    (
                        (f"{view.view_name}_{view.frame.image_id}_dino"),
                        (f"{view.view_name}_{view.frame.image_id}_surface"),
                    )
                    for view in prepared_views
                )

            def fake_align(
                *,
                generated_states: object,
                output_root: Path,
                **_: object,
            ) -> tuple[SimpleNamespace, ...]:
                return tuple(
                    SimpleNamespace(
                        generated=state,
                        self_alignment=(SimpleNamespace(candidate_index=0)),
                        self_evaluation=(
                            SimpleNamespace(
                                summary_path=(output_root / "self_summary.json")
                            )
                        ),
                    )
                    for state in generated_states
                )

            def fake_pair(
                *,
                output_root: Path,
                **kwargs: object,
            ) -> main.PairPipelineOutcome:
                pair_roots.append(output_root)
                pair_dino_inputs.append(
                    (
                        kwargs["reference_dino"],
                        kwargs["reference_surface"],
                        kwargs["query_dino"],
                        kwargs["query_surface"],
                    )
                )
                return main.PairPipelineOutcome(
                    final_status="CONSISTENT",
                    summary_path=(output_root / "final" / "final_selection.json"),
                    pose_path=(output_root / "final" / "final_relative_pose.npy"),
                    pose_accepted=True,
                    visualization_path=(output_root / "visualizations" / "index.html"),
                )

            generator_context = MagicMock()
            generator_context.__enter__.return_value = object()
            generator_context.__exit__.return_value = False

            with (
                patch.object(
                    instantmesh_generator,
                    "InstantMeshGenerator",
                    return_value=generator_context,
                ),
                patch.object(
                    main,
                    "_prepare_linemod_view",
                    side_effect=fake_prepare,
                ),
                patch.object(
                    main,
                    "_extract_dino_inputs",
                    side_effect=fake_extract_dino,
                ),
                patch.object(
                    main,
                    "_build_generated_state",
                    side_effect=fake_generate,
                ),
                patch.object(
                    main,
                    "_self_align_generated_states",
                    side_effect=fake_align,
                ),
                patch.object(
                    main,
                    "_refine_aligned_states_independent_axis_scale",
                    side_effect=lambda *, aligned_states, **_: tuple(aligned_states),
                ),
                patch.object(
                    main,
                    "_run_aligned_pair",
                    side_effect=fake_pair,
                ),
                patch.object(
                    main,
                    "validate_config",
                    return_value=None,
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                exit_code = main.run_batch_pipeline(config)

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                prepared_frames,
                [
                    ("reference", 0),
                    ("query", 1),
                    ("query", 2),
                ],
            )
            self.assertEqual(
                generated_frames,
                prepared_frames,
            )
            self.assertEqual(len(pair_roots), 2)
            self.assertEqual(
                pair_dino_inputs,
                [
                    (
                        "reference_0_dino",
                        "reference_0_surface",
                        "query_1_dino",
                        "query_1_surface",
                    ),
                    (
                        "reference_0_dino",
                        "reference_0_surface",
                        "query_2_dino",
                        "query_2_surface",
                    ),
                ],
            )
            self.assertEqual(
                pair_roots[0].name,
                "q000008_000001_i00",
            )
            self.assertEqual(
                pair_roots[1].name,
                "q000008_000002_i00",
            )

            summary = json.loads(
                (output_root / "batch_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["reference"]["status"],
                "completed",
            )
            self.assertEqual(
                summary["completed_count"],
                2,
            )
            self.assertEqual(
                summary["failed_count"],
                0,
            )

    def test_query_failure_does_not_stop_later_queries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            output_root = Path(temp_dir) / "batch"
            config = replace(
                main.build_config(
                    main.parse_args(
                        [
                            "--query-image-ids",
                            "1",
                            "2",
                            "--disable-dino",
                        ]
                    )
                ),
                output_root=output_root,
                storage_results_only=False,
                visible_scale_policy="independent",
                enable_shared_axis_scale_refinement=False,
            )

            def fake_prepare(
                *,
                view_name: str,
                frame: main.FrameSpec,
                **_: object,
            ) -> SimpleNamespace:
                return SimpleNamespace(
                    view_name=view_name,
                    frame=frame,
                )

            def fake_generate(
                *,
                view_name: str,
                frame: main.FrameSpec,
                **_: object,
            ) -> SimpleNamespace:
                return SimpleNamespace(
                    view_name=view_name,
                    frame=frame,
                )

            def fake_align(
                *,
                generated_states: object,
                output_root: Path,
                **_: object,
            ) -> tuple[SimpleNamespace, ...]:
                return tuple(
                    SimpleNamespace(
                        generated=state,
                        self_alignment=(SimpleNamespace(candidate_index=0)),
                        self_evaluation=(
                            SimpleNamespace(
                                summary_path=(output_root / "self_summary.json")
                            )
                        ),
                    )
                    for state in generated_states
                )

            pair_call_count = 0

            def fake_pair(
                *,
                output_root: Path,
                **_: object,
            ) -> main.PairPipelineOutcome:
                nonlocal pair_call_count
                pair_call_count += 1

                if pair_call_count == 1:
                    raise RuntimeError("intentional query failure")

                return main.PairPipelineOutcome(
                    final_status="CONSISTENT",
                    summary_path=(output_root / "final" / "final_selection.json"),
                    pose_path=(output_root / "final" / "final_relative_pose.npy"),
                    pose_accepted=True,
                    visualization_path=None,
                    visualization_error=("RuntimeError: visualization"),
                )

            generator_context = MagicMock()
            generator_context.__enter__.return_value = object()
            generator_context.__exit__.return_value = False

            with (
                patch.object(
                    instantmesh_generator,
                    "InstantMeshGenerator",
                    return_value=generator_context,
                ),
                patch.object(
                    main,
                    "_prepare_linemod_view",
                    side_effect=fake_prepare,
                ),
                patch.object(
                    main,
                    "_build_generated_state",
                    side_effect=fake_generate,
                ),
                patch.object(
                    main,
                    "_self_align_generated_states",
                    side_effect=fake_align,
                ),
                patch.object(
                    main,
                    "_refine_aligned_states_independent_axis_scale",
                    side_effect=lambda *, aligned_states, **_: tuple(aligned_states),
                ),
                patch.object(
                    main,
                    "_run_aligned_pair",
                    side_effect=fake_pair,
                ),
                patch.object(
                    main,
                    "validate_config",
                    return_value=None,
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                exit_code = main.run_batch_pipeline(config)

            self.assertEqual(exit_code, 1)
            self.assertEqual(pair_call_count, 2)

            summary = json.loads(
                (output_root / "batch_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [record["status"] for record in summary["queries"]],
                ["failed", "completed"],
            )
            self.assertEqual(
                summary["completed_count"],
                1,
            )
            self.assertEqual(
                summary["failed_count"],
                1,
            )
            self.assertIsNone(summary["queries"][1]["visualization_path"])
            self.assertEqual(
                summary["queries"][1]["visualization_error"],
                "RuntimeError: visualization",
            )


if __name__ == "__main__":
    unittest.main()
