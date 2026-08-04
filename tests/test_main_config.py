from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import main


TEST_DIRECTORY = Path(__file__).resolve().parent


class MainConfigTests(unittest.TestCase):
    def test_object_id_uses_semantic_sam3_defaults(self) -> None:
        config = main.build_config(
            main.parse_args(
                [
                    "--object-id",
                    "9",
                    "--reference-scene-id",
                    "9",
                    "--query-scene-id",
                    "9",
                ]
            )
        )

        self.assertEqual(config.object_name, "duck")
        self.assertEqual(
            config.sam3_prompt,
            "a small yellow rubber duck",
        )

    def test_explicit_sam3_prompt_overrides_object_default(self) -> None:
        config = main.build_config(
            main.parse_args(
                [
                    "--object-id",
                    "9",
                    "--sam3-prompt",
                    "yellow bath duck",
                ]
            )
        )

        self.assertEqual(config.object_name, "duck")
        self.assertEqual(config.sam3_prompt, "yellow bath duck")

    def test_default_instantmesh_config_is_low_vram(self) -> None:
        config_path = main.DEFAULT_INSTANTMESH_CONFIG

        self.assertEqual(
            config_path.stem,
            "instant-mesh-large",
        )
        self.assertIn(
            "grid_res: 64",
            config_path.read_text(
                encoding="utf-8",
            ),
        )

    def test_cli_builds_reference_and_query_config(self) -> None:
        args = main.parse_args(
            [
                "--object-id",
                "8",
                "--reference-scene-id",
                "8",
                "--reference-image-id",
                "1",
                "--query-scene-id",
                "8",
                "--query-image-id",
                "2",
            ]
        )

        config = main.build_config(args)

        self.assertEqual(config.object_name, "driller")
        self.assertEqual(config.sam3_prompt, "power drill")
        self.assertEqual(config.reference.image_id, 1)
        self.assertEqual(config.query.image_id, 2)
        self.assertEqual(config.mask_type, "mask_visib")
        self.assertEqual(
            config.instantmesh_config.name,
            "instant-mesh-large.yaml",
        )
        self.assertEqual(
            config.instantmesh_config,
            main.DEFAULT_INSTANTMESH_CONFIG.resolve(),
        )
        self.assertEqual(
            config.foundationpose_workers,
            2,
        )
        self.assertTrue(config.dino_enabled)
        self.assertEqual(
            config.dinov3_repository,
            main.DEFAULT_DINOV3_REPOSITORY.resolve(),
        )
        self.assertEqual(
            config.dinov3_checkpoint.name,
            (
                "dinov3-vitl16-"
                "pretrain-lvd1689m"
            ),
        )
        self.assertEqual(
            config.dinov3_model,
            "dinov3_vitl16",
        )
        self.assertIsNone(
            config.batch_query_image_ids
        )
        self.assertIsInstance(config.dataset_root, Path)

    def test_cli_can_disable_dino(self) -> None:
        config = main.build_config(
            main.parse_args(["--disable-dino"])
        )

        self.assertFalse(config.dino_enabled)
        self.assertEqual(
            main._extract_dino_inputs(
                config=config,
                prepared_views=(object(),),
                output_roots=(Path("."),),
            ),
            ((None, None),),
        )

    @unittest.skip(
        "DINO is only reachable via pose_path in "
        "{'combined', 'self_cross'}, both removed; the default "
        "pose_path is now self_mesh, which never triggers DINO. "
        "DINO subsystem cleanup is a separate, deferred task."
    )
    def test_hf_model_directory_does_not_require_hubconf(
        self,
    ) -> None:
        config = replace(
            main.build_config(
                main.parse_args(
                    [
                        "--dinov3-model",
                        "dinov3_vitl16",
                        "--dinov3-checkpoint",
                        ".",
                    ]
                )
            ),
            dinov3_repository=Path(
                "missing_dinov3_repository"
            ),
        )

        with (
            patch.object(
                main,
                "_require_directory",
            ) as require_directory,
            patch.object(
                main,
                "_require_file",
            ) as require_file,
        ):
            main.validate_config(config)

        directory_descriptions = {
            call.args[1]
            for call in (
                require_directory.call_args_list
            )
        }
        file_descriptions = {
            call.args[1]
            for call in require_file.call_args_list
        }

        self.assertNotIn(
            "DINOv3 repository",
            directory_descriptions,
        )
        self.assertTrue(
            {
                "Hugging Face DINOv3 config.json",
                (
                    "Hugging Face DINOv3 "
                    "model.safetensors"
                ),
                (
                    "Hugging Face DINOv3 "
                    "preprocessor_config.json"
                ),
            }.issubset(file_descriptions)
        )

    @unittest.skip(
        "DINO is only reachable via pose_path in "
        "{'combined', 'self_cross'}, both removed; the default "
        "pose_path is now self_mesh, which never triggers DINO. "
        "DINO subsystem cleanup is a separate, deferred task."
    )
    def test_dino_inputs_share_one_model_load(
        self,
    ) -> None:
        config = main.build_config(
            main.parse_args([])
        )

        reference_view = SimpleNamespace(
            view=SimpleNamespace(
                source=SimpleNamespace(
                    name="reference"
                )
            )
        )
        query_view = SimpleNamespace(
            view=SimpleNamespace(
                source=SimpleNamespace(
                    name="query"
                )
            )
        )

        reference_dense = SimpleNamespace(
            view_name="reference"
        )
        query_dense = SimpleNamespace(
            view_name="query"
        )
        reference_surface = SimpleNamespace(
            view_name="reference"
        )
        query_surface = SimpleNamespace(
            view_name="query"
        )

        extractor = MagicMock()
        extractor.extract.side_effect = (
            reference_dense,
            query_dense,
        )
        extractor_context = MagicMock()
        extractor_context.__enter__.return_value = (
            extractor
        )
        extractor_context.__exit__.return_value = (
            False
        )

        output_root = (
            TEST_DIRECTORY / "unused_dino_output"
        )

        with (
            patch(
                "features.dinov3_extractor"
                ".DINOv3Extractor",
                return_value=extractor_context,
            ) as extractor_factory,
            patch(
                "features.observed_surface_features"
                ".build_observed_surface_features",
                side_effect=(
                    reference_surface,
                    query_surface,
                ),
            ) as surface_builder,
        ):
            results = main._extract_dino_inputs(
                config=config,
                prepared_views=(
                    reference_view,
                    query_view,
                ),
                output_roots=(
                    output_root,
                    output_root,
                ),
            )

        self.assertEqual(
            results,
            (
                (
                    reference_dense,
                    reference_surface,
                ),
                (
                    query_dense,
                    query_surface,
                ),
            ),
        )
        extractor_factory.assert_called_once_with(
            repository_path=(
                config.dinov3_repository
            ),
            checkpoint_path=(
                config.dinov3_checkpoint
            ),
            model_name=config.dinov3_model,
            device=config.device,
            target_long_side=(
                config.dinov3_target_long_side
            ),
            use_amp=config.dinov3_use_amp,
            save_dtype=config.dinov3_save_dtype,
        )
        self.assertEqual(
            extractor.extract.call_count,
            2,
        )
        self.assertEqual(
            surface_builder.call_count,
            2,
        )

    def test_cli_overrides_sam3_prompt(self) -> None:
        config = main.build_config(
            main.parse_args(
                [
                    "--sam3-prompt",
                    " cordless drill ",
                ]
            )
        )

        self.assertEqual(
            config.sam3_prompt,
            "cordless drill",
        )

    def test_cli_builds_multi_query_config(self) -> None:
        config = main.build_config(
            main.parse_args(
                [
                    "--query-scene-id",
                    "8",
                    "--query-image-ids",
                    "1",
                    "5",
                    "9",
                ]
            )
        )

        self.assertEqual(
            config.batch_query_image_ids,
            (1, 5, 9),
        )
        self.assertEqual(config.query.image_id, 1)
        self.assertEqual(
            [
                frame.image_id
                for frame in main._query_frames(config)
            ],
            [1, 5, 9],
        )
        self.assertEqual(
            config.output_root.name,
            (
                "batch_object_08_r000008_000000"
                "_ri00_qs000008_qi00"
                "_q000001-000005-000009"
                "_self_mesh"
            ),
        )

    def test_batch_output_root_distinguishes_query_sets(
        self,
    ) -> None:
        first = main.build_config(
            main.parse_args(
                [
                    "--query-image-ids",
                    "1",
                    "2",
                ]
            )
        )
        second = main.build_config(
            main.parse_args(
                [
                    "--query-image-ids",
                    "1",
                    "3",
                ]
            )
        )

        self.assertNotEqual(
            first.output_root,
            second.output_root,
        )

    def test_query_cli_modes_are_mutually_exclusive(
        self,
    ) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main.parse_args(
                    [
                        "--query-image-id",
                        "1",
                        "--query-image-ids",
                        "2",
                        "3",
                    ]
                )

    def test_main_dispatches_multi_query_mode(
        self,
    ) -> None:
        with patch.object(
            main,
            "run_batch_pipeline",
            return_value=0,
        ) as batch_runner:
            exit_code = main.main(
                [
                    "--query-image-ids",
                    "1",
                    "3",
                ]
            )

        self.assertEqual(exit_code, 0)
        batch_runner.assert_called_once()
        called_config = (
            batch_runner.call_args.args[0]
        )
        self.assertEqual(
            called_config.batch_query_image_ids,
            (1, 3),
        )

    def test_object_cli_modes_are_mutually_exclusive(
        self,
    ) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main.parse_args(
                    [
                        "--object-id",
                        "8",
                        "--object-ids",
                        "8",
                        "9",
                    ]
                )

    def test_main_runs_objects_in_separate_processes(
        self,
    ) -> None:
        completed = SimpleNamespace(returncode=0)

        with patch.object(
            main.subprocess,
            "run",
            side_effect=(
                SimpleNamespace(returncode=1),
                completed,
            ),
        ) as process_runner:
            with redirect_stderr(io.StringIO()):
                exit_code = main.main(
                    [
                        "--object-ids",
                        "8",
                        "9",
                        "--query-image-ids",
                        "1",
                        "3",
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            process_runner.call_count,
            2,
        )

        first_command = (
            process_runner.call_args_list[0]
            .args[0]
        )
        second_command = (
            process_runner.call_args_list[1]
            .args[0]
        )

        self.assertNotIn(
            "--object-ids",
            first_command,
        )
        self.assertEqual(
            first_command[
                first_command.index("--object-id")
                + 1
            ],
            "8",
        )
        self.assertEqual(
            second_command[
                second_command.index("--object-id")
                + 1
            ],
            "9",
        )
        self.assertEqual(
            second_command[
            second_command.index("--sam3-prompt")
                + 1
            ],
            "a small yellow rubber duck",
        )
        self.assertEqual(
            first_command[
                first_command.index(
                    "--reference-scene-id"
                )
                + 1
            ],
            "8",
        )
        self.assertEqual(
            second_command[
                second_command.index(
                    "--query-scene-id"
                )
                + 1
            ],
            "9",
        )
        self.assertIn(
            "--query-image-ids",
            first_command,
        )
        process_runner.assert_any_call(
            first_command,
            check=False,
        )

    def test_multi_object_ids_must_be_unique(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            main.main(
                [
                    "--object-ids",
                    "8",
                    "8",
                ]
            )

    def test_multi_object_rejects_unsupported_id(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported LINEMOD object ID",
        ):
            main.main(
                [
                    "--object-ids",
                    "8",
                    "99",
                ]
            )

    def test_multi_object_rejects_derived_cli_fields(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "--sam3-prompt",
        ):
            main.main(
                [
                    "--object-ids",
                    "8",
                    "9",
                    "--sam3-prompt",
                    "custom object",
                ]
            )

    def test_multi_object_nests_explicit_output_root(
        self,
    ) -> None:
        completed = SimpleNamespace(returncode=0)
        output_root = (
            TEST_DIRECTORY
            / "custom_multi_object_output"
        )

        with patch.object(
            main.subprocess,
            "run",
            return_value=completed,
        ) as process_runner:
            exit_code = main.main(
                [
                    "--object-ids",
                    "8",
                    "9",
                    "--output-root",
                    str(output_root),
                ]
            )

        self.assertEqual(exit_code, 0)

        for call, object_id in zip(
            process_runner.call_args_list,
            (8, 9),
            strict=True,
        ):
            command = call.args[0]
            output_option_index = max(
                index
                for index, argument in enumerate(
                    command
                )
                if argument == "--output-root"
            )
            self.assertEqual(
                Path(
                    command[
                        output_option_index + 1
                    ]
                ),
                output_root.resolve()
                / f"object_{object_id:02d}",
            )

    def test_multi_query_ids_must_be_unique(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            main.build_config(
                main.parse_args(
                    [
                        "--query-image-ids",
                        "2",
                        "2",
                    ]
                )
            )

    def test_cli_sets_foundationpose_worker_count(self) -> None:
        config = main.build_config(
            main.parse_args(
                [
                    "--foundationpose-workers",
                    "3",
                ]
            )
        )

        self.assertEqual(
            config.foundationpose_workers,
            3,
        )

    def test_no_argument_defaults_and_conda_sibling(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            envs_root = Path(temp_dir) / "envs"
            current_prefix = envs_root / "sam3_ros"
            instantmesh_python = (
                envs_root
                / "instantmesh_clean"
                / "bin"
                / "python"
            )

            current_prefix.mkdir(parents=True)
            instantmesh_python.parent.mkdir(
                parents=True
            )
            instantmesh_python.touch()

            with patch.dict(
                main.os.environ,
                {
                    "CONDA_PREFIX": str(
                        current_prefix
                    )
                },
            ):
                config = main.build_config(
                    main.parse_args([])
                )

            self.assertEqual(config.object_id, 8)
            self.assertEqual(
                config.object_name,
                "driller",
            )
            self.assertEqual(
                config.reference,
                main.FrameSpec(
                    scene_id=8,
                    image_id=0,
                ),
            )
            self.assertEqual(
                config.query,
                main.FrameSpec(
                    scene_id=8,
                    image_id=1,
                ),
            )
            self.assertEqual(
                config.instantmesh_python,
                instantmesh_python.resolve(),
            )

    def test_frame_spec_rejects_boolean_identifier(self) -> None:
        with self.assertRaises(ValueError):
            main.FrameSpec(
                scene_id=True,
                image_id=1,
            )


if __name__ == "__main__":
    unittest.main()
