from __future__ import annotations

import json
import unittest
from pathlib import Path

from linemod_all_runner import (
    _remove_single_value_option,
    build_object_command,
    load_object_image_ids,
)


class LinemodAllRunnerTest(unittest.TestCase):
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

    def test_builds_sequential_object_command(
        self,
    ) -> None:
        command = build_object_command(
            python_executable="/env/bin/python",
            entrypoint_path=Path(
                "/project/main_cross_mesh.py"
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
                        "/project/main_cross_mesh.py"
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


if __name__ == "__main__":
    unittest.main()
