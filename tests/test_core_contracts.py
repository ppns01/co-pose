from __future__ import annotations

import unittest
from pathlib import Path

from core.types import MeshGenerationResult, ViewInput


class ViewInputContractTests(unittest.TestCase):
    def test_accepts_optional_object_id_and_trims_name(self) -> None:
        view = ViewInput(
            name="reference",
            rgb_path=Path("rgb.png"),
            depth_path=Path("depth.png"),
            intrinsics_path=Path("scene_camera.json"),
            object_name="  driller  ",
            scene_id=1,
            image_id=2,
            object_id=None,
        )

        self.assertEqual(view.object_name, "driller")
        self.assertIsNone(view.object_id)

    def test_rejects_invalid_identifiers(self) -> None:
        common = {
            "name": "query",
            "rgb_path": Path("rgb.png"),
            "depth_path": Path("depth.png"),
            "intrinsics_path": Path("scene_camera.json"),
            "object_name": "driller",
            "scene_id": 1,
            "image_id": 2,
            "object_id": 1,
        }

        for field_name, bad_value in (
            ("scene_id", -1),
            ("image_id", True),
            ("object_id", -1),
        ):
            with self.subTest(field_name=field_name):
                values = dict(common)
                values[field_name] = bad_value

                with self.assertRaises(ValueError):
                    ViewInput(**values)


class MeshGenerationResultContractTests(unittest.TestCase):
    def test_primary_output_and_legacy_alias_match(self) -> None:
        result = MeshGenerationResult(
            generator_name="instantmesh",
            output_dir=Path("outputs"),
            primary_output_path=Path("outputs/proxy.obj"),
            artifact_paths=(Path("outputs/proxy.obj"),),
        )

        self.assertEqual(
            result.mesh_path,
            result.primary_output_path,
        )

    def test_rejects_unsupported_mesh_suffix(self) -> None:
        with self.assertRaises(ValueError):
            MeshGenerationResult(
                generator_name="instantmesh",
                output_dir=Path("outputs"),
                primary_output_path=Path("outputs/proxy.txt"),
            )


if __name__ == "__main__":
    unittest.main()
