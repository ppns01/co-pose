from __future__ import annotations

import unittest

from main import (
    LINEMOD_OBJECT_METADATA,
    build_config,
    parse_args,
)


class PosePathConfigTest(unittest.TestCase):
    def test_sam3_prompts_use_visual_aliases(
        self,
    ) -> None:
        expected_prompts = {
            1: "brown toy",
            5: "white watering can",
            6: "pink cat figurine",
            10: "plastic case",
            12: "blue paper hole punch",
            15: "phone handset",
        }

        self.assertEqual(
            {
                object_id: (
                    LINEMOD_OBJECT_METADATA[
                        object_id
                    ][1]
                )
                for object_id
                in expected_prompts
            },
            expected_prompts,
        )

    def test_default_main_uses_self_mesh_mode(
        self,
    ) -> None:
        config = build_config(parse_args([]))

        self.assertEqual(
            config.pose_path,
            "self_mesh",
        )

    def test_method_override_is_applied(
        self,
    ) -> None:
        config = build_config(
            parse_args(
                [
                    "--pose-path",
                    "self_mesh",
                ]
            )
        )

        self.assertEqual(
            config.pose_path,
            "self_mesh",
        )
        self.assertTrue(
            config.output_root.name.endswith(
                "_self_mesh"
            )
        )

    def test_removed_cross_mesh_path_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--pose-path",
                    "cross_mesh",
                ]
            )


if __name__ == "__main__":
    unittest.main()
