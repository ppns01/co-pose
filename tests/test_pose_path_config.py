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
            1: "red toy monkey",
            5: "blue tin can",
            6: "pink cat figurine",
            10: "white egg carton",
            12: "blue paper hole punch",
            15: "cordless phone",
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

    def test_default_main_preserves_combined_mode(
        self,
    ) -> None:
        config = build_config(parse_args([]))

        self.assertEqual(
            config.pose_path,
            "combined",
        )

    def test_method_override_is_applied(
        self,
    ) -> None:
        config = build_config(
            parse_args(
                [
                    "--pose-path",
                    "cross_mesh",
                ]
            )
        )

        self.assertEqual(
            config.pose_path,
            "cross_mesh",
        )
        self.assertTrue(
            config.output_root.name.endswith(
                "_cross_mesh"
            )
        )


if __name__ == "__main__":
    unittest.main()
