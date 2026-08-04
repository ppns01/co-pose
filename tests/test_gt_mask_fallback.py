from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main


class GTMaskFallbackTests(unittest.TestCase):
    def test_uses_gt_mask_and_records_provenance_when_sam3_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rgb_path = root / "rgb.png"
            depth_path = root / "depth.png"
            mask_path = root / "mask.png"
            selected_input = SimpleNamespace(
                rgb_path=rgb_path,
                depth_path=depth_path,
                mask_path=mask_path,
            )
            loaded_view = SimpleNamespace(
                source=SimpleNamespace(
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                )
            )
            fallback_segmentation = SimpleNamespace(source="gt_fallback")
            prepared_view = SimpleNamespace(
                segmentation=fallback_segmentation
            )
            config = replace(
                main.build_config(main.parse_args([])),
                output_root=root / "output",
                gt_mask_fallback_on_sam3_failure=True,
            )

            with (
                patch(
                    "input_selector.build_view_input",
                    return_value=selected_input,
                ),
                patch(
                    "dataset_io.linemod_loader.load_linemod_view",
                    return_value=loaded_view,
                ),
                patch(
                    "mask_provider.generate_sam3_segmentation",
                    side_effect=RuntimeError("no SAM3 mask"),
                ),
                patch(
                    "mask_provider.load_existing_segmentation",
                    return_value=fallback_segmentation,
                ) as load_gt,
                patch(
                    "preprocessing.mask_processing.prepare_masked_view",
                    return_value=prepared_view,
                ),
            ):
                actual = main._prepare_linemod_view(
                    config=config,
                    view_name="query",
                    frame=main.FrameSpec(8, 1, 0),
                )

            self.assertIs(actual, prepared_view)
            self.assertEqual(load_gt.call_args.kwargs["mask_path"], mask_path)
            self.assertEqual(
                load_gt.call_args.kwargs["source"],
                "gt_fallback",
            )
            selection = json.loads(
                (
                    config.output_root
                    / "views"
                    / "query"
                    / "segmentation_selection.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(selection["selected_source"], "gt_fallback")
            self.assertTrue(selection["gt_assisted"])
            self.assertIn("no SAM3 mask", selection["sam3_error"])

    def test_does_not_use_gt_when_fallback_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_paths = SimpleNamespace(
                rgb_path=root / "rgb.png",
                depth_path=root / "depth.png",
                mask_path=root / "mask.png",
            )
            loaded_view = SimpleNamespace(
                source=SimpleNamespace(
                    rgb_path=input_paths.rgb_path,
                    depth_path=input_paths.depth_path,
                )
            )
            config = replace(
                main.build_config(main.parse_args([])),
                output_root=root / "output",
                gt_mask_fallback_on_sam3_failure=False,
            )

            with (
                patch(
                    "input_selector.build_view_input",
                    return_value=input_paths,
                ),
                patch(
                    "dataset_io.linemod_loader.load_linemod_view",
                    return_value=loaded_view,
                ),
                patch(
                    "mask_provider.generate_sam3_segmentation",
                    side_effect=RuntimeError("no SAM3 mask"),
                ),
                patch(
                    "mask_provider.load_existing_segmentation"
                ) as load_gt,
            ):
                with self.assertRaisesRegex(RuntimeError, "no SAM3 mask"):
                    main._prepare_linemod_view(
                        config=config,
                        view_name="query",
                        frame=main.FrameSpec(8, 1, 0),
                    )

            load_gt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
