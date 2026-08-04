from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from core.types import LoadedView, ViewInput
from mask_provider import load_existing_segmentation


TEST_DIRECTORY = Path(__file__).resolve().parent


class ExistingMaskAdapterTests(unittest.TestCase):
    def test_converts_bop_mask_to_segmentation_result(self) -> None:
        height = 6
        width = 8

        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            temp_root = Path(temp_dir)
            source_mask_path = temp_root / "source_mask.png"
            output_directory = temp_root / "segmentation"

            source_mask = np.zeros(
                (height, width),
                dtype=np.uint8,
            )
            source_mask[1:5, 2:7] = 255
            Image.fromarray(source_mask).save(source_mask_path)

            view = LoadedView(
                source=ViewInput(
                    name="reference",
                    rgb_path=temp_root / "rgb.png",
                    depth_path=temp_root / "depth.png",
                    intrinsics_path=(
                        temp_root / "scene_camera.json"
                    ),
                    object_name="driller",
                    scene_id=1,
                    image_id=0,
                    object_id=1,
                ),
                rgb=np.zeros(
                    (height, width, 3),
                    dtype=np.uint8,
                ),
                depth_m=np.ones(
                    (height, width),
                    dtype=np.float32,
                ),
                camera_matrix=np.array(
                    [
                        [500.0, 0.0, width / 2.0],
                        [0.0, 500.0, height / 2.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                depth_scale_to_m=0.001,
            )

            result = load_existing_segmentation(
                view=view,
                mask_path=source_mask_path,
                output_directory=output_directory,
            )

            self.assertEqual(
                result.mask_bool.dtype,
                np.bool_,
            )
            self.assertEqual(
                result.mask_rgb.shape,
                (height, width, 3),
            )
            self.assertTrue(result.mask_bool[2, 3])
            self.assertFalse(result.mask_bool[0, 0])
            self.assertTrue(result.mask_bool_path.is_file())
            self.assertTrue(result.mask_rgb_path.is_file())
            self.assertIsNotNone(result.metadata_path)
            self.assertEqual(result.source, "existing_mask")

            metadata = json.loads(
                result.metadata_path.read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(
                metadata["foreground_pixel_count"],
                20,
            )
            self.assertEqual(metadata["source"], "existing_mask")

    def test_rejects_mask_with_wrong_image_size(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            temp_root = Path(temp_dir)
            source_mask_path = temp_root / "source_mask.png"
            Image.fromarray(
                np.ones((2, 2), dtype=np.uint8) * 255
            ).save(source_mask_path)

            view = LoadedView(
                source=ViewInput(
                    name="query",
                    rgb_path=temp_root / "rgb.png",
                    depth_path=temp_root / "depth.png",
                    intrinsics_path=(
                        temp_root / "scene_camera.json"
                    ),
                    object_name="driller",
                    scene_id=1,
                    image_id=1,
                    object_id=1,
                ),
                rgb=np.zeros((3, 4, 3), dtype=np.uint8),
                depth_m=np.ones((3, 4), dtype=np.float32),
                camera_matrix=np.eye(3, dtype=np.float32),
                depth_scale_to_m=0.001,
            )

            with self.assertRaises(ValueError):
                load_existing_segmentation(
                    view=view,
                    mask_path=source_mask_path,
                    output_directory=(
                        temp_root / "segmentation"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
