from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from pose.foundationpose_runner import (
    FoundationPoseRunner,
)


class FoundationPoseInputTests(unittest.TestCase):
    def test_camera_matrix_uses_foundationpose_float64_boundary(
        self,
    ) -> None:
        camera_matrix = np.array(
            [
                [572.4, 0.0, 325.3],
                [0.0, 573.5, 242.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        prepared_view = SimpleNamespace(
            view=SimpleNamespace(
                rgb=np.zeros(
                    (3, 4, 3),
                    dtype=np.uint8,
                ),
                depth_m=np.ones(
                    (3, 4),
                    dtype=np.float32,
                ),
                camera_matrix=camera_matrix,
            ),
            segmentation=SimpleNamespace(
                mask_bool=np.ones(
                    (3, 4),
                    dtype=np.bool_,
                ),
            ),
        )

        (
            rgb,
            depth_m,
            prepared_camera_matrix,
            mask_uint8,
        ) = FoundationPoseRunner._prepare_inputs(
            prepared_view,
        )

        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(depth_m.dtype, np.float32)
        self.assertEqual(
            prepared_camera_matrix.dtype,
            np.float64,
        )
        self.assertEqual(mask_uint8.dtype, np.uint8)
        self.assertTrue(
            prepared_camera_matrix.flags.c_contiguous
        )
        np.testing.assert_array_equal(
            prepared_camera_matrix,
            camera_matrix.astype(np.float64),
        )


if __name__ == "__main__":
    unittest.main()
