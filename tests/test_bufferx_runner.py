from __future__ import annotations

import unittest

import numpy as np

from pose.bufferx_runner import (
    compose_bufferx_relative_pose,
)


def _pose(
    rotation: np.ndarray,
    translation: tuple[float, float, float],
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


class BufferXCompositionTests(unittest.TestCase):
    def test_identity_proxy_registration_reduces_to_self_poses(
        self,
    ) -> None:
        reference_self = _pose(
            np.eye(3),
            (0.1, -0.2, 0.3),
        )
        query_self = _pose(
            np.eye(3),
            (-0.4, 0.5, 0.6),
        )

        actual = compose_bufferx_relative_pose(
            reference_pose_camera_from_proxy=(
                reference_self
            ),
            query_pose_camera_from_proxy=query_self,
            proxy_pose_query_from_reference=np.eye(4),
        )
        expected = (
            query_self
            @ np.linalg.inv(reference_self)
        )

        np.testing.assert_allclose(
            actual,
            expected,
            atol=1e-12,
        )

    def test_proxy_registration_is_composed_between_self_poses(
        self,
    ) -> None:
        quarter_turn_z = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        reference_self = _pose(
            np.eye(3),
            (0.2, 0.0, 0.0),
        )
        query_self = _pose(
            quarter_turn_z,
            (0.0, 0.3, 0.0),
        )
        proxy_registration = _pose(
            quarter_turn_z.T,
            (0.01, -0.02, 0.03),
        )

        actual = compose_bufferx_relative_pose(
            reference_pose_camera_from_proxy=(
                reference_self
            ),
            query_pose_camera_from_proxy=query_self,
            proxy_pose_query_from_reference=(
                proxy_registration
            ),
        )
        expected = (
            query_self
            @ proxy_registration
            @ np.linalg.inv(reference_self)
        )

        np.testing.assert_allclose(
            actual,
            expected,
            atol=1e-12,
        )

    def test_invalid_proxy_transform_is_rejected(
        self,
    ) -> None:
        invalid = np.eye(4)
        invalid[0, 0] = 2.0

        with self.assertRaisesRegex(
            ValueError,
            "valid rotation",
        ):
            compose_bufferx_relative_pose(
                reference_pose_camera_from_proxy=(
                    np.eye(4)
                ),
                query_pose_camera_from_proxy=np.eye(4),
                proxy_pose_query_from_reference=invalid,
            )


if __name__ == "__main__":
    unittest.main()
