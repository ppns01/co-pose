from __future__ import annotations

import unittest

import numpy as np


def _translation_pose(
    x: float,
    y: float,
    z: float,
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = (x, y, z)
    return pose


class RelativePoseFormulaTests(unittest.TestCase):
    def test_dgedi_local_proxy_pose_is_composed_with_self_poses(
        self,
    ) -> None:
        from pose.dgedi_runner import compose_dgedi_relative_pose

        reference_self = _translation_pose(1.0, 2.0, 3.0)
        query_self = _translation_pose(4.0, 6.0, 8.0)
        proxy_pose = _translation_pose(0.5, -0.25, 0.75)

        actual = compose_dgedi_relative_pose(
            reference_pose_camera_from_proxy=reference_self,
            query_pose_camera_from_proxy=query_self,
            proxy_pose_query_from_reference=proxy_pose,
        )
        expected = (
            query_self
            @ proxy_pose
            @ np.linalg.inv(reference_self)
        )

        np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
