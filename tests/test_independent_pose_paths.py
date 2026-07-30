from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from pose.bufferx_runner import (
    compose_bufferx_relative_pose,
)
from pose.independent_pose_paths import (
    compose_cross_mesh_relative_pose,
    save_independent_pose_path,
)


def _pose(
    angle_deg: float,
    translation: tuple[float, float, float],
) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result[:3, 3] = translation
    return result


class IndependentPosePathTest(
    unittest.TestCase
):
    def test_self_mesh_composition(self) -> None:
        reference_self = _pose(
            12.0,
            (0.1, -0.2, 0.8),
        )
        query_self = _pose(
            -7.0,
            (-0.3, 0.4, 1.1),
        )
        proxy_registration = _pose(
            5.0,
            (0.02, -0.01, 0.03),
        )
        expected = (
            query_self
            @ proxy_registration
            @ np.linalg.inv(reference_self)
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

        np.testing.assert_allclose(
            actual,
            expected,
            atol=1e-9,
        )

    def test_cross_mesh_composition_direction(
        self,
    ) -> None:
        reference_cross = _pose(
            18.0,
            (0.2, 0.1, 0.9),
        )
        query_cross = _pose(
            -11.0,
            (-0.1, 0.3, 1.0),
        )
        proxy_registration = _pose(
            6.0,
            (0.04, 0.02, -0.01),
        )
        expected = (
            reference_cross
            @ np.linalg.inv(proxy_registration)
            @ np.linalg.inv(query_cross)
        )

        actual = (
            compose_cross_mesh_relative_pose(
                reference_proxy_to_query_camera=(
                    reference_cross
                ),
                query_proxy_to_reference_camera=(
                    query_cross
                ),
                proxy_pose_query_from_reference=(
                    proxy_registration
                ),
            )
        )

        np.testing.assert_allclose(
            actual,
            expected,
            atol=1e-9,
        )

    def test_independent_summary_forbids_fusion(
        self,
    ) -> None:
        output_directory = (
            Path(__file__).parent
            / "_independent_pose_output"
        )
        output_directory.mkdir(exist_ok=False)

        try:
            result = save_independent_pose_path(
                method="cross_mesh",
                relative_pose_query_from_reference=(
                    np.eye(4)
                ),
                output_directory=output_directory,
                composition="B @ inv(M) @ inv(C)",
                sources={"example": "source"},
            )

            self.assertTrue(result.pose_path.is_file())
            with result.summary_path.open(
                encoding="utf-8",
            ) as file:
                payload = json.load(file)
        finally:
            for path in output_directory.iterdir():
                path.unlink()
            output_directory.rmdir()

        self.assertEqual(
            payload["method"],
            "cross_mesh",
        )
        self.assertIn(
            "no comparison",
            payload["selection_policy"],
        )


if __name__ == "__main__":
    unittest.main()
