from __future__ import annotations

import unittest

import numpy as np

from scale.local_scale_refiner import (
    estimate_scale_translation_fixed_rotation,
)


class VisibleScaleRefinerTests(unittest.TestCase):
    def test_fixed_rotation_fit_recovers_scale_with_outliers(
        self,
    ) -> None:
        random = np.random.default_rng(42)
        proxy_points = random.normal(
            size=(1000, 3)
        )
        angle = np.deg2rad(25.0)
        rotation = np.array(
            [
                [
                    np.cos(angle),
                    -np.sin(angle),
                    0.0,
                ],
                [
                    np.sin(angle),
                    np.cos(angle),
                    0.0,
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected_scale = 0.237
        expected_translation = np.array(
            [0.04, -0.03, 0.72],
            dtype=np.float64,
        )
        observed_points = (
            expected_scale
            * (rotation @ proxy_points.T).T
            + expected_translation
        )
        observed_points += random.normal(
            scale=0.001,
            size=observed_points.shape,
        )
        observed_points[:40] += random.normal(
            scale=0.08,
            size=(40, 3),
        )

        (
            scale,
            translation,
            residuals,
            weights,
        ) = estimate_scale_translation_fixed_rotation(
            proxy_points_object_unscaled=(
                proxy_points
            ),
            observed_points_camera=observed_points,
            rotation_object_to_camera=rotation,
            irls_iterations=3,
            huber_delta_m=0.01,
        )

        self.assertAlmostEqual(
            scale,
            expected_scale,
            delta=0.001,
        )
        np.testing.assert_allclose(
            translation,
            expected_translation,
            atol=0.002,
        )
        self.assertEqual(
            residuals.shape,
            observed_points.shape,
        )
        self.assertEqual(
            weights.shape,
            (observed_points.shape[0],),
        )
        self.assertLess(
            float(np.median(weights[:40])),
            float(np.median(weights[40:])),
        )

    def test_fixed_rotation_fit_rejects_degenerate_points(
        self,
    ) -> None:
        points = np.zeros(
            (10, 3),
            dtype=np.float64,
        )

        with self.assertRaisesRegex(
            ValueError,
            "분산",
        ):
            estimate_scale_translation_fixed_rotation(
                proxy_points_object_unscaled=points,
                observed_points_camera=points,
                rotation_object_to_camera=np.eye(3),
            )


if __name__ == "__main__":
    unittest.main()
