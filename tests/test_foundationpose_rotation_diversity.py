from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pose.foundationpose_runner import FoundationPoseRunner


def _pose_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return pose


class FoundationPoseRotationDiversityTests(unittest.TestCase):
    def test_selects_separated_rotations_before_same_basin_fill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = FoundationPoseRunner(
                repository_path=Path(directory),
                output_root=Path(directory) / "out",
                top_k=3,
                rotation_diversity_threshold_deg=40.0,
                device="cuda:0",
            )
            poses = np.stack(
                [_pose_z(value) for value in (0.0, 5.0, 10.0, 90.0, 180.0)]
            )
            runner._estimator = SimpleNamespace(
                poses=poses,
                scores=np.asarray((5.0, 4.0, 3.0, 2.0, 1.0)),
                get_tf_to_centered_mesh=lambda: np.eye(4, dtype=np.float64),
            )
            hypotheses = runner._extract_hypotheses(
                returned_best_pose=poses[0]
            )

        self.assertEqual([item.rank for item in hypotheses], [0, 1, 2])
        self.assertEqual([item.score for item in hypotheses], [5.0, 2.0, 1.0])
        self.assertEqual(
            [item.source_score_rank for item in hypotheses],
            [0, 3, 4],
        )


if __name__ == "__main__":
    unittest.main()
