from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose.independent_pose_paths import (
    save_independent_pose_path,
    save_rejected_pose_path,
)


class IndependentPosePathTest(
    unittest.TestCase
):
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
                method="self_mesh",
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
            "self_mesh",
        )
        self.assertIn(
            "no comparison",
            payload["selection_policy"],
        )

    def test_rejected_summary_uses_canonical_path_without_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            summary_path = save_rejected_pose_path(
                method="self_mesh",
                output_directory=output_directory,
                reason="registration failed",
                sources={"diagnostics": "failure.json"},
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(
                summary_path,
                output_directory.resolve() / "final_selection.json",
            )
            self.assertFalse(
                (output_directory / "final_relative_pose.npy").exists()
            )
            self.assertEqual(payload["status"], "REJECT")
            self.assertIsNone(payload["relative_pose_query_from_reference"])
            self.assertIn("no same-proxy", payload["selection_policy"])

    def test_continuous_selection_policy_can_be_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = save_independent_pose_path(
                method="self_mesh",
                relative_pose_query_from_reference=np.eye(4),
                output_directory=Path(temporary_directory),
                composition="compare H0 and H1",
                sources={"selected_candidate": "H0_baseline"},
                selection_policy=(
                    "continuous H0/H1 score comparison; no hard reject"
                ),
            )
            payload = json.loads(
                result.summary_path.read_text(encoding="utf-8")
            )

            self.assertTrue(payload["pose_accepted"])
            self.assertIn("continuous H0/H1", payload["selection_policy"])

    def test_rejected_summary_can_preserve_unaccepted_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            pose = np.eye(4, dtype=np.float64)
            pose[0, 3] = 0.1

            summary_path = save_rejected_pose_path(
                method="self_mesh",
                output_directory=output_directory,
                reason="observation validation failed",
                sources={"diagnostics": "validation.json"},
                relative_pose_query_from_reference=pose,
            )

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            rejected_pose_path = (
                output_directory / "rejected_relative_pose.npy"
            )
            self.assertTrue(rejected_pose_path.is_file())
            np.testing.assert_allclose(
                np.load(rejected_pose_path, allow_pickle=False),
                pose,
            )
            self.assertFalse(payload["pose_accepted"])
            np.testing.assert_allclose(
                payload["relative_pose_query_from_reference"],
                pose,
            )


if __name__ == "__main__":
    unittest.main()
