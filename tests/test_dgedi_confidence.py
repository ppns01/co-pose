from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose.dgedi_confidence import (
    PoseCandidateScore,
    compute_registration_confidence,
    select_pose_candidates,
)
from pose.dgedi_observation_validator import (
    DGeDiObservationValidationResult,
)


def _validation(root: Path) -> DGeDiObservationValidationResult:
    metrics = {
        "reference_cross": {
            "total_loss": 0.20,
            "mask_iou": 0.80,
            "depth_residual_normalized": 0.05,
            "free_space_loss": 0.10,
        },
        "query_cross": {
            "total_loss": 0.20,
            "mask_iou": 0.80,
            "depth_residual_normalized": 0.05,
            "free_space_loss": 0.10,
        },
    }
    return DGeDiObservationValidationResult(
        accepted=False,
        reasons=("legacy threshold miss",),
        summary_path=root / "validation.json",
        reference_render_path=root / "reference.npz",
        query_render_path=root / "query.npz",
        metrics=metrics,
    )


def _candidate_score(
    name: str,
    total_score: float,
    registration_confidence: float,
) -> PoseCandidateScore:
    return PoseCandidateScore(
        name=name,
        total_score=total_score,
        observation_loss=total_score,
        rotation_penalty=0.0,
        uncertainty_penalty=1.0 - registration_confidence,
        deformation_penalty=0.0,
        registration_confidence=registration_confidence,
    )


class DgediConfidenceTests(unittest.TestCase):
    def test_registration_evidence_becomes_continuous_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_diagnostics = root / "reference.json"
            query_diagnostics = root / "query.json"
            for path in (reference_diagnostics, query_diagnostics):
                path.write_text(
                    json.dumps({"point_count_saved": 1000}),
                    encoding="utf-8",
                )
            metadata_path = root / "registration.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "normalization_diameter_m": 0.1,
                        "reference_surface_diagnostics": str(
                            reference_diagnostics
                        ),
                        "query_surface_diagnostics": str(query_diagnostics),
                        "icp": {
                            "fitness": 0.5,
                            "inlier_rmse_m": 0.001,
                            "correspondence_count": 100,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = compute_registration_confidence(
                registration_metadata_path=metadata_path,
                observation_validation=_validation(root),
                output_path=root / "confidence.json",
                component_weights={
                    "correspondence": 1.0,
                    "inlier": 1.0,
                    "rmse": 1.0,
                    "mask": 1.0,
                    "depth": 1.0,
                    "free_space": 1.0,
                },
                target_correspondence_fraction=0.10,
                good_inlier_fitness=0.50,
                maximum_normalized_rmse=0.05,
                maximum_normalized_depth_residual=0.20,
            )

            self.assertAlmostEqual(result.components["correspondence"], 1.0)
            self.assertAlmostEqual(result.components["inlier"], 1.0)
            self.assertAlmostEqual(result.components["rmse"], 0.8)
            self.assertAlmostEqual(result.components["mask"], 0.8)
            self.assertAlmostEqual(result.components["depth"], 0.75)
            self.assertAlmostEqual(result.components["free_space"], 0.9)
            self.assertAlmostEqual(result.confidence, 0.875)
            payload = json.loads(
                result.summary_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["method"],
                "continuous_dgedi_registration_confidence",
            )

    def test_g0_depth_or_free_space_failure_collapses_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_diagnostics = root / "reference.json"
            query_diagnostics = root / "query.json"
            for path in (reference_diagnostics, query_diagnostics):
                path.write_text(
                    json.dumps({"point_count_saved": 1000}),
                    encoding="utf-8",
                )
            metadata_path = root / "registration.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "normalization_diameter_m": 0.1,
                        "reference_surface_diagnostics": str(
                            reference_diagnostics
                        ),
                        "query_surface_diagnostics": str(query_diagnostics),
                        "icp": {
                            "fitness": 0.5,
                            "inlier_rmse_m": 0.001,
                            "correspondence_count": 100,
                        },
                    }
                ),
                encoding="utf-8",
            )

            for failed_component in ("depth", "free_space"):
                validation = _validation(root)
                if failed_component == "depth":
                    for view in ("reference_cross", "query_cross"):
                        validation.metrics[view][
                            "depth_residual_normalized"
                        ] = 0.20
                else:
                    for view in ("reference_cross", "query_cross"):
                        validation.metrics[view]["free_space_loss"] = 1.0

                result = compute_registration_confidence(
                    registration_metadata_path=metadata_path,
                    observation_validation=validation,
                    output_path=root / f"confidence_{failed_component}.json",
                    component_weights={
                        "correspondence": 1.0,
                        "inlier": 1.0,
                        "rmse": 1.0,
                        "mask": 1.0,
                        "depth": 1.0,
                        "free_space": 1.0,
                    },
                    target_correspondence_fraction=0.10,
                    good_inlier_fitness=0.50,
                    maximum_normalized_rmse=0.05,
                    maximum_normalized_depth_residual=0.20,
                    apply_depth_free_space_gate=True,
                )

                self.assertEqual(result.components[failed_component], 0.0)
                self.assertEqual(result.confidence, 0.0)
                payload = json.loads(
                    result.summary_path.read_text(encoding="utf-8")
                )
                self.assertTrue(
                    payload["aggregation"][
                        "depth_free_space_gate_applied"
                    ]
                )
                self.assertEqual(
                    payload["aggregation"]["depth_free_space_gate"],
                    0.0,
                )

            near_zero_validation = _validation(root)
            for view in ("reference_cross", "query_cross"):
                near_zero_validation.metrics[view][
                    "depth_residual_normalized"
                ] = 0.198
            near_zero_result = compute_registration_confidence(
                registration_metadata_path=metadata_path,
                observation_validation=near_zero_validation,
                output_path=root / "confidence_near_zero_depth.json",
                component_weights={
                    "correspondence": 1.0,
                    "inlier": 1.0,
                    "rmse": 1.0,
                    "mask": 1.0,
                    "depth": 1.0,
                    "free_space": 1.0,
                },
                target_correspondence_fraction=0.10,
                good_inlier_fitness=0.50,
                maximum_normalized_rmse=0.05,
                maximum_normalized_depth_residual=0.20,
                apply_depth_free_space_gate=True,
            )
            self.assertAlmostEqual(
                near_zero_result.components["depth"],
                0.01,
            )
            self.assertGreater(near_zero_result.confidence, 0.0)
            self.assertLess(near_zero_result.confidence, 0.10)

    def test_lower_score_is_selected_without_quality_reject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_pose = np.eye(4, dtype=np.float64)
            refined_pose = np.eye(4, dtype=np.float64)
            refined_pose[0, 3] = 0.01
            selection = select_pose_candidates(
                baseline_pose_query_from_reference=baseline_pose,
                baseline_score=_candidate_score("H0_baseline", 0.20, 0.8),
                refined_pose_query_from_reference=refined_pose,
                refined_score=_candidate_score(
                    "H1_confidence_shared_D",
                    0.10,
                    0.7,
                ),
                near_tie_margin=0.01,
                output_path=root / "selection.json",
            )

            self.assertEqual(selection.selected_name, "H1_confidence_shared_D")
            np.testing.assert_allclose(
                selection.selected_pose_query_from_reference,
                refined_pose,
            )
            payload = json.loads(
                selection.summary_path.read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "SELECTED")
            self.assertIsNotNone(payload["baseline_pose_query_from_reference"])
            self.assertIsNotNone(payload["refined_pose_query_from_reference"])

    def test_semantic_g1_failure_keeps_h0_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection = select_pose_candidates(
                baseline_pose_query_from_reference=np.eye(4),
                baseline_score=_candidate_score("H0_baseline", 0.20, 0.8),
                refined_pose_query_from_reference=None,
                refined_score=None,
                near_tie_margin=0.01,
                output_path=Path(directory) / "selection.json",
                refined_failure="DGeDiSemanticFailure: insufficient overlap",
            )

            self.assertEqual(selection.selected_name, "H0_baseline")
            self.assertAlmostEqual(selection.confidence, 0.8)
            self.assertFalse(selection.near_tie)


if __name__ == "__main__":
    unittest.main()
