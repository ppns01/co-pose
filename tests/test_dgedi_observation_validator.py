from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pose.alignment_scorer import (
    AlignmentScoreResult,
    AlignmentScoreWeights,
)
from pose.dgedi_observation_validator import (
    DGeDiObservationValidationResult,
    _view_rejection_reasons,
    select_dgedi_with_foundationpose_fallback,
    validate_dgedi_against_observations,
)


def _score(
    *,
    total_loss: float,
    mask_iou: float,
    depth_ratio: float,
    depth_overlap: int,
    rendered_pixels: int = 1000,
) -> AlignmentScoreResult:
    return AlignmentScoreResult(
        total_loss=total_loss,
        mask_loss=1.0 - mask_iou,
        depth_loss=min(depth_ratio, 1.0),
        free_space_loss=0.0,
        boundary_loss=0.0,
        mask_iou=mask_iou,
        depth_residual_m=depth_ratio * 0.1,
        depth_residual_normalized=depth_ratio,
        overlap_pixel_count=depth_overlap,
        valid_depth_overlap_count=depth_overlap,
        rendered_pixel_count=rendered_pixels,
        free_space_violation_count=0,
    )


def _validation(
    mean_cross_loss: float,
    *,
    accepted: bool = True,
) -> DGeDiObservationValidationResult:
    return DGeDiObservationValidationResult(
        accepted=accepted,
        reasons=() if accepted else ("observation gate failed",),
        summary_path=Path("summary.json"),
        reference_render_path=Path("reference.npz"),
        query_render_path=Path("query.npz"),
        metrics={
            "reference_cross": {"total_loss": mean_cross_loss},
            "query_cross": {"total_loss": mean_cross_loss},
        },
    )


def _pose(rotation_deg: float = 0.0, translation_x: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(rotation_deg)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = (
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
    pose[0, 3] = translation_x
    return pose


class DgediObservationValidatorTests(unittest.TestCase):
    def test_diagnostic_mode_uses_one_fixed_scale_without_rejecting(self) -> None:
        captured_scales: list[float] = []

        def fake_score_alignment(**kwargs: object) -> AlignmentScoreResult:
            captured_scales.append(float(kwargs["object_scale_m"]))
            return _score(
                total_loss=0.90,
                mask_iou=0.10,
                depth_ratio=0.50,
                depth_overlap=100,
            )

        def fake_raycast(**kwargs: object) -> dict[str, np.ndarray]:
            shape = (
                int(kwargs["image_height"]),
                int(kwargs["image_width"]),
            )
            return {
                "rendered_mask": np.ones(shape, dtype=bool),
                "rendered_depth": np.ones(shape, dtype=np.float32),
            }

        points = np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            dtype=np.float64,
        )
        triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory, patch(
            "pose.dgedi_observation_validator._load_mesh",
            return_value=(points, triangles),
        ), patch(
            "pose.dgedi_observation_validator.score_alignment",
            side_effect=fake_score_alignment,
        ):
            result = validate_dgedi_against_observations(
                reference_mesh_path=Path("reference.obj"),
                query_mesh_path=Path("query.obj"),
                relative_pose_query_from_reference=np.eye(4),
                reference_camera_k=np.eye(3),
                query_camera_k=np.eye(3),
                reference_mask_bool=np.ones((10, 10), dtype=bool),
                query_mask_bool=np.ones((10, 10), dtype=bool),
                reference_depth_m=np.ones((10, 10), dtype=np.float32),
                query_depth_m=np.ones((10, 10), dtype=np.float32),
                output_directory=Path(directory),
                weights=AlignmentScoreWeights(),
                depth_trim_quantile=0.9,
                minimum_depth_overlap_pixels=50,
                free_space_absolute_tolerance_m=0.005,
                free_space_relative_tolerance=0.02,
                object_scale_m=0.2,
                diagnostic_only=True,
                raycast_function=fake_raycast,
            )

            self.assertFalse(result.accepted)
            self.assertEqual(captured_scales, [0.2, 0.2, 0.2, 0.2])
            payload = json.loads(
                result.summary_path.read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "EVALUATED")
            self.assertEqual(payload["decision_mode"], "diagnostic_only")
            self.assertFalse(payload["legacy_gate_would_accept"])

    def test_bidirectional_view_gate_accepts_close_cross_render(self) -> None:
        reasons = _view_rejection_reasons(
            view_name="query",
            baseline=_score(
                total_loss=0.10,
                mask_iou=0.85,
                depth_ratio=0.03,
                depth_overlap=900,
            ),
            cross=_score(
                total_loss=0.22,
                mask_iou=0.72,
                depth_ratio=0.08,
                depth_overlap=700,
            ),
            minimum_depth_overlap_pixels=50,
        )
        self.assertEqual(reasons, [])

    def test_bidirectional_view_gate_rejects_empty_wrong_branch(self) -> None:
        reasons = _view_rejection_reasons(
            view_name="reference",
            baseline=_score(
                total_loss=0.10,
                mask_iou=0.85,
                depth_ratio=0.03,
                depth_overlap=900,
            ),
            cross=_score(
                total_loss=0.90,
                mask_iou=0.05,
                depth_ratio=1.0,
                depth_overlap=0,
                rendered_pixels=0,
            ),
            minimum_depth_overlap_pixels=50,
        )
        self.assertTrue(any("empty" in reason for reason in reasons))
        self.assertTrue(any("depth overlap" in reason for reason in reasons))
        self.assertTrue(any("mask IoU" in reason for reason in reasons))

    def test_view_gate_uses_configurable_mask_floor(self) -> None:
        reasons = _view_rejection_reasons(
            view_name="query",
            baseline=_score(
                total_loss=0.10,
                mask_iou=0.85,
                depth_ratio=0.03,
                depth_overlap=900,
            ),
            cross=_score(
                total_loss=0.20,
                mask_iou=0.72,
                depth_ratio=0.05,
                depth_overlap=700,
            ),
            minimum_depth_overlap_pixels=50,
            minimum_mask_iou=0.75,
        )
        self.assertTrue(any("mask IoU" in reason for reason in reasons))

    def test_large_unhelpful_dgedi_correction_falls_back(self) -> None:
        decision = select_dgedi_with_foundationpose_fallback(
            dgedi_pose_query_from_reference=_pose(8.0, 0.03),
            foundationpose_pose_query_from_reference=_pose(),
            dgedi_validation=_validation(0.20),
            foundationpose_validation=_validation(0.18),
        )

        self.assertTrue(decision.used_foundationpose_fallback)
        self.assertEqual(
            decision.selected_method,
            "foundationpose_only_fallback",
        )
        np.testing.assert_allclose(
            decision.selected_pose_query_from_reference,
            np.eye(4),
        )

    def test_large_correction_is_kept_when_loss_clearly_improves(self) -> None:
        dgedi_pose = _pose(8.0, 0.03)
        decision = select_dgedi_with_foundationpose_fallback(
            dgedi_pose_query_from_reference=dgedi_pose,
            foundationpose_pose_query_from_reference=_pose(),
            dgedi_validation=_validation(0.15),
            foundationpose_validation=_validation(0.20),
        )

        self.assertFalse(decision.used_foundationpose_fallback)
        self.assertTrue(decision.observation_improved_clearly)
        np.testing.assert_allclose(
            decision.selected_pose_query_from_reference,
            dgedi_pose,
        )

    def test_exact_two_centimeter_boundary_triggers_fallback(self) -> None:
        decision = select_dgedi_with_foundationpose_fallback(
            dgedi_pose_query_from_reference=_pose(0.0, 0.020),
            foundationpose_pose_query_from_reference=_pose(),
            dgedi_validation=_validation(0.20),
            foundationpose_validation=_validation(0.20),
        )

        self.assertTrue(decision.used_foundationpose_fallback)

    def test_rejected_dgedi_validation_cannot_override_fallback(self) -> None:
        decision = select_dgedi_with_foundationpose_fallback(
            dgedi_pose_query_from_reference=_pose(8.0, 0.03),
            foundationpose_pose_query_from_reference=_pose(),
            dgedi_validation=_validation(0.10, accepted=False),
            foundationpose_validation=_validation(0.20),
        )

        self.assertTrue(decision.used_foundationpose_fallback)
        self.assertFalse(decision.observation_improved_clearly)
        self.assertEqual(
            decision.selected_method,
            "foundationpose_only_fallback",
        )


if __name__ == "__main__":
    unittest.main()
