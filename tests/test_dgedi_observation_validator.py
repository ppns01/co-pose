from __future__ import annotations

import unittest

from pose.alignment_scorer import AlignmentScoreResult
from pose.dgedi_observation_validator import _view_rejection_reasons


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


class DgediObservationValidatorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
