from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh

from pose.alignment_scorer import AlignmentScoreResult, AlignmentScoreWeights
from scale.independent_axis_scale_refiner import (
    _depth_observability,
    _score_rendered_candidates,
    build_axis_scale_factor_grid,
    centered_axis_scale_affine,
    centered_camera_axis_scale_affine,
    fit_independent_axis_scale,
)
from scale.mesh_scaler import ScaledMeshCandidate


def _score(loss: float) -> AlignmentScoreResult:
    return AlignmentScoreResult(
        total_loss=loss,
        mask_loss=loss,
        depth_loss=loss,
        free_space_loss=0.0,
        boundary_loss=0.0,
        mask_iou=1.0 - loss,
        depth_residual_m=loss * 0.01,
        depth_residual_normalized=loss,
        overlap_pixel_count=100,
        valid_depth_overlap_count=100,
        rendered_pixel_count=100,
        free_space_violation_count=0,
    )


class _Renderer:
    def __init__(self) -> None:
        self.transforms: np.ndarray | None = None

    def render_affine_depth_mask_batch(
        self,
        *,
        mesh_path: Path,
        transforms_camera_from_proxy: np.ndarray,
        camera_matrix: np.ndarray,
        image_height: int,
        image_width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del mesh_path, camera_matrix
        self.transforms = transforms_camera_from_proxy.copy()
        count = transforms_camera_from_proxy.shape[0]
        masks = np.ones((count, image_height, image_width), dtype=bool)
        depths = np.ones_like(masks, dtype=np.float32)
        return masks, depths


class IndependentAxisScaleRefinerTests(unittest.TestCase):
    @staticmethod
    def _candidate(root: Path) -> ScaledMeshCandidate:
        mesh_path = root / "mesh.obj"
        metadata_path = root / "mesh.json"
        mesh = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
        mesh.apply_translation((0.03, -0.04, 0.05))
        mesh.export(mesh_path)
        metadata_path.write_text("{}\n", encoding="utf-8")
        return ScaledMeshCandidate(
            candidate_index=4,
            scale_m=float(np.linalg.norm((0.1, 0.2, 0.3))),
            normalized_mesh_path=mesh_path,
            scaled_mesh_path=mesh_path,
            metadata_path=metadata_path,
            scale_transform=np.eye(4, dtype=np.float64),
        )

    @staticmethod
    def _view() -> SimpleNamespace:
        mask = np.ones((8, 8), dtype=bool)
        return SimpleNamespace(
            view=SimpleNamespace(
                source=SimpleNamespace(name="reference"),
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                depth_m=np.ones((8, 8), dtype=np.float32),
                camera_matrix=np.eye(3, dtype=np.float64),
            ),
            segmentation=SimpleNamespace(mask_bool=mask),
        )

    def test_grid_is_xyz_cartesian_and_identity_first(self) -> None:
        grid = build_axis_scale_factor_grid(
            minimum_factor=0.85,
            maximum_factor=1.15,
            grid_step_count=5,
        )
        self.assertEqual(len(grid), 125)
        self.assertEqual(grid[0], (1.0, 1.0, 1.0))
        self.assertIn((0.85, 1.0, 1.15), grid)

    def test_centered_scale_preserves_local_center(self) -> None:
        center = np.asarray((3.0, -2.0, 1.0), dtype=np.float64)
        affine = centered_axis_scale_affine(
            scale_factors=(0.8, 1.1, 1.2),
            center_m=tuple(center),
        )
        transformed = affine[:3, :3] @ center + affine[:3, 3]
        np.testing.assert_allclose(transformed, center)

    def test_camera_axis_affine_is_rotated_back_to_proxy_frame(self) -> None:
        rotation = np.asarray(
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        affine = centered_camera_axis_scale_affine(
            scale_factors_uvd=(2.0, 1.0, 1.0),
            center_proxy_m=(0.0, 0.0, 0.0),
            rotation_camera_from_proxy=rotation,
        )
        np.testing.assert_allclose(
            affine[:3, :3],
            np.diag((1.0, 2.0, 1.0)),
            atol=1e-12,
        )

    def test_candidate_scores_keep_pair_object_scale_fixed(self) -> None:
        masks = np.ones((3, 8, 8), dtype=bool)
        depths = np.ones((3, 8, 8), dtype=np.float32)
        with patch(
            "scale.independent_axis_scale_refiner.score_alignment",
            return_value=_score(0.1),
        ) as score_alignment_mock:
            results = _score_rendered_candidates(
                prepared_view=self._view(),
                rendered_masks=masks,
                rendered_depth_m=depths,
                object_scale_m=0.15,
                weights=AlignmentScoreWeights(),
                depth_trim_quantile=0.9,
                minimum_depth_overlap_pixels=1,
                free_space_absolute_tolerance_m=0.005,
                free_space_relative_tolerance=0.02,
            )
        self.assertEqual(len(results), 3)
        self.assertEqual(
            [
                call.kwargs["object_scale_m"]
                for call in score_alignment_mock.call_args_list
            ],
            [0.15, 0.15, 0.15],
        )

    def test_depth_observability_uses_trimmed_span_and_valid_fraction(self) -> None:
        prepared = self._view()
        prepared.view.depth_m[:] = np.linspace(
            1.0, 1.1, prepared.view.depth_m.size, dtype=np.float32
        ).reshape(prepared.view.depth_m.shape)
        support = _depth_observability(
            prepared_view=prepared,
            object_scale_m=0.1,
            quantile_low=0.05,
            quantile_high=0.95,
            minimum_valid_points=10,
        )
        self.assertEqual(support["valid_depth_fraction"], 1.0)
        self.assertAlmostEqual(support["normalized_depth_span"], 0.9, places=5)
        self.assertAlmostEqual(support["observability"], 0.9, places=5)

    def test_identity_is_kept_without_measurable_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            renderer = _Renderer()
            with patch(
                "scale.independent_axis_scale_refiner."
                "_score_rendered_candidates",
                return_value=tuple(_score(0.1) for _ in range(27)),
            ):
                result = fit_independent_axis_scale(
                    view_name="reference",
                    source_candidate=self._candidate(root),
                    prepared_view=self._view(),
                    fixed_pose_camera_from_proxy=np.eye(4),
                    renderer=renderer,
                    output_directory=root / "out",
                    weights=AlignmentScoreWeights(),
                    depth_trim_quantile=0.9,
                    minimum_depth_overlap_pixels=1,
                    free_space_absolute_tolerance_m=0.005,
                    free_space_relative_tolerance=0.02,
                    quantile_low=0.01,
                    quantile_high=0.99,
                    sample_count=1000,
                    random_seed=0,
                    minimum_factor=0.9,
                    maximum_factor=1.1,
                    grid_step_count=3,
                    scale_penalty_weight=0.02,
                    uncertainty_scale_penalty_weight=0.10,
                    depth_quantile_low=0.05,
                    depth_quantile_high=0.95,
                    depth_minimum_valid_points=1,
                    minimum_loss_improvement_ratio=0.01,
                )
            self.assertFalse(result.applied)
            self.assertEqual(result.selected_scale_factors, (1.0, 1.0, 1.0))
            self.assertEqual(result.selected_candidate, result.source_candidate)

    def test_best_independent_xyz_candidate_is_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grid = build_axis_scale_factor_grid(
                minimum_factor=0.9,
                maximum_factor=1.1,
                grid_step_count=3,
            )
            target = (1.1, 0.9, 1.0)
            target_index = grid.index(target)
            scores = [_score(0.2) for _ in grid]
            scores[target_index] = _score(0.1)
            renderer = _Renderer()
            with patch(
                "scale.independent_axis_scale_refiner."
                "_score_rendered_candidates",
                return_value=tuple(scores),
            ):
                result = fit_independent_axis_scale(
                    view_name="query",
                    source_candidate=self._candidate(root),
                    prepared_view=self._view(),
                    fixed_pose_camera_from_proxy=np.eye(4),
                    renderer=renderer,
                    output_directory=root / "out",
                    weights=AlignmentScoreWeights(),
                    depth_trim_quantile=0.9,
                    minimum_depth_overlap_pixels=1,
                    free_space_absolute_tolerance_m=0.005,
                    free_space_relative_tolerance=0.02,
                    quantile_low=0.01,
                    quantile_high=0.99,
                    sample_count=1000,
                    random_seed=0,
                    minimum_factor=0.9,
                    maximum_factor=1.1,
                    grid_step_count=3,
                    scale_penalty_weight=0.02,
                    uncertainty_scale_penalty_weight=0.10,
                    depth_quantile_low=0.05,
                    depth_quantile_high=0.95,
                    depth_minimum_valid_points=1,
                    minimum_loss_improvement_ratio=0.01,
                )
            self.assertTrue(result.applied)
            self.assertEqual(result.selected_scale_factors, target)
            self.assertEqual(
                result.selected_candidate.scale_m,
                result.source_candidate.scale_m,
            )
            self.assertTrue(result.selected_candidate.scaled_mesh_path.is_file())
            metadata = json.loads(
                result.selected_candidate.metadata_path.read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["method"], "independent_camera_axis_scale")
            self.assertEqual(metadata["scale_factors_uvd"], list(target))
            self.assertEqual(renderer.transforms.shape, (27, 4, 4))


if __name__ == "__main__":
    unittest.main()
