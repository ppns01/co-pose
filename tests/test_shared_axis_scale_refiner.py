from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh

from pose.alignment_scorer import AlignmentScoreResult, AlignmentScoreWeights
from scale.mesh_scaler import ScaledMeshCandidate
from scale.shared_axis_scale_refiner import (
    centered_shared_axis_scale_affine,
    fit_shared_axis_scale,
)


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
        count = transforms_camera_from_proxy.shape[0]
        masks = np.ones((count, image_height, image_width), dtype=bool)
        depths = np.ones_like(masks, dtype=np.float32)
        return masks, depths


def _candidate(root: Path, name: str) -> ScaledMeshCandidate:
    mesh_path = root / f"{name}.obj"
    metadata_path = root / f"{name}.json"
    mesh = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    mesh.export(mesh_path)
    metadata_path.write_text("{}\n", encoding="utf-8")
    return ScaledMeshCandidate(
        candidate_index=0,
        scale_m=float(np.linalg.norm((0.1, 0.2, 0.3))),
        normalized_mesh_path=mesh_path,
        scaled_mesh_path=mesh_path,
        metadata_path=metadata_path,
        scale_transform=np.eye(4, dtype=np.float64),
    )


def _view() -> SimpleNamespace:
    mask = np.ones((8, 8), dtype=bool)
    return SimpleNamespace(
        view=SimpleNamespace(
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            depth_m=np.ones((8, 8), dtype=np.float32),
            camera_matrix=np.eye(3, dtype=np.float64),
        ),
        segmentation=SimpleNamespace(mask_bool=mask),
    )


_ROTATION_90_Z = np.asarray(
    ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)


class SharedAxisScaleAffineTests(unittest.TestCase):
    def test_reference_side_is_plain_diagonal(self) -> None:
        # Cr = D: reference is expressed in its own proxy axes directly.
        from scale.independent_axis_scale_refiner import (
            centered_axis_scale_affine,
        )

        affine = centered_axis_scale_affine(
            scale_factors=(2.0, 0.5, 1.0), center_m=(0.0, 0.0, 0.0)
        )
        np.testing.assert_allclose(
            affine[:3, :3], np.diag((2.0, 0.5, 1.0)), atol=1e-12
        )

    def test_query_side_is_R_D_RT_not_RT_D_R(self) -> None:
        # Cq = R0 D R0^T -- the opposite multiplication order from
        # centered_camera_axis_scale_affine's R^T D R.
        affine = centered_shared_axis_scale_affine(
            scale_factors_reference_uvd=(2.0, 1.0, 1.0),
            center_proxy_m=(0.0, 0.0, 0.0),
            rotation_query_from_reference=_ROTATION_90_Z,
        )
        expected = _ROTATION_90_Z @ np.diag((2.0, 1.0, 1.0)) @ _ROTATION_90_Z.T
        np.testing.assert_allclose(affine[:3, :3], expected, atol=1e-12)
        # A 90 degree rotation about Z swaps which axis the x2 factor lands
        # on: the reference's x-stretch becomes a query y-stretch.
        np.testing.assert_allclose(
            affine[:3, :3], np.diag((1.0, 2.0, 1.0)), atol=1e-12
        )

    def test_query_side_matches_direct_derivation_for_a_point(self) -> None:
        # For any point p_r in reference axes with query-axes image
        # p_q = R @ p_r, deforming p_r by D and re-expressing in query axes
        # must equal applying Cq to p_q directly.
        rotation = _ROTATION_90_Z
        factors = (2.0, 0.5, 1.3)
        p_r = np.asarray((0.3, -0.1, 0.05))
        p_q = rotation @ p_r
        deformed_in_reference_axes = np.diag(factors) @ p_r
        deformed_in_query_axes_expected = rotation @ deformed_in_reference_axes

        affine = centered_shared_axis_scale_affine(
            scale_factors_reference_uvd=factors,
            center_proxy_m=(0.0, 0.0, 0.0),
            rotation_query_from_reference=rotation,
        )
        deformed_in_query_axes_actual = affine[:3, :3] @ p_q + affine[:3, 3]
        np.testing.assert_allclose(
            deformed_in_query_axes_actual, deformed_in_query_axes_expected, atol=1e-12
        )

    def test_rejects_improper_rotation(self) -> None:
        reflection = np.diag((1.0, 1.0, -1.0))
        with self.assertRaises(ValueError):
            centered_shared_axis_scale_affine(
                scale_factors_reference_uvd=(1.0, 1.0, 1.0),
                center_proxy_m=(0.0, 0.0, 0.0),
                rotation_query_from_reference=reflection,
            )

    def test_centered_preserves_local_center(self) -> None:
        center = (3.0, -2.0, 1.0)
        affine = centered_shared_axis_scale_affine(
            scale_factors_reference_uvd=(0.8, 1.1, 1.2),
            center_proxy_m=center,
            rotation_query_from_reference=_ROTATION_90_Z,
        )
        center_array = np.asarray(center, dtype=np.float64)
        transformed = affine[:3, :3] @ center_array + affine[:3, 3]
        np.testing.assert_allclose(transformed, center_array)


class FitSharedAxisScaleTests(unittest.TestCase):
    def test_identity_is_kept_without_measurable_joint_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "scale.shared_axis_scale_refiner.score_alignment",
                return_value=_score(0.1),
            ):
                result = fit_shared_axis_scale(
                    reference_source_candidate=_candidate(root, "reference"),
                    reference_prepared_view=_view(),
                    reference_fixed_pose_camera_from_proxy=np.eye(4),
                    query_source_candidate=_candidate(root, "query"),
                    query_prepared_view=_view(),
                    query_fixed_pose_camera_from_proxy=np.eye(4),
                    rotation_query_from_reference=_ROTATION_90_Z,
                    renderer=_Renderer(),
                    output_directory=root / "out",
                    weights=AlignmentScoreWeights(),
                    depth_trim_quantile=0.9,
                    minimum_depth_overlap_pixels=1,
                    free_space_absolute_tolerance_m=0.005,
                    free_space_relative_tolerance=0.02,
                    quantile_low=0.01,
                    quantile_high=0.99,
                    sample_count=200,
                    random_seed=0,
                    minimum_factor=0.85,
                    maximum_factor=1.15,
                    grid_step_count=3,
                    scale_penalty_weight=0.01,
                    minimum_loss_improvement_ratio=0.01,
                )
        self.assertFalse(result.applied)
        self.assertEqual(result.selected_scale_factors, (1.0, 1.0, 1.0))
        self.assertIs(
            result.reference_selected_candidate,
            result.reference_selected_candidate,
        )

    def test_applies_shared_factors_to_both_views_when_it_helps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            grid_size = 3**3
            call_count = {"n": 0}

            def _scripted_score(**kwargs):
                del kwargs
                call_count["n"] += 1
                # Reference and query each iterate the same grid_size-point
                # grid in the same order (identity first). Penalize identity
                # for both views identically so the joint objective picks a
                # non-identity D* regardless of exact interleaving.
                position_in_grid = (call_count["n"] - 1) % grid_size
                loss = 0.2 if position_in_grid == 0 else 0.05
                return _score(loss)

            with patch(
                "scale.shared_axis_scale_refiner.score_alignment",
                side_effect=_scripted_score,
            ):
                result = fit_shared_axis_scale(
                    reference_source_candidate=_candidate(root, "reference"),
                    reference_prepared_view=_view(),
                    reference_fixed_pose_camera_from_proxy=np.eye(4),
                    query_source_candidate=_candidate(root, "query"),
                    query_prepared_view=_view(),
                    query_fixed_pose_camera_from_proxy=np.eye(4),
                    rotation_query_from_reference=_ROTATION_90_Z,
                    renderer=_Renderer(),
                    output_directory=root / "out",
                    weights=AlignmentScoreWeights(),
                    depth_trim_quantile=0.9,
                    minimum_depth_overlap_pixels=1,
                    free_space_absolute_tolerance_m=0.005,
                    free_space_relative_tolerance=0.02,
                    quantile_low=0.01,
                    quantile_high=0.99,
                    sample_count=200,
                    random_seed=0,
                    minimum_factor=0.85,
                    maximum_factor=1.15,
                    grid_step_count=3,
                    scale_penalty_weight=0.0,
                    minimum_loss_improvement_ratio=0.01,
                )
            self.assertTrue(result.applied)
            self.assertNotEqual(result.selected_scale_factors, (1.0, 1.0, 1.0))
            self.assertTrue(
                result.reference_selected_candidate.scaled_mesh_path.is_file()
            )
            self.assertTrue(
                result.query_selected_candidate.scaled_mesh_path.is_file()
            )
            # Both proxies received a scaled mesh distinct from the source.
            self.assertNotEqual(
                result.reference_selected_candidate.scaled_mesh_path,
                _candidate(root, "reference").scaled_mesh_path,
            )


if __name__ == "__main__":
    unittest.main()
