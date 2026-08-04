from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

from pose.alignment_scorer import AlignmentScoreWeights
from pose.dgedi_runner import compose_dgedi_relative_pose
from scale.confidence_shared_dimension_refiner import (
    _canonical_affine,
    _scale_rotation_toward_identity,
    build_confidence_factor_grid,
    fit_confidence_shared_dimensions,
)
from scale.mesh_scaler import ScaledMeshCandidate


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


class _ConstantRenderer:
    def render_affine_depth_mask_batch(self, **kwargs: object):
        transforms = np.asarray(kwargs["transforms_camera_from_proxy"])
        height = int(kwargs["image_height"])
        width = int(kwargs["image_width"])
        count = transforms.shape[0]
        return (
            np.ones((count, height, width), dtype=bool),
            np.ones((count, height, width), dtype=np.float32),
        )


class ConfidenceSharedDimensionRefinerTests(unittest.TestCase):
    def test_rotation_influence_scales_continuously_with_confidence(self) -> None:
        rotation = _rotation_z(120.0)
        np.testing.assert_allclose(
            _scale_rotation_toward_identity(rotation, 0.0),
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            _scale_rotation_toward_identity(rotation, 0.25),
            _rotation_z(30.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            _scale_rotation_toward_identity(rotation, 1.0),
            rotation,
            atol=1e-12,
        )

    def test_rebaked_a1_b1_g1_preserve_camera_pose_contract(self) -> None:
        rotation = _rotation_z(90.0)
        reference_common = _canonical_affine(
            linear=np.eye(3),
            center=np.asarray([0.1, -0.2, 0.3]),
        )
        query_common = _canonical_affine(
            linear=rotation.T,
            center=np.asarray([-0.3, 0.4, 0.2]),
        )

        a0 = np.eye(4, dtype=np.float64)
        a0[:3, :3] = _rotation_z(15.0)
        a0[:3, 3] = (0.1, 0.2, 0.8)
        b0 = np.eye(4, dtype=np.float64)
        b0[:3, :3] = _rotation_z(-20.0)
        b0[:3, 3] = (-0.1, 0.3, 1.0)
        g0 = np.eye(4, dtype=np.float64)
        g0[:3, :3] = _rotation_z(35.0)
        g0[:3, 3] = (0.02, -0.03, 0.01)

        a1 = a0 @ np.linalg.inv(reference_common)
        b1 = b0 @ np.linalg.inv(query_common)
        g1 = query_common @ g0 @ np.linalg.inv(reference_common)
        h0 = compose_dgedi_relative_pose(
            reference_pose_camera_from_proxy=a0,
            query_pose_camera_from_proxy=b0,
            proxy_pose_query_from_reference=g0,
        )
        h1 = compose_dgedi_relative_pose(
            reference_pose_camera_from_proxy=a1,
            query_pose_camera_from_proxy=b1,
            proxy_pose_query_from_reference=g1,
        )

        np.testing.assert_allclose(h1, h0, atol=1e-12)

    def test_grid_shrinks_linearly_to_identity(self) -> None:
        zero_grid = build_confidence_factor_grid(
            minimum_factor=0.70,
            maximum_factor=1.15,
            grid_step_count=5,
            confidence=0.0,
        )
        self.assertEqual(zero_grid, ((1.0, 1.0, 1.0),))

        half_grid = build_confidence_factor_grid(
            minimum_factor=0.70,
            maximum_factor=1.15,
            grid_step_count=5,
            confidence=0.5,
        )
        half_values = [value for factors in half_grid for value in factors]
        self.assertAlmostEqual(min(half_values), 0.85)
        self.assertAlmostEqual(max(half_values), 1.075)

        full_grid = build_confidence_factor_grid(
            minimum_factor=0.70,
            maximum_factor=1.15,
            grid_step_count=5,
            confidence=1.0,
        )
        self.assertEqual(len(full_grid), 125)
        full_values = [value for factors in full_grid for value in factors]
        self.assertAlmostEqual(min(full_values), 0.70)
        self.assertAlmostEqual(max(full_values), 1.15)

    def test_query_rotation_is_transport_to_reference_common_axes(self) -> None:
        rotation = _rotation_z(90.0)
        reference_points = np.asarray(
            [
                [-1.0, -0.5, -0.25],
                [1.0, -0.5, -0.25],
                [1.0, 0.5, 0.25],
                [-1.0, 0.5, 0.25],
            ],
            dtype=np.float64,
        )
        query_points = reference_points @ rotation.T
        factors = np.asarray([1.2, 0.8, 1.1], dtype=np.float64)
        reference_affine = _canonical_affine(
            linear=np.diag(factors),
            center=np.zeros(3),
        )
        query_affine = _canonical_affine(
            linear=np.diag(factors) @ rotation.T,
            center=np.zeros(3),
        )

        np.testing.assert_allclose(
            _transform_points(reference_points, reference_affine),
            _transform_points(query_points, query_affine),
            atol=1e-12,
        )

    def test_full_fit_writes_two_meshes_in_one_common_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rotation = _rotation_z(90.0)
            reference_mesh = trimesh.creation.box(extents=(0.12, 0.08, 0.05))
            query_mesh = reference_mesh.copy()
            query_transform = np.eye(4, dtype=np.float64)
            query_transform[:3, :3] = rotation
            query_mesh.apply_transform(query_transform)
            reference_path = root / "reference.obj"
            query_path = root / "query.obj"
            reference_mesh.export(reference_path)
            query_mesh.export(query_path)

            def candidate(path: Path, name: str) -> ScaledMeshCandidate:
                metadata_path = root / f"{name}.json"
                metadata_path.write_text("{}\n", encoding="utf-8")
                return ScaledMeshCandidate(
                    candidate_index=0,
                    scale_m=0.15,
                    normalized_mesh_path=path,
                    scaled_mesh_path=path,
                    metadata_path=metadata_path,
                    scale_transform=np.eye(4, dtype=np.float64),
                )

            observed = SimpleNamespace(
                view=SimpleNamespace(
                    rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                    depth_m=np.ones((8, 8), dtype=np.float32),
                    camera_matrix=np.eye(3, dtype=np.float64),
                ),
                segmentation=SimpleNamespace(
                    mask_bool=np.ones((8, 8), dtype=bool)
                ),
            )
            pose = np.eye(4, dtype=np.float64)
            pose[2, 3] = 1.0
            reference_candidate = candidate(reference_path, "reference")
            query_candidate = candidate(query_path, "query")

            def run_fit(confidence: float, name: str):
                return fit_confidence_shared_dimensions(
                    reference_source_candidate=reference_candidate,
                    query_source_candidate=query_candidate,
                    reference_prepared_view=observed,
                    query_prepared_view=observed,
                    reference_pose_camera_from_proxy=pose,
                    query_pose_camera_from_proxy=pose,
                    rotation_query_from_reference_axes=rotation,
                    confidence=confidence,
                    renderer=_ConstantRenderer(),
                    output_directory=root / name,
                    weights=AlignmentScoreWeights(),
                    depth_trim_quantile=0.9,
                    minimum_depth_overlap_pixels=1,
                    free_space_absolute_tolerance_m=0.005,
                    free_space_relative_tolerance=0.02,
                    quantile_low=0.0,
                    quantile_high=1.0,
                    sample_count=1000,
                    random_seed=0,
                    minimum_factor=0.90,
                    maximum_factor=1.10,
                    grid_step_count=3,
                    minimum_scale_penalty_weight=0.02,
                    uncertainty_scale_penalty_weight=0.10,
                )

            result = run_fit(1.0, "fit")

            reference_final = trimesh.load_mesh(
                result.reference_candidate.scaled_mesh_path,
                process=False,
            )
            query_final = trimesh.load_mesh(
                result.query_candidate.scaled_mesh_path,
                process=False,
            )
            np.testing.assert_allclose(
                reference_final.extents,
                query_final.extents,
                atol=1e-6,
            )
            payload = json.loads(
                result.summary_path.read_text(encoding="utf-8")
            )
            self.assertTrue(payload["pair_shared_scale_verified"])
            self.assertEqual(payload["candidate_count"], 27)
            self.assertIn("no hard fallback", payload["selection_policy"])

            zero_result = run_fit(0.0, "fit_zero")
            np.testing.assert_allclose(
                zero_result.reference_scale_factors_common,
                (1.0, 1.0, 1.0),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                zero_result.query_scale_factors_common,
                (1.0, 1.0, 1.0),
                atol=1e-12,
            )
            zero_payload = json.loads(
                zero_result.summary_path.read_text(encoding="utf-8")
            )
            self.assertEqual(zero_payload["candidate_count"], 1)
            np.testing.assert_allclose(
                zero_payload["rotation_query_from_reference_axes"],
                np.eye(3),
                atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
