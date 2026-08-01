from __future__ import annotations

import inspect
import unittest

import numpy as np
import scipy.ndimage

from mesh_refinement.weighted_visible_arap_refiner import (
    _apply_solver_trust_region,
    _adjacency,
    _contour_metrics,
    _depth_constraint_system,
    _huber_irls_weights,
    _safe_increment_line_search,
    _scalar_barycentric_system,
    _silhouette_constraint_system,
    _soft_unseen_confidence,
    _stratified_sample_pixels,
    _unique_edges,
    refine_mesh_with_weighted_visible_arap,
)


class WeightedVisibleArapRefinerTests(unittest.TestCase):
    def test_default_depth_sampling_stride_is_one(self) -> None:
        signature = inspect.signature(
            refine_mesh_with_weighted_visible_arap
        )
        self.assertEqual(
            signature.parameters["depth_sample_stride_px"].default,
            1,
        )

    def test_solver_trust_region_limits_global_step(self) -> None:
        current = np.zeros((2, 3), dtype=np.float64)
        proposed = np.asarray(
            [[0.030, 0.0, 0.0], [0.015, 0.0, 0.0]],
            dtype=np.float64,
        )
        constrained, diagnostics = _apply_solver_trust_region(
            current_points=current,
            proposed_points=proposed,
            maximum_displacement_m=0.003,
        )
        self.assertAlmostEqual(
            diagnostics["unconstrained_step_max_m"],
            0.030,
        )
        self.assertAlmostEqual(
            diagnostics["trust_region_step_max_m"],
            0.003,
        )
        self.assertAlmostEqual(
            diagnostics["trust_region_scale"],
            0.1,
        )
        np.testing.assert_allclose(
            constrained,
            proposed * 0.1,
            atol=1e-12,
        )

    def test_huber_irls_downweights_large_residual(self) -> None:
        residual = np.asarray(
            [0.0, 0.002, 0.006, 0.012],
            dtype=np.float64,
        )
        weights = _huber_irls_weights(
            residual,
            0.006,
        )
        np.testing.assert_allclose(
            weights,
            np.asarray([1.0, 1.0, 1.0, 0.5]),
            atol=1e-12,
        )

    def test_stratified_sampling_is_capped_and_deterministic(self) -> None:
        mask = np.ones((12, 12), dtype=bool)
        first_v, first_u = _stratified_sample_pixels(
            mask,
            stride_px=3,
            maximum_sample_count=7,
        )
        second_v, second_u = _stratified_sample_pixels(
            mask,
            stride_px=3,
            maximum_sample_count=7,
        )
        self.assertEqual(len(first_v), 7)
        np.testing.assert_array_equal(first_v, second_v)
        np.testing.assert_array_equal(first_u, second_u)
        self.assertTrue(mask[first_v, first_u].all())

    def test_scalar_barycentric_system_distributes_normal_row(self) -> None:
        triangle_vertices = np.asarray([[0, 1, 2]], dtype=np.int64)
        barycentric = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64)
        direction = np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64)
        matrix, rhs = _scalar_barycentric_system(
            vertex_count=3,
            triangle_vertices=triangle_vertices,
            barycentric=barycentric,
            directions=direction,
            right_hand_side=np.asarray([-1.1]),
        )
        points = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.assertAlmostEqual(
            float((matrix @ points.reshape(-1))[0]),
            -1.0,
        )
        self.assertAlmostEqual(float(rhs[0]), -1.1)

    def test_depth_system_uses_pixel_hit_triangle(self) -> None:
        points = np.asarray(
            [
                [-0.1, -0.1, 1.0],
                [0.1, -0.1, 1.0],
                [0.0, 0.1, 1.0],
            ],
            dtype=np.float64,
        )
        triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
        depth = np.zeros((7, 7), dtype=np.float64)
        depth[2:5, 2:5] = 1.1
        observed_mask = depth > 0.0
        rendered_mask = np.zeros((7, 7), dtype=bool)
        rendered_mask[3, 3] = True
        primitive_ids = np.full((7, 7), -1, dtype=np.int64)
        primitive_ids[3, 3] = 0
        primitive_uvs = np.zeros((7, 7, 2), dtype=np.float64)
        primitive_uvs[3, 3] = np.asarray([0.25, 0.25])
        raycast = {
            "rendered_mask": rendered_mask,
            "primitive_ids": primitive_ids,
            "primitive_uvs": primitive_uvs,
        }
        camera_k = np.asarray(
            [
                [10.0, 0.0, 3.0],
                [0.0, 10.0, 3.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        matrix, rhs, confidence, stats = _depth_constraint_system(
            points=points,
            triangles=triangles,
            camera_k=camera_k,
            raycast=raycast,
            observed_depth_m=depth,
            observed_mask=observed_mask,
            mask_erosion_px=0,
            sample_stride_px=1,
            maximum_sample_count=32,
            maximum_neighbor_depth_delta_m=0.02,
            maximum_projective_residual_m=0.20,
            grazing_cosine_floor=0.15,
        )

        self.assertEqual(matrix.shape, (1, 9))
        self.assertEqual(rhs.shape, (1,))
        self.assertEqual(confidence.shape, (1,))
        self.assertEqual(stats["depth_sample_count"], 1)
        residual = float((matrix @ points.reshape(-1) - rhs)[0])
        self.assertAlmostEqual(abs(residual), 0.1, places=8)

    def test_silhouette_system_pushes_inner_contour_outward(self) -> None:
        points = np.asarray(
            [
                [0.10, 0.00, 1.0],
                [0.11, 0.00, 1.0],
                [0.10, 0.01, 1.0],
            ],
            dtype=np.float64,
        )
        triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
        camera_k = np.asarray(
            [
                [20.0, 0.0, 4.0],
                [0.0, 20.0, 4.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        observed_mask = np.zeros((9, 9), dtype=bool)
        observed_mask[1:8, 1:8] = True
        rendered_boundary = np.zeros((9, 9), dtype=bool)
        rendered_boundary[4, 6] = True
        primitive_ids = np.full((9, 9), -1, dtype=np.int64)
        primitive_ids[4, 6] = 0
        primitive_uvs = np.zeros((9, 9, 2), dtype=np.float64)

        matrix, rhs, _, _, stats = _silhouette_constraint_system(
            points=points,
            triangles=triangles,
            camera_k=camera_k,
            raycast={
                "rendered_boundary": rendered_boundary,
                "primitive_ids": primitive_ids,
                "primitive_uvs": primitive_uvs,
            },
            observed_mask=observed_mask,
            contour_sample_stride=1,
            maximum_sample_count=32,
            minimum_residual_px=0.15,
            maximum_residual_px=30.0,
        )

        residual = matrix @ points.reshape(-1) - rhs
        self.assertEqual(stats["active_contour_sample_count"], 1)
        self.assertLess(float(residual[0]), 0.0)
        self.assertGreater(float(matrix[0, 0]), 0.0)

    def test_soft_unseen_anchor_has_smooth_transition(self) -> None:
        edges = np.asarray(
            [[0, 1], [1, 2], [2, 3]],
            dtype=np.int64,
        )
        adjacency = _adjacency(edges, vertex_count=4)
        confidence = _soft_unseen_confidence(
            visible_mask=np.asarray([True, False, False, False]),
            adjacency=adjacency,
            transition_ring_count=2,
            visible_confidence=0.0,
            transition_confidence=0.1,
            far_hidden_confidence=1.0,
        )
        np.testing.assert_allclose(
            confidence,
            np.asarray([0.0, 0.4, 0.7, 1.0]),
            atol=1e-12,
        )

    def test_line_search_accepts_coherent_safe_translation(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.01, 0.0, 1.0],
                [0.0, 0.01, 1.0],
            ],
            dtype=np.float64,
        )
        triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
        proposed = points + np.asarray([0.001, 0.0, 0.0])
        edges = _unique_edges(triangles)
        accepted, stats = _safe_increment_line_search(
            original_points=points,
            current_points=points,
            proposed_points=proposed,
            triangles=triangles,
            edges=edges,
            maximum_step_displacement_m=0.003,
            maximum_cumulative_displacement_m=0.008,
            minimum_edge_ratio=0.55,
            maximum_edge_ratio=1.80,
            minimum_area_ratio=0.25,
            maximum_area_ratio=4.00,
            minimum_step_scale=1.0 / 4096.0,
        )
        self.assertTrue(stats["topology_safe"])
        self.assertEqual(stats["step_scale"], 1.0)
        np.testing.assert_allclose(accepted, proposed, atol=1e-12)

    def test_contour_metrics_are_symmetric(self) -> None:
        first = np.zeros((9, 9), dtype=bool)
        second = np.zeros((9, 9), dtype=bool)
        first[2:7, 2:7] = True
        second[2:7, 3:8] = True

        forward = _contour_metrics(first, second)
        reverse = _contour_metrics(second, first)

        self.assertAlmostEqual(
            forward["symmetric_boundary_distance_mean_px"],
            reverse["symmetric_boundary_distance_mean_px"],
        )

    def test_full_solver_preserves_matching_planar_observation(self) -> None:
        points = np.asarray(
            [
                [0.05, 0.15, 1.0],
                [0.15, 0.15, 1.0],
                [0.15, 0.25, 1.0],
                [0.05, 0.25, 1.0],
            ],
            dtype=np.float64,
        )
        triangles = np.asarray(
            [[0, 1, 2], [0, 2, 3]],
            dtype=np.int64,
        )
        observed_mask = np.zeros((9, 9), dtype=bool)
        observed_mask[2:7, 2:7] = True
        observed_depth = np.zeros((9, 9), dtype=np.float64)
        observed_depth[observed_mask] = 1.0
        camera_k = np.asarray(
            [
                [20.0, 0.0, 4.0],
                [0.0, 20.0, 4.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        def fake_raycast(**_arguments: object) -> dict[str, np.ndarray]:
            primitive_ids = np.full((9, 9), -1, dtype=np.int64)
            primitive_ids[observed_mask] = 0
            primitive_uvs = np.zeros((9, 9, 2), dtype=np.float64)
            rendered_depth = np.zeros((9, 9), dtype=np.float64)
            rendered_depth[observed_mask] = 1.0
            boundary = observed_mask & ~np.asarray(
                scipy.ndimage.binary_erosion(
                    observed_mask,
                    iterations=1,
                    border_value=0,
                ),
                dtype=bool,
            )
            return {
                "rendered_mask": observed_mask.copy(),
                "rendered_boundary": boundary,
                "rendered_depth": rendered_depth,
                "primitive_ids": primitive_ids,
                "primitive_uvs": primitive_uvs,
                "visible_vertex_mask": np.ones(4, dtype=bool),
            }

        def diameter(values: np.ndarray) -> float:
            difference = values[:, None, :] - values[None, :, :]
            return float(np.linalg.norm(difference, axis=2).max())

        result = refine_mesh_with_weighted_visible_arap(
            points_camera=points,
            triangles=triangles,
            mask_bool=observed_mask,
            camera_k=camera_k,
            masked_depth_m=observed_depth,
            target_scale_m=diameter(points),
            diameter_fn=diameter,
            depth_mask_erosion_px=0,
            depth_sample_stride_px=1,
            outer_iteration_count=1,
            local_global_iteration_count=2,
            linear_solver_iteration_count=200,
            raycast_function=fake_raycast,
        )

        np.testing.assert_allclose(
            result.refined_points_camera,
            points,
            atol=1e-5,
        )
        self.assertEqual(result.diagnostics["raster_iou_after"], 1.0)
        self.assertGreater(
            result.diagnostics["depth_correspondence_count"],
            0,
        )
        self.assertEqual(
            result.diagnostics["outer_iterations"][0]["status"],
            "converged",
        )

    def test_unsafe_final_scale_correction_is_skipped(self) -> None:
        points = np.asarray(
            [
                [0.05, 0.15, 1.0],
                [0.15, 0.15, 1.0],
                [0.15, 0.25, 1.0],
                [0.05, 0.25, 1.0],
            ],
            dtype=np.float64,
        )
        triangles = np.asarray(
            [[0, 1, 2], [0, 2, 3]],
            dtype=np.int64,
        )
        observed_mask = np.zeros((9, 9), dtype=bool)
        observed_mask[2:7, 2:7] = True
        observed_depth = np.zeros((9, 9), dtype=np.float64)
        observed_depth[observed_mask] = 1.0
        camera_k = np.asarray(
            [
                [20.0, 0.0, 4.0],
                [0.0, 20.0, 4.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        def fake_raycast(**_arguments: object) -> dict[str, np.ndarray]:
            primitive_ids = np.full((9, 9), -1, dtype=np.int64)
            primitive_ids[observed_mask] = 0
            boundary = observed_mask & ~np.asarray(
                scipy.ndimage.binary_erosion(
                    observed_mask,
                    iterations=1,
                    border_value=0,
                ),
                dtype=bool,
            )
            rendered_depth = np.zeros((9, 9), dtype=np.float64)
            rendered_depth[observed_mask] = 1.0
            return {
                "rendered_mask": observed_mask.copy(),
                "rendered_boundary": boundary,
                "rendered_depth": rendered_depth,
                "primitive_ids": primitive_ids,
                "primitive_uvs": np.zeros((9, 9, 2), dtype=np.float64),
                "visible_vertex_mask": np.ones(4, dtype=bool),
            }

        def diameter(values: np.ndarray) -> float:
            difference = values[:, None, :] - values[None, :, :]
            return float(np.linalg.norm(difference, axis=2).max())

        result = refine_mesh_with_weighted_visible_arap(
            points_camera=points,
            triangles=triangles,
            mask_bool=observed_mask,
            camera_k=camera_k,
            masked_depth_m=observed_depth,
            target_scale_m=diameter(points) * 0.998,
            diameter_fn=diameter,
            depth_mask_erosion_px=0,
            outer_iteration_count=1,
            local_global_iteration_count=1,
            scale_weight=0.0,
            minimum_edge_ratio=0.999,
            raycast_function=fake_raycast,
        )

        self.assertTrue(result.diagnostics["scale_correction_attempted"])
        self.assertFalse(result.diagnostics["scale_correction_applied"])
        self.assertIsNotNone(
            result.diagnostics["scale_correction_rejected_reason"]
        )
        self.assertTrue(
            result.diagnostics["final_cumulative_topology"]["topology_safe"]
        )


if __name__ == "__main__":
    unittest.main()
