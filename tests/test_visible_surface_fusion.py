from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mesh_fusion.visible_surface_fusion import (
    build_depth_surface_mesh,
    fuse_aligned_surface_meshes,
    transform_surface_mesh,
    write_triangle_mesh_ply,
)


def _camera_matrix() -> np.ndarray:
    return np.asarray(
        [
            [100.0, 0.0, 1.0],
            [0.0, 100.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class VisibleSurfaceFusionTests(unittest.TestCase):
    def test_planar_depth_builds_pixel_grid_surface(self) -> None:
        depth = np.ones((3, 3), dtype=np.float32)
        mask = np.ones((3, 3), dtype=bool)
        rgb = np.full((3, 3, 3), 127, dtype=np.uint8)

        mesh = build_depth_surface_mesh(
            masked_depth_m=depth,
            mask_bool=mask,
            camera_k=_camera_matrix(),
            rgb=rgb,
        )

        self.assertEqual(mesh.vertices_m.shape, (9, 3))
        self.assertEqual(mesh.triangles.shape, (8, 3))
        self.assertTrue(np.all(mesh.vertex_normals[:, 2] < -0.99))

    def test_depth_discontinuity_triangles_are_removed(self) -> None:
        depth = np.ones((3, 3), dtype=np.float32)
        depth[0, 0] = 1.1
        mask = np.ones((3, 3), dtype=bool)
        rgb = np.zeros((3, 3, 3), dtype=np.uint8)

        mesh = build_depth_surface_mesh(
            masked_depth_m=depth,
            mask_bool=mask,
            camera_k=_camera_matrix(),
            rgb=rgb,
            maximum_triangle_depth_delta_m=0.01,
            maximum_triangle_edge_length_m=0.2,
        )

        self.assertLess(len(mesh.triangles), 8)
        self.assertGreater(len(mesh.triangles), 0)

    def test_inverse_pose_alignment_merges_coincident_surfaces(self) -> None:
        depth = np.ones((3, 3), dtype=np.float32)
        mask = np.ones((3, 3), dtype=bool)
        rgb = np.full((3, 3, 3), 64, dtype=np.uint8)
        reference_mesh = build_depth_surface_mesh(
            masked_depth_m=depth,
            mask_bool=mask,
            camera_k=_camera_matrix(),
            rgb=rgb,
        )
        transform_query_from_reference = np.eye(4, dtype=np.float64)
        transform_query_from_reference[0, 3] = 0.2
        query_mesh = transform_surface_mesh(
            reference_mesh,
            transform_query_from_reference,
        )
        query_mesh_in_reference = transform_surface_mesh(
            query_mesh,
            np.linalg.inv(transform_query_from_reference),
        )

        result = fuse_aligned_surface_meshes(
            reference_mesh=reference_mesh,
            query_mesh_in_reference=query_mesh_in_reference,
            merge_distance_m=1e-5,
        )

        self.assertEqual(len(result.mesh.vertices_m), 9)
        self.assertEqual(len(result.mesh.triangles), 8)
        np.testing.assert_array_equal(
            result.reference_observation_count,
            np.ones(9, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            result.query_observation_count,
            np.ones(9, dtype=np.int64),
        )
        self.assertEqual(
            result.diagnostics["matched_query_vertex_fraction"],
            1.0,
        )

    def test_ply_writer_preserves_partial_mesh(self) -> None:
        depth = np.ones((2, 2), dtype=np.float32)
        mask = np.ones((2, 2), dtype=bool)
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        mesh = build_depth_surface_mesh(
            masked_depth_m=depth,
            mask_bool=mask,
            camera_k=_camera_matrix(),
            rgb=rgb,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_triangle_mesh_ply(
                Path(temporary_directory) / "surface.ply",
                mesh,
            )
            content = path.read_text(encoding="ascii")

        self.assertIn("element vertex 4", content)
        self.assertIn("element face 2", content)


if __name__ == "__main__":
    unittest.main()
