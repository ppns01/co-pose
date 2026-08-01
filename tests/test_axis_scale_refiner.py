from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from scale.axis_scale_refiner import (
    _load_scene,
    _write_candidate,
    build_volume_preserving_axis_scale_grid,
)
from scale.mesh_scaler import ScaledMeshCandidate


class AxisScaleRefinerTests(unittest.TestCase):
    def test_grid_contains_identity_and_preserves_volume(self) -> None:
        candidates = build_volume_preserving_axis_scale_grid(
            minimum_scale=0.85,
            maximum_scale=1.15,
            grid_step_count=5,
        )

        self.assertIn((1.0, 1.0, 1.0), candidates)
        self.assertGreater(len(candidates), 1)
        for scales in candidates:
            self.assertTrue(all(0.85 <= value <= 1.15 for value in scales))
            self.assertTrue(
                math.isclose(
                    math.prod(scales),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )

    def test_invalid_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_volume_preserving_axis_scale_grid(
                minimum_scale=1.01,
                maximum_scale=1.15,
            )

    def test_candidate_scales_local_axes_about_normalized_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.obj"
            metadata_path = root / "source.json"
            trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(
                source_path
            )
            metadata_path.write_text("{}", encoding="utf-8")
            source = ScaledMeshCandidate(
                candidate_index=0,
                scale_m=0.2,
                normalized_mesh_path=source_path,
                scaled_mesh_path=source_path,
                metadata_path=metadata_path,
                scale_transform=np.diag((0.2, 0.2, 0.2, 1.0)),
            )
            axis_scales = (1.1, 1.0, 1.0 / 1.1)
            candidate = _write_candidate(
                source_scene=_load_scene(source_path),
                source_candidate=source,
                candidate_index=0,
                axis_scales=axis_scales,
                output_directory=root / "axis_candidates",
            )
            loaded = trimesh.load(
                candidate.scaled_mesh_path,
                force="mesh",
                process=False,
            )

            np.testing.assert_allclose(
                loaded.extents,
                np.asarray((1.0, 2.0, 3.0))
                * np.asarray(axis_scales),
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
