from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pose.dgedi_runner import (
    DGeDiSemanticFailure,
    _save_depth_consistent_proxy_surface_cloud,
    _save_registration_cloud_pair_quality,
    _worker_failure_error_type,
    run_dgedi_registration,
)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (x, y, z)
    return pose


class DgediRunnerTests(unittest.TestCase):
    def test_worker_failure_taxonomy_does_not_hide_infrastructure(self) -> None:
        self.assertIs(
            _worker_failure_error_type("Too few correspondences after RANSAC"),
            DGeDiSemanticFailure,
        )
        self.assertIs(
            _worker_failure_error_type("CUDA out of memory"),
            RuntimeError,
        )

    def test_pair_quality_records_degenerate_ratio_without_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_path = root / "reference.json"
            query_path = root / "query.json"
            reference_path.write_text(
                json.dumps({"point_count_saved": 1000, "diameter_m": 0.1}),
                encoding="utf-8",
            )
            query_path.write_text(
                json.dumps({"point_count_saved": 50, "diameter_m": 0.1}),
                encoding="utf-8",
            )
            output_path = root / "pair_quality.json"

            _, diagnostics = _save_registration_cloud_pair_quality(
                reference_diagnostics_path=reference_path,
                query_diagnostics_path=query_path,
                output_path=output_path,
                minimum_point_count_ratio=0.1,
                minimum_diameter_ratio=0.1,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "EVALUATED")
            self.assertFalse(payload["accepted"])
            self.assertFalse(diagnostics["meets_recommended_minimum"])
            self.assertIn(
                "point_count_ratio_below_threshold",
                payload["reasons"],
            )

    def test_depth_consistent_proxy_hits_stay_in_proxy_frame(self) -> None:
        try:
            import open3d as o3d
        except ImportError:
            self.skipTest("open3d is unavailable")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mesh_path = root / "final_proxy.ply"
            cloud_path = root / "surface_hits.ply"

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(
                np.array(
                    [
                        [-1.5, -1.5, 0.0],
                        [1.5, -1.5, 0.0],
                        [1.5, 1.5, 0.0],
                        [-1.5, 1.5, 0.0],
                    ],
                    dtype=np.float64,
                )
            )
            mesh.triangles = o3d.utility.Vector3iVector(
                np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            )
            self.assertTrue(o3d.io.write_triangle_mesh(str(mesh_path), mesh))

            pose = _translation(0.25, -0.10, 2.0)
            intrinsic = np.array(
                [
                    [20.0, 0.0, 9.5],
                    [0.0, 20.0, 9.5],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            mask = np.ones((20, 20), dtype=bool)
            depth = np.full((20, 20), 2.0, dtype=np.float64)

            saved_cloud, diagnostics_path = (
                _save_depth_consistent_proxy_surface_cloud(
                    local_mesh_path=mesh_path,
                    pose_camera_from_proxy=pose,
                    camera_matrix=intrinsic,
                    observed_mask_bool=mask,
                    observed_depth_m=depth,
                    output_cloud_path=cloud_path,
                    sample_count=1000,
                    maximum_depth_residual_m=0.005,
                    minimum_consistent_pixels=256,
                )
            )

            points = np.asarray(
                o3d.io.read_point_cloud(str(saved_cloud)).points,
                dtype=np.float64,
            )
            self.assertEqual(len(points), 400)
            np.testing.assert_allclose(points[:, 2], 0.0, atol=1e-5)
            camera_points = points @ pose[:3, :3].T + pose[:3, 3]
            np.testing.assert_allclose(
                camera_points[:, 2],
                2.0,
                atol=1e-5,
            )
            diagnostics = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostics["coordinate_frame"], "proxy_local")
            self.assertEqual(diagnostics["consistent_pixel_count"], 400)
            self.assertIn("final_proxy_first_ray_hits", diagnostics["point_source"])

            with self.assertRaisesRegex(
                DGeDiSemanticFailure,
                "Too few depth-consistent visible pixels",
            ):
                _save_depth_consistent_proxy_surface_cloud(
                    local_mesh_path=mesh_path,
                    pose_camera_from_proxy=pose,
                    camera_matrix=intrinsic,
                    observed_mask_bool=mask,
                    observed_depth_m=np.full((20, 20), 2.02),
                    output_cloud_path=root / "rejected.ply",
                    sample_count=1000,
                    maximum_depth_residual_m=0.005,
                    minimum_consistent_pixels=256,
                )

    def test_proxy_surface_local_pose_is_composed_with_self_poses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = root / "repository"
            repository.mkdir()
            python_executable = root / "python"
            config_path = root / "config.yaml"
            reference_mesh = root / "reference.obj"
            query_mesh = root / "query.obj"
            for path in (
                python_executable,
                config_path,
                reference_mesh,
                query_mesh,
            ):
                path.touch()

            reference_pose = _translation(1.0, 2.0, 3.0)
            query_pose = _translation(4.0, 5.0, 6.0)
            proxy_pose = _translation(0.5, -0.25, 0.75)
            captured_command: list[str] = []
            saved_mesh_paths: list[Path] = []
            surface_cloud_paths: list[Path] = []
            surface_cloud_calls: list[dict[str, object]] = []

            def fake_save_self_aligned_mesh(**kwargs: object) -> Path:
                output_path = Path(kwargs["output_mesh_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
                resolved = output_path.resolve()
                saved_mesh_paths.append(resolved)
                return resolved

            def fake_save_surface_cloud(
                **kwargs: object,
            ) -> tuple[Path, Path]:
                output_path = Path(kwargs["output_cloud_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
                diagnostics_path = output_path.with_suffix(".json")
                diagnostics_path.write_text(
                    json.dumps(
                        {
                            "point_count_saved": 1000,
                            "diameter_m": 0.1,
                        }
                    ),
                    encoding="utf-8",
                )
                surface_cloud_paths.append(output_path.resolve())
                surface_cloud_calls.append(dict(kwargs))
                return output_path.resolve(), diagnostics_path.resolve()

            def fake_run(command: list[str], **_: object) -> SimpleNamespace:
                captured_command.extend(command)
                output = Path(
                    command[command.index("--output-directory") + 1]
                )
                np.save(
                    output
                    / "dgedi_proxy_pose_query_from_reference.npy",
                    proxy_pose,
                    allow_pickle=False,
                )
                (output / "dgedi_registration.json").write_text(
                    json.dumps({"status": "completed"}),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch(
                    "pose.dgedi_runner._save_self_aligned_mesh",
                    side_effect=fake_save_self_aligned_mesh,
                ),
                patch(
                    "pose.dgedi_runner._save_depth_consistent_proxy_surface_cloud",
                    side_effect=fake_save_surface_cloud,
                ),
                patch(
                    "pose.dgedi_runner.subprocess.run",
                    side_effect=fake_run,
                ),
            ):
                result = run_dgedi_registration(
                    repository_path=repository,
                    python_executable=python_executable,
                    config_path=config_path,
                    reference_self_alignment=SimpleNamespace(
                        scaled_mesh_path=reference_mesh,
                        pose_camera_from_proxy=reference_pose,
                    ),
                    query_self_alignment=SimpleNamespace(
                        scaled_mesh_path=query_mesh,
                        pose_camera_from_proxy=query_pose,
                    ),
                    reference_camera_matrix=np.eye(3),
                    query_camera_matrix=np.eye(3),
                    reference_mask_bool=np.ones((2, 2), dtype=bool),
                    query_mask_bool=np.ones((2, 2), dtype=bool),
                    reference_depth_m=np.ones((2, 2), dtype=np.float32),
                    query_depth_m=np.ones((2, 2), dtype=np.float32),
                    output_directory=root / "output",
                    device="cpu",
                    maximum_surface_depth_residual_m=0.007,
                    minimum_visible_depth_pixels=256,
                )

            self.assertEqual(
                Path(
                    captured_command[
                        captured_command.index("--reference-mesh") + 1
                    ]
                ),
                surface_cloud_paths[0],
            )
            self.assertEqual(
                Path(
                    captured_command[
                        captured_command.index("--query-mesh") + 1
                    ]
                ),
                surface_cloud_paths[1],
            )
            np.testing.assert_allclose(
                result.proxy_pose_query_from_reference,
                proxy_pose,
            )
            np.testing.assert_allclose(
                result.relative_pose_query_from_reference,
                query_pose @ proxy_pose @ np.linalg.inv(reference_pose),
            )
            self.assertEqual(
                result.reference_self_aligned_mesh_path,
                saved_mesh_paths[0],
            )
            self.assertEqual(
                result.query_self_aligned_mesh_path,
                saved_mesh_paths[1],
            )
            self.assertEqual(
                result.reference_registration_cloud_path,
                surface_cloud_paths[0],
            )
            self.assertEqual(
                result.query_registration_cloud_path,
                surface_cloud_paths[1],
            )
            self.assertEqual(
                surface_cloud_calls[0]["minimum_consistent_pixels"],
                256,
            )
            self.assertEqual(
                surface_cloud_calls[0]["local_mesh_path"],
                reference_mesh.resolve(),
            )
            self.assertEqual(
                surface_cloud_calls[1]["local_mesh_path"],
                query_mesh.resolve(),
            )
            self.assertEqual(
                surface_cloud_calls[0]["maximum_depth_residual_m"],
                0.007,
            )
            np.testing.assert_allclose(
                surface_cloud_calls[0]["pose_camera_from_proxy"],
                reference_pose,
            )
            np.testing.assert_allclose(
                surface_cloud_calls[1]["pose_camera_from_proxy"],
                query_pose,
            )


if __name__ == "__main__":
    unittest.main()
