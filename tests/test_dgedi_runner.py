from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pose.dgedi_runner import run_dgedi_registration


def _translation(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (x, y, z)
    return pose


class DgediRunnerTests(unittest.TestCase):
    def test_local_meshes_are_registered_then_composed(self) -> None:
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

            def fake_save_self_aligned_mesh(**kwargs: object) -> Path:
                output_path = Path(kwargs["output_mesh_path"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
                return output_path.resolve()

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
                    output_directory=root / "output",
                    device="cpu",
                )

            self.assertEqual(
                Path(
                    captured_command[
                        captured_command.index("--reference-mesh") + 1
                    ]
                ),
                reference_mesh.resolve(),
            )
            self.assertEqual(
                Path(
                    captured_command[
                        captured_command.index("--query-mesh") + 1
                    ]
                ),
                query_mesh.resolve(),
            )
            np.testing.assert_allclose(
                result.proxy_pose_query_from_reference,
                proxy_pose,
            )
            np.testing.assert_allclose(
                result.relative_pose_query_from_reference,
                query_pose @ proxy_pose @ np.linalg.inv(reference_pose),
            )


if __name__ == "__main__":
    unittest.main()
