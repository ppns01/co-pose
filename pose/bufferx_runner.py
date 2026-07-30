from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BufferXRegistrationResult:
    """Independent BUFFER-X mesh-to-mesh registration result."""

    proxy_pose_query_from_reference: np.ndarray
    relative_pose_query_from_reference: np.ndarray
    proxy_pose_path: Path
    relative_pose_path: Path
    metadata_path: Path


def _as_rigid_transform(
    matrix: Any,
    *,
    name: str,
) -> np.ndarray:
    transform = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if transform.shape != (4, 4):
        raise ValueError(
            f"{name} must have shape (4, 4): "
            f"{transform.shape}"
        )

    if not np.all(np.isfinite(transform)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if not np.allclose(
        transform[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name} has an invalid homogeneous row."
        )

    rotation = transform[:3, :3]

    if (
        not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=5e-3,
            rtol=0.0,
        )
        or np.linalg.det(rotation) <= 0.0
    ):
        raise ValueError(
            f"{name} does not contain a valid rotation."
        )

    return transform


def compose_bufferx_relative_pose(
    *,
    reference_pose_camera_from_proxy: Any,
    query_pose_camera_from_proxy: Any,
    proxy_pose_query_from_reference: Any,
) -> np.ndarray:
    """
    Compose the camera relative pose from two self poses and BUFFER-X.

    T_Cq_Cr = T_Cq_Pq @ T_Pq_Pr @ inv(T_Cr_Pr)
    """

    reference_self = _as_rigid_transform(
        reference_pose_camera_from_proxy,
        name="reference_pose_camera_from_proxy",
    )
    query_self = _as_rigid_transform(
        query_pose_camera_from_proxy,
        name="query_pose_camera_from_proxy",
    )
    proxy_registration = _as_rigid_transform(
        proxy_pose_query_from_reference,
        name="proxy_pose_query_from_reference",
    )

    relative_pose = (
        query_self
        @ proxy_registration
        @ np.linalg.inv(reference_self)
    )

    return _as_rigid_transform(
        relative_pose,
        name="relative_pose_query_from_reference",
    )


def run_bufferx_registration(
    *,
    repository_path: Path,
    python_executable: Path,
    reference_self_alignment: Any,
    query_self_alignment: Any,
    output_directory: Path,
    pose_estimator: str,
    sample_count: int,
    icp_refine: bool,
    random_seed: int,
) -> BufferXRegistrationResult:
    """Run BUFFER-X in its own Python environment."""

    repository = (
        Path(repository_path)
        .expanduser()
        .resolve()
    )
    python_path = (
        Path(python_executable)
        .expanduser()
        .resolve()
    )
    output_root = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )
    inference_script = (
        Path(__file__)
        .resolve()
        .with_name("bufferx_inference.py")
    )

    if not repository.is_dir():
        raise FileNotFoundError(
            f"BUFFER-X repository not found: {repository}"
        )

    if not python_path.is_file():
        raise FileNotFoundError(
            f"BUFFER-X Python not found: {python_path}"
        )

    if not inference_script.is_file():
        raise FileNotFoundError(
            "BUFFER-X inference script not found: "
            f"{inference_script}"
        )

    reference_mesh_path = Path(
        reference_self_alignment.scaled_mesh_path
    ).resolve()
    query_mesh_path = Path(
        query_self_alignment.scaled_mesh_path
    ).resolve()

    for name, mesh_path in (
        ("Reference scaled mesh", reference_mesh_path),
        ("Query scaled mesh", query_mesh_path),
    ):
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"{name} not found: {mesh_path}"
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        str(python_path),
        str(inference_script),
        "--repository",
        str(repository),
        "--reference-mesh",
        str(reference_mesh_path),
        "--query-mesh",
        str(query_mesh_path),
        "--output-directory",
        str(output_root),
        "--pose-estimator",
        pose_estimator,
        "--sample-count",
        str(sample_count),
        "--random-seed",
        str(random_seed),
    ]

    if icp_refine:
        command.append("--icp-refine")
    else:
        command.append("--no-icp-refine")

    completed = subprocess.run(
        command,
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "BUFFER-X execution failed.\n"
            f"Exit code: {completed.returncode}\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    proxy_pose_path = (
        output_root
        / "bufferx_proxy_pose_query_from_reference.npy"
    )
    metadata_path = (
        output_root / "bufferx_registration.json"
    )

    if not proxy_pose_path.is_file():
        raise FileNotFoundError(
            "BUFFER-X did not save its proxy pose: "
            f"{proxy_pose_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "BUFFER-X did not save its metadata: "
            f"{metadata_path}"
        )

    proxy_pose = _as_rigid_transform(
        np.load(
            proxy_pose_path,
            allow_pickle=False,
        ),
        name="proxy_pose_query_from_reference",
    )
    reference_self_pose = _as_rigid_transform(
        reference_self_alignment
        .pose_camera_from_proxy,
        name="reference_pose_camera_from_proxy",
    )
    query_self_pose = _as_rigid_transform(
        query_self_alignment
        .pose_camera_from_proxy,
        name="query_pose_camera_from_proxy",
    )
    relative_pose = compose_bufferx_relative_pose(
        reference_pose_camera_from_proxy=(
            reference_self_pose
        ),
        query_pose_camera_from_proxy=(
            query_self_pose
        ),
        proxy_pose_query_from_reference=proxy_pose,
    )
    relative_pose_path = (
        output_root
        / "bufferx_relative_pose_query_from_reference.npy"
    )
    np.save(
        relative_pose_path,
        relative_pose.astype(np.float64),
        allow_pickle=False,
    )

    with metadata_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    metadata.update(
        {
            "status": "completed",
            "pose_convention": (
                "T_query_camera_from_reference_camera"
            ),
            "proxy_pose_convention": (
                "T_query_proxy_from_reference_proxy"
            ),
            "reference_scaled_mesh_path": str(
                reference_mesh_path
            ),
            "query_scaled_mesh_path": str(
                query_mesh_path
            ),
            "reference_pose_camera_from_proxy": (
                reference_self_pose.tolist()
            ),
            "query_pose_camera_from_proxy": (
                query_self_pose.tolist()
            ),
            "relative_pose_query_from_reference": (
                relative_pose.tolist()
            ),
            "relative_pose_path": str(
                relative_pose_path
            ),
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return BufferXRegistrationResult(
        proxy_pose_query_from_reference=proxy_pose,
        relative_pose_query_from_reference=(
            relative_pose
        ),
        proxy_pose_path=proxy_pose_path,
        relative_pose_path=relative_pose_path,
        metadata_path=metadata_path,
    )

