from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class IndependentPosePathResult:
    method: str
    relative_pose_query_from_reference: np.ndarray
    pose_path: Path
    summary_path: Path


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


def compose_cross_mesh_relative_pose(
    *,
    reference_proxy_to_query_camera: Any,
    query_proxy_to_reference_camera: Any,
    proxy_pose_query_from_reference: Any,
) -> np.ndarray:
    """
    Compose Cross + mesh-to-mesh without either self pose.

    B = T_Cq_from_Pr
    C = T_Cr_from_Pq
    M = T_Pq_from_Pr

    T_Cq_from_Cr = B @ inv(M) @ inv(C)
    """

    reference_cross = _as_rigid_transform(
        reference_proxy_to_query_camera,
        name="reference_proxy_to_query_camera",
    )
    query_cross = _as_rigid_transform(
        query_proxy_to_reference_camera,
        name="query_proxy_to_reference_camera",
    )
    proxy_registration = _as_rigid_transform(
        proxy_pose_query_from_reference,
        name="proxy_pose_query_from_reference",
    )

    relative_pose = (
        reference_cross
        @ np.linalg.inv(proxy_registration)
        @ np.linalg.inv(query_cross)
    )

    return _as_rigid_transform(
        relative_pose,
        name="relative_pose_query_from_reference",
    )


def save_independent_pose_path(
    *,
    method: str,
    relative_pose_query_from_reference: Any,
    output_directory: Path,
    composition: str,
    sources: Mapping[str, Any],
) -> IndependentPosePathResult:
    pose = _as_rigid_transform(
        relative_pose_query_from_reference,
        name="relative_pose_query_from_reference",
    )
    output_root = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    pose_path = (
        output_root / "final_relative_pose.npy"
    )
    summary_path = (
        output_root / "final_selection.json"
    )

    np.save(
        pose_path,
        pose,
        allow_pickle=False,
    )

    payload = {
        "status": "COMPLETED",
        "method": method,
        "pose_convention": (
            "T_query_camera_from_reference_camera"
        ),
        "composition": composition,
        "relative_pose_query_from_reference": (
            pose.tolist()
        ),
        "sources": dict(sources),
        "selection_policy": (
            "independent; no comparison, averaging, "
            "consensus, or rejection against another method"
        ),
    }

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return IndependentPosePathResult(
        method=method,
        relative_pose_query_from_reference=pose,
        pose_path=pose_path,
        summary_path=summary_path,
    )
