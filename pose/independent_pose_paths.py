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


def save_independent_pose_path(
    *,
    method: str,
    relative_pose_query_from_reference: Any,
    output_directory: Path,
    composition: str,
    sources: Mapping[str, Any],
    selection_policy: str | None = None,
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
        "pose_accepted": True,
        "sources": dict(sources),
        "selection_policy": (
            selection_policy
            or (
                "independent; no comparison, averaging, "
                "consensus, or rejection against another method"
            )
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


def save_rejected_pose_path(
    *,
    method: str,
    output_directory: Path,
    reason: str,
    sources: Mapping[str, Any],
    relative_pose_query_from_reference: Any | None = None,
    composition: str | None = None,
) -> Path:
    """Write a REJECT summary and optionally preserve its raw estimate."""
    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "final_selection.json"
    rejected_pose_path: Path | None = None
    rejected_pose: np.ndarray | None = None
    if relative_pose_query_from_reference is not None:
        rejected_pose = _as_rigid_transform(
            relative_pose_query_from_reference,
            name="rejected relative_pose_query_from_reference",
        )
        rejected_pose_path = output_root / "rejected_relative_pose.npy"
        np.save(
            rejected_pose_path,
            rejected_pose,
            allow_pickle=False,
        )

    payload = {
        "status": "REJECT",
        "method": method,
        "pose_convention": "T_query_camera_from_reference_camera",
        "composition": composition,
        "pose_accepted": False,
        "relative_pose_query_from_reference": (
            rejected_pose.tolist()
            if rejected_pose is not None
            else None
        ),
        "rejected_pose_path": (
            str(rejected_pose_path)
            if rejected_pose_path is not None
            else None
        ),
        "reason": str(reason),
        "sources": dict(sources),
        "selection_policy": (
            "independent dual-proxy result only; no same-proxy or "
            "B*A^-1 fallback"
        ),
    }
    with summary_path.open(mode="w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    return summary_path
