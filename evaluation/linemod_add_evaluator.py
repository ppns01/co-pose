from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evaluation.relative_pose_evaluator import (
    BOPFrameGTSpec,
    load_bop_linemod_absolute_gt_pose,
)


ADD_AUC_MAXIMUM_THRESHOLD_M = 0.1


@dataclass(frozen=True)
class LinemodADDEvaluationResult:
    """ADD/ADD-S evaluation derived from a relative pose estimate."""

    add_m: float
    adds_m: float
    add_normalized: float
    adds_normalized: float
    add_or_adds_m: float
    add_or_adds_normalized: float
    add_metric_used: str
    add_or_adds_0_1d_threshold_m: float
    add_or_adds_0_1d_passed: bool
    object_diameter_m: float
    symmetry_aware: bool
    symmetry_type: str
    model_point_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "add_m": self.add_m,
            "adds_m": self.adds_m,
            "add_normalized": self.add_normalized,
            "adds_normalized": self.adds_normalized,
            "add_or_adds_m": self.add_or_adds_m,
            "add_or_adds_normalized": (
                self.add_or_adds_normalized
            ),
            "add_metric_used": self.add_metric_used,
            "add_or_adds_0_1d_threshold_m": (
                self.add_or_adds_0_1d_threshold_m
            ),
            "add_or_adds_0_1d_passed": (
                self.add_or_adds_0_1d_passed
            ),
            "object_diameter_m": self.object_diameter_m,
            "symmetry_aware": self.symmetry_aware,
            "symmetry_type": self.symmetry_type,
            "model_point_count": self.model_point_count,
        }


def calculate_add_auc(
    errors: list[float] | tuple[float, ...] | NDArray[np.floating],
    *,
    maximum_threshold: float = ADD_AUC_MAXIMUM_THRESHOLD_M,
    target_count: int | None = None,
) -> float | None:
    """Calculate normalized area under an empirical ADD recall curve.

    Errors at or above ``maximum_threshold`` and missing targets contribute
    zero area. The result is normalized to [0, 1]. This is equivalent to the
    standard VOC-style integral of recall over thresholds from zero to the
    selected maximum.
    """
    threshold = float(maximum_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            "maximum_threshold must be finite and positive: "
            f"{maximum_threshold}"
        )

    error_array = np.asarray(errors, dtype=np.float64).reshape(-1)
    if np.any(error_array < 0.0):
        raise ValueError("ADD errors must be non-negative.")
    finite_errors = error_array[np.isfinite(error_array)]

    denominator = (
        int(target_count)
        if target_count is not None
        else int(error_array.size)
    )
    if denominator < finite_errors.size:
        raise ValueError(
            "target_count cannot be smaller than the number of finite "
            f"errors: {denominator} < {finite_errors.size}"
        )
    if denominator <= 0:
        return None

    per_target_area = np.clip(
        1.0 - finite_errors / threshold,
        0.0,
        1.0,
    )
    return float(np.sum(per_target_area) / denominator)


def _validate_rigid_pose(
    pose: NDArray[np.floating],
    name: str,
) -> NDArray[np.float64]:
    pose_array = np.asarray(pose, dtype=np.float64)
    if pose_array.shape != (4, 4):
        raise ValueError(
            f"{name} must have shape (4, 4): {pose_array.shape}"
        )
    if not np.all(np.isfinite(pose_array)):
        raise ValueError(f"{name} contains NaN or Inf.")
    if not np.allclose(
        pose_array[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(f"{name} is not a homogeneous transform.")
    rotation = pose_array[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(f"{name} rotation is not orthonormal.")
    if not np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(f"{name} rotation determinant is not 1.")
    return np.ascontiguousarray(pose_array)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


@lru_cache(maxsize=8)
def _load_models_info(model_root_value: str) -> dict[str, Any]:
    return _load_json_object(Path(model_root_value) / "models_info.json")


def _load_bop_model_vertices_m(model_path: Path) -> NDArray[np.float64]:
    if not model_path.is_file():
        raise FileNotFoundError(f"BOP evaluation model not found: {model_path}")

    import trimesh

    loaded = trimesh.load(
        model_path,
        force="scene",
        process=False,
    )
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Evaluation model is empty: {model_path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded

    vertices_mm = np.asarray(mesh.vertices, dtype=np.float64)
    if (
        vertices_mm.ndim != 2
        or vertices_mm.shape[1] != 3
        or vertices_mm.shape[0] == 0
        or not np.all(np.isfinite(vertices_mm))
    ):
        raise ValueError(
            f"Evaluation model has invalid vertices: {model_path}"
        )

    # BOP model coordinates and models_info diameters are millimetres.
    return np.ascontiguousarray(vertices_mm * 0.001)


@lru_cache(maxsize=64)
def _load_model_inputs(
    model_root_value: str,
    object_id: int,
) -> tuple[NDArray[np.float64], float, str]:
    model_root = Path(model_root_value)
    models_info = _load_models_info(model_root_value)
    object_key = str(object_id)
    model_info = models_info.get(object_key)
    if not isinstance(model_info, dict):
        raise KeyError(
            f"Object {object_key} is absent from "
            f"{model_root / 'models_info.json'}"
        )

    diameter_mm = float(model_info.get("diameter", 0.0))
    if not np.isfinite(diameter_mm) or diameter_mm <= 0.0:
        raise ValueError(
            f"Invalid model diameter for object {object_key}: {diameter_mm}"
        )
    model_path = model_root / f"obj_{object_id:06d}.ply"
    return (
        _load_bop_model_vertices_m(model_path),
        diameter_mm * 0.001,
        _symmetry_type(model_info),
    )


def _symmetry_type(model_info: dict[str, Any]) -> str:
    has_discrete = bool(model_info.get("symmetries_discrete"))
    has_continuous = bool(model_info.get("symmetries_continuous"))
    if has_discrete and has_continuous:
        return "discrete+continuous"
    if has_continuous:
        return "continuous"
    if has_discrete:
        return "discrete"
    return "none"


def _transform_points(
    points: NDArray[np.float64],
    pose_camera_from_object: NDArray[np.float64],
) -> NDArray[np.float64]:
    return (
        points @ pose_camera_from_object[:3, :3].T
        + pose_camera_from_object[:3, 3]
    )


def evaluate_linemod_add_metrics(
    *,
    predicted_relative_pose: NDArray[np.floating],
    reference_frame: BOPFrameGTSpec,
    query_frame: BOPFrameGTSpec,
    models_directory: Path | None = None,
) -> LinemodADDEvaluationResult:
    """
    Evaluate a relative pose with LINEMOD ADD/ADD-S at 0.1 diameter.

    The prediction is converted to an absolute query object pose only for
    evaluation:

        T_query_from_object_hat
            = H_query_from_reference_hat @ T_reference_from_object_gt
    """
    if reference_frame.object_id != query_frame.object_id:
        raise ValueError(
            "Reference and query object IDs differ: "
            f"{reference_frame.object_id} != {query_frame.object_id}"
        )

    predicted_relative = _validate_rigid_pose(
        predicted_relative_pose,
        "predicted_relative_pose",
    )
    reference_absolute = _validate_rigid_pose(
        load_bop_linemod_absolute_gt_pose(reference_frame),
        "reference_absolute_pose",
    )
    query_absolute = _validate_rigid_pose(
        load_bop_linemod_absolute_gt_pose(query_frame),
        "query_absolute_pose",
    )
    predicted_query_absolute = _validate_rigid_pose(
        predicted_relative @ reference_absolute,
        "predicted_query_absolute_pose",
    )

    model_root = (
        Path(models_directory).expanduser().resolve()
        if models_directory is not None
        else (
            Path(reference_frame.dataset_root).expanduser().resolve()
            / "models_eval"
        )
    )
    model_points, diameter_m, symmetry_type = _load_model_inputs(
        str(model_root),
        reference_frame.object_id,
    )
    predicted_points = _transform_points(
        model_points,
        predicted_query_absolute,
    )
    ground_truth_points = _transform_points(
        model_points,
        query_absolute,
    )

    add_m = float(
        np.mean(
            np.linalg.norm(
                predicted_points - ground_truth_points,
                axis=1,
            )
        )
    )

    from scipy.spatial import cKDTree

    nearest_distances, _ = cKDTree(ground_truth_points).query(
        predicted_points,
        k=1,
    )
    adds_m = float(np.mean(nearest_distances))

    symmetry_aware = symmetry_type != "none"
    selected_error_m = adds_m if symmetry_aware else add_m
    selected_normalized = selected_error_m / diameter_m
    threshold_m = 0.1 * diameter_m

    return LinemodADDEvaluationResult(
        add_m=add_m,
        adds_m=adds_m,
        add_normalized=add_m / diameter_m,
        adds_normalized=adds_m / diameter_m,
        add_or_adds_m=selected_error_m,
        add_or_adds_normalized=selected_normalized,
        add_metric_used="ADD-S" if symmetry_aware else "ADD",
        add_or_adds_0_1d_threshold_m=threshold_m,
        add_or_adds_0_1d_passed=(selected_error_m < threshold_m),
        object_diameter_m=diameter_m,
        symmetry_aware=symmetry_aware,
        symmetry_type=symmetry_type,
        model_point_count=int(model_points.shape[0]),
    )
