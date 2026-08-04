from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pose.dgedi_observation_validator import (
    DGeDiObservationValidationResult,
)


@dataclass(frozen=True)
class RegistrationConfidenceResult:
    """Continuous evidence for one dGeDi registration."""

    confidence: float
    components: dict[str, float]
    component_weights: dict[str, float]
    raw_metrics: dict[str, float | int]
    summary_path: Path


@dataclass(frozen=True)
class PoseCandidateScore:
    """One final-pose hypothesis scored without a hard gate."""

    name: str
    total_score: float
    observation_loss: float
    rotation_penalty: float
    uncertainty_penalty: float
    deformation_penalty: float
    registration_confidence: float


@dataclass(frozen=True)
class PoseCandidateSelection:
    selected_name: str
    selected_pose_query_from_reference: np.ndarray
    baseline_score: PoseCandidateScore
    refined_score: PoseCandidateScore | None
    absolute_score_margin: float
    relative_score_margin: float
    near_tie: bool
    confidence: float
    summary_path: Path


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _mean_metric(
    validation: DGeDiObservationValidationResult,
    key: str,
) -> float:
    return float(
        (
            float(validation.metrics["reference_cross"][key])
            + float(validation.metrics["query_cross"][key])
        )
        * 0.5
    )


def mean_cross_observation_loss(
    validation: DGeDiObservationValidationResult,
) -> float:
    return _mean_metric(validation, "total_loss")


def _load_json_object(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {resolved}")
    return payload


def compute_registration_confidence(
    *,
    registration_metadata_path: Path,
    observation_validation: DGeDiObservationValidationResult,
    output_path: Path,
    component_weights: Mapping[str, float],
    target_correspondence_fraction: float,
    good_inlier_fitness: float,
    maximum_normalized_rmse: float,
    maximum_normalized_depth_residual: float,
    apply_depth_free_space_gate: bool = False,
) -> RegistrationConfidenceResult:
    """Combine registration and RGB-D evidence into c in [0, 1]."""
    if not isinstance(apply_depth_free_space_gate, bool):
        raise TypeError("apply_depth_free_space_gate must be boolean")
    expected_components = {
        "correspondence",
        "inlier",
        "rmse",
        "mask",
        "depth",
        "free_space",
    }
    weights = {
        str(key): float(value)
        for key, value in component_weights.items()
    }
    if set(weights) != expected_components:
        raise ValueError(
            "Registration-confidence weights must contain exactly "
            f"{sorted(expected_components)}"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("Registration-confidence weights must be finite >= 0")
    weight_sum = float(sum(weights.values()))
    if weight_sum <= 0.0:
        raise ValueError("Registration-confidence weight sum must be positive")
    for name, value in (
        ("target_correspondence_fraction", target_correspondence_fraction),
        ("good_inlier_fitness", good_inlier_fitness),
        ("maximum_normalized_rmse", maximum_normalized_rmse),
        (
            "maximum_normalized_depth_residual",
            maximum_normalized_depth_residual,
        ),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    metadata = _load_json_object(registration_metadata_path)
    icp = metadata.get("icp")
    if not isinstance(icp, dict):
        raise KeyError("dGeDi metadata is missing structured ICP metrics")

    reference_diagnostics = _load_json_object(
        Path(str(metadata["reference_surface_diagnostics"]))
    )
    query_diagnostics = _load_json_object(
        Path(str(metadata["query_surface_diagnostics"]))
    )
    reference_count = int(reference_diagnostics["point_count_saved"])
    query_count = int(query_diagnostics["point_count_saved"])
    minimum_point_count = max(min(reference_count, query_count), 1)
    correspondence_count = int(icp.get("correspondence_count", 0))
    target_correspondence_count = max(
        minimum_point_count * float(target_correspondence_fraction),
        1.0,
    )
    correspondence_confidence = _clip01(
        correspondence_count / target_correspondence_count
    )

    inlier_fitness = float(icp.get("fitness", 0.0))
    inlier_confidence = _clip01(inlier_fitness / good_inlier_fitness)

    normalization_diameter_m = float(metadata["normalization_diameter_m"])
    inlier_rmse_m = float(icp.get("inlier_rmse_m", math.inf))
    normalized_rmse = inlier_rmse_m / max(normalization_diameter_m, 1e-12)
    rmse_confidence = _clip01(
        1.0 - normalized_rmse / maximum_normalized_rmse
    )

    mean_mask_iou = _mean_metric(observation_validation, "mask_iou")
    mask_confidence = _clip01(mean_mask_iou)

    mean_depth_residual_normalized = _mean_metric(
        observation_validation,
        "depth_residual_normalized",
    )
    depth_confidence = _clip01(
        1.0
        - mean_depth_residual_normalized
        / maximum_normalized_depth_residual
    )

    mean_free_space_loss = _mean_metric(
        observation_validation,
        "free_space_loss",
    )
    free_space_confidence = _clip01(1.0 - mean_free_space_loss)

    components = {
        "correspondence": correspondence_confidence,
        "inlier": inlier_confidence,
        "rmse": rmse_confidence,
        "mask": mask_confidence,
        "depth": depth_confidence,
        "free_space": free_space_confidence,
    }
    weighted_mean_confidence = _clip01(
        sum(weights[key] * components[key] for key in components)
        / weight_sum
    )
    depth_free_space_gate = math.sqrt(
        depth_confidence * free_space_confidence
    )
    confidence = _clip01(
        weighted_mean_confidence
        * (depth_free_space_gate if apply_depth_free_space_gate else 1.0)
    )
    raw_metrics: dict[str, float | int] = {
        "reference_point_count": reference_count,
        "query_point_count": query_count,
        "minimum_point_count": minimum_point_count,
        "correspondence_count": correspondence_count,
        "target_correspondence_count": float(target_correspondence_count),
        "inlier_fitness": inlier_fitness,
        "inlier_rmse_m": inlier_rmse_m,
        "normalization_diameter_m": normalization_diameter_m,
        "normalized_inlier_rmse": normalized_rmse,
        "mean_cross_mask_iou": mean_mask_iou,
        "mean_cross_depth_residual_normalized": (
            mean_depth_residual_normalized
        ),
        "mean_cross_free_space_loss": mean_free_space_loss,
        "mean_cross_observation_loss": mean_cross_observation_loss(
            observation_validation
        ),
    }
    resolved_output = Path(output_path).expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(
        json.dumps(
            {
                "method": "continuous_dgedi_registration_confidence",
                "confidence": confidence,
                "components": components,
                "component_weights": weights,
                "aggregation": {
                    "weighted_mean_confidence": weighted_mean_confidence,
                    "depth_free_space_gate_applied": (
                        apply_depth_free_space_gate
                    ),
                    "depth_free_space_gate": depth_free_space_gate,
                    "formula": (
                        "weighted_mean * sqrt(depth * free_space)"
                        if apply_depth_free_space_gate
                        else "weighted_mean"
                    ),
                },
                "raw_metrics": raw_metrics,
                "source_registration_metadata": str(
                    Path(registration_metadata_path).expanduser().resolve()
                ),
                "source_observation_metrics": str(
                    observation_validation.summary_path
                ),
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return RegistrationConfidenceResult(
        confidence=confidence,
        components=components,
        component_weights=weights,
        raw_metrics=raw_metrics,
        summary_path=resolved_output,
    )


def _rotation_angle_deg(pose: np.ndarray) -> float:
    matrix = np.asarray(pose, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Proxy pose must be a finite 4x4 matrix")
    cosine = float(np.clip((np.trace(matrix[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def score_pose_candidate(
    *,
    name: str,
    proxy_pose_query_from_reference: np.ndarray,
    observation_validation: DGeDiObservationValidationResult,
    registration_confidence: RegistrationConfidenceResult,
    deformation_penalty: float,
    rotation_penalty_weight: float,
    uncertainty_penalty_weight: float,
    deformation_penalty_weight: float,
) -> PoseCandidateScore:
    for value_name, value in (
        ("deformation_penalty", deformation_penalty),
        ("rotation_penalty_weight", rotation_penalty_weight),
        ("uncertainty_penalty_weight", uncertainty_penalty_weight),
        ("deformation_penalty_weight", deformation_penalty_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{value_name} must be finite and non-negative")
    observation_loss = mean_cross_observation_loss(observation_validation)
    rotation_penalty = _rotation_angle_deg(
        proxy_pose_query_from_reference
    ) / 180.0
    uncertainty_penalty = 1.0 - registration_confidence.confidence
    total_score = float(
        observation_loss
        + rotation_penalty_weight * rotation_penalty
        + uncertainty_penalty_weight * uncertainty_penalty
        + deformation_penalty_weight * deformation_penalty
    )
    return PoseCandidateScore(
        name=name,
        total_score=total_score,
        observation_loss=observation_loss,
        rotation_penalty=rotation_penalty,
        uncertainty_penalty=uncertainty_penalty,
        deformation_penalty=float(deformation_penalty),
        registration_confidence=registration_confidence.confidence,
    )


def select_pose_candidates(
    *,
    baseline_pose_query_from_reference: np.ndarray,
    baseline_score: PoseCandidateScore,
    refined_pose_query_from_reference: np.ndarray | None,
    refined_score: PoseCandidateScore | None,
    near_tie_margin: float,
    output_path: Path,
    refined_failure: str | None = None,
) -> PoseCandidateSelection:
    if not math.isfinite(near_tie_margin) or near_tie_margin <= 0.0:
        raise ValueError("near_tie_margin must be finite and positive")
    if (refined_pose_query_from_reference is None) != (refined_score is None):
        raise ValueError("Refined pose and score must be both present or absent")

    selected_pose = np.asarray(
        baseline_pose_query_from_reference,
        dtype=np.float64,
    )
    selected_score = baseline_score
    if (
        refined_score is not None
        and refined_pose_query_from_reference is not None
        and refined_score.total_score < baseline_score.total_score
    ):
        selected_score = refined_score
        selected_pose = np.asarray(
            refined_pose_query_from_reference,
            dtype=np.float64,
        )

    competing_scores = [baseline_score.total_score]
    if refined_score is not None:
        competing_scores.append(refined_score.total_score)
    if len(competing_scores) == 2:
        absolute_margin = float(abs(competing_scores[0] - competing_scores[1]))
        relative_margin = float(
            absolute_margin / max(min(competing_scores), 1e-12)
        )
    else:
        absolute_margin = 0.0
        relative_margin = 0.0
    near_tie = bool(
        refined_score is not None and absolute_margin < near_tie_margin
    )
    margin_confidence = (
        _clip01(absolute_margin / near_tie_margin)
        if refined_score is not None
        else 1.0
    )
    final_confidence = _clip01(
        selected_score.registration_confidence * margin_confidence
    )

    resolved_output = Path(output_path).expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "SELECTED",
        "selection_policy": (
            "continuous H0/H1 score comparison; no observation hard reject"
        ),
        "selected_name": selected_score.name,
        "selected_pose_query_from_reference": selected_pose.tolist(),
        "baseline_pose_query_from_reference": np.asarray(
            baseline_pose_query_from_reference,
            dtype=np.float64,
        ).tolist(),
        "refined_pose_query_from_reference": (
            np.asarray(
                refined_pose_query_from_reference,
                dtype=np.float64,
            ).tolist()
            if refined_pose_query_from_reference is not None
            else None
        ),
        "baseline_score": asdict(baseline_score),
        "refined_score": (
            asdict(refined_score) if refined_score is not None else None
        ),
        "absolute_score_margin": absolute_margin,
        "relative_score_margin": relative_margin,
        "near_tie_margin": near_tie_margin,
        "near_tie": near_tie,
        "confidence": final_confidence,
        "refined_failure": refined_failure,
    }
    resolved_output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return PoseCandidateSelection(
        selected_name=selected_score.name,
        selected_pose_query_from_reference=np.ascontiguousarray(selected_pose),
        baseline_score=baseline_score,
        refined_score=refined_score,
        absolute_score_margin=absolute_margin,
        relative_score_margin=relative_margin,
        near_tie=near_tie,
        confidence=final_confidence,
        summary_path=resolved_output,
    )
