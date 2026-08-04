from __future__ import annotations

import itertools
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from pose.alignment_scorer import (
    AlignmentScoreResult,
    AlignmentScoreWeights,
)
from scale.independent_axis_scale_refiner import (
    _load_scene,
    _robust_dimensions_and_center,
    _sample_surface,
    _score_rendered_candidates,
)
from scale.mesh_scaler import ScaledMeshCandidate


@dataclass(frozen=True)
class ConfidenceSharedDimensionResult:
    confidence: float
    rotation_query_from_reference_axes: np.ndarray
    reference_candidate: ScaledMeshCandidate
    query_candidate: ScaledMeshCandidate
    reference_source_dimensions_m: tuple[float, float, float]
    query_source_dimensions_common_m: tuple[float, float, float]
    raw_shared_dimensions_m: tuple[float, float, float]
    reference_target_dimensions_m: tuple[float, float, float]
    query_target_dimensions_m: tuple[float, float, float]
    reference_scale_factors_common: tuple[float, float, float]
    query_scale_factors_common: tuple[float, float, float]
    reference_score: AlignmentScoreResult
    query_score: AlignmentScoreResult
    deformation_penalty: float
    scale_penalty_weight: float
    objective: float
    summary_path: Path


def build_confidence_factor_grid(
    *,
    minimum_factor: float,
    maximum_factor: float,
    grid_step_count: int,
    confidence: float,
) -> tuple[tuple[float, float, float], ...]:
    """Shrink a log-scale Cartesian grid continuously toward identity."""
    if not (
        math.isfinite(minimum_factor)
        and math.isfinite(maximum_factor)
        and 0.0 < minimum_factor <= 1.0 <= maximum_factor
    ):
        raise ValueError("Dimension-factor bounds must contain 1")
    if isinstance(grid_step_count, bool) or grid_step_count < 2:
        raise ValueError("grid_step_count must be at least 2")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0,1]")

    minimum = 1.0 - confidence * (1.0 - minimum_factor)
    maximum = 1.0 + confidence * (maximum_factor - 1.0)
    values = np.linspace(
        minimum,
        maximum,
        num=grid_step_count,
        dtype=np.float64,
    )
    values[int(np.argmin(np.abs(values - 1.0)))] = 1.0
    unique = {
        tuple(round(float(value), 12) for value in factors)
        for factors in itertools.product(values, repeat=3)
    }
    return tuple(
        sorted(
            unique,
            key=lambda factors: (
                sum(abs(math.log(value)) for value in factors),
                factors,
            ),
        )
    )


def _validate_rotation(value: np.ndarray) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Common-frame rotation must be finite 3x3")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-3):
        raise ValueError("Common-frame rotation must be orthonormal")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("Common-frame rotation determinant must be positive")
    return np.ascontiguousarray(rotation)


def _scale_rotation_toward_identity(
    rotation: np.ndarray,
    confidence: float,
) -> np.ndarray:
    """Scale an SO(3) rotation angle continuously by confidence."""
    validated = _validate_rotation(rotation)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    if confidence == 0.0:
        return np.eye(3, dtype=np.float64)
    if confidence == 1.0:
        return validated.copy()

    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :3] = validated
    quaternion = np.asarray(
        trimesh.transformations.quaternion_from_matrix(homogeneous),
        dtype=np.float64,
    )
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm <= 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = quaternion[1:] / vector_norm
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[0]))
    scaled_half_angle = 0.5 * confidence * angle
    scaled_quaternion = np.concatenate(
        (
            np.asarray([math.cos(scaled_half_angle)], dtype=np.float64),
            axis * math.sin(scaled_half_angle),
        )
    )
    scaled_rotation = trimesh.transformations.quaternion_matrix(
        scaled_quaternion
    )[:3, :3]
    return _validate_rotation(scaled_rotation)


def _centered_affine(
    *,
    linear: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    transform[:3, 3] = center - linear @ center
    return np.ascontiguousarray(transform)


def _canonical_affine(
    *,
    linear: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    transform[:3, 3] = -linear @ center
    return np.ascontiguousarray(transform)


def _deformation_penalty(
    reference_factors: np.ndarray,
    query_factors: np.ndarray,
) -> float:
    logs = np.concatenate(
        (np.log(reference_factors), np.log(query_factors))
    )
    return float(np.mean(logs * logs))


def _write_common_candidate(
    *,
    view_name: str,
    source_candidate: ScaledMeshCandidate,
    source_scene: trimesh.Scene,
    canonical_affine: np.ndarray,
    source_dimensions_m: np.ndarray,
    target_dimensions_m: np.ndarray,
    scale_factors_common: np.ndarray,
    rotation_query_from_reference_axes: np.ndarray,
    output_directory: Path,
) -> ScaledMeshCandidate:
    candidate_directory = Path(output_directory).expanduser().resolve()
    if candidate_directory.exists():
        shutil.rmtree(candidate_directory)
    candidate_directory.mkdir(parents=True, exist_ok=False)

    scene = source_scene.copy()
    scene.apply_transform(canonical_affine)
    suffix = source_candidate.scaled_mesh_path.suffix.lower()
    if not suffix:
        raise ValueError("Shared-dimension source mesh has no suffix")
    mesh_path = candidate_directory / f"mesh_common_dimensions{suffix}"
    scene.export(mesh_path, file_type=suffix.lstrip("."))
    mesh_path = mesh_path.resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Common-dimension mesh was not created: {mesh_path}"
        )

    combined_transform = canonical_affine @ np.asarray(
        source_candidate.scale_transform,
        dtype=np.float64,
    )
    metadata_path = candidate_directory / "common_dimension_candidate.json"
    metadata_path.write_text(
        json.dumps(
            {
                "method": "confidence_aware_pair_common_dimensions",
                "view": view_name,
                "source_candidate_index": source_candidate.candidate_index,
                "pair_shared_isotropic_scale_m": source_candidate.scale_m,
                "source_dimensions_m": source_dimensions_m.tolist(),
                "target_dimensions_m": target_dimensions_m.tolist(),
                "scale_factors_common": scale_factors_common.tolist(),
                "rotation_query_from_reference_axes": (
                    rotation_query_from_reference_axes.tolist()
                ),
                "canonical_affine": canonical_affine.tolist(),
                "combined_scale_transform": combined_transform.tolist(),
                "source_scaled_mesh_path": str(
                    source_candidate.scaled_mesh_path
                ),
                "scaled_mesh_path": str(mesh_path),
                "coordinate_frame": "pair_common_centered",
                "unit": "meter",
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return ScaledMeshCandidate(
        candidate_index=source_candidate.candidate_index,
        scale_m=source_candidate.scale_m,
        normalized_mesh_path=source_candidate.normalized_mesh_path,
        scaled_mesh_path=mesh_path,
        metadata_path=metadata_path.resolve(),
        scale_transform=np.ascontiguousarray(combined_transform),
        artifact_paths=(metadata_path.resolve(),),
    )


def fit_confidence_shared_dimensions(
    *,
    reference_source_candidate: ScaledMeshCandidate,
    query_source_candidate: ScaledMeshCandidate,
    reference_prepared_view: Any,
    query_prepared_view: Any,
    reference_pose_camera_from_proxy: np.ndarray,
    query_pose_camera_from_proxy: np.ndarray,
    rotation_query_from_reference_axes: np.ndarray,
    confidence: float,
    renderer: Any,
    output_directory: Path,
    weights: AlignmentScoreWeights,
    depth_trim_quantile: float,
    minimum_depth_overlap_pixels: int,
    free_space_absolute_tolerance_m: float,
    free_space_relative_tolerance: float,
    quantile_low: float,
    quantile_high: float,
    sample_count: int,
    random_seed: int,
    minimum_factor: float,
    maximum_factor: float,
    grid_step_count: int,
    minimum_scale_penalty_weight: float,
    uncertainty_scale_penalty_weight: float,
) -> ConfidenceSharedDimensionResult:
    """Search pair-common dimensions with deformation controlled by c0."""
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    for name, value in (
        ("minimum_scale_penalty_weight", minimum_scale_penalty_weight),
        ("uncertainty_scale_penalty_weight", uncertainty_scale_penalty_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    input_rotation = _validate_rotation(rotation_query_from_reference_axes)
    rotation = _scale_rotation_toward_identity(
        input_rotation,
        confidence,
    )
    reference_pose = np.asarray(reference_pose_camera_from_proxy, dtype=np.float64)
    query_pose = np.asarray(query_pose_camera_from_proxy, dtype=np.float64)
    for name, pose in (("reference", reference_pose), ("query", query_pose)):
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{name} self pose must be finite 4x4")

    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reference_scene = _load_scene(
        reference_source_candidate.scaled_mesh_path
    )
    query_scene = _load_scene(query_source_candidate.scaled_mesh_path)
    reference_samples = _sample_surface(
        scene=reference_scene,
        sample_count=sample_count,
        random_seed=random_seed,
    )
    query_samples = _sample_surface(
        scene=query_scene,
        sample_count=sample_count,
        random_seed=random_seed,
    )
    reference_dimensions_tuple, reference_center_tuple = (
        _robust_dimensions_and_center(
            reference_samples,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )
    )
    query_samples_common = query_samples @ rotation
    query_dimensions_tuple, query_center_common_tuple = (
        _robust_dimensions_and_center(
            query_samples_common,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )
    )
    reference_dimensions = np.asarray(
        reference_dimensions_tuple,
        dtype=np.float64,
    )
    query_dimensions = np.asarray(
        query_dimensions_tuple,
        dtype=np.float64,
    )
    reference_center = np.asarray(reference_center_tuple, dtype=np.float64)
    query_center_common = np.asarray(
        query_center_common_tuple,
        dtype=np.float64,
    )
    query_center = query_center_common @ rotation.T

    raw_shared_dimensions = np.sqrt(
        reference_dimensions * query_dimensions
    )
    reference_base_target = np.exp(
        (1.0 - confidence) * np.log(reference_dimensions)
        + confidence * np.log(raw_shared_dimensions)
    )
    query_base_target = np.exp(
        (1.0 - confidence) * np.log(query_dimensions)
        + confidence * np.log(raw_shared_dimensions)
    )
    grid = build_confidence_factor_grid(
        minimum_factor=minimum_factor,
        maximum_factor=maximum_factor,
        grid_step_count=grid_step_count,
        confidence=confidence,
    )
    scale_penalty_weight = float(
        minimum_scale_penalty_weight
        + (1.0 - confidence) * uncertainty_scale_penalty_weight
    )

    records: list[dict[str, Any]] = []
    reference_physical_affines: list[np.ndarray] = []
    query_physical_affines: list[np.ndarray] = []
    reference_targets: list[np.ndarray] = []
    query_targets: list[np.ndarray] = []
    reference_factor_values: list[np.ndarray] = []
    query_factor_values: list[np.ndarray] = []
    deformation_penalties: list[float] = []

    for grid_factors_tuple in grid:
        grid_factors = np.asarray(grid_factors_tuple, dtype=np.float64)
        reference_target = reference_base_target * grid_factors
        query_target = query_base_target * grid_factors
        reference_factors = reference_target / reference_dimensions
        query_factors = query_target / query_dimensions

        reference_linear = np.diag(reference_factors)
        query_linear = (
            rotation
            @ np.diag(query_factors)
            @ rotation.T
        )
        reference_physical_affines.append(
            _centered_affine(
                linear=reference_linear,
                center=reference_center,
            )
        )
        query_physical_affines.append(
            _centered_affine(
                linear=query_linear,
                center=query_center,
            )
        )
        reference_targets.append(reference_target)
        query_targets.append(query_target)
        reference_factor_values.append(reference_factors)
        query_factor_values.append(query_factors)
        deformation_penalties.append(
            _deformation_penalty(reference_factors, query_factors)
        )

    reference_transforms = np.stack(
        [
            reference_pose @ affine
            for affine in reference_physical_affines
        ]
    )
    query_transforms = np.stack(
        [query_pose @ affine for affine in query_physical_affines]
    )
    reference_height, reference_width = (
        reference_prepared_view.view.rgb.shape[:2]
    )
    query_height, query_width = query_prepared_view.view.rgb.shape[:2]
    reference_masks, reference_depths = (
        renderer.render_affine_depth_mask_batch(
            mesh_path=reference_source_candidate.scaled_mesh_path,
            transforms_camera_from_proxy=reference_transforms,
            camera_matrix=reference_prepared_view.view.camera_matrix,
            image_height=reference_height,
            image_width=reference_width,
        )
    )
    query_masks, query_depths = renderer.render_affine_depth_mask_batch(
        mesh_path=query_source_candidate.scaled_mesh_path,
        transforms_camera_from_proxy=query_transforms,
        camera_matrix=query_prepared_view.view.camera_matrix,
        image_height=query_height,
        image_width=query_width,
    )
    reference_scores = _score_rendered_candidates(
        prepared_view=reference_prepared_view,
        rendered_masks=reference_masks,
        rendered_depth_m=reference_depths,
        object_scale_m=reference_source_candidate.scale_m,
        weights=weights,
        depth_trim_quantile=depth_trim_quantile,
        minimum_depth_overlap_pixels=minimum_depth_overlap_pixels,
        free_space_absolute_tolerance_m=free_space_absolute_tolerance_m,
        free_space_relative_tolerance=free_space_relative_tolerance,
    )
    query_scores = _score_rendered_candidates(
        prepared_view=query_prepared_view,
        rendered_masks=query_masks,
        rendered_depth_m=query_depths,
        object_scale_m=query_source_candidate.scale_m,
        weights=weights,
        depth_trim_quantile=depth_trim_quantile,
        minimum_depth_overlap_pixels=minimum_depth_overlap_pixels,
        free_space_absolute_tolerance_m=free_space_absolute_tolerance_m,
        free_space_relative_tolerance=free_space_relative_tolerance,
    )

    for index, grid_factors in enumerate(grid):
        observation_loss = float(
            (
                reference_scores[index].total_loss
                + query_scores[index].total_loss
            )
            * 0.5
        )
        penalty = deformation_penalties[index]
        objective = observation_loss + scale_penalty_weight * penalty
        records.append(
            {
                "grid_index": index,
                "grid_factors_common": list(grid_factors),
                "reference_target_dimensions_m": (
                    reference_targets[index].tolist()
                ),
                "query_target_dimensions_m": query_targets[index].tolist(),
                "reference_scale_factors_common": (
                    reference_factor_values[index].tolist()
                ),
                "query_scale_factors_common": (
                    query_factor_values[index].tolist()
                ),
                "reference_alignment_score": asdict(reference_scores[index]),
                "query_alignment_score": asdict(query_scores[index]),
                "mean_observation_loss": observation_loss,
                "deformation_penalty": penalty,
                "scale_penalty_weight": scale_penalty_weight,
                "objective": objective,
            }
        )

    selected_record = min(
        records,
        key=lambda record: (
            record["objective"],
            record["mean_observation_loss"],
            record["deformation_penalty"],
            record["grid_index"],
        ),
    )
    selected_index = int(selected_record["grid_index"])
    selected_reference_factors = reference_factor_values[selected_index]
    selected_query_factors = query_factor_values[selected_index]
    reference_common_affine = _canonical_affine(
        linear=np.diag(selected_reference_factors),
        center=reference_center,
    )
    query_common_affine = _canonical_affine(
        linear=np.diag(selected_query_factors) @ rotation.T,
        center=query_center,
    )
    reference_candidate = _write_common_candidate(
        view_name="reference",
        source_candidate=reference_source_candidate,
        source_scene=reference_scene,
        canonical_affine=reference_common_affine,
        source_dimensions_m=reference_dimensions,
        target_dimensions_m=reference_targets[selected_index],
        scale_factors_common=selected_reference_factors,
        rotation_query_from_reference_axes=rotation,
        output_directory=output_root / "selected" / "reference",
    )
    query_candidate = _write_common_candidate(
        view_name="query",
        source_candidate=query_source_candidate,
        source_scene=query_scene,
        canonical_affine=query_common_affine,
        source_dimensions_m=query_dimensions,
        target_dimensions_m=query_targets[selected_index],
        scale_factors_common=selected_query_factors,
        rotation_query_from_reference_axes=rotation,
        output_directory=output_root / "selected" / "query",
    )

    summary_path = output_root / "selection.json"
    summary_path.write_text(
        json.dumps(
            {
                "method": "confidence_aware_pair_common_dimensions",
                "confidence": confidence,
                "input_rotation_query_from_reference_axes": (
                    input_rotation.tolist()
                ),
                "rotation_query_from_reference_axes": rotation.tolist(),
                "rotation_influence_policy": (
                    "SO(3) shortest-arc interpolation from identity by "
                    "confidence"
                ),
                "pair_shared_scale_verified": math.isclose(
                    reference_source_candidate.scale_m,
                    query_source_candidate.scale_m,
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                ),
                "reference_source_dimensions_m": (
                    reference_dimensions.tolist()
                ),
                "query_source_dimensions_common_m": (
                    query_dimensions.tolist()
                ),
                "raw_shared_dimensions_m": raw_shared_dimensions.tolist(),
                "reference_base_target_dimensions_m": (
                    reference_base_target.tolist()
                ),
                "query_base_target_dimensions_m": query_base_target.tolist(),
                "effective_grid_bounds": {
                    "minimum": (
                        1.0
                        - confidence * (1.0 - minimum_factor)
                    ),
                    "maximum": (
                        1.0
                        + confidence * (maximum_factor - 1.0)
                    ),
                },
                "grid_step_count": grid_step_count,
                "candidate_count": len(grid),
                "minimum_scale_penalty_weight": (
                    minimum_scale_penalty_weight
                ),
                "uncertainty_scale_penalty_weight": (
                    uncertainty_scale_penalty_weight
                ),
                "effective_scale_penalty_weight": scale_penalty_weight,
                "selected_record": selected_record,
                "selected_reference_mesh": str(
                    reference_candidate.scaled_mesh_path
                ),
                "selected_query_mesh": str(
                    query_candidate.scaled_mesh_path
                ),
                "selection_policy": (
                    "minimum pair mean RGB-D loss plus confidence-aware "
                    "log-deformation penalty; no hard fallback"
                ),
                "candidates": records,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return ConfidenceSharedDimensionResult(
        confidence=confidence,
        rotation_query_from_reference_axes=rotation,
        reference_candidate=reference_candidate,
        query_candidate=query_candidate,
        reference_source_dimensions_m=tuple(reference_dimensions),
        query_source_dimensions_common_m=tuple(query_dimensions),
        raw_shared_dimensions_m=tuple(raw_shared_dimensions),
        reference_target_dimensions_m=tuple(
            reference_targets[selected_index]
        ),
        query_target_dimensions_m=tuple(query_targets[selected_index]),
        reference_scale_factors_common=tuple(
            selected_reference_factors
        ),
        query_scale_factors_common=tuple(selected_query_factors),
        reference_score=reference_scores[selected_index],
        query_score=query_scores[selected_index],
        deformation_penalty=deformation_penalties[selected_index],
        scale_penalty_weight=scale_penalty_weight,
        objective=float(selected_record["objective"]),
        summary_path=summary_path.resolve(),
    )
