from __future__ import annotations

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
    score_alignment,
)
from scale.independent_axis_scale_refiner import (
    _load_scene,
    _robust_dimensions_and_center,
    _sample_surface,
    build_axis_scale_factor_grid,
    centered_axis_scale_affine,
)
from scale.mesh_scaler import ScaledMeshCandidate


@dataclass(frozen=True)
class SharedAxisScaleResult:
    """One pair-shared physical deformation D*, expressed independently in
    the reference and query proxy-local frames via R0."""

    reference_selected_candidate: ScaledMeshCandidate
    query_selected_candidate: ScaledMeshCandidate
    selected_scale_factors: tuple[float, float, float]
    reference_source_dimensions_m: tuple[float, float, float]
    query_source_dimensions_m: tuple[float, float, float]
    reference_baseline_score: AlignmentScoreResult
    query_baseline_score: AlignmentScoreResult
    reference_selected_score: AlignmentScoreResult
    query_selected_score: AlignmentScoreResult
    baseline_joint_loss: float
    selected_joint_loss: float
    improvement_ratio: float
    applied: bool
    summary_path: Path


def centered_shared_axis_scale_affine(
    *,
    scale_factors_reference_uvd: tuple[float, float, float],
    center_proxy_m: tuple[float, float, float],
    rotation_query_from_reference: np.ndarray,
) -> np.ndarray:
    """Express a reference-proxy-frame diag(D) as an affine in query-proxy coordinates.

    dGeDi's G convention is query_proxy = R @ reference_proxy (+t) -- register_one
    is called with source=reference, target=query, and the returned transformation
    maps source (reference) points into the target (query) frame. So R maps
    reference-proxy-frame vectors into query-proxy-frame vectors: for any point
    expressed in reference axes p_r and its query-axes image p_q = R @ p_r,
    deforming p_r by D and re-expressing the result in query axes gives
    R @ (D @ p_r) = R @ D @ R.T @ p_q. The query-side affine is therefore
    R D R.T -- the opposite multiplication order from
    centered_camera_axis_scale_affine's R.T D R (which converts a
    camera-frame scale into a *single view's own* proxy frame, not a
    cross-view proxy-to-proxy scale).
    """
    rotation = np.asarray(rotation_query_from_reference, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_query_from_reference must be finite 3x3")
    if (
        not np.allclose(
            rotation.T @ rotation,
            np.eye(3, dtype=np.float64),
            atol=5e-3,
            rtol=0.0,
        )
        or np.linalg.det(rotation) <= 0.0
    ):
        raise ValueError("rotation_query_from_reference is not a proper rotation")

    factors = np.asarray(scale_factors_reference_uvd, dtype=np.float64)
    center = np.asarray(center_proxy_m, dtype=np.float64)
    if (
        factors.shape != (3,)
        or not np.all(np.isfinite(factors))
        or np.any(factors <= 0.0)
    ):
        raise ValueError("Shared-axis scale factors must be finite and positive")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Shared-axis scale center must be finite")

    linear = rotation @ np.diag(factors) @ rotation.T
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = linear
    affine[:3, 3] = center - linear @ center
    return np.ascontiguousarray(affine, dtype=np.float64)


def _scale_penalty(factors: tuple[float, float, float]) -> float:
    logs = np.log(np.asarray(factors, dtype=np.float64))
    return float(np.mean(logs * logs))


def _render_and_score_one_view(
    *,
    grid: tuple[tuple[float, float, float], ...],
    affine_builder: Any,
    source_dimensions: tuple[float, float, float],
    pose: np.ndarray,
    source_candidate: ScaledMeshCandidate,
    renderer: Any,
    prepared_view: Any,
    weights: AlignmentScoreWeights,
    depth_trim_quantile: float,
    minimum_depth_overlap_pixels: int,
    free_space_absolute_tolerance_m: float,
    free_space_relative_tolerance: float,
) -> tuple[tuple[np.ndarray, ...], tuple[AlignmentScoreResult, ...]]:
    affines = tuple(affine_builder(factors) for factors in grid)
    transforms = np.stack([pose @ affine for affine in affines], axis=0)
    image_height, image_width = prepared_view.view.rgb.shape[:2]
    rendered_masks, rendered_depth_m = renderer.render_affine_depth_mask_batch(
        mesh_path=source_candidate.scaled_mesh_path,
        transforms_camera_from_proxy=transforms,
        camera_matrix=prepared_view.view.camera_matrix,
        image_height=image_height,
        image_width=image_width,
    )
    dimensions = np.asarray(source_dimensions, dtype=np.float64)
    scores = tuple(
        score_alignment(
            observed_mask=prepared_view.segmentation.mask_bool,
            observed_depth_m=prepared_view.view.depth_m,
            rendered_mask=rendered_masks[index],
            rendered_depth_m=rendered_depth_m[index],
            # Keep the pair-shared S* fixed throughout this search so a
            # larger candidate isn't rewarded merely for a depth residual
            # divided by a larger value.
            object_scale_m=source_candidate.scale_m,
            weights=weights,
            depth_trim_quantile=depth_trim_quantile,
            min_depth_overlap_pixels=minimum_depth_overlap_pixels,
            free_space_absolute_tolerance_m=free_space_absolute_tolerance_m,
            free_space_relative_tolerance=free_space_relative_tolerance,
        )
        for index in range(rendered_masks.shape[0])
    )
    del dimensions
    return affines, scores


def _write_shared_scaled_candidate(
    *,
    view_name: str,
    source_candidate: ScaledMeshCandidate,
    source_scene: trimesh.Scene,
    geometry_affine: np.ndarray,
    scale_factors: tuple[float, float, float],
    source_dimensions_m: tuple[float, float, float],
    source_center_m: tuple[float, float, float],
    rotation_query_from_reference: np.ndarray,
    output_directory: Path,
) -> ScaledMeshCandidate:
    candidate_directory = output_directory / "selected_candidate"
    if candidate_directory.exists():
        shutil.rmtree(candidate_directory)
    candidate_directory.mkdir(parents=True, exist_ok=False)

    fitted_scene = source_scene.copy()
    fitted_scene.apply_transform(geometry_affine)
    suffix = source_candidate.scaled_mesh_path.suffix.lower()
    if not suffix:
        raise ValueError("Shared-axis-scale source mesh has no suffix")
    mesh_path = candidate_directory / f"mesh_shared_axis_scaled{suffix}"
    fitted_scene.export(mesh_path, file_type=suffix.lstrip("."))
    mesh_path = mesh_path.resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Shared-axis-scale mesh was not created: {mesh_path}"
        )

    combined_transform = geometry_affine @ np.asarray(
        source_candidate.scale_transform,
        dtype=np.float64,
    )
    metadata_path = candidate_directory / "shared_axis_scale_candidate.json"
    metadata = {
        "method": "shared_dgedi_axis_scale",
        "view": view_name,
        "source_candidate_index": source_candidate.candidate_index,
        "source_object_scale_m": source_candidate.scale_m,
        "scale_factors_reference_uvd": list(scale_factors),
        "source_dimensions_m": list(source_dimensions_m),
        "source_center_m": list(source_center_m),
        "rotation_query_from_reference": (
            np.asarray(rotation_query_from_reference, dtype=np.float64).tolist()
        ),
        "geometry_affine": geometry_affine.tolist(),
        "combined_scale_transform": combined_transform.tolist(),
        "source_scaled_mesh_path": str(source_candidate.scaled_mesh_path),
        "scaled_mesh_path": str(mesh_path),
        "unit": "meter",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
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


def fit_shared_axis_scale(
    *,
    reference_source_candidate: ScaledMeshCandidate,
    reference_prepared_view: Any,
    reference_fixed_pose_camera_from_proxy: np.ndarray,
    query_source_candidate: ScaledMeshCandidate,
    query_prepared_view: Any,
    query_fixed_pose_camera_from_proxy: np.ndarray,
    rotation_query_from_reference: np.ndarray,
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
    scale_penalty_weight: float,
    minimum_loss_improvement_ratio: float,
) -> SharedAxisScaleResult:
    """Search one shared D over BOTH proxies at once (R0-conjugated for query)."""
    rotation = np.asarray(rotation_query_from_reference, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_query_from_reference must be finite 3x3")
    if not math.isfinite(scale_penalty_weight) or scale_penalty_weight < 0.0:
        raise ValueError("Shared-axis-scale penalty weight must be finite and >= 0")
    if not 0.0 <= minimum_loss_improvement_ratio < 1.0:
        raise ValueError("Shared-axis-scale minimum improvement must be in [0,1)")

    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reference_pose = np.asarray(
        reference_fixed_pose_camera_from_proxy, dtype=np.float64
    )
    query_pose = np.asarray(query_fixed_pose_camera_from_proxy, dtype=np.float64)
    for name, pose in (("reference", reference_pose), ("query", query_pose)):
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(f"Shared-axis-scale {name} fixed pose must be finite 4x4")

    reference_scene = _load_scene(reference_source_candidate.scaled_mesh_path)
    query_scene = _load_scene(query_source_candidate.scaled_mesh_path)

    reference_samples = _sample_surface(
        scene=reference_scene, sample_count=sample_count, random_seed=random_seed
    )
    query_samples = _sample_surface(
        scene=query_scene, sample_count=sample_count, random_seed=random_seed
    )
    _, reference_center = _robust_dimensions_and_center(
        reference_samples, quantile_low=quantile_low, quantile_high=quantile_high
    )
    _, query_center = _robust_dimensions_and_center(
        query_samples, quantile_low=quantile_low, quantile_high=quantile_high
    )

    # Dimensions are measured in each view's own proxy-local axes (D is
    # defined in reference-proxy axes, so that is where its scale grid is
    # dimensionally meaningful for the penalty/observability terms).
    reference_dimensions, _ = _robust_dimensions_and_center(
        reference_samples - np.asarray(reference_center, dtype=np.float64),
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    query_dimensions, _ = _robust_dimensions_and_center(
        query_samples - np.asarray(query_center, dtype=np.float64),
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )

    grid = build_axis_scale_factor_grid(
        minimum_factor=minimum_factor,
        maximum_factor=maximum_factor,
        grid_step_count=grid_step_count,
    )

    def _reference_affine(factors: tuple[float, float, float]) -> np.ndarray:
        return centered_axis_scale_affine(
            scale_factors=factors, center_m=reference_center
        )

    def _query_affine(factors: tuple[float, float, float]) -> np.ndarray:
        return centered_shared_axis_scale_affine(
            scale_factors_reference_uvd=factors,
            center_proxy_m=query_center,
            rotation_query_from_reference=rotation,
        )

    reference_affines, reference_scores = _render_and_score_one_view(
        grid=grid,
        affine_builder=_reference_affine,
        source_dimensions=reference_dimensions,
        pose=reference_pose,
        source_candidate=reference_source_candidate,
        renderer=renderer,
        prepared_view=reference_prepared_view,
        weights=weights,
        depth_trim_quantile=depth_trim_quantile,
        minimum_depth_overlap_pixels=minimum_depth_overlap_pixels,
        free_space_absolute_tolerance_m=free_space_absolute_tolerance_m,
        free_space_relative_tolerance=free_space_relative_tolerance,
    )
    query_affines, query_scores = _render_and_score_one_view(
        grid=grid,
        affine_builder=_query_affine,
        source_dimensions=query_dimensions,
        pose=query_pose,
        source_candidate=query_source_candidate,
        renderer=renderer,
        prepared_view=query_prepared_view,
        weights=weights,
        depth_trim_quantile=depth_trim_quantile,
        minimum_depth_overlap_pixels=minimum_depth_overlap_pixels,
        free_space_absolute_tolerance_m=free_space_absolute_tolerance_m,
        free_space_relative_tolerance=free_space_relative_tolerance,
    )

    records: list[dict[str, Any]] = []
    for index, factors in enumerate(grid):
        penalty = _scale_penalty(factors)
        joint_loss = (
            reference_scores[index].total_loss + query_scores[index].total_loss
        )
        records.append(
            {
                "grid_index": index,
                "scale_factors_reference_uvd": list(factors),
                "reference_alignment_score": asdict(reference_scores[index]),
                "query_alignment_score": asdict(query_scores[index]),
                "joint_loss": joint_loss,
                "scale_penalty": penalty,
                "objective": (
                    joint_loss + scale_penalty_weight * penalty
                ),
            }
        )

    baseline_index = grid.index((1.0, 1.0, 1.0))
    baseline_record = records[baseline_index]
    best_record = min(
        records,
        key=lambda record: (
            record["objective"],
            record["joint_loss"],
            record["scale_penalty"],
            record["grid_index"],
        ),
    )
    baseline_joint_loss = float(baseline_record["joint_loss"])
    best_joint_loss = float(best_record["joint_loss"])
    improvement_ratio = (
        (baseline_joint_loss - best_joint_loss) / max(baseline_joint_loss, 1e-12)
    )
    requested_factors = tuple(
        float(value) for value in best_record["scale_factors_reference_uvd"]
    )
    applied = bool(
        requested_factors != (1.0, 1.0, 1.0)
        and improvement_ratio >= minimum_loss_improvement_ratio
    )
    selected_record = best_record if applied else baseline_record
    selected_index = int(selected_record["grid_index"])
    selected_factors = grid[selected_index]

    reference_selected_candidate = reference_source_candidate
    query_selected_candidate = query_source_candidate
    if applied:
        reference_selected_candidate = _write_shared_scaled_candidate(
            view_name="reference",
            source_candidate=reference_source_candidate,
            source_scene=reference_scene,
            geometry_affine=reference_affines[selected_index],
            scale_factors=selected_factors,
            source_dimensions_m=reference_dimensions,
            source_center_m=reference_center,
            rotation_query_from_reference=rotation,
            output_directory=output_root / "reference",
        )
        query_selected_candidate = _write_shared_scaled_candidate(
            view_name="query",
            source_candidate=query_source_candidate,
            source_scene=query_scene,
            geometry_affine=query_affines[selected_index],
            scale_factors=selected_factors,
            source_dimensions_m=query_dimensions,
            source_center_m=query_center,
            rotation_query_from_reference=rotation,
            output_directory=output_root / "query",
        )

    summary_path = output_root / "shared_axis_scale_search.json"
    summary = {
        "method": "shared_dgedi_axis_scale",
        "rotation_query_from_reference": rotation.tolist(),
        "reference_fixed_pose_camera_from_proxy": reference_pose.tolist(),
        "query_fixed_pose_camera_from_proxy": query_pose.tolist(),
        "reference_source_dimensions_m": list(reference_dimensions),
        "query_source_dimensions_m": list(query_dimensions),
        "parameters": {
            "quantile_low": quantile_low,
            "quantile_high": quantile_high,
            "surface_sample_count": sample_count,
            "random_seed": random_seed,
            "minimum_factor": minimum_factor,
            "maximum_factor": maximum_factor,
            "grid_step_count": grid_step_count,
            "candidate_count": len(grid),
            "scale_penalty_weight": scale_penalty_weight,
            "minimum_loss_improvement_ratio": minimum_loss_improvement_ratio,
        },
        "baseline_record": baseline_record,
        "best_requested_record": best_record,
        "selected_record": selected_record,
        "requested_improvement_ratio": improvement_ratio,
        "applied": applied,
        "selected_scale_factors_reference_uvd": list(selected_factors),
        "reference_selected_scaled_mesh_path": str(
            reference_selected_candidate.scaled_mesh_path
        ),
        "query_selected_scaled_mesh_path": str(
            query_selected_candidate.scaled_mesh_path
        ),
        "selection_policy": (
            "minimum joint (reference+query) fixed-pose RGB-D objective, "
            "one D shared by both proxies via Cr=D / Cq=R0 D R0^T; "
            "identity fallback when joint loss improvement is insufficient"
        ),
        "candidates": records,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return SharedAxisScaleResult(
        reference_selected_candidate=reference_selected_candidate,
        query_selected_candidate=query_selected_candidate,
        selected_scale_factors=selected_factors,
        reference_source_dimensions_m=reference_dimensions,
        query_source_dimensions_m=query_dimensions,
        reference_baseline_score=reference_scores[baseline_index],
        query_baseline_score=query_scores[baseline_index],
        reference_selected_score=reference_scores[selected_index],
        query_selected_score=query_scores[selected_index],
        baseline_joint_loss=baseline_joint_loss,
        selected_joint_loss=best_joint_loss if applied else baseline_joint_loss,
        improvement_ratio=float(improvement_ratio),
        applied=applied,
        summary_path=summary_path.resolve(),
    )
