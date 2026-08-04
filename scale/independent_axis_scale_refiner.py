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
    score_alignment,
)
from scale.mesh_scaler import ScaledMeshCandidate


@dataclass(frozen=True)
class IndependentAxisScaleResult:
    """One view's camera-axis scale search with its initial pose held fixed."""

    view_name: str
    source_candidate: ScaledMeshCandidate
    selected_candidate: ScaledMeshCandidate
    selected_scale_factors: tuple[float, float, float]
    source_center_m: tuple[float, float, float]
    source_dimensions_m: tuple[float, float, float]
    selected_dimensions_m: tuple[float, float, float]
    depth_observability: float
    axis_penalty_weights: tuple[float, float, float]
    baseline_score: AlignmentScoreResult
    selected_score: AlignmentScoreResult
    scale_penalty: float
    objective: float
    improvement_ratio: float
    applied: bool
    summary_path: Path


def build_axis_scale_factor_grid(
    *,
    minimum_factor: float,
    maximum_factor: float,
    grid_step_count: int,
) -> tuple[tuple[float, float, float], ...]:
    """Build an XYZ Cartesian grid, ordered with identity first."""
    if not (
        math.isfinite(minimum_factor)
        and math.isfinite(maximum_factor)
        and 0.0 < minimum_factor <= 1.0 <= maximum_factor
    ):
        raise ValueError("Axis-scale factor bounds must contain 1")
    if isinstance(grid_step_count, bool) or grid_step_count < 2:
        raise ValueError("Axis-scale grid_step_count must be >= 2")

    values = np.linspace(
        minimum_factor,
        maximum_factor,
        num=grid_step_count,
        dtype=np.float64,
    )
    values[int(np.argmin(np.abs(values - 1.0)))] = 1.0
    factors = tuple(
        tuple(float(value) for value in item)
        for item in itertools.product(values, repeat=3)
    )
    return tuple(
        sorted(
            factors,
            key=lambda item: (
                sum(abs(math.log(value)) for value in item),
                item,
            ),
        )
    )


def build_local_axis_scale_factor_grid(
    *,
    center_factors: tuple[float, float, float],
    radius: float,
    grid_step_count: int,
    minimum_factor: float,
    maximum_factor: float,
) -> tuple[tuple[float, float, float], ...]:
    """Build a fine XYZ grid around one coarse candidate, clipped to [min,max]."""
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Local axis-scale radius must be finite and positive")
    if isinstance(grid_step_count, bool) or grid_step_count < 2:
        raise ValueError("Local axis-scale grid_step_count must be >= 2")
    center = np.asarray(center_factors, dtype=np.float64)
    if (
        center.shape != (3,)
        or not np.all(np.isfinite(center))
        or np.any(center <= 0.0)
    ):
        raise ValueError("Local axis-scale center must be finite and positive")

    per_axis_values = []
    for value in center:
        lower = max(float(value) - radius, minimum_factor)
        upper = min(float(value) + radius, maximum_factor)
        if upper <= lower:
            lower = upper = float(np.clip(value, minimum_factor, maximum_factor))
        per_axis_values.append(
            np.linspace(lower, upper, num=grid_step_count, dtype=np.float64)
        )

    factors = {
        tuple(round(float(value), 12) for value in combo)
        for combo in itertools.product(*per_axis_values)
    }
    return tuple(
        sorted(
            factors,
            key=lambda item: (
                sum(
                    (math.log(value) - math.log(c)) ** 2
                    for value, c in zip(item, center, strict=True)
                ),
                item,
            ),
        )
    )


def _load_scene(mesh_path: Path) -> trimesh.Scene:
    resolved = Path(mesh_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Axis-scale source mesh: {resolved}")
    scene = trimesh.load_scene(resolved, process=False)
    if scene.is_empty:
        raise ValueError(f"Axis-scale source mesh is empty: {resolved}")
    return scene


def _sample_surface(
    *,
    scene: trimesh.Scene,
    sample_count: int,
    random_seed: int,
) -> np.ndarray:
    if sample_count < 100:
        raise ValueError("Axis-scale sample_count must be >= 100")
    if random_seed < 0:
        raise ValueError("Axis-scale random_seed must be >= 0")
    mesh = scene.to_mesh()
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("Axis-scale statistics mesh is empty")
    try:
        points, _ = trimesh.sample.sample_surface(
            mesh=mesh,
            count=sample_count,
            seed=random_seed,
        )
    except TypeError as error:
        raise RuntimeError(
            "The installed trimesh lacks deterministic surface sampling"
        ) from error
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (sample_count, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Invalid axis-scale surface samples")
    return np.ascontiguousarray(points, dtype=np.float64)


def _robust_dimensions_and_center(
    points: np.ndarray,
    *,
    quantile_low: float,
    quantile_high: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("Axis-scale quantiles are invalid")
    values = np.asarray(points, dtype=np.float64)
    lower = np.quantile(values, quantile_low, axis=0)
    upper = np.quantile(values, quantile_high, axis=0)
    dimensions = upper - lower
    center = (lower + upper) * 0.5
    if (
        not np.all(np.isfinite(dimensions))
        or np.any(dimensions <= 0.0)
        or not np.all(np.isfinite(center))
    ):
        raise ValueError("Axis-scale robust bounds are invalid")
    return (
        tuple(float(value) for value in dimensions),
        tuple(float(value) for value in center),
    )


def centered_axis_scale_affine(
    *,
    scale_factors: tuple[float, float, float],
    center_m: tuple[float, float, float],
) -> np.ndarray:
    factors = np.asarray(scale_factors, dtype=np.float64)
    center = np.asarray(center_m, dtype=np.float64)
    if (
        factors.shape != (3,)
        or not np.all(np.isfinite(factors))
        or np.any(factors <= 0.0)
    ):
        raise ValueError("Axis-scale factors must be finite and positive")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Axis-scale center must be finite")
    linear = np.diag(factors)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = linear
    affine[:3, 3] = center - linear @ center
    return np.ascontiguousarray(affine, dtype=np.float64)


def centered_camera_axis_scale_affine(
    *,
    scale_factors_uvd: tuple[float, float, float],
    center_proxy_m: tuple[float, float, float],
    rotation_camera_from_proxy: np.ndarray,
) -> np.ndarray:
    """Express a camera-(u,v,depth) scale as an affine in proxy coordinates."""
    rotation = np.asarray(rotation_camera_from_proxy, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Camera-from-proxy rotation must be finite 3x3")
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        atol=5e-3,
        rtol=0.0,
    ) or np.linalg.det(rotation) <= 0.0:
        raise ValueError("Camera-from-proxy rotation is not a proper rotation")

    factors = np.asarray(scale_factors_uvd, dtype=np.float64)
    center = np.asarray(center_proxy_m, dtype=np.float64)
    if (
        factors.shape != (3,)
        or not np.all(np.isfinite(factors))
        or np.any(factors <= 0.0)
    ):
        raise ValueError("Camera-axis scale factors must be finite and positive")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("Camera-axis scale center must be finite")

    linear = rotation.T @ np.diag(factors) @ rotation
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = linear
    affine[:3, 3] = center - linear @ center
    return np.ascontiguousarray(affine, dtype=np.float64)


def _score_rendered_candidates(
    *,
    prepared_view: Any,
    rendered_masks: np.ndarray,
    rendered_depth_m: np.ndarray,
    object_scale_m: float,
    weights: AlignmentScoreWeights,
    depth_trim_quantile: float,
    minimum_depth_overlap_pixels: int,
    free_space_absolute_tolerance_m: float,
    free_space_relative_tolerance: float,
) -> tuple[AlignmentScoreResult, ...]:
    if rendered_masks.shape != rendered_depth_m.shape:
        raise ValueError("Axis-scale render mask/depth shapes differ")
    if not math.isfinite(object_scale_m) or object_scale_m <= 0.0:
        raise ValueError("Invalid axis-scale object scale")
    return tuple(
        score_alignment(
            observed_mask=prepared_view.segmentation.mask_bool,
            observed_depth_m=prepared_view.view.depth_m,
            rendered_mask=rendered_masks[index],
            rendered_depth_m=rendered_depth_m[index],
            # S* denotes the same physical object throughout this local-axis
            # search. Keeping it fixed avoids rewarding a larger candidate
            # merely because its depth residual is divided by a larger value.
            object_scale_m=object_scale_m,
            weights=weights,
            depth_trim_quantile=depth_trim_quantile,
            min_depth_overlap_pixels=minimum_depth_overlap_pixels,
            free_space_absolute_tolerance_m=(
                free_space_absolute_tolerance_m
            ),
            free_space_relative_tolerance=free_space_relative_tolerance,
        )
        for index in range(rendered_masks.shape[0])
    )


def _render_and_score_grid(
    *,
    grid: tuple[tuple[float, float, float], ...],
    source_dimensions: tuple[float, float, float],
    source_center: tuple[float, float, float],
    rotation_camera_from_proxy: np.ndarray,
    pose: np.ndarray,
    source_candidate: ScaledMeshCandidate,
    renderer: Any,
    prepared_view: Any,
    weights: AlignmentScoreWeights,
    depth_trim_quantile: float,
    minimum_depth_overlap_pixels: int,
    free_space_absolute_tolerance_m: float,
    free_space_relative_tolerance: float,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[AlignmentScoreResult, ...],
    np.ndarray,
]:
    """Batch-render one factor grid against the fixed pose and score it."""
    affines = tuple(
        centered_camera_axis_scale_affine(
            scale_factors_uvd=factors,
            center_proxy_m=source_center,
            rotation_camera_from_proxy=rotation_camera_from_proxy,
        )
        for factors in grid
    )
    transforms = np.stack(
        [pose @ affine for affine in affines],
        axis=0,
    )
    dimensions = np.asarray(source_dimensions, dtype=np.float64)
    candidate_dimensions = np.stack(
        [dimensions * np.asarray(factors, dtype=np.float64) for factors in grid]
    )
    image_height, image_width = prepared_view.view.rgb.shape[:2]
    rendered_masks, rendered_depth_m = renderer.render_affine_depth_mask_batch(
        mesh_path=source_candidate.scaled_mesh_path,
        transforms_camera_from_proxy=transforms,
        camera_matrix=prepared_view.view.camera_matrix,
        image_height=image_height,
        image_width=image_width,
    )
    scores = _score_rendered_candidates(
        prepared_view=prepared_view,
        rendered_masks=rendered_masks,
        rendered_depth_m=rendered_depth_m,
        object_scale_m=source_candidate.scale_m,
        weights=weights,
        depth_trim_quantile=depth_trim_quantile,
        minimum_depth_overlap_pixels=minimum_depth_overlap_pixels,
        free_space_absolute_tolerance_m=free_space_absolute_tolerance_m,
        free_space_relative_tolerance=free_space_relative_tolerance,
    )
    return affines, scores, candidate_dimensions


def _scale_penalty(factors: tuple[float, float, float]) -> float:
    logs = np.log(np.asarray(factors, dtype=np.float64))
    return float(np.mean(logs * logs))


def _weighted_scale_penalty(
    factors: tuple[float, float, float],
    axis_weights: tuple[float, float, float],
) -> float:
    logs = np.log(np.asarray(factors, dtype=np.float64))
    weights = np.asarray(axis_weights, dtype=np.float64)
    if (
        weights.shape != (3,)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("Axis-scale penalty weights must be finite and >= 0")
    return float(np.mean(weights * logs * logs))


def _depth_observability(
    *,
    prepared_view: Any,
    object_scale_m: float,
    quantile_low: float,
    quantile_high: float,
    minimum_valid_points: int,
) -> dict[str, float | int]:
    """Estimate how strongly one RGB-D view constrains camera-depth scale."""
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("Depth observability quantiles are invalid")
    if minimum_valid_points < 1:
        raise ValueError("Depth observability minimum_valid_points must be >= 1")
    if not math.isfinite(object_scale_m) or object_scale_m <= 0.0:
        raise ValueError("Depth observability object scale must be positive")

    mask = np.asarray(prepared_view.segmentation.mask_bool, dtype=bool)
    depth_source = getattr(prepared_view, "masked_depth_m", None)
    if depth_source is None:
        depth_source = prepared_view.view.depth_m
    depth_m = np.asarray(depth_source, dtype=np.float64)
    if depth_m.shape != mask.shape:
        raise ValueError("Depth observability mask/depth shapes differ")
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.0)
    mask_count = int(np.count_nonzero(mask))
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = float(valid_count / max(mask_count, 1))

    depth_span_m = 0.0
    if valid_count >= minimum_valid_points:
        valid_depth = depth_m[valid]
        lower, upper = np.quantile(
            valid_depth,
            (quantile_low, quantile_high),
        )
        depth_span_m = max(float(upper - lower), 0.0)
    normalized_span = depth_span_m / object_scale_m
    observability = float(
        np.clip(valid_fraction * normalized_span, 0.0, 1.0)
    )
    return {
        "observability": observability,
        "mask_pixel_count": mask_count,
        "valid_depth_pixel_count": valid_count,
        "valid_depth_fraction": valid_fraction,
        "depth_span_m": depth_span_m,
        "normalized_depth_span": normalized_span,
    }


def _write_scaled_candidate(
    *,
    view_name: str,
    source_candidate: ScaledMeshCandidate,
    source_scene: trimesh.Scene,
    geometry_affine: np.ndarray,
    scale_factors: tuple[float, float, float],
    source_dimensions_m: tuple[float, float, float],
    selected_dimensions_m: tuple[float, float, float],
    source_center_m: tuple[float, float, float],
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
        raise ValueError("Axis-scale source mesh has no suffix")
    mesh_path = candidate_directory / f"mesh_axis_scaled{suffix}"
    fitted_scene.export(mesh_path, file_type=suffix.lstrip("."))
    mesh_path = mesh_path.resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Axis-scale mesh was not created: {mesh_path}")

    combined_transform = geometry_affine @ np.asarray(
        source_candidate.scale_transform,
        dtype=np.float64,
    )
    metadata_path = candidate_directory / "axis_scale_candidate.json"
    metadata = {
        "method": "independent_camera_axis_scale",
        "view": view_name,
        "source_candidate_index": source_candidate.candidate_index,
        "source_object_scale_m": source_candidate.scale_m,
        "scale_factors_uvd": list(scale_factors),
        "scale_factors_xyz": list(scale_factors),
        "source_dimensions_m": list(source_dimensions_m),
        "selected_dimensions_m": list(selected_dimensions_m),
        "source_center_m": list(source_center_m),
        "geometry_affine": geometry_affine.tolist(),
        "combined_scale_transform": combined_transform.tolist(),
        "source_scaled_mesh_path": str(source_candidate.scaled_mesh_path),
        "scaled_mesh_path": str(mesh_path),
        "unit": "meter",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return ScaledMeshCandidate(
        candidate_index=source_candidate.candidate_index,
        # Preserve the pair-shared S*. The independent Sx/Sy/Sz correction is
        # represented by the mesh geometry and combined scale_transform.
        scale_m=source_candidate.scale_m,
        normalized_mesh_path=source_candidate.normalized_mesh_path,
        scaled_mesh_path=mesh_path,
        metadata_path=metadata_path.resolve(),
        scale_transform=np.ascontiguousarray(combined_transform),
        artifact_paths=(metadata_path.resolve(),),
    )


def fit_independent_axis_scale(
    *,
    view_name: str,
    source_candidate: ScaledMeshCandidate,
    prepared_view: Any,
    fixed_pose_camera_from_proxy: np.ndarray,
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
    uncertainty_scale_penalty_weight: float,
    depth_quantile_low: float,
    depth_quantile_high: float,
    depth_minimum_valid_points: int,
    minimum_loss_improvement_ratio: float,
    coarse_shortlist_count: int = 5,
    fine_factor_radius: float = 0.10,
    fine_grid_step_count: int = 5,
) -> IndependentAxisScaleResult:
    """Fit camera-frame (Su,Sv,Sd) while keeping the initial A0/B0 fixed."""
    if view_name not in {"reference", "query"}:
        raise ValueError(f"Unsupported axis-scale view: {view_name}")
    if not math.isfinite(scale_penalty_weight) or scale_penalty_weight < 0.0:
        raise ValueError("Axis-scale penalty weight must be finite and >= 0")
    if (
        not math.isfinite(uncertainty_scale_penalty_weight)
        or uncertainty_scale_penalty_weight < 0.0
    ):
        raise ValueError(
            "Axis-scale uncertainty penalty weight must be finite and >= 0"
        )
    if not 0.0 <= minimum_loss_improvement_ratio < 1.0:
        raise ValueError("Axis-scale minimum improvement must be in [0,1)")
    if isinstance(coarse_shortlist_count, bool) or coarse_shortlist_count < 1:
        raise ValueError("Axis-scale coarse_shortlist_count must be >= 1")
    if not math.isfinite(fine_factor_radius) or fine_factor_radius <= 0.0:
        raise ValueError("Axis-scale fine_factor_radius must be finite and positive")
    if isinstance(fine_grid_step_count, bool) or fine_grid_step_count < 2:
        raise ValueError("Axis-scale fine_grid_step_count must be >= 2")
    pose = np.asarray(fixed_pose_camera_from_proxy, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("Axis-scale fixed pose must be finite 4x4")

    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scene = _load_scene(source_candidate.scaled_mesh_path)
    samples = _sample_surface(
        scene=scene,
        sample_count=sample_count,
        random_seed=random_seed,
    )
    _, source_center = _robust_dimensions_and_center(
        samples,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    rotation_camera_from_proxy = pose[:3, :3]
    centered_samples_camera = (
        rotation_camera_from_proxy
        @ (
            samples
            - np.asarray(source_center, dtype=np.float64)
        ).T
    ).T
    source_dimensions, _ = _robust_dimensions_and_center(
        centered_samples_camera,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    depth_support = _depth_observability(
        prepared_view=prepared_view,
        object_scale_m=source_candidate.scale_m,
        quantile_low=depth_quantile_low,
        quantile_high=depth_quantile_high,
        minimum_valid_points=depth_minimum_valid_points,
    )
    depth_observability = float(depth_support["observability"])
    axis_penalty_weights = (
        float(scale_penalty_weight),
        float(scale_penalty_weight),
        float(
            scale_penalty_weight
            + uncertainty_scale_penalty_weight * (1.0 - depth_observability)
        ),
    )
    coarse_grid = build_axis_scale_factor_grid(
        minimum_factor=minimum_factor,
        maximum_factor=maximum_factor,
        grid_step_count=grid_step_count,
    )
    coarse_affines, coarse_scores, coarse_dimensions = _render_and_score_grid(
        grid=coarse_grid,
        source_dimensions=source_dimensions,
        source_center=source_center,
        rotation_camera_from_proxy=rotation_camera_from_proxy,
        pose=pose,
        source_candidate=source_candidate,
        renderer=renderer,
        prepared_view=prepared_view,
        weights=weights,
        depth_trim_quantile=depth_trim_quantile,
        minimum_depth_overlap_pixels=minimum_depth_overlap_pixels,
        free_space_absolute_tolerance_m=free_space_absolute_tolerance_m,
        free_space_relative_tolerance=free_space_relative_tolerance,
    )

    def _build_records(
        grid_slice: tuple[tuple[float, float, float], ...],
        scores_slice: tuple[AlignmentScoreResult, ...],
        dimensions_slice: np.ndarray,
        stage: str,
        index_offset: int,
    ) -> list[dict[str, Any]]:
        built: list[dict[str, Any]] = []
        for local_index, (factors, score) in enumerate(
            zip(grid_slice, scores_slice, strict=True)
        ):
            penalty = _scale_penalty(factors)
            penalty_term = _weighted_scale_penalty(
                factors,
                axis_penalty_weights,
            )
            built.append(
                {
                    "grid_index": index_offset + local_index,
                    "stage": stage,
                    "scale_factors_uvd": list(factors),
                    "scale_factors_xyz": list(factors),
                    "target_dimensions_m": (
                        dimensions_slice[local_index].tolist()
                    ),
                    "scale_penalty": penalty,
                    "scale_penalty_term": penalty_term,
                    "objective": score.total_loss + penalty_term,
                    "alignment_score": asdict(score),
                }
            )
        return built

    coarse_records = _build_records(
        coarse_grid, coarse_scores, coarse_dimensions, "coarse", 0
    )

    # Coarse-to-fine: refine a ±fine_factor_radius neighborhood (step
    # fine_grid_step_count) around each of the coarse_shortlist_count best
    # coarse candidates, so the final grid resolution isn't limited to the
    # coarse grid_step_count spacing.
    coarse_ranked = sorted(
        coarse_records,
        key=lambda record: (
            record["objective"],
            record["alignment_score"]["total_loss"],
            record["scale_penalty"],
            record["grid_index"],
        ),
    )
    coarse_shortlist = coarse_ranked[:coarse_shortlist_count]
    coarse_factor_set = set(coarse_grid)
    fine_factor_set: set[tuple[float, float, float]] = set()
    for shortlisted in coarse_shortlist:
        local_grid = build_local_axis_scale_factor_grid(
            center_factors=tuple(shortlisted["scale_factors_xyz"]),
            radius=fine_factor_radius,
            grid_step_count=fine_grid_step_count,
            minimum_factor=minimum_factor,
            maximum_factor=maximum_factor,
        )
        fine_factor_set.update(local_grid)
    fine_grid = tuple(sorted(fine_factor_set - coarse_factor_set))

    if fine_grid:
        fine_affines, fine_scores, fine_dimensions = _render_and_score_grid(
            grid=fine_grid,
            source_dimensions=source_dimensions,
            source_center=source_center,
            rotation_camera_from_proxy=rotation_camera_from_proxy,
            pose=pose,
            source_candidate=source_candidate,
            renderer=renderer,
            prepared_view=prepared_view,
            weights=weights,
            depth_trim_quantile=depth_trim_quantile,
            minimum_depth_overlap_pixels=minimum_depth_overlap_pixels,
            free_space_absolute_tolerance_m=free_space_absolute_tolerance_m,
            free_space_relative_tolerance=free_space_relative_tolerance,
        )
        fine_records = _build_records(
            fine_grid, fine_scores, fine_dimensions, "fine", len(coarse_grid)
        )
    else:
        fine_affines, fine_scores = (), ()
        fine_dimensions = np.zeros((0, 3), dtype=np.float64)
        fine_records = []

    grid = coarse_grid + fine_grid
    affines = coarse_affines + fine_affines
    scores = coarse_scores + fine_scores
    candidate_dimensions = (
        np.concatenate([coarse_dimensions, fine_dimensions], axis=0)
        if fine_grid
        else coarse_dimensions
    )
    records = coarse_records + fine_records
    baseline_index = grid.index((1.0, 1.0, 1.0))
    baseline_record = records[baseline_index]
    best_record = min(
        records,
        key=lambda record: (
            record["objective"],
            record["alignment_score"]["total_loss"],
            record["scale_penalty"],
            record["grid_index"],
        ),
    )
    baseline_loss = float(baseline_record["alignment_score"]["total_loss"])
    best_loss = float(best_record["alignment_score"]["total_loss"])
    improvement_ratio = (
        (baseline_loss - best_loss) / max(baseline_loss, 1e-12)
    )
    requested_factors = tuple(
        float(value) for value in best_record["scale_factors_xyz"]
    )
    applied = bool(
        requested_factors != (1.0, 1.0, 1.0)
        and improvement_ratio >= minimum_loss_improvement_ratio
    )
    selected_record = best_record if applied else baseline_record
    selected_index = int(selected_record["grid_index"])
    selected_factors = grid[selected_index]
    selected_score = scores[selected_index]
    selected_dimensions = tuple(
        float(value) for value in candidate_dimensions[selected_index]
    )

    selected_candidate = source_candidate
    if applied:
        selected_candidate = _write_scaled_candidate(
            view_name=view_name,
            source_candidate=source_candidate,
            source_scene=scene,
            geometry_affine=affines[selected_index],
            scale_factors=selected_factors,
            source_dimensions_m=source_dimensions,
            selected_dimensions_m=selected_dimensions,
            source_center_m=source_center,
            output_directory=output_root,
        )

    summary_path = output_root / "search.json"
    summary = {
        "method": "independent_camera_axis_scale",
        "view": view_name,
        "fixed_pose_camera_from_proxy": pose.tolist(),
        "camera_axis_convention": "(u=image horizontal, v=image vertical, d=camera depth)",
        "source_candidate_index": source_candidate.candidate_index,
        "source_object_scale_m": source_candidate.scale_m,
        "source_scaled_mesh_path": str(source_candidate.scaled_mesh_path),
        "source_dimensions_m": list(source_dimensions),
        "source_center_m": list(source_center),
        "depth_observability": depth_support,
        "axis_penalty_weights_uvd": list(axis_penalty_weights),
        "parameters": {
            "quantile_low": quantile_low,
            "quantile_high": quantile_high,
            "surface_sample_count": sample_count,
            "random_seed": random_seed,
            "minimum_factor": minimum_factor,
            "maximum_factor": maximum_factor,
            "grid_step_count": grid_step_count,
            "coarse_candidate_count": len(coarse_grid),
            "coarse_shortlist_count": coarse_shortlist_count,
            "fine_factor_radius": fine_factor_radius,
            "fine_grid_step_count": fine_grid_step_count,
            "fine_candidate_count": len(fine_grid),
            "candidate_count": len(grid),
            "scale_penalty_weight": scale_penalty_weight,
            "uncertainty_scale_penalty_weight": (
                uncertainty_scale_penalty_weight
            ),
            "depth_quantile_low": depth_quantile_low,
            "depth_quantile_high": depth_quantile_high,
            "depth_minimum_valid_points": depth_minimum_valid_points,
            "minimum_loss_improvement_ratio": minimum_loss_improvement_ratio,
        },
        "coarse_shortlist": coarse_shortlist,
        "baseline_record": baseline_record,
        "best_requested_record": best_record,
        "selected_record": selected_record,
        "requested_improvement_ratio": improvement_ratio,
        "applied": applied,
        "selected_scale_factors_uvd": list(selected_factors),
        "selected_scale_factors_xyz": list(selected_factors),
        "selected_dimensions_m": list(selected_dimensions),
        "selected_scaled_mesh_path": str(selected_candidate.scaled_mesh_path),
        "selection_policy": (
            "minimum fixed-pose RGB-D objective with depth-observability-aware "
            "soft log-scale penalty; "
            "identity fallback when raw loss improvement is insufficient"
        ),
        "candidates": records,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return IndependentAxisScaleResult(
        view_name=view_name,
        source_candidate=source_candidate,
        selected_candidate=selected_candidate,
        selected_scale_factors=selected_factors,
        source_center_m=source_center,
        source_dimensions_m=source_dimensions,
        selected_dimensions_m=selected_dimensions,
        depth_observability=depth_observability,
        axis_penalty_weights=axis_penalty_weights,
        baseline_score=scores[baseline_index],
        selected_score=selected_score,
        scale_penalty=float(selected_record["scale_penalty"]),
        objective=float(selected_record["objective"]),
        improvement_ratio=float(improvement_ratio),
        applied=applied,
        summary_path=summary_path.resolve(),
    )
