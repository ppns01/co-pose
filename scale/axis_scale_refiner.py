from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from core.types import PreparedView
from pose.alignment_scorer import (
    AlignmentScoreResult,
    AlignmentScoreWeights,
    score_alignment,
)
from pose.mesh_renderer import FoundationPoseMeshRenderer
from scale.mesh_scaler import ScaledMeshCandidate


@dataclass(frozen=True)
class AxisScaleSearchResult:
    candidate: ScaledMeshCandidate
    axis_scales: tuple[float, float, float]
    alignment_score: AlignmentScoreResult
    regularization_loss: float
    objective: float
    summary_path: Path


def build_volume_preserving_axis_scale_grid(
    *,
    minimum_scale: float = 0.85,
    maximum_scale: float = 1.15,
    grid_step_count: int = 5,
) -> tuple[tuple[float, float, float], ...]:
    """Build bounded Sx/Sy/Sz candidates with Sx*Sy*Sz == 1."""
    if not (
        0.0 < minimum_scale <= 1.0 <= maximum_scale
    ):
        raise ValueError(
            "Axis scale bounds must contain 1: "
            f"minimum={minimum_scale}, maximum={maximum_scale}"
        )
    if grid_step_count < 2:
        raise ValueError("grid_step_count must be at least 2")

    scale_values = np.linspace(
        minimum_scale,
        maximum_scale,
        num=grid_step_count,
        dtype=np.float64,
    )
    candidates: set[tuple[float, float, float]] = {
        (1.0, 1.0, 1.0)
    }

    for scale_x in scale_values:
        for scale_y in scale_values:
            scale_z = 1.0 / (float(scale_x) * float(scale_y))
            if not (
                minimum_scale - 1e-12
                <= scale_z
                <= maximum_scale + 1e-12
            ):
                continue
            scales = (
                float(scale_x),
                float(scale_y),
                scale_z,
            )
            candidates.add(
                tuple(round(value, 12) for value in scales)
            )

    def sort_key(
        scales: tuple[float, float, float],
    ) -> tuple[float, float, float, float]:
        logs = tuple(abs(math.log(value)) for value in scales)
        return (sum(logs), scales[0], scales[1], scales[2])

    return tuple(sorted(candidates, key=sort_key))


def _load_scene(mesh_path: Path) -> trimesh.Scene:
    resolved = Path(mesh_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Axis-scale source mesh: {resolved}")
    scene = trimesh.load_scene(resolved, process=False)
    if scene.is_empty:
        raise ValueError(f"Axis-scale source mesh is empty: {resolved}")
    return scene


def _write_candidate(
    *,
    source_scene: trimesh.Scene,
    source_candidate: ScaledMeshCandidate,
    candidate_index: int,
    axis_scales: tuple[float, float, float],
    output_directory: Path,
) -> ScaledMeshCandidate:
    candidate_directory = (
        output_directory / f"candidate_{candidate_index:02d}"
    )
    if candidate_directory.exists():
        shutil.rmtree(candidate_directory)
    candidate_directory.mkdir(parents=True, exist_ok=False)

    axis_transform = np.eye(4, dtype=np.float64)
    axis_transform[0, 0] = axis_scales[0]
    axis_transform[1, 1] = axis_scales[1]
    axis_transform[2, 2] = axis_scales[2]

    candidate_scene = source_scene.copy()
    candidate_scene.apply_transform(axis_transform)

    suffix = source_candidate.scaled_mesh_path.suffix.lower()
    if not suffix:
        raise ValueError("Axis-scale source mesh has no suffix")
    mesh_path = candidate_directory / f"mesh_axis_scaled{suffix}"
    candidate_scene.export(
        file_obj=mesh_path,
        file_type=suffix.lstrip("."),
    )
    mesh_path = mesh_path.resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Axis-scaled mesh was not created: {mesh_path}"
        )

    combined_transform = (
        axis_transform
        @ np.asarray(
            source_candidate.scale_transform,
            dtype=np.float64,
        )
    )
    metadata_path = candidate_directory / "axis_scale_candidate.json"
    metadata_path.write_text(
        json.dumps(
            {
                "candidate_index": candidate_index,
                "source_candidate_index": (
                    source_candidate.candidate_index
                ),
                "shared_isotropic_scale_m": source_candidate.scale_m,
                "axis_scales": list(axis_scales),
                "axis_scale_product": float(np.prod(axis_scales)),
                "source_scaled_mesh_path": str(
                    source_candidate.scaled_mesh_path
                ),
                "scaled_mesh_path": str(mesh_path),
                "axis_transform": axis_transform.tolist(),
                "combined_scale_transform": (
                    combined_transform.tolist()
                ),
                "coordinate_rule": (
                    "p_axis_scaled = diag(Sx,Sy,Sz) "
                    "@ p_shared_isotropic_scaled"
                ),
                "unit": "meter",
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = tuple(
        sorted(
            path.resolve()
            for path in candidate_directory.rglob("*")
            if path.is_file()
            and path.resolve() not in {
                mesh_path,
                metadata_path.resolve(),
            }
        )
    )
    return ScaledMeshCandidate(
        candidate_index=candidate_index,
        scale_m=float(source_candidate.scale_m),
        normalized_mesh_path=(
            source_candidate.normalized_mesh_path
        ),
        scaled_mesh_path=mesh_path,
        metadata_path=metadata_path.resolve(),
        scale_transform=np.ascontiguousarray(
            combined_transform,
            dtype=np.float64,
        ),
        artifact_paths=artifacts,
    )


def refine_axis_scale_against_observation(
    *,
    prepared_view: PreparedView,
    source_candidate: ScaledMeshCandidate,
    pose_camera_from_proxy: Any,
    renderer: FoundationPoseMeshRenderer,
    output_directory: Path,
    weights: AlignmentScoreWeights,
    depth_trim_quantile: float,
    minimum_depth_overlap_pixels: int,
    free_space_absolute_tolerance_m: float,
    free_space_relative_tolerance: float,
    minimum_scale: float = 0.85,
    maximum_scale: float = 1.15,
    grid_step_count: int = 5,
    regularization_weight: float = 0.02,
) -> AxisScaleSearchResult:
    """Fit bounded local-axis residual scales to one RGB-D observation."""
    if regularization_weight < 0.0:
        raise ValueError("regularization_weight must be non-negative")

    output_root = Path(output_directory).expanduser().resolve()
    candidate_root = output_root / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    source_scene = _load_scene(source_candidate.scaled_mesh_path)
    axis_scale_grid = build_volume_preserving_axis_scale_grid(
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
        grid_step_count=grid_step_count,
    )
    pose = np.asarray(pose_camera_from_proxy, dtype=np.float32)
    maximum_log_delta = max(
        abs(math.log(minimum_scale)),
        abs(math.log(maximum_scale)),
        1e-12,
    )

    records: list[dict[str, Any]] = []
    evaluated: list[
        tuple[
            float,
            float,
            int,
            ScaledMeshCandidate,
            tuple[float, float, float],
            AlignmentScoreResult,
        ]
    ] = []

    for candidate_index, axis_scales in enumerate(axis_scale_grid):
        candidate = _write_candidate(
            source_scene=source_scene,
            source_candidate=source_candidate,
            candidate_index=candidate_index,
            axis_scales=axis_scales,
            output_directory=candidate_root,
        )
        render = renderer.render(
            mesh_path=candidate.scaled_mesh_path,
            poses_camera_from_proxy=pose,
            camera_matrix=prepared_view.view.camera_matrix,
            image_height=prepared_view.view.rgb.shape[0],
            image_width=prepared_view.view.rgb.shape[1],
        )
        alignment_score = score_alignment(
            observed_mask=prepared_view.segmentation.mask_bool,
            observed_depth_m=prepared_view.view.depth_m,
            rendered_mask=render.rendered_masks[0],
            rendered_depth_m=render.rendered_depth_m[0],
            object_scale_m=source_candidate.scale_m,
            weights=weights,
            depth_trim_quantile=depth_trim_quantile,
            min_depth_overlap_pixels=minimum_depth_overlap_pixels,
            free_space_absolute_tolerance_m=(
                free_space_absolute_tolerance_m
            ),
            free_space_relative_tolerance=(
                free_space_relative_tolerance
            ),
        )
        regularization_loss = float(
            np.mean(
                [
                    (math.log(value) / maximum_log_delta) ** 2
                    for value in axis_scales
                ]
            )
        )
        objective = float(
            alignment_score.total_loss
            + regularization_weight * regularization_loss
        )
        evaluated.append(
            (
                objective,
                regularization_loss,
                candidate_index,
                candidate,
                axis_scales,
                alignment_score,
            )
        )
        records.append(
            {
                "candidate_index": candidate_index,
                "axis_scales": list(axis_scales),
                "axis_scale_product": float(np.prod(axis_scales)),
                "alignment_score": asdict(alignment_score),
                "regularization_loss": regularization_loss,
                "regularization_weight": regularization_weight,
                "objective": objective,
                "mesh_path": str(candidate.scaled_mesh_path),
            }
        )

    selected = min(
        evaluated,
        key=lambda item: (item[0], item[1], item[2]),
    )
    (
        objective,
        regularization_loss,
        _,
        candidate,
        axis_scales,
        alignment_score,
    ) = selected

    selected_render_root = output_root / "selected_render"
    renderer.render(
        mesh_path=candidate.scaled_mesh_path,
        poses_camera_from_proxy=pose,
        camera_matrix=prepared_view.view.camera_matrix,
        image_height=prepared_view.view.rgb.shape[0],
        image_width=prepared_view.view.rgb.shape[1],
        output_directory=selected_render_root,
        filename_prefix="axis_scaled",
    )

    summary_path = output_root / "axis_scale_selection.json"
    summary_path.write_text(
        json.dumps(
            {
                "view": prepared_view.view.source.name,
                "method": "bounded_volume_preserving_axis_scale",
                "shared_isotropic_scale_m": source_candidate.scale_m,
                "bounds": [minimum_scale, maximum_scale],
                "grid_step_count": grid_step_count,
                "candidate_count": len(records),
                "selection_policy": (
                    "minimum mask+depth+free-space+boundary loss "
                    "plus log-axis prior"
                ),
                "selected_candidate_index": candidate.candidate_index,
                "selected_axis_scales": list(axis_scales),
                "selected_axis_scale_product": float(
                    np.prod(axis_scales)
                ),
                "selected_alignment_score": asdict(alignment_score),
                "selected_regularization_loss": regularization_loss,
                "selected_objective": objective,
                "selected_mesh_path": str(candidate.scaled_mesh_path),
                "candidates": records,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return AxisScaleSearchResult(
        candidate=candidate,
        axis_scales=axis_scales,
        alignment_score=alignment_score,
        regularization_loss=regularization_loss,
        objective=objective,
        summary_path=summary_path.resolve(),
    )
