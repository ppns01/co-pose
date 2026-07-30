from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


POSE_DIRECTION = "query_from_reference"
POSE_CONVENTION = (
    "T_query_camera_from_reference_camera"
)
TRANSLATION_UNIT = "meter"
CAMERA_CONVENTION = "opencv"
DEFAULT_RANDOM_SEED = 42


PAIR_RESULT_FIELDS = (
    "run_id",
    "pair_id",
    "timestamp",
    "dataset",
    "object_id",
    "object_name",
    "split",
    "reference_frame_id",
    "reference_scene_id",
    "reference_image_id",
    "reference_instance_index",
    "query_frame_id",
    "query_scene_id",
    "query_image_id",
    "query_instance_index",
    "random_seed",
    "method",
    "path",
    "generator",
    "segmentation_mode",
    "segmentation_model",
    "mask_type",
    "source_validation_enabled",
    "bidirectional_enabled",
    "rejection_enabled",
    "pose_direction",
    "pose_convention",
    "translation_unit",
    "depth_unit",
    "mesh_unit",
    "camera_convention",
    "gt_relative_rotation_deg",
    "gt_relative_translation_cm",
    "gt_tx_cm",
    "gt_ty_cm",
    "gt_tz_cm",
    "gt_pose_path",
    "estimated_rotation_deg",
    "estimated_translation_cm",
    "est_tx_cm",
    "est_ty_cm",
    "est_tz_cm",
    "estimated_pose_path",
    "rotation_error_deg",
    "translation_error_cm",
    "translation_error_x_cm",
    "translation_error_y_cm",
    "translation_error_z_cm",
    "success_5deg_5cm",
    "success_2deg_2cm",
    "success_1deg_1cm",
    "success_5deg_2cm",
    "success_10deg_5cm",
    "success_15deg_5cm",
    "catastrophic_failure",
    "add",
    "adds",
    "add_normalized",
    "adds_normalized",
    "success_add_10pct_diameter",
    "ref_path_rotation_error_deg",
    "ref_path_translation_error_cm",
    "query_path_rotation_error_deg",
    "query_path_translation_error_cm",
    "bidirectional_rotation_disagreement_deg",
    "bidirectional_translation_disagreement_cm",
    "selected_path",
    "selector_correct",
    "selector_correct_definition",
    "confidence_raw",
    "confidence_calibrated",
    "rejected",
    "rejection_reason",
    "bidirectional_consistency_score",
    "mask_score",
    "depth_score",
    "source_validation_score",
    "final_selection_score",
    "top1_top2_margin",
    "reference_mesh_scale",
    "query_mesh_scale",
    "reference_source_mask_iou",
    "reference_source_depth_residual_cm",
    "query_source_mask_iou",
    "query_source_depth_residual_cm",
    "ref_to_query_mask_iou",
    "ref_to_query_depth_residual_cm",
    "ref_to_query_foundationpose_score",
    "query_to_ref_mask_iou",
    "query_to_ref_depth_residual_cm",
    "query_to_ref_foundationpose_score",
    "segmentation_time_sec",
    "generation_time_ref_sec",
    "generation_time_query_sec",
    "generation_time_sec",
    "source_anchor_time_sec",
    "cross_alignment_time_sec",
    "foundationpose_time_sec",
    "relative_pose_time_sec",
    "consistency_time_sec",
    "cross_evidence_time_sec",
    "selection_time_sec",
    "scoring_time_sec",
    "visualization_time_sec",
    "total_time_sec",
    "total_time_scope",
    "shared_reference_time_sec",
    "standalone_equivalent_time_sec",
    "gpu_name",
    "peak_gpu_memory_mb",
    "git_commit",
    "config_hash",
    "experiment_config_path",
    "instantmesh_commit",
    "foundationpose_commit",
)


PATH_RESULT_FIELDS = (
    "run_id",
    "pair_id",
    "timestamp",
    "dataset",
    "object_id",
    "reference_frame_id",
    "query_frame_id",
    "method",
    "path",
    "candidate_index",
    "selected_for_final",
    "estimated_pose_path",
    "gt_pose_path",
    "pose_direction",
    "translation_unit",
    "rotation_error_deg",
    "translation_error_cm",
    "translation_error_x_cm",
    "translation_error_y_cm",
    "translation_error_z_cm",
    "success_5deg_5cm",
    "success_2deg_2cm",
    "success_1deg_1cm",
    "catastrophic_failure",
    "scale_m",
    "scaled_mesh_path",
    "self_candidate_index",
    "self_hypothesis_rank",
    "self_foundationpose_score",
    "self_alignment_loss",
    "cross_candidate_index",
    "cross_hypothesis_rank",
    "cross_foundationpose_score",
    "cross_total_loss",
    "cross_mask_iou",
    "cross_mask_score",
    "cross_depth_loss",
    "cross_depth_score",
    "cross_depth_residual_cm",
    "cross_free_space_loss",
    "cross_boundary_loss",
    "cross_valid_depth_overlap_count",
    "cross_rendered_pixel_count",
    "cross_free_space_violation_count",
    "dino_available",
    "dino_loss",
    "path_loss",
    "bidirectional_rotation_disagreement_deg",
    "bidirectional_translation_disagreement_cm",
    "consistency_loss",
    "normalized_consistency_loss",
    "passes_rotation_gate",
    "passes_translation_gate",
    "passes_scale_gate",
    "passes_hard_gate",
    "selected_path",
    "selector_correct",
)


PROXY_RESULT_FIELDS = (
    "run_id",
    "pair_id",
    "timestamp",
    "dataset",
    "object_id",
    "reference_frame_id",
    "query_frame_id",
    "side",
    "generator",
    "generated_mesh_path",
    "scaled_mesh_path",
    "mesh_scale_m",
    "candidate_index",
    "hypothesis_rank",
    "foundationpose_score",
    "source_alignment_total_loss",
    "source_mask_iou",
    "source_depth_loss",
    "source_depth_residual_cm",
    "source_free_space_loss",
    "source_free_space_violation_ratio",
    "source_boundary_loss",
    "trusted_surface_ratio",
    "mask_type",
    "segmentation_mode",
    "segmentation_model",
    "segmentation_confidence",
    "mask_area_px",
    "mask_path",
    "mask_iou_gt",
    "vertex_count",
    "face_count",
    "bbox_x_m",
    "bbox_y_m",
    "bbox_z_m",
    "bbox_ratio_xy",
    "bbox_ratio_xz",
    "mesh_geometry_error",
)


@dataclass(frozen=True)
class ResearchRunContext:
    run_id: str
    started_at: str
    output_directory: Path
    config_path: Path
    config_hash: str
    git_commit: str
    instantmesh_commit: str
    foundationpose_commit: str
    python_version: str
    torch_version: str
    cuda_version: str
    gpu_name: str
    random_seed: int = DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class ResearchLoggingResult:
    pair_results_path: Path
    path_results_path: Path
    proxy_results_path: Path
    reference_pose_path: Path
    query_pose_path: Path
    final_pose_path: Path | None
    ground_truth_pose_path: Path


def _now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def _canonical_config_hash(
    config_path: Path,
) -> str:
    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def _git_commit(repository_path: Path) -> str:
    repository_path = Path(repository_path)

    if not (repository_path / ".git").exists():
        return ""

    try:
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(repository_path),
                "rev-parse",
                "HEAD",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return ""

    return completed.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _cuda_version_from_torch_version(
    torch_version: str,
) -> str:
    marker = "+cu"

    if marker not in torch_version:
        return ""

    encoded = torch_version.split(
        marker,
        maxsplit=1,
    )[1]

    if len(encoded) < 3 or not encoded.isdigit():
        return encoded

    return (
        f"{int(encoded[:-1])}."
        f"{int(encoded[-1])}"
    )


def _gpu_name_from_nvidia_smi() -> str:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return ""

    names = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    return ";".join(names)


def initialize_research_run(
    *,
    output_root: Path,
    config_path: Path,
    project_root: Path,
    instantmesh_repository: Path,
    foundationpose_repository: Path,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> ResearchRunContext:
    config_path = Path(config_path).resolve()
    config_hash = _canonical_config_hash(
        config_path
    )
    started_at = _now_local_iso()
    compact_time = (
        datetime.now()
        .astimezone()
        .strftime("%Y%m%dT%H%M%S%f%z")
    )
    run_id = (
        f"run_{compact_time}_{config_hash[:8]}"
    )
    research_root = (
        Path(output_root).resolve() / "research"
    )
    research_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch_version = _package_version("torch")
    context = ResearchRunContext(
        run_id=run_id,
        started_at=started_at,
        output_directory=research_root,
        config_path=config_path,
        config_hash=config_hash,
        git_commit=_git_commit(project_root),
        instantmesh_commit=_git_commit(
            instantmesh_repository
        ),
        foundationpose_commit=_git_commit(
            foundationpose_repository
        ),
        python_version=sys.version.split()[0],
        torch_version=torch_version,
        cuda_version=(
            _cuda_version_from_torch_version(
                torch_version
            )
        ),
        gpu_name=_gpu_name_from_nvidia_smi(),
        random_seed=random_seed,
    )

    metadata_directory = (
        research_root / "run_metadata"
    )
    metadata_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    metadata_path = (
        metadata_directory
        / f"{run_id}.json"
    )
    metadata = {
        **asdict(context),
        "output_directory": str(
            context.output_directory
        ),
        "config_path": str(context.config_path),
        "peak_gpu_memory_mb": None,
        "peak_gpu_memory_note": (
            "Unavailable: FoundationPose and InstantMesh "
            "run in separate processes."
        ),
    }

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

    definitions_path = (
        research_root
        / "metric_definitions.json"
    )
    definitions = {
        "pose_direction": POSE_DIRECTION,
        "pose_equation": (
            "X_query = H_query_from_reference "
            "@ X_reference"
        ),
        "translation_error_axis": (
            "signed estimate minus ground truth"
        ),
        "catastrophic_failure": (
            "rotation_error_deg > 20 or "
            "translation_error_cm > 10"
        ),
        "mask_score": "cross-render mask IoU",
        "depth_score": (
            "1 - normalized cross depth loss"
        ),
        "source_validation_score": (
            "1 - self-alignment total loss"
        ),
        "bidirectional_consistency_score": (
            "1 - normalized consistency loss"
        ),
        "selector_correct": (
            "Pareto dominance on GT rotation and "
            "translation errors; blank for a tie or "
            "rotation/translation trade-off"
        ),
        "confidence_raw": (
            "relative candidate confidence; "
            "not a calibrated probability"
        ),
        "runtime_scope": (
            "Measured end-to-end timings are recorded "
            "only on the dual method row. Ref-only and "
            "query-only rows are analytical path "
            "projections, not separately timed runs."
        ),
        "batch_runtime_accounting": (
            "In batch mode total_time_sec is the "
            "observed query wall time excluding the "
            "once-only shared reference. Use "
            "shared_reference_time_sec once per batch, "
            "or standalone_equivalent_time_sec for an "
            "independent-pair cost estimate."
        ),
        "csv_join_model": (
            "Join pair_results, path_results, "
            "proxy_results, and run_metadata by run_id "
            "and pair_id. Full provenance lives in the "
            "pair table and run metadata."
        ),
        "path_candidate_scope": (
            "Reference/query path rows are the two "
            "components of the best accepted dual pair, "
            "or the best overall pair when rejected."
        ),
        "intentionally_unavailable": [
            "ADD/ADD-S until an evaluation object "
            "point-set and symmetry policy are defined",
            "confidence_calibrated until calibration "
            "is fitted on multiple pairs",
            "trusted_surface_ratio until trusted "
            "surface construction is implemented",
            "cross-process peak GPU memory",
        ],
    }

    with definitions_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            definitions,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return context


def _frame_id(frame: Any) -> str:
    return (
        f"{frame.scene_id:06d}_"
        f"{frame.image_id:06d}_"
        f"i{frame.instance_index:02d}"
    )


def _pair_id(
    *,
    object_id: int,
    reference_frame: Any,
    query_frame: Any,
) -> str:
    return (
        f"linemod_obj{object_id:02d}_"
        f"r{_frame_id(reference_frame)}_"
        f"q{_frame_id(query_frame)}"
    )


def _append_csv_rows(
    *,
    csv_path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    expected_header = list(fields)

    if csv_path.is_file():
        with csv_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.reader(file)
            actual_header = next(reader, [])

        if actual_header != expected_header:
            raise ValueError(
                "Existing research CSV schema differs "
                f"from the current schema: {csv_path}"
            )

    write_header = (
        not csv_path.exists()
        or csv_path.stat().st_size == 0
    )

    with csv_path.open(
        mode="a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=expected_header,
            extrasaction="ignore",
        )

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in expected_header
                }
            )


def _threshold_map(
    evaluation: Any,
) -> dict[str, bool]:
    return {
        item.threshold.name: item.success
        for item in evaluation.threshold_results
    }


def _evaluation_values(
    evaluation: Any,
) -> dict[str, Any]:
    thresholds = _threshold_map(evaluation)

    return {
        "gt_relative_rotation_deg": (
            evaluation
            .ground_truth_relative_rotation_deg
        ),
        "gt_relative_translation_cm": (
            evaluation
            .ground_truth_relative_translation_cm
        ),
        "gt_tx_cm": (
            evaluation.ground_truth_tx_m * 100.0
        ),
        "gt_ty_cm": (
            evaluation.ground_truth_ty_m * 100.0
        ),
        "gt_tz_cm": (
            evaluation.ground_truth_tz_m * 100.0
        ),
        "estimated_rotation_deg": (
            evaluation.estimated_rotation_deg
        ),
        "estimated_translation_cm": (
            evaluation.estimated_translation_cm
        ),
        "est_tx_cm": (
            evaluation.estimated_tx_m * 100.0
        ),
        "est_ty_cm": (
            evaluation.estimated_ty_m * 100.0
        ),
        "est_tz_cm": (
            evaluation.estimated_tz_m * 100.0
        ),
        "rotation_error_deg": (
            evaluation.rotation_error_deg
        ),
        "translation_error_cm": (
            evaluation.translation_error_cm
        ),
        "translation_error_x_cm": (
            evaluation.translation_error_x_cm
        ),
        "translation_error_y_cm": (
            evaluation.translation_error_y_cm
        ),
        "translation_error_z_cm": (
            evaluation.translation_error_z_cm
        ),
        "success_5deg_5cm": thresholds.get(
            "5deg_5cm"
        ),
        "success_2deg_2cm": thresholds.get(
            "2deg_2cm"
        ),
        "success_1deg_1cm": thresholds.get(
            "1deg_1cm"
        ),
        "success_5deg_2cm": thresholds.get(
            "5deg_2cm"
        ),
        "success_10deg_5cm": thresholds.get(
            "10deg_5cm"
        ),
        "success_15deg_5cm": thresholds.get(
            "15deg_5cm"
        ),
        "catastrophic_failure": (
            evaluation.catastrophic_failure
        ),
    }


def _alignment_for_selection(
    evaluation_result: Any,
    selection: Any,
) -> Any:
    for item in evaluation_result.evaluations:
        if (
            item.candidate_result.candidate_index
            == selection.candidate_index
            and item.hypothesis.rank
            == selection.hypothesis_rank
        ):
            return item.alignment_score

    raise KeyError(
        "Selected self-alignment score was not found: "
        f"candidate={selection.candidate_index}, "
        f"rank={selection.hypothesis_rank}"
    )


def _cross_evidence(
    path_result: Any,
    candidate_index: int,
) -> Any:
    for evidence in path_result.evidences:
        if evidence.candidate_index == candidate_index:
            return evidence

    raise KeyError(
        "Cross evidence was not found: "
        f"path={path_result.path_name}, "
        f"candidate={candidate_index}"
    )


def _free_space_violation_ratio(
    alignment: Any,
) -> float | None:
    if alignment.rendered_pixel_count <= 0:
        return None

    return (
        alignment.free_space_violation_count
        / alignment.rendered_pixel_count
    )


def _mesh_geometry(
    mesh_path: Path,
) -> dict[str, Any]:
    empty = {
        "vertex_count": None,
        "face_count": None,
        "bbox_x_m": None,
        "bbox_y_m": None,
        "bbox_z_m": None,
        "bbox_ratio_xy": None,
        "bbox_ratio_xz": None,
        "mesh_geometry_error": None,
    }

    try:
        import trimesh

        loaded = trimesh.load(
            mesh_path,
            force="scene",
            process=False,
        )

        if isinstance(loaded, trimesh.Scene):
            geometries = tuple(
                loaded.geometry.values()
            )

            if not geometries:
                raise ValueError(
                    "Mesh scene contains no geometry."
                )

            mesh = trimesh.util.concatenate(
                geometries
            )
        else:
            mesh = loaded

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        extents = np.asarray(
            mesh.bounding_box.extents,
            dtype=np.float64,
        )

        if extents.shape != (3,):
            raise ValueError(
                f"Unexpected mesh extents: {extents}"
            )

        x_size, y_size, z_size = (
            float(extents[0]),
            float(extents[1]),
            float(extents[2]),
        )

        return {
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
            "bbox_x_m": x_size,
            "bbox_y_m": y_size,
            "bbox_z_m": z_size,
            "bbox_ratio_xy": (
                x_size / y_size
                if y_size > 0.0
                else None
            ),
            "bbox_ratio_xz": (
                x_size / z_size
                if z_size > 0.0
                else None
            ),
            "mesh_geometry_error": None,
        }

    except Exception as error:
        return {
            **empty,
            "mesh_geometry_error": (
                f"{type(error).__name__}: {error}"
            ),
        }


def _pareto_selector_correct(
    *,
    selected_path: str | None,
    reference_evaluation: Any,
    query_evaluation: Any,
) -> bool | None:
    if selected_path is None:
        return None

    reference_values = (
        reference_evaluation.rotation_error_deg,
        reference_evaluation.translation_error_m,
    )
    query_values = (
        query_evaluation.rotation_error_deg,
        query_evaluation.translation_error_m,
    )

    reference_dominates = (
        reference_values[0] <= query_values[0]
        and reference_values[1] <= query_values[1]
        and reference_values != query_values
    )
    query_dominates = (
        query_values[0] <= reference_values[0]
        and query_values[1] <= reference_values[1]
        and query_values != reference_values
    )

    if reference_dominates:
        return selected_path == "reference_proxy"

    if query_dominates:
        return selected_path == "query_proxy"

    return None


def _timing_values(
    timings: Mapping[str, Any] | None,
) -> dict[str, Any]:
    timings = timings or {}
    fields = (
        "segmentation_time_sec",
        "generation_time_ref_sec",
        "generation_time_query_sec",
        "generation_time_sec",
        "source_anchor_time_sec",
        "cross_alignment_time_sec",
        "foundationpose_time_sec",
        "relative_pose_time_sec",
        "consistency_time_sec",
        "cross_evidence_time_sec",
        "selection_time_sec",
        "scoring_time_sec",
        "visualization_time_sec",
        "total_time_sec",
        "total_time_scope",
        "shared_reference_time_sec",
        "standalone_equivalent_time_sec",
    )

    return {
        field: timings.get(field)
        for field in fields
    }


def save_pair_research_results(
    *,
    context: ResearchRunContext,
    pair_output_root: Path,
    dataset_root: Path,
    split: str,
    object_id: int,
    object_name: str,
    reference_frame: Any,
    query_frame: Any,
    mask_type: str,
    segmentation_mode: str = "ground_truth",
    segmentation_model: str | None = None,
    reference_prepared_view: Any,
    query_prepared_view: Any,
    reference_mesh_result: Any,
    query_mesh_result: Any,
    reference_self_evaluation: Any,
    query_self_evaluation: Any,
    reference_self_alignment: Any,
    query_self_alignment: Any,
    cross_evidence: Any,
    final_result: Any,
    timings: Mapping[str, Any] | None = None,
) -> ResearchLoggingResult:
    from evaluation.relative_pose_evaluator import (
        BOPFrameGTSpec,
        evaluate_relative_pose,
    )

    if not final_result.evaluated_pair_scores:
        raise ValueError(
            "Final selection contains no evaluated pairs."
        )

    pair_score = (
        final_result.best_pair_score
        if final_result.best_pair_score is not None
        else final_result.evaluated_pair_scores[0]
    )
    consistency_pair = (
        pair_score.consistency_pair
    )
    reference_candidate = (
        consistency_pair.reference_candidate
    )
    query_candidate = (
        consistency_pair.query_candidate
    )

    pair_id = _pair_id(
        object_id=object_id,
        reference_frame=reference_frame,
        query_frame=query_frame,
    )
    pair_research_root = (
        Path(pair_output_root).resolve()
        / "research"
    )
    pose_directory = (
        pair_research_root / "poses"
    )
    pose_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_pose_path = (
        pose_directory
        / "H_query_from_reference_reference_proxy.npy"
    )
    query_pose_path = (
        pose_directory
        / "H_query_from_reference_query_proxy.npy"
    )
    final_pose_path = (
        pose_directory
        / "H_query_from_reference_final.npy"
    )
    ground_truth_pose_path = (
        pose_directory
        / "H_query_from_reference_ground_truth.npy"
    )

    np.save(
        reference_pose_path,
        np.asarray(
            reference_candidate
            .relative_pose_query_from_reference,
            dtype=np.float64,
        ),
        allow_pickle=False,
    )
    np.save(
        query_pose_path,
        np.asarray(
            query_candidate
            .relative_pose_query_from_reference,
            dtype=np.float64,
        ),
        allow_pickle=False,
    )

    final_pose_output: Path | None = None
    if (
        final_result
        .selected_relative_pose_query_from_reference
        is not None
    ):
        np.save(
            final_pose_path,
            np.asarray(
                final_result
                .selected_relative_pose_query_from_reference,
                dtype=np.float64,
            ),
            allow_pickle=False,
        )
        final_pose_output = final_pose_path

    reference_gt = BOPFrameGTSpec(
        dataset_root=Path(dataset_root),
        split=split,
        scene_id=reference_frame.scene_id,
        image_id=reference_frame.image_id,
        object_id=object_id,
        instance_index=(
            reference_frame.instance_index
        ),
    )
    query_gt = BOPFrameGTSpec(
        dataset_root=Path(dataset_root),
        split=split,
        scene_id=query_frame.scene_id,
        image_id=query_frame.image_id,
        object_id=object_id,
        instance_index=query_frame.instance_index,
    )
    evaluation_root = (
        pair_research_root / "evaluations"
    )
    reference_evaluation = evaluate_relative_pose(
        predicted_relative_pose=(
            reference_candidate
            .relative_pose_query_from_reference
        ),
        reference_frame=reference_gt,
        query_frame=query_gt,
        output_directory=(
            evaluation_root / "reference_proxy"
        ),
    )
    query_evaluation = evaluate_relative_pose(
        predicted_relative_pose=(
            query_candidate
            .relative_pose_query_from_reference
        ),
        reference_frame=reference_gt,
        query_frame=query_gt,
        output_directory=(
            evaluation_root / "query_proxy"
        ),
    )
    np.save(
        ground_truth_pose_path,
        np.asarray(
            reference_evaluation
            .ground_truth_relative_pose,
            dtype=np.float64,
        ),
        allow_pickle=False,
    )

    selected_path = final_result.selected_path_name
    final_evaluation = (
        reference_evaluation
        if selected_path == "reference_proxy"
        else query_evaluation
        if selected_path == "query_proxy"
        else None
    )
    selector_correct = _pareto_selector_correct(
        selected_path=selected_path,
        reference_evaluation=(
            reference_evaluation
        ),
        query_evaluation=query_evaluation,
    )

    reference_cross = _cross_evidence(
        cross_evidence.reference_proxy,
        consistency_pair.reference_candidate_index,
    ).cross_alignment
    query_cross = _cross_evidence(
        cross_evidence.query_proxy,
        consistency_pair.query_candidate_index,
    ).cross_alignment
    reference_source = _alignment_for_selection(
        reference_self_evaluation,
        reference_self_alignment,
    )
    query_source = _alignment_for_selection(
        query_self_evaluation,
        query_self_alignment,
    )

    reference_frame_id = _frame_id(
        reference_frame
    )
    query_frame_id = _frame_id(query_frame)
    timestamp = _now_local_iso()
    generators = sorted(
        {
            reference_mesh_result.generator_name,
            query_mesh_result.generator_name,
        }
    )
    generator_name = "+".join(generators)
    common = {
        "run_id": context.run_id,
        "pair_id": pair_id,
        "timestamp": timestamp,
        "dataset": "linemod",
        "object_id": object_id,
        "object_name": object_name,
        "split": split,
        "reference_frame_id": (
            reference_frame_id
        ),
        "reference_scene_id": (
            reference_frame.scene_id
        ),
        "reference_image_id": (
            reference_frame.image_id
        ),
        "reference_instance_index": (
            reference_frame.instance_index
        ),
        "query_frame_id": query_frame_id,
        "query_scene_id": query_frame.scene_id,
        "query_image_id": query_frame.image_id,
        "query_instance_index": (
            query_frame.instance_index
        ),
        "random_seed": context.random_seed,
        "generator": generator_name,
        "segmentation_mode": segmentation_mode,
        "segmentation_model": segmentation_model,
        "mask_type": mask_type,
        "source_validation_enabled": True,
        "pose_direction": POSE_DIRECTION,
        "pose_convention": POSE_CONVENTION,
        "translation_unit": TRANSLATION_UNIT,
        "depth_unit": TRANSLATION_UNIT,
        "mesh_unit": TRANSLATION_UNIT,
        "camera_convention": CAMERA_CONVENTION,
        "gt_pose_path": str(
            ground_truth_pose_path
        ),
        "ref_path_rotation_error_deg": (
            reference_evaluation
            .rotation_error_deg
        ),
        "ref_path_translation_error_cm": (
            reference_evaluation
            .translation_error_cm
        ),
        "query_path_rotation_error_deg": (
            query_evaluation.rotation_error_deg
        ),
        "query_path_translation_error_cm": (
            query_evaluation
            .translation_error_cm
        ),
        "bidirectional_rotation_disagreement_deg": (
            consistency_pair.rotation_difference_deg
        ),
        "bidirectional_translation_disagreement_cm": (
            consistency_pair.translation_difference_m
            * 100.0
        ),
        "selected_path": selected_path,
        "selector_correct": selector_correct,
        "selector_correct_definition": (
            "pareto_rotation_translation"
        ),
        "confidence_raw": final_result.confidence,
        "confidence_calibrated": None,
        "rejected": (
            final_result.status == "REJECT"
        ),
        "rejection_reason": (
            "no_pair_passed_bidirectional_hard_gate"
            if final_result.status == "REJECT"
            else None
        ),
        "bidirectional_consistency_score": (
            1.0
            - pair_score.normalized_consistency_loss
        ),
        "final_selection_score": (
            1.0 - final_result.final_loss
        ),
        "top1_top2_margin": (
            final_result.score_margin
        ),
        "reference_mesh_scale": (
            reference_candidate.scale_m
        ),
        "query_mesh_scale": (
            query_candidate.scale_m
        ),
        "reference_source_mask_iou": (
            reference_source.mask_iou
        ),
        "reference_source_depth_residual_cm": (
            reference_source.depth_residual_m
            * 100.0
        ),
        "query_source_mask_iou": (
            query_source.mask_iou
        ),
        "query_source_depth_residual_cm": (
            query_source.depth_residual_m
            * 100.0
        ),
        "ref_to_query_mask_iou": (
            reference_cross.mask_iou
        ),
        "ref_to_query_depth_residual_cm": (
            reference_cross.depth_residual_m
            * 100.0
        ),
        "ref_to_query_foundationpose_score": (
            reference_candidate
            .cross_foundationpose_score
        ),
        "query_to_ref_mask_iou": (
            query_cross.mask_iou
        ),
        "query_to_ref_depth_residual_cm": (
            query_cross.depth_residual_m
            * 100.0
        ),
        "query_to_ref_foundationpose_score": (
            query_candidate
            .cross_foundationpose_score
        ),
        "gpu_name": context.gpu_name,
        "peak_gpu_memory_mb": None,
        "git_commit": context.git_commit,
        "config_hash": context.config_hash,
        "experiment_config_path": str(
            context.config_path
        ),
        "instantmesh_commit": (
            context.instantmesh_commit
        ),
        "foundationpose_commit": (
            context.foundationpose_commit
        ),
        **_timing_values(timings),
    }

    path_specs = (
        (
            "ref_only",
            "reference_proxy",
            reference_candidate,
            reference_evaluation,
            reference_pose_path,
            pair_score.reference_score,
            reference_cross,
            reference_source,
            consistency_pair
            .reference_candidate_index,
        ),
        (
            "query_only",
            "query_proxy",
            query_candidate,
            query_evaluation,
            query_pose_path,
            pair_score.query_score,
            query_cross,
            query_source,
            consistency_pair.query_candidate_index,
        ),
    )

    pair_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []

    for (
        method,
        path_name,
        candidate,
        evaluation,
        pose_path,
        composite_score,
        cross_alignment,
        source_alignment,
        candidate_index,
    ) in path_specs:
        evaluation_values = _evaluation_values(
            evaluation
        )
        pair_rows.append(
            {
                **common,
                **evaluation_values,
                "method": method,
                "path": path_name,
                "bidirectional_enabled": False,
                "rejection_enabled": False,
                "estimated_pose_path": str(
                    pose_path
                ),
                "confidence_raw": None,
                "rejected": False,
                "rejection_reason": None,
                "mask_score": (
                    cross_alignment.mask_iou
                ),
                "depth_score": (
                    1.0
                    - cross_alignment.depth_loss
                ),
                "source_validation_score": (
                    1.0
                    - source_alignment.total_loss
                ),
                "final_selection_score": None,
                "top1_top2_margin": None,
                **_timing_values(None),
            }
        )
        path_rows.append(
            {
                **common,
                **evaluation_values,
                "method": method,
                "path": path_name,
                "candidate_index": candidate_index,
                "selected_for_final": (
                    selected_path == path_name
                ),
                "estimated_pose_path": str(
                    pose_path
                ),
                "scale_m": candidate.scale_m,
                "scaled_mesh_path": str(
                    candidate.scaled_mesh_path
                ),
                "self_candidate_index": (
                    candidate.self_candidate_index
                ),
                "self_hypothesis_rank": (
                    candidate.self_hypothesis_rank
                ),
                "self_foundationpose_score": (
                    candidate
                    .self_foundationpose_score
                ),
                "self_alignment_loss": (
                    candidate.self_alignment_loss
                ),
                "cross_candidate_index": (
                    candidate.cross_candidate_index
                ),
                "cross_hypothesis_rank": (
                    candidate.cross_hypothesis_rank
                ),
                "cross_foundationpose_score": (
                    candidate
                    .cross_foundationpose_score
                ),
                "cross_total_loss": (
                    cross_alignment.total_loss
                ),
                "cross_mask_iou": (
                    cross_alignment.mask_iou
                ),
                "cross_mask_score": (
                    cross_alignment.mask_iou
                ),
                "cross_depth_loss": (
                    cross_alignment.depth_loss
                ),
                "cross_depth_score": (
                    1.0
                    - cross_alignment.depth_loss
                ),
                "cross_depth_residual_cm": (
                    cross_alignment.depth_residual_m
                    * 100.0
                ),
                "cross_free_space_loss": (
                    cross_alignment.free_space_loss
                ),
                "cross_boundary_loss": (
                    cross_alignment.boundary_loss
                ),
                "cross_valid_depth_overlap_count": (
                    cross_alignment
                    .valid_depth_overlap_count
                ),
                "cross_rendered_pixel_count": (
                    cross_alignment
                    .rendered_pixel_count
                ),
                "cross_free_space_violation_count": (
                    cross_alignment
                    .free_space_violation_count
                ),
                "dino_available": (
                    composite_score.dino_available
                ),
                "dino_loss": (
                    composite_score.dino_loss
                ),
                "path_loss": (
                    composite_score.path_loss
                ),
                "consistency_loss": (
                    consistency_pair.consistency_loss
                ),
                "normalized_consistency_loss": (
                    pair_score
                    .normalized_consistency_loss
                ),
                "passes_rotation_gate": (
                    consistency_pair
                    .passes_rotation_gate
                ),
                "passes_translation_gate": (
                    consistency_pair
                    .passes_translation_gate
                ),
                "passes_scale_gate": (
                    consistency_pair
                    .passes_scale_gate
                ),
                "passes_hard_gate": (
                    consistency_pair
                    .passes_hard_gate
                ),
            }
        )

    if final_evaluation is not None:
        final_evaluation_values = (
            _evaluation_values(final_evaluation)
        )
    else:
        final_evaluation_values = {
            field: None
            for field in (
                "gt_relative_rotation_deg",
                "gt_relative_translation_cm",
                "gt_tx_cm",
                "gt_ty_cm",
                "gt_tz_cm",
                "estimated_rotation_deg",
                "estimated_translation_cm",
                "est_tx_cm",
                "est_ty_cm",
                "est_tz_cm",
                "rotation_error_deg",
                "translation_error_cm",
                "translation_error_x_cm",
                "translation_error_y_cm",
                "translation_error_z_cm",
                "success_5deg_5cm",
                "success_2deg_2cm",
                "success_1deg_1cm",
                "success_5deg_2cm",
                "success_10deg_5cm",
                "success_15deg_5cm",
                "catastrophic_failure",
            )
        }
        final_evaluation_values.update(
            {
                "gt_relative_rotation_deg": (
                    reference_evaluation
                    .ground_truth_relative_rotation_deg
                ),
                "gt_relative_translation_cm": (
                    reference_evaluation
                    .ground_truth_relative_translation_cm
                ),
                "gt_tx_cm": (
                    reference_evaluation
                    .ground_truth_tx_m
                    * 100.0
                ),
                "gt_ty_cm": (
                    reference_evaluation
                    .ground_truth_ty_m
                    * 100.0
                ),
                "gt_tz_cm": (
                    reference_evaluation
                    .ground_truth_tz_m
                    * 100.0
                ),
            }
        )

    pair_rows.append(
        {
            **common,
            **final_evaluation_values,
            "method": (
                "dual_validated_reject"
                if final_result.status == "REJECT"
                else "dual_validated"
            ),
            "path": "bidirectional",
            "bidirectional_enabled": True,
            "rejection_enabled": True,
            "estimated_pose_path": (
                str(final_pose_output)
                if final_pose_output is not None
                else None
            ),
            "mask_score": (
                (
                    reference_cross.mask_iou
                    + query_cross.mask_iou
                )
                / 2.0
            ),
            "depth_score": (
                (
                    (1.0 - reference_cross.depth_loss)
                    + (1.0 - query_cross.depth_loss)
                )
                / 2.0
            ),
            "source_validation_score": (
                (
                    (1.0 - reference_source.total_loss)
                    + (1.0 - query_source.total_loss)
                )
                / 2.0
            ),
        }
    )

    proxy_rows: list[dict[str, Any]] = []
    proxy_specs = (
        (
            "reference",
            reference_mesh_result,
            reference_self_alignment,
            reference_source,
            reference_prepared_view,
            reference_candidate.scale_m,
        ),
        (
            "query",
            query_mesh_result,
            query_self_alignment,
            query_source,
            query_prepared_view,
            query_candidate.scale_m,
        ),
    )

    for (
        side,
        mesh_result,
        self_alignment,
        source_alignment,
        prepared_view,
        mesh_scale,
    ) in proxy_specs:
        segmentation = (
            prepared_view.segmentation
        )
        geometry = _mesh_geometry(
            self_alignment.scaled_mesh_path
        )
        proxy_rows.append(
            {
                **common,
                "side": side,
                "generator": (
                    mesh_result.generator_name
                ),
                "generated_mesh_path": str(
                    mesh_result.primary_output_path
                ),
                "scaled_mesh_path": str(
                    self_alignment.scaled_mesh_path
                ),
                "mesh_scale_m": mesh_scale,
                "candidate_index": (
                    self_alignment.candidate_index
                ),
                "hypothesis_rank": (
                    self_alignment.hypothesis_rank
                ),
                "foundationpose_score": (
                    self_alignment
                    .foundationpose_score
                ),
                "source_alignment_total_loss": (
                    source_alignment.total_loss
                ),
                "source_mask_iou": (
                    source_alignment.mask_iou
                ),
                "source_depth_loss": (
                    source_alignment.depth_loss
                ),
                "source_depth_residual_cm": (
                    source_alignment
                    .depth_residual_m
                    * 100.0
                ),
                "source_free_space_loss": (
                    source_alignment.free_space_loss
                ),
                "source_free_space_violation_ratio": (
                    _free_space_violation_ratio(
                        source_alignment
                    )
                ),
                "source_boundary_loss": (
                    source_alignment.boundary_loss
                ),
                "trusted_surface_ratio": None,
                "segmentation_confidence": (
                    segmentation.score
                ),
                "mask_area_px": int(
                    np.count_nonzero(
                        segmentation.mask_bool
                    )
                ),
                "mask_path": str(
                    segmentation.mask_bool_path
                ),
                "mask_iou_gt": None,
                **geometry,
            }
        )

    pair_results_path = (
        context.output_directory
        / "pair_results.csv"
    )
    path_results_path = (
        context.output_directory
        / "path_results.csv"
    )
    proxy_results_path = (
        context.output_directory
        / "proxy_results.csv"
    )
    _append_csv_rows(
        csv_path=pair_results_path,
        fields=PAIR_RESULT_FIELDS,
        rows=pair_rows,
    )
    _append_csv_rows(
        csv_path=path_results_path,
        fields=PATH_RESULT_FIELDS,
        rows=path_rows,
    )
    _append_csv_rows(
        csv_path=proxy_results_path,
        fields=PROXY_RESULT_FIELDS,
        rows=proxy_rows,
    )

    pair_summary_path = (
        pair_research_root
        / "research_result_paths.json"
    )
    pair_summary = {
        "run_id": context.run_id,
        "pair_id": pair_id,
        "pair_results_csv": str(
            pair_results_path
        ),
        "path_results_csv": str(
            path_results_path
        ),
        "proxy_results_csv": str(
            proxy_results_path
        ),
        "reference_pose_path": str(
            reference_pose_path
        ),
        "query_pose_path": str(
            query_pose_path
        ),
        "final_pose_path": (
            str(final_pose_output)
            if final_pose_output is not None
            else None
        ),
        "ground_truth_pose_path": str(
            ground_truth_pose_path
        ),
        "reference_evaluation_path": str(
            reference_evaluation.metadata_path
        ),
        "query_evaluation_path": str(
            query_evaluation.metadata_path
        ),
        "all_candidate_pose_artifacts": {
            "json": str(
                Path(pair_output_root)
                / "relative_pose_candidates"
                / "relative_pose_candidates.json"
            ),
            "reference_npy": str(
                Path(pair_output_root)
                / "relative_pose_candidates"
                / "reference_proxy_relative_poses.npy"
            ),
            "query_npy": str(
                Path(pair_output_root)
                / "relative_pose_candidates"
                / "query_proxy_relative_poses.npy"
            ),
        },
    }

    with pair_summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            pair_summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return ResearchLoggingResult(
        pair_results_path=pair_results_path,
        path_results_path=path_results_path,
        proxy_results_path=proxy_results_path,
        reference_pose_path=(
            reference_pose_path
        ),
        query_pose_path=query_pose_path,
        final_pose_path=final_pose_output,
        ground_truth_pose_path=(
            ground_truth_pose_path
        ),
    )
