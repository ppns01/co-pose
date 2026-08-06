from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.types import EvaluationStatus


@dataclass(frozen=True)
class CompactResultPaths:
    result_directory: Path
    result_path: Path
    pose_path: Path | None
    evaluation_path: Path | None
    inference_status: str = EvaluationStatus.FAILED.value
    relative_evaluation_status: str = EvaluationStatus.NOT_APPLICABLE.value
    add_evaluation_status: str = EvaluationStatus.NOT_APPLICABLE.value
    all_required_evaluations_completed: bool = False
    pose_produced: bool = False
    quality_gate_mode: str = "not_run"
    quality_gate_passed: bool | None = None
    gt_assisted: bool = False


def compact_query_label(
    *,
    scene_id: int,
    image_id: int,
    instance_index: int,
) -> str:
    return f"q{scene_id:06d}_{image_id:06d}_i{instance_index:02d}"


def _read_summary_fields(
    summary_path: Path | None,
) -> dict[str, Any]:
    if summary_path is None or not Path(summary_path).is_file():
        return {}

    try:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    allowed_fields = (
        "status",
        "method",
        "selected_method",
        "pose_accepted",
        "pose_convention",
        "composition",
        "reason",
        "selection_policy",
    )
    summary_fields = {
        field: payload[field] for field in allowed_fields if field in payload
    }
    sources = payload.get("sources")
    if isinstance(sources, dict):
        axis_selection = sources.get("axis_scale_refinement")
        if isinstance(axis_selection, dict):
            # Preserve only the per-view numerical Sx/Sy/Sz decisions.
            summary_fields["axis_scale_refinement"] = axis_selection
        shared_dimensions = sources.get("confidence_shared_dimensions")
        compact_selection = {
            "selected_candidate": sources.get("selected_candidate"),
            "selected_dgedi_stage": sources.get("selected_dgedi_stage"),
            "final_selection_confidence": sources.get(
                "final_selection_confidence"
            ),
            "h0_score": sources.get("h0_score"),
            "h1_score": sources.get("h1_score"),
            "confidence_shared_dimensions": shared_dimensions,
        }
        if any(value is not None for value in compact_selection.values()):
            summary_fields["continuous_pose_selection"] = compact_selection
    return summary_fields


def _evaluation_values(evaluation: Any) -> dict[str, Any]:
    return {
        "rotation_error_deg": evaluation.rotation_error_deg,
        "translation_error_m": evaluation.translation_error_m,
        "translation_error_cm": evaluation.translation_error_cm,
        "translation_error_x_cm": evaluation.translation_error_x_cm,
        "translation_error_y_cm": evaluation.translation_error_y_cm,
        "translation_error_z_cm": evaluation.translation_error_z_cm,
        "reference_origin_translation_error_cm": (
            evaluation.reference_origin_translation_error_cm
        ),
        "catastrophic_failure": evaluation.catastrophic_failure,
        "threshold_results": [
            {
                "name": item.threshold.name,
                "maximum_rotation_error_deg": (
                    item.threshold.maximum_rotation_error_deg
                ),
                "maximum_translation_error_m": (
                    item.threshold.maximum_translation_error_m
                ),
                "success": item.success,
            }
            for item in evaluation.threshold_results
        ],
    }


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def save_compact_pair_result(
    *,
    output_root: Path,
    dataset_root: Path,
    split: str,
    object_id: int,
    object_name: str,
    reference_scene_id: int,
    reference_image_id: int,
    reference_instance_index: int,
    query_scene_id: int,
    query_image_id: int,
    query_instance_index: int,
    pipeline_status: str,
    final_status: str,
    pose_produced: bool,
    quality_gate_mode: str = "not_run",
    quality_gate_passed: bool | None = None,
    reference_mask_source: str | None = None,
    query_mask_source: str | None = None,
    summary_path: Path | None = None,
    pose_path: Path | None = None,
    error_type: str | None = None,
    error: str | None = None,
) -> CompactResultPaths:
    """Persist an accepted or rejected estimate and numerical evaluation."""
    result_directory = (
        Path(output_root).expanduser().resolve()
        / "results"
        / compact_query_label(
            scene_id=query_scene_id,
            image_id=query_image_id,
            instance_index=query_instance_index,
        )
    )
    result_directory.mkdir(parents=True, exist_ok=True)

    summary_fields = _read_summary_fields(summary_path)
    compact_pose_path: Path | None = None
    evaluation_path: Path | None = None
    pose_values: list[list[float]] | None = None
    evaluation_values: dict[str, Any] | None = None
    evaluation_error: str | None = None
    add_evaluation_error: str | None = None

    if pose_path is not None:
        source_pose_path = Path(pose_path).expanduser().resolve()
        if not source_pose_path.is_file():
            raise FileNotFoundError(
                f"Final relative pose not found: {source_pose_path}"
            )
        pose = np.asarray(
            np.load(source_pose_path, allow_pickle=False),
            dtype=np.float64,
        )
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("Final relative pose must be a finite (4,4) matrix.")
        compact_pose_path = result_directory / (
            "final_relative_pose.npy" if pose_produced else "rejected_relative_pose.npy"
        )
        np.save(compact_pose_path, pose, allow_pickle=False)
        pose_values = pose.tolist()

        from evaluation.relative_pose_evaluator import BOPFrameGTSpec

        reference_frame = BOPFrameGTSpec(
            dataset_root=dataset_root,
            split=split,
            scene_id=reference_scene_id,
            image_id=reference_image_id,
            object_id=object_id,
            instance_index=reference_instance_index,
        )
        query_frame = BOPFrameGTSpec(
            dataset_root=dataset_root,
            split=split,
            scene_id=query_scene_id,
            image_id=query_image_id,
            object_id=object_id,
            instance_index=query_instance_index,
        )

        try:
            from evaluation.relative_pose_evaluator import (
                evaluate_relative_pose,
                load_bop_linemod_absolute_gt_pose,
            )

            evaluation = evaluate_relative_pose(
                predicted_relative_pose=pose,
                reference_frame=reference_frame,
                query_frame=query_frame,
                output_directory=result_directory / "evaluation",
            )
            evaluation_values = _evaluation_values(evaluation)
            evaluation_path = evaluation.metadata_path

            # Generate GT pose visualization
            try:
                from evaluation.mesh_on_photo_visualizer import render_mesh_with_gt_pose

                # Load GT poses
                reference_gt_pose = load_bop_linemod_absolute_gt_pose(reference_frame)
                query_gt_pose = load_bop_linemod_absolute_gt_pose(query_frame)

                # Find CAD mesh path
                models_dir = Path(dataset_root) / "models"
                cad_mesh_path = models_dir / f"obj_{object_id:06d}.ply"

                if cad_mesh_path.is_file():
                    # Need to load view data for RGB and masks
                    from core.types import FrameSpec
                    from preparation.view_loader import load_view

                    ref_frame_spec = FrameSpec(
                        dataset_root=dataset_root,
                        split=split,
                        scene_id=reference_scene_id,
                        image_id=reference_image_id,
                        object_id=object_id,
                        instance_index=reference_instance_index,
                    )
                    query_frame_spec = FrameSpec(
                        dataset_root=dataset_root,
                        split=split,
                        scene_id=query_scene_id,
                        image_id=query_image_id,
                        object_id=object_id,
                        instance_index=query_instance_index,
                    )

                    ref_view = load_view(ref_frame_spec)
                    query_view = load_view(query_frame_spec)

                    gt_viz_path = render_mesh_with_gt_pose(
                        cad_mesh_path=cad_mesh_path,
                        reference_gt_pose=reference_gt_pose,
                        query_gt_pose=query_gt_pose,
                        reference_camera_k=np.asarray(ref_view.camera_matrix, dtype=np.float64),
                        query_camera_k=np.asarray(query_view.camera_matrix, dtype=np.float64),
                        reference_rgb=np.asarray(ref_view.rgb),
                        query_rgb=np.asarray(query_view.rgb),
                        reference_mask_bool=np.asarray(ref_view.mask_bool, dtype=bool),
                        query_mask_bool=np.asarray(query_view.mask_bool, dtype=bool),
                        output_path=(
                            result_directory / "evaluation" / f"gt_mesh_visualization_obj{object_id}.png"
                        ),
                        title=f"Object {object_id}: Ground Truth Pose Visualization",
                    )
            except Exception as gt_viz_error:
                # GT visualization is optional, don't fail the whole evaluation
                pass

        except Exception as evaluation_exception:
            evaluation_error = (
                f"{type(evaluation_exception).__name__}: {evaluation_exception}"
            )

        try:
            from evaluation.linemod_add_evaluator import (
                evaluate_linemod_add_metrics,
            )

            add_evaluation = evaluate_linemod_add_metrics(
                predicted_relative_pose=pose,
                reference_frame=reference_frame,
                query_frame=query_frame,
            )
            if evaluation_values is None:
                evaluation_values = {}
            evaluation_values.update(add_evaluation.to_dict())
        except Exception as add_exception:
            add_evaluation_error = f"{type(add_exception).__name__}: {add_exception}"

    # inference_status/relative_evaluation_status/add_evaluation_status are an
    # axis fully separate from pipeline/quality-gate state. A pair with
    # all_required_evaluations_completed=false stays in the denominator of
    # paper averages and counts as a failure; it is never dropped.
    inference_status = (
        EvaluationStatus.COMPLETED
        if str(pipeline_status).lower() == "completed"
        else EvaluationStatus.FAILED
    )
    if pose_values is None:
        relative_evaluation_status = EvaluationStatus.NOT_APPLICABLE
        add_evaluation_status = EvaluationStatus.NOT_APPLICABLE
    else:
        relative_evaluation_status = (
            EvaluationStatus.FAILED
            if evaluation_error is not None
            else EvaluationStatus.COMPLETED
        )
        add_evaluation_status = (
            EvaluationStatus.FAILED
            if add_evaluation_error is not None
            else EvaluationStatus.COMPLETED
        )
    all_required_evaluations_completed = bool(
        relative_evaluation_status is EvaluationStatus.COMPLETED
        and add_evaluation_status is EvaluationStatus.COMPLETED
    )
    gt_assisted = "gt_fallback" in {reference_mask_source, query_mask_source}

    selected_method = summary_fields.get("selected_method") or summary_fields.get(
        "method"
    )
    result_path = result_directory / "result.json"
    payload = {
        "dataset": "BOP-LINEMOD",
        "object_id": object_id,
        "object_name": object_name,
        "reference": {
            "scene_id": reference_scene_id,
            "image_id": reference_image_id,
            "instance_index": reference_instance_index,
        },
        "query": {
            "scene_id": query_scene_id,
            "image_id": query_image_id,
            "instance_index": query_instance_index,
        },
        "pipeline_status": pipeline_status,
        "final_status": final_status,
        # Deprecated duplicates of pose_produced/relative_evaluation_error,
        # kept for linemod_all_runner.py backward compatibility.
        "pose_accepted": bool(pose_produced and pose_values is not None),
        "pose_status": (
            "accepted"
            if pose_produced and pose_values is not None
            else ("rejected" if pose_values is not None else "missing")
        ),
        "selected_method": selected_method,
        "segmentation": {
            "reference_source": reference_mask_source,
            "query_source": query_mask_source,
            "gt_assisted": gt_assisted,
        },
        "pose_convention": "T_query_camera_from_reference_camera",
        "relative_pose_query_from_reference": pose_values,
        "evaluation": evaluation_values,
        "evaluation_error": evaluation_error,
        "add_evaluation_error": add_evaluation_error,
        "inference_status": inference_status.value,
        "relative_evaluation_status": relative_evaluation_status.value,
        "add_evaluation_status": add_evaluation_status.value,
        "all_required_evaluations_completed": all_required_evaluations_completed,
        "relative_evaluation_error": evaluation_error,
        "pose_produced": bool(pose_produced and pose_values is not None),
        "quality_gate_mode": quality_gate_mode,
        "quality_gate_passed": quality_gate_passed,
        "error_type": error_type,
        "error": error,
        "final_summary": summary_fields,
        "artifacts": {
            "pose": (compact_pose_path.name if compact_pose_path is not None else None),
            "evaluation": (
                str(evaluation_path.relative_to(result_directory))
                if evaluation_path is not None
                else None
            ),
        },
    }
    result_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return CompactResultPaths(
        result_directory=result_directory,
        result_path=result_path,
        pose_path=compact_pose_path,
        evaluation_path=evaluation_path,
        inference_status=inference_status.value,
        relative_evaluation_status=relative_evaluation_status.value,
        add_evaluation_status=add_evaluation_status.value,
        all_required_evaluations_completed=all_required_evaluations_completed,
        pose_produced=bool(pose_produced and pose_values is not None),
        quality_gate_mode=quality_gate_mode,
        quality_gate_passed=quality_gate_passed,
        gt_assisted=gt_assisted,
    )


def backfill_compact_result_add_metrics(
    *,
    result_path: Path,
    dataset_root: Path,
    split: str = "test",
    models_directory: Path | None = None,
) -> dict[str, Any]:
    """Add ADD/ADD-S fields to one existing compact result in place."""
    resolved_result_path = Path(result_path).expanduser().resolve()
    payload = json.loads(resolved_result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Compact result must be a JSON object: {resolved_result_path}"
        )

    pose_values = payload.get("relative_pose_query_from_reference")
    if pose_values is None:
        raise ValueError(
            f"Compact result has no estimated pose: {resolved_result_path}"
        )
    pose = np.asarray(pose_values, dtype=np.float64)

    reference = payload.get("reference")
    query = payload.get("query")
    if not isinstance(reference, dict) or not isinstance(query, dict):
        raise ValueError(
            f"Compact result has invalid frame metadata: {resolved_result_path}"
        )

    object_id = int(payload["object_id"])
    from evaluation.relative_pose_evaluator import BOPFrameGTSpec
    from evaluation.linemod_add_evaluator import (
        evaluate_linemod_add_metrics,
    )

    add_evaluation = evaluate_linemod_add_metrics(
        predicted_relative_pose=pose,
        reference_frame=BOPFrameGTSpec(
            dataset_root=dataset_root,
            split=split,
            scene_id=int(reference["scene_id"]),
            image_id=int(reference["image_id"]),
            object_id=object_id,
            instance_index=int(reference.get("instance_index", 0)),
        ),
        query_frame=BOPFrameGTSpec(
            dataset_root=dataset_root,
            split=split,
            scene_id=int(query["scene_id"]),
            image_id=int(query["image_id"]),
            object_id=object_id,
            instance_index=int(query.get("instance_index", 0)),
        ),
        models_directory=models_directory,
    )

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
        payload["evaluation"] = evaluation
    evaluation.update(add_evaluation.to_dict())
    payload["add_evaluation_error"] = None
    payload["add_evaluation_status"] = EvaluationStatus.COMPLETED.value

    relative_evaluation_status = payload.get("relative_evaluation_status")
    if relative_evaluation_status is None:
        # Legacy result.json predates this field; derive it conservatively
        # from the pre-existing error strings rather than fabricating one.
        relative_evaluation_status = (
            EvaluationStatus.COMPLETED.value
            if (
                payload.get("relative_evaluation_error") is None
                and payload.get("evaluation_error") is None
                and isinstance(payload.get("evaluation"), dict)
            )
            else EvaluationStatus.FAILED.value
        )
        payload["relative_evaluation_status"] = relative_evaluation_status
    payload["all_required_evaluations_completed"] = (
        relative_evaluation_status == EvaluationStatus.COMPLETED.value
    )

    _write_json_atomic(resolved_result_path, payload)
    return payload


def remove_pipeline_intermediate_directory(
    *,
    path: Path,
    output_root: Path,
) -> bool:
    """Remove only an exact query workdir or the shared-reference workdir."""
    root = Path(output_root).expanduser().resolve()
    target = Path(path).expanduser().resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Refusing to remove a path outside output_root: {target}"
        ) from error

    allowed = relative == Path("shared_reference") or (
        len(relative.parts) == 2
        and relative.parts[0] == "queries"
        and relative.parts[1].startswith("q")
    )
    if not allowed:
        raise ValueError(f"Refusing to remove a non-intermediate path: {target}")
    if not target.exists():
        return False
    if not target.is_dir() or target.is_symlink():
        raise ValueError(f"Intermediate target must be a real directory: {target}")

    shutil.rmtree(target)
    if relative.parts[0] == "queries":
        try:
            target.parent.rmdir()
        except OSError:
            pass
    return True
