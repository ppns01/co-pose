from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from main import LINEMOD_OBJECT_METADATA


_CONFLICTING_OPTIONS = (
    "--object-id",
    "--object-name",
    "--sam3-prompt",
    "--reference-scene-id",
    "--reference-image-id",
    "--reference-instance-index",
    "--query-scene-id",
    "--query-image-id",
    "--query-image-ids",
    "--query-instance-index",
)

_PROCESS_GROUP_TERMINATION_GRACE_SECONDS = 2.0
_PROCESS_GROUP_POLL_INTERVAL_SECONDS = 0.05


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    return True


def _send_process_group_signal(
    process_group_id: int,
    signal_number: signal.Signals,
) -> bool:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        return False
    except OSError as error:
        print(
            "[LINEMOD process cleanup warning] "
            f"pgid={process_group_id}, "
            f"signal={signal_number.name}, "
            f"error={type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return False

    print(
        "[LINEMOD process cleanup] "
        f"pgid={process_group_id}, "
        f"signal={signal_number.name}"
    )
    return True


def _terminate_process_group(
    process: subprocess.Popen[Any],
) -> None:
    """Terminate an object process and all inherited workers."""

    process_group_id = process.pid

    if (
        process_group_id <= 0
        or process_group_id == os.getpgrp()
    ):
        print(
            "[LINEMOD process cleanup warning] "
            "Refusing to signal the current process group: "
            f"pgid={process_group_id}",
            file=sys.stderr,
        )
        return

    if not _send_process_group_signal(
        process_group_id,
        signal.SIGTERM,
    ):
        return

    deadline = (
        time.monotonic()
        + _PROCESS_GROUP_TERMINATION_GRACE_SECONDS
    )

    while time.monotonic() < deadline:
        process.poll()

        if not _process_group_exists(process_group_id):
            return

        time.sleep(_PROCESS_GROUP_POLL_INTERVAL_SECONDS)

    if not _send_process_group_signal(
        process_group_id,
        signal.SIGKILL,
    ):
        return

    if process.poll() is None:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            print(
                "[LINEMOD process cleanup warning] "
                "Object process did not exit after SIGKILL: "
                f"pgid={process_group_id}",
                file=sys.stderr,
            )


def _run_object_process(command: Sequence[str]) -> int:
    process = subprocess.Popen(
        list(command),
        start_new_session=True,
    )

    try:
        return_code = process.wait()
    except BaseException:
        _terminate_process_group(process)
        raise

    if return_code != 0:
        _terminate_process_group(process)

    return return_code


def _option_is_present(
    argv: Sequence[str],
    option: str,
) -> bool:
    return any(
        argument == option
        or argument.startswith(f"{option}=")
        for argument in argv
    )


def _remove_single_value_option(
    argv: Sequence[str],
    option: str,
) -> list[str]:
    cleaned: list[str] = []
    index = 0

    while index < len(argv):
        argument = argv[index]

        if argument.startswith(f"{option}="):
            index += 1
            continue

        if argument == option:
            if index + 1 >= len(argv):
                raise ValueError(
                    f"{option} requires a value."
                )
            index += 2
            continue

        cleaned.append(argument)
        index += 1

    return cleaned


def _remove_object_ids_option(
    argv: Sequence[str],
) -> list[str]:
    """Remove --object-ids and its integer values from child arguments."""
    cleaned: list[str] = []
    index = 0

    while index < len(argv):
        argument = argv[index]

        if argument.startswith("--object-ids="):
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                index += 1
            continue

        if argument == "--object-ids":
            index += 1
            value_count = 0
            while index < len(argv) and not argv[index].startswith("--"):
                index += 1
                value_count += 1
            if value_count == 0:
                raise ValueError("--object-ids requires at least one value.")
            continue

        cleaned.append(argument)
        index += 1

    return cleaned


def load_object_image_ids(
    *,
    dataset_root: Path,
    split: str,
    object_id: int,
) -> tuple[int, ...]:
    scene_gt_path = (
        Path(dataset_root)
        / split
        / f"{object_id:06d}"
        / "scene_gt.json"
    )

    if not scene_gt_path.is_file():
        raise FileNotFoundError(
            f"scene_gt.json not found: {scene_gt_path}"
        )

    with scene_gt_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise TypeError(
            "scene_gt.json root must be an object: "
            f"{scene_gt_path}"
        )

    image_ids: list[int] = []

    for raw_image_id, raw_instances in payload.items():
        if not isinstance(raw_instances, list):
            continue

        if any(
            isinstance(instance, dict)
            and instance.get("obj_id") == object_id
            for instance in raw_instances
        ):
            image_ids.append(int(raw_image_id))

    if not image_ids:
        raise ValueError(
            f"No frames for object {object_id}: "
            f"{scene_gt_path}"
        )

    return tuple(sorted(set(image_ids)))


def build_object_command(
    *,
    python_executable: str,
    entrypoint_path: Path,
    base_argv: Sequence[str],
    object_id: int,
    object_name: str,
    sam3_prompt: str,
    reference_image_id: int,
    query_image_ids: Sequence[int],
    output_root: Path,
) -> list[str]:
    if not query_image_ids:
        raise ValueError(
            "query_image_ids must not be empty."
        )

    return [
        python_executable,
        str(
            Path(entrypoint_path)
            .expanduser()
            .resolve()
        ),
        *base_argv,
        "--object-id",
        str(object_id),
        "--object-name",
        object_name,
        "--sam3-prompt",
        sam3_prompt,
        "--reference-scene-id",
        str(object_id),
        "--reference-image-id",
        str(reference_image_id),
        "--reference-instance-index",
        "0",
        "--query-scene-id",
        str(object_id),
        "--query-instance-index",
        "0",
        "--query-image-ids",
        *(
            str(image_id)
            for image_id in query_image_ids
        ),
        "--output-root",
        str(output_root),
    ]


def _save_summary(
    *,
    output_root: Path,
    pose_path: str,
    records: Sequence[dict[str, Any]],
) -> Path:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path = (
        output_root / "linemod_all_summary.json"
    )
    failed_count = sum(
        record["status"] != "completed"
        for record in records
    )
    payload = {
        "pose_path": pose_path,
        "execution": (
            "objects sequential; queries sequential "
            "inside each object batch"
        ),
        "completed_objects": (
            len(records) - failed_count
        ),
        "failed_objects": failed_count,
        "records": list(records),
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

    return summary_path


def _collect_current_results(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in records:
        object_output_root = record.get("output_root")
        query_image_ids = record.get("query_image_ids")
        if object_output_root is None or not isinstance(
            query_image_ids,
            list,
        ):
            continue

        for query_image_id in query_image_ids:
            result_path = (
                Path(object_output_root)
                / "results"
                / (
                    f"q{record['object_id']:06d}_"
                    f"{int(query_image_id):06d}_i00"
                )
                / "result.json"
            )
            if not result_path.is_file():
                continue
            try:
                payload = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            payload["result_path"] = str(result_path.resolve())
            results.append(payload)

    return results


def _add_or_adds_summary(
    *,
    results: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    def summarize(
        subset: Sequence[dict[str, Any]],
        target_count: int,
    ) -> dict[str, Any]:
        pose_result_count = sum(
            result.get("relative_pose_query_from_reference") is not None
            for result in subset
        )
        evaluated = [
            result
            for result in subset
            if isinstance(result.get("evaluation"), dict)
            and isinstance(
                result["evaluation"].get(
                    "add_or_adds_0_1d_passed"
                ),
                bool,
            )
        ]
        passed_count = sum(
            result["evaluation"]["add_or_adds_0_1d_passed"]
            for result in evaluated
        )
        return {
            "target_count": target_count,
            "pose_result_count": pose_result_count,
            "evaluated_count": len(evaluated),
            "passed_count": passed_count,
            "recall_over_all_targets": (
                passed_count / target_count
                if target_count > 0
                else None
            ),
            "recall_over_evaluated_poses": (
                passed_count / len(evaluated)
                if evaluated
                else None
            ),
        }

    per_object: list[dict[str, Any]] = []
    for record in records:
        query_image_ids = record.get("query_image_ids")
        if not isinstance(query_image_ids, list):
            continue
        object_id = int(record["object_id"])
        object_results = [
            result
            for result in results
            if result.get("object_id") == object_id
        ]
        per_object.append(
            {
                "object_id": object_id,
                "object_name": record.get("object_name"),
                **summarize(
                    object_results,
                    len(query_image_ids),
                ),
            }
        )

    target_count = sum(
        len(record["query_image_ids"])
        for record in records
        if isinstance(record.get("query_image_ids"), list)
    )
    overall = summarize(results, target_count)
    object_recalls = [
        item["recall_over_all_targets"]
        for item in per_object
        if item["recall_over_all_targets"] is not None
    ]
    overall["macro_recall_over_all_targets"] = (
        sum(object_recalls) / len(object_recalls)
        if object_recalls
        else None
    )
    return {
        "protocol": (
            "BOP models_eval vertices; ADD for asymmetric objects, "
            "ADD-S for objects with a BOP symmetry declaration; "
            "success when error < 0.1 * object diameter"
        ),
        "gt_usage": (
            "evaluation only: predicted query object pose = predicted "
            "relative pose @ reference GT object pose"
        ),
        "overall": overall,
        "per_object": per_object,
    }


def _add_auc_summary(
    *,
    results: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    from evaluation.linemod_add_evaluator import calculate_add_auc

    metric_specs = {
        "add_auc_0_1m": ("add_m", 0.1),
        "adds_auc_0_1m": ("adds_m", 0.1),
        "add_or_adds_auc_0_1m": ("add_or_adds_m", 0.1),
        "add_or_adds_auc_0_1d": (
            "add_or_adds_normalized",
            0.1,
        ),
    }

    def summarize(
        subset: Sequence[dict[str, Any]],
        target_count: int,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {"target_count": target_count}
        for output_name, (field_name, maximum_threshold) in (
            metric_specs.items()
        ):
            errors = []
            for result in subset:
                evaluation = result.get("evaluation")
                if not isinstance(evaluation, dict):
                    continue
                value = evaluation.get(field_name)
                if isinstance(value, (int, float)):
                    errors.append(float(value))
            summary[f"{output_name}_evaluated_count"] = len(errors)
            summary[output_name] = calculate_add_auc(
                errors,
                maximum_threshold=maximum_threshold,
                target_count=target_count,
            )
        return summary

    per_object: list[dict[str, Any]] = []
    for record in records:
        query_image_ids = record.get("query_image_ids")
        if not isinstance(query_image_ids, list):
            continue
        object_id = int(record["object_id"])
        object_results = [
            result
            for result in results
            if result.get("object_id") == object_id
        ]
        per_object.append(
            {
                "object_id": object_id,
                "object_name": record.get("object_name"),
                **summarize(object_results, len(query_image_ids)),
            }
        )

    target_count = sum(
        len(record["query_image_ids"])
        for record in records
        if isinstance(record.get("query_image_ids"), list)
    )
    overall = summarize(results, target_count)
    for output_name in metric_specs:
        object_values = [
            item[output_name]
            for item in per_object
            if item[output_name] is not None
        ]
        overall[f"macro_{output_name}"] = (
            sum(object_values) / len(object_values)
            if object_values
            else None
        )
    return {
        "protocol": (
            "Normalized area under empirical ADD recall curve; "
            "missing poses contribute zero. Absolute AUC cutoff is "
            "0.1 m; scale-normalized ADD(-S) AUC cutoff is 0.1d."
        ),
        "gt_usage": (
            "evaluation only: predicted query object pose = predicted "
            "relative pose @ reference GT object pose"
        ),
        "overall": overall,
        "per_object": per_object,
    }


def _load_official_bop19_scores(output_root: Path) -> dict[str, Any]:
    scores_path = (
        Path(output_root)
        / "evaluation"
        / "bop19"
        / "official_eval"
        / "selfmesh_lm-test"
        / "scores_bop19.json"
    )
    if not scores_path.is_file():
        return {
            "status": "not_available",
            "scores_path": str(scores_path.resolve()),
            "scores": None,
        }
    payload = json.loads(
        scores_path.read_text(encoding="utf-8"),
        parse_constant=lambda _value: None,
    )
    if not isinstance(payload, dict):
        raise ValueError(
            f"Official BOP score file must be a JSON object: {scores_path}"
        )
    return {
        "status": "completed",
        "scores_path": str(scores_path.resolve()),
        "scores": payload,
    }


def _save_aggregate_results(
    *,
    output_root: Path,
    records: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    results = _collect_current_results(records)
    json_path = output_root / "linemod_all_results.json"
    csv_path = output_root / "linemod_all_results.csv"
    json_path.write_text(
        json.dumps(
            {
                "result_count": len(results),
                "add_or_adds_0_1d_summary": _add_or_adds_summary(
                    results=results,
                    records=records,
                ),
                "add_auc_summary": _add_auc_summary(
                    results=results,
                    records=records,
                ),
                "bop19_official": _load_official_bop19_scores(
                    output_root
                ),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    fields = (
        "object_id",
        "object_name",
        "reference_scene_id",
        "reference_image_id",
        "query_scene_id",
        "query_image_id",
        "pipeline_status",
        "final_status",
        "pose_accepted",
        "pose_status",
        "selected_method",
        "reference_mask_source",
        "query_mask_source",
        "gt_assisted",
        "rotation_error_deg",
        "translation_error_cm",
        "translation_error_x_cm",
        "translation_error_y_cm",
        "translation_error_z_cm",
        "catastrophic_failure",
        "add_metric_used",
        "symmetry_aware",
        "symmetry_type",
        "object_diameter_m",
        "model_point_count",
        "add_m",
        "adds_m",
        "add_normalized",
        "adds_normalized",
        "add_or_adds_m",
        "add_or_adds_normalized",
        "add_or_adds_0_1d_threshold_m",
        "add_or_adds_0_1d_passed",
        "add_evaluation_error",
        "relative_pose_query_from_reference",
        "result_path",
        "error_type",
        "error",
    )
    with csv_path.open(mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            reference = result.get("reference") or {}
            query = result.get("query") or {}
            evaluation = result.get("evaluation") or {}
            segmentation = result.get("segmentation") or {}
            row = {
                "object_id": result.get("object_id"),
                "object_name": result.get("object_name"),
                "reference_scene_id": reference.get("scene_id"),
                "reference_image_id": reference.get("image_id"),
                "query_scene_id": query.get("scene_id"),
                "query_image_id": query.get("image_id"),
                "pipeline_status": result.get("pipeline_status"),
                "final_status": result.get("final_status"),
                "pose_accepted": result.get("pose_accepted"),
                "pose_status": result.get("pose_status"),
                "selected_method": result.get("selected_method"),
                "reference_mask_source": segmentation.get(
                    "reference_source"
                ),
                "query_mask_source": segmentation.get(
                    "query_source"
                ),
                "gt_assisted": segmentation.get("gt_assisted"),
                "rotation_error_deg": evaluation.get(
                    "rotation_error_deg"
                ),
                "translation_error_cm": evaluation.get(
                    "translation_error_cm"
                ),
                "translation_error_x_cm": evaluation.get(
                    "translation_error_x_cm"
                ),
                "translation_error_y_cm": evaluation.get(
                    "translation_error_y_cm"
                ),
                "translation_error_z_cm": evaluation.get(
                    "translation_error_z_cm"
                ),
                "catastrophic_failure": evaluation.get(
                    "catastrophic_failure"
                ),
                "add_metric_used": evaluation.get(
                    "add_metric_used"
                ),
                "symmetry_aware": evaluation.get(
                    "symmetry_aware"
                ),
                "symmetry_type": evaluation.get(
                    "symmetry_type"
                ),
                "object_diameter_m": evaluation.get(
                    "object_diameter_m"
                ),
                "model_point_count": evaluation.get(
                    "model_point_count"
                ),
                "add_m": evaluation.get("add_m"),
                "adds_m": evaluation.get("adds_m"),
                "add_normalized": evaluation.get(
                    "add_normalized"
                ),
                "adds_normalized": evaluation.get(
                    "adds_normalized"
                ),
                "add_or_adds_m": evaluation.get(
                    "add_or_adds_m"
                ),
                "add_or_adds_normalized": evaluation.get(
                    "add_or_adds_normalized"
                ),
                "add_or_adds_0_1d_threshold_m": evaluation.get(
                    "add_or_adds_0_1d_threshold_m"
                ),
                "add_or_adds_0_1d_passed": evaluation.get(
                    "add_or_adds_0_1d_passed"
                ),
                "add_evaluation_error": result.get(
                    "add_evaluation_error"
                ),
                "relative_pose_query_from_reference": json.dumps(
                    result.get("relative_pose_query_from_reference"),
                    ensure_ascii=False,
                ),
                "result_path": result.get("result_path"),
                "error_type": result.get("error_type"),
                "error": result.get("error"),
            }
            writer.writerow(row)

    return json_path, csv_path


def rebuild_linemod_aggregate_results(
    output_root: Path,
) -> tuple[Path, Path]:
    """Rebuild aggregate JSON/CSV from a completed runner summary."""
    resolved_output_root = Path(output_root).expanduser().resolve()
    summary_path = resolved_output_root / "linemod_all_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError(
            f"LINEMOD summary has no records list: {summary_path}"
        )
    return _save_aggregate_results(
        output_root=resolved_output_root,
        records=records,
    )


def _run_pose_metric_postprocessing(
    *,
    output_root: Path,
    dataset_root: Path,
    split: str,
    records: Sequence[dict[str, Any]],
) -> Path | None:
    if not any(
        isinstance(record.get("query_image_ids"), list)
        and bool(record["query_image_ids"])
        for record in records
    ):
        return None

    try:
        from scripts.evaluate_linemod_pose_metrics import (
            evaluate_output_root,
        )

        report = evaluate_output_root(
            output_root=output_root,
            dataset_root=dataset_root,
            split=split,
            rebuild_aggregate=False,
        )
    except Exception as error:
        print(
            "[LINEMOD metric postprocessing warning] "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return None
    if report["add_failure_count"]:
        print(
            "[LINEMOD metric postprocessing warning] "
            f"{report['add_failure_count']} pose(s) could not be "
            "evaluated for ADD/ADD-S; "
            "inference results remain saved.",
            file=sys.stderr,
        )
    report_path = Path(report["report_path"])
    print(f"[LINEMOD BOP19 evaluation] {report_path}")
    return report_path


def run_all_linemod_sequence(
    *,
    pose_path: str,
    entrypoint_path: Path,
    argv: Sequence[str],
) -> int:
    conflicts = tuple(
        option
        for option in _CONFLICTING_OPTIONS
        if _option_is_present(argv, option)
    )

    if conflicts:
        raise ValueError(
            "--all-linemod derives object/reference/query "
            "arguments and cannot be combined with "
            + ", ".join(conflicts)
            + "."
        )

    from main import build_config, parse_args

    parsed = parse_args(
        [
            *argv,
            "--pose-path",
            pose_path,
        ]
    )
    config = build_config(parsed)
    configured_output_root = (
        parsed.output_root is not None
    )
    output_root = (
        config.output_root
        if configured_output_root
        else (
            Path(__file__).resolve().parent
            / "outputs"
            / "linemod_all"
            / pose_path
        )
    )
    child_base_argv = _remove_object_ids_option(
        _remove_single_value_option(
            argv,
            "--output-root",
        )
    )
    records: list[dict[str, Any]] = []

    requested_object_ids = getattr(
        parsed,
        "object_ids",
        None,
    )
    object_ids = (
        tuple(requested_object_ids)
        if requested_object_ids is not None
        else config.linemod_all_object_ids
    )
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("--object-ids must not contain duplicates.")
    unsupported_ids = tuple(
        object_id
        for object_id in object_ids
        if object_id not in LINEMOD_OBJECT_METADATA
    )
    if unsupported_ids:
        raise ValueError(
            "Unsupported LINEMOD object ID(s): "
            + ", ".join(str(object_id) for object_id in unsupported_ids)
        )
    print(
        f"[LINEMOD all] {len(object_ids)} objects run sequentially. "
        "Each object's reference is reused for all "
        "remaining frames."
    )

    for sequence_index, object_id in enumerate(
        object_ids,
        start=1,
    ):
        object_name, sam3_prompt = (
            LINEMOD_OBJECT_METADATA[object_id]
        )
        record: dict[str, Any] = {
            "object_id": object_id,
            "object_name": object_name,
            "status": "running",
        }
        records.append(record)

        try:
            image_ids = load_object_image_ids(
                dataset_root=config.dataset_root,
                split=config.split,
                object_id=object_id,
            )
            reference_image_id = (
                config.reference.image_id
                if config.reference.image_id in image_ids
                else image_ids[0]
            )
            query_image_ids = tuple(
                image_id
                for image_id in image_ids
                if image_id != reference_image_id
            )
            query_image_ids = query_image_ids[
                :: config.linemod_all_query_stride
            ]
            if (
                config.linemod_all_maximum_queries_per_object
                is not None
            ):
                query_image_ids = query_image_ids[
                    : config.linemod_all_maximum_queries_per_object
                ]

            if not query_image_ids:
                raise ValueError(
                    f"Object {object_id} has no query frames."
                )

            object_output_root = (
                output_root
                / f"object_{object_id:02d}_{object_name}"
            )
            record.update(
                {
                    "reference_image_id": reference_image_id,
                    "query_count": len(query_image_ids),
                    "query_image_ids": list(query_image_ids),
                    "output_root": str(object_output_root),
                }
            )
            command = build_object_command(
                python_executable=sys.executable,
                entrypoint_path=entrypoint_path,
                base_argv=child_base_argv,
                object_id=object_id,
                object_name=object_name,
                sam3_prompt=sam3_prompt,
                reference_image_id=(
                    reference_image_id
                ),
                query_image_ids=query_image_ids,
                output_root=object_output_root,
            )

            print(
                "\n[LINEMOD object "
                f"{sequence_index}/{len(object_ids)}] "
                f"{object_id}: {object_name}, "
                f"reference={reference_image_id}, "
                f"queries={len(query_image_ids)}"
            )
            return_code = _run_object_process(command)

            if return_code != 0:
                raise RuntimeError(
                    "Object process exited with code "
                    f"{return_code}."
                )

            record.update(
                {
                    "status": "completed",
                }
            )

        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "error_type": (
                        type(error).__name__
                    ),
                    "error": str(error),
                }
            )
            print(
                "[LINEMOD object warning] "
                f"object={object_id}: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            if not config.linemod_all_continue_on_error:
                _save_summary(
                    output_root=output_root,
                    pose_path=pose_path,
                    records=records,
                )
                _save_aggregate_results(
                    output_root=output_root,
                    records=records,
                )
                break

        _save_summary(
            output_root=output_root,
            pose_path=pose_path,
            records=records,
        )
        _save_aggregate_results(
            output_root=output_root,
            records=records,
        )

    summary_path = _save_summary(
        output_root=output_root,
        pose_path=pose_path,
        records=records,
    )
    _run_pose_metric_postprocessing(
        output_root=output_root,
        dataset_root=config.dataset_root,
        split=config.split,
        records=records,
    )
    results_json_path, results_csv_path = _save_aggregate_results(
        output_root=output_root,
        records=records,
    )
    failed_count = sum(
        record["status"] != "completed"
        for record in records
    )
    print(
        "\n[LINEMOD all complete] "
        f"completed={len(records) - failed_count}, "
        f"failed={failed_count}"
    )
    print(f"[LINEMOD all summary] {summary_path}")
    print(f"[LINEMOD all results JSON] {results_json_path}")
    print(f"[LINEMOD all results CSV] {results_csv_path}")

    return 1 if failed_count else 0


def _cli_main(argv: Sequence[str]) -> int:
    """Direct `python3 linemod_all_runner.py ...` entrypoint.

    Equivalent to `python3 main_self_mesh.py --all-linemod ...` (or
    main_self_cross.py, via --pose-path) without going through
    pose_path_entrypoint.py's --all-linemod flag.

    run_all_linemod_sequence spawns one child process per object by
    re-invoking this same script (build_object_command) with per-object
    flags such as --object-id appended and no --all-linemod. Those child
    invocations must fall straight through to main.main(...) for that one
    object -- calling run_all_linemod_sequence again would trip its own
    _CONFLICTING_OPTIONS check.
    """
    import argparse

    pose_path_parser = argparse.ArgumentParser(add_help=False)
    pose_path_parser.add_argument(
        "--pose-path",
        choices=("self_mesh",),
        default="self_mesh",
    )
    known, remaining_argv = pose_path_parser.parse_known_args(argv)

    if any(
        _option_is_present(remaining_argv, option)
        for option in _CONFLICTING_OPTIONS
    ):
        from main import main as run_single_object

        return run_single_object(
            [*remaining_argv, "--pose-path", known.pose_path]
        )

    return run_all_linemod_sequence(
        pose_path=known.pose_path,
        entrypoint_path=Path(__file__),
        argv=remaining_argv,
    )


if __name__ == "__main__":
    raise SystemExit(_cli_main(sys.argv[1:]))
