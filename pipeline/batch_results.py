from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from core.types import FrameSpec, PairPipelineOutcome, PipelineConfig


def _save_batch_summary(
    *,
    config: PipelineConfig,
    reference_record: dict[str, Any],
    query_records: Sequence[dict[str, Any]],
) -> Path:
    summary_path = (
        config.output_root / "batch_summary.json"
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = list(query_records)
    payload = {
        "mode": "multi_query",
        "reference": reference_record,
        "requested_query_image_ids": list(
            config.batch_query_image_ids or ()
        ),
        "foundationpose_workers": (
            config.foundationpose_workers
        ),
        "completed_count": sum(
            record.get("status") == "completed"
            for record in records
        ),
        "failed_count": sum(
            record.get("status") == "failed"
            for record in records
        ),
        "queries": records,
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


def _persist_compact_batch_result(
    *,
    config: PipelineConfig,
    query_frame: FrameSpec,
    record: dict[str, Any],
    outcome: PairPipelineOutcome | None,
    error: Exception | None = None,
) -> None:
    from result_storage import (
        save_compact_pair_result,
    )

    paths = save_compact_pair_result(
        output_root=config.output_root,
        dataset_root=config.dataset_root,
        split=config.split,
        object_id=config.object_id,
        object_name=config.object_name,
        reference_scene_id=config.reference.scene_id,
        reference_image_id=config.reference.image_id,
        reference_instance_index=(
            config.reference.instance_index
        ),
        query_scene_id=query_frame.scene_id,
        query_image_id=query_frame.image_id,
        query_instance_index=query_frame.instance_index,
        pipeline_status=(
            "completed"
            if error is None
            else "failed"
        ),
        final_status=(
            outcome.final_status
            if outcome is not None
            else "FAILED"
        ),
        pose_accepted=(
            outcome.pose_accepted
            if outcome is not None
            else False
        ),
        reference_mask_source=record.get(
            "reference_mask_source"
        ),
        query_mask_source=record.get(
            "query_mask_source"
        ),
        summary_path=(
            outcome.summary_path
            if outcome is not None
            else None
        ),
        pose_path=(
            outcome.pose_path
            if outcome is not None
            else None
        ),
        error_type=(
            type(error).__name__
            if error is not None
            else None
        ),
        error=(
            str(error)
            if error is not None
            else None
        ),
    )
    record.update(
        {
            "compact_result_path": str(
                paths.result_path
            ),
            "final_summary_path": str(
                paths.result_path
            ),
            "pose_path": (
                str(paths.pose_path)
                if paths.pose_path is not None
                else None
            ),
            "evaluation_path": (
                str(paths.evaluation_path)
                if paths.evaluation_path
                is not None
                else None
            ),
            "visualization_path": None,
        }
    )

