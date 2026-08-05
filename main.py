from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from core.types import (
    AlignedProxyState,
    FrameSpec,
    GeneratedProxyState,
    PairPipelineOutcome,
    PipelineConfig,
)
from pipeline.config_defaults import (
    DEFAULT_DINOV3_REPOSITORY,
    DEFAULT_INSTANTMESH_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    DINOV3_CHECKPOINT_FILENAMES,
    DINOV3_HF_MODEL_DIRECTORIES,
    LINEMOD_OBJECT_METADATA,
    POSE_PATH_CHOICES,
    PROJECT_ROOT,
    _default_dinov3_weights_path,
    _default_instantmesh_python,
)
from pipeline.config_build import (
    _pose_path_uses_dino,
    _query_frames,
    _require_directory,
    _require_file,
    build_config,
)
from pipeline.config_cli import (
    _batch_query_output_label,
    parse_args,
)
from pipeline.config_persist import (
    _initialize_research_run,
    _print_runtime_config,
    _save_run_config,
)
from pipeline.config_validate import validate_config
from pipeline.scale_refinement_dispatch import (
    _apply_pair_scale_refinement,
)
from pipeline.scale_refinement_shared import (
    _candidate_by_index,
    _visible_scale_enabled_for_view,
    _visible_scale_loss_improved,
)
from pipeline.batch_results import (
    _persist_compact_batch_result,
    _save_batch_summary,
)
from pipeline.dgedi_final_stage import (
    _load_compact_axis_scale_selection,
    _run_aligned_pair,
    _run_self_mesh_final_dgedi,
)
from pipeline.proxy_generation import (
    _build_generated_state,
    _generate_scaled_candidates,
    _self_align_generated_states,
)
from pipeline.view_preparation import (
    _extract_dino_inputs,
    _prepare_linemod_view,
    _prepared_mask_source,
)






def run_pipeline(
    config: PipelineConfig,
) -> Path | None:
    pipeline_started_at = time.perf_counter()
    validate_config(config)
    _print_runtime_config(config)

    if config.batch_query_image_ids is not None:
        raise ValueError(
            "run_pipeline only accepts single-query configs."
        )

    from generators.instantmesh_generator import (
        InstantMeshGenerator,
    )
    from mask_provider import (
        release_sam3_processor,
    )

    config_path = _save_run_config(config)
    research_context = _initialize_research_run(
        config=config,
        config_path=config_path,
    )
    timing_values: dict[str, Any] = {}

    print(f"[Config] {config_path}")
    print("[1/8] Prepare LINEMOD RGB-D and masks")

    segmentation_started_at = time.perf_counter()
    try:
        reference_view = _prepare_linemod_view(
            config=config,
            view_name="reference",
            frame=config.reference,
        )
        query_view = _prepare_linemod_view(
            config=config,
            view_name="query",
            frame=config.query,
        )
    finally:
        release_sam3_processor()
    timing_values["segmentation_time_sec"] = (
        time.perf_counter()
        - segmentation_started_at
    )

    dino_started_at = time.perf_counter()
    (
        (
            reference_dino,
            reference_surface,
        ),
        (
            query_dino,
            query_surface,
        ),
    ) = _extract_dino_inputs(
        config=config,
        prepared_views=(
            reference_view,
            query_view,
        ),
        output_roots=(
            config.output_root,
            config.output_root,
        ),
    )
    timing_values["dino_feature_time_sec"] = (
        time.perf_counter() - dino_started_at
    )

    print(
        "[2/8] Generate InstantMesh proxies and scale candidates"
    )

    with InstantMeshGenerator(
        repository_path=(
            config.instantmesh_repository
        ),
        python_executable=(
            config.instantmesh_python
        ),
        config_path=config.instantmesh_config,
        use_rembg=(config.instantmesh_use_rembg),
        offline=config.instantmesh_offline,
        seed=config.random_seed,
        diffusion_steps=(
            config.instantmesh_diffusion_steps
        ),
        view_count=(
            config.instantmesh_view_count
        ),
        model_scale=(
            config.instantmesh_model_scale
        ),
        render_distance=(
            config.instantmesh_render_distance
        ),
        export_texture_map=(
            config.instantmesh_export_texture_map
        ),
        save_video=(
            config.instantmesh_save_video
        ),
    ) as generator:
        generation_started_at = (
            time.perf_counter()
        )
        reference_generated = (
            _build_generated_state(
                config=config,
                generator=generator,
                view_name="reference",
                frame=config.reference,
                prepared_view=reference_view,
                output_root=config.output_root,
            )
        )
        timing_values[
            "generation_time_ref_sec"
        ] = (
            time.perf_counter()
            - generation_started_at
        )

        generation_started_at = (
            time.perf_counter()
        )
        query_generated = _build_generated_state(
            config=config,
            generator=generator,
            view_name="query",
            frame=config.query,
            prepared_view=query_view,
            output_root=config.output_root,
        )
        timing_values[
            "generation_time_query_sec"
        ] = (
            time.perf_counter()
            - generation_started_at
        )

    print(
        "[3/8] Self-align proxies to their RGB-D"
    )
    print(
        "[4/8] Evaluate self poses with mask + depth"
    )

    source_anchor_started_at = time.perf_counter()
    reference_state, query_state = (
        _self_align_generated_states(
            config=config,
            generated_states=(
                reference_generated,
                query_generated,
            ),
            output_root=config.output_root,
        )
    )
    timing_values["source_anchor_time_sec"] = (
        time.perf_counter()
        - source_anchor_started_at
    )

    outcome = _run_aligned_pair(
        config=config,
        reference_state=reference_state,
        query_state=query_state,
        reference_dino=reference_dino,
        reference_surface=reference_surface,
        query_dino=query_dino,
        query_surface=query_surface,
        output_root=config.output_root,
        research_context=research_context,
        timings=timing_values,
        pipeline_started_at=pipeline_started_at,
    )

    if (
        config.evaluation_enabled
        and outcome.pose_path is not None
    ):
        try:
            import numpy as np

            from evaluation.relative_pose_evaluator import (
                BOPFrameGTSpec,
                evaluate_relative_pose,
            )

            predicted_relative_pose = np.load(
                outcome.pose_path
            )

            reference_frame = BOPFrameGTSpec(
                dataset_root=config.dataset_root,
                split=config.split,
                scene_id=config.reference.scene_id,
                image_id=config.reference.image_id,
                object_id=config.object_id,
                instance_index=(
                    config.reference.instance_index
                ),
            )
            query_frame = BOPFrameGTSpec(
                dataset_root=config.dataset_root,
                split=config.split,
                scene_id=config.query.scene_id,
                image_id=config.query.image_id,
                object_id=config.object_id,
                instance_index=(
                    config.query.instance_index
                ),
            )

            evaluation = evaluate_relative_pose(
                predicted_relative_pose=(
                    predicted_relative_pose
                ),
                reference_frame=reference_frame,
                query_frame=query_frame,
                output_directory=(
                    config.output_root / "evaluation"
                ),
            )

            print(
                "[Evaluation] rotation_error_deg="
                f"{evaluation.rotation_error_deg:.3f}, "
                "translation_error_cm="
                f"{evaluation.translation_error_cm:.3f}"
            )
        except Exception as evaluation_error:
            print(
                "[Evaluation warning] Failed to score "
                "the predicted relative pose against GT "
                "(the completed run is unaffected): "
                f"{type(evaluation_error).__name__}: "
                f"{evaluation_error}"
            )

    return outcome.pose_path




def run_batch_pipeline(
    config: PipelineConfig,
) -> int:
    validate_config(config)
    _print_runtime_config(config)

    if config.batch_query_image_ids is None:
        raise ValueError(
            "run_batch_pipeline requires --query-image-ids."
        )

    from generators.instantmesh_generator import (
        InstantMeshGenerator,
    )
    from mask_provider import (
        release_sam3_processor,
    )

    config_path = _save_run_config(config)
    research_context = _initialize_research_run(
        config=config,
        config_path=config_path,
    )
    reference_timings: dict[str, Any] = {}
    reference_root = (
        config.output_root / "shared_reference"
    )
    reference_record: dict[str, Any] = {
        "scene_id": config.reference.scene_id,
        "image_id": config.reference.image_id,
        "instance_index": (
            config.reference.instance_index
        ),
        "output_root": str(reference_root),
        "status": "running",
    }
    query_records: list[dict[str, Any]] = []

    resumed_completed_keys: set[
        tuple[int, int, int]
    ] = set()
    if config.resume:
        previous_summary_path = (
            config.output_root
            / "batch_summary.json"
        )
        if previous_summary_path.is_file():
            with previous_summary_path.open(
                mode="r",
                encoding="utf-8",
            ) as file:
                previous_summary = json.load(file)

            for (
                previous_record
            ) in previous_summary.get(
                "queries",
                [],
            ):
                if (
                    previous_record.get("status")
                    != "completed"
                ):
                    continue

                resumed_completed_keys.add(
                    (
                        previous_record.get(
                            "scene_id"
                        ),
                        previous_record.get(
                            "image_id"
                        ),
                        previous_record.get(
                            "instance_index"
                        ),
                    )
                )
                query_records.append(
                    previous_record
                )

            if resumed_completed_keys:
                print(
                    "[Batch resume] "
                    f"{len(resumed_completed_keys)} "
                    "previously completed quer(y/ies) will "
                    f"be skipped: {previous_summary_path}"
                )

    summary_path = _save_batch_summary(
        config=config,
        reference_record=reference_record,
        query_records=query_records,
    )

    print(f"[Config] {config_path}")
    print(f"[Batch summary] {summary_path}")
    print(
        "[Batch] Queries run sequentially; candidate workers apply inside each query."
    )

    with InstantMeshGenerator(
        repository_path=(
            config.instantmesh_repository
        ),
        python_executable=(
            config.instantmesh_python
        ),
        config_path=config.instantmesh_config,
        use_rembg=(config.instantmesh_use_rembg),
        offline=config.instantmesh_offline,
        seed=config.random_seed,
        diffusion_steps=(
            config.instantmesh_diffusion_steps
        ),
        view_count=(
            config.instantmesh_view_count
        ),
        model_scale=(
            config.instantmesh_model_scale
        ),
        render_distance=(
            config.instantmesh_render_distance
        ),
        export_texture_map=(
            config.instantmesh_export_texture_map
        ),
        save_video=(
            config.instantmesh_save_video
        ),
    ) as generator:
        try:
            reference_work_started_at = (
                time.perf_counter()
            )
            print(
                "[Batch reference 1/4] Prepare RGB-D and mask"
            )
            stage_started_at = time.perf_counter()
            try:
                reference_view = _prepare_linemod_view(
                    config=config,
                    view_name="reference",
                    frame=config.reference,
                    output_root=reference_root,
                )
            finally:
                release_sam3_processor()
            reference_timings[
                "segmentation_time_sec"
            ] = (
                time.perf_counter()
                - stage_started_at
            )

            stage_started_at = time.perf_counter()
            (
                (
                    reference_dino,
                    reference_surface,
                ),
            ) = _extract_dino_inputs(
                config=config,
                prepared_views=(reference_view,),
                output_roots=(reference_root,),
            )
            reference_timings[
                "dino_feature_time_sec"
            ] = (
                time.perf_counter()
                - stage_started_at
            )

            print(
                "[Batch reference 2/4] Generate proxy and scale candidates"
            )
            stage_started_at = time.perf_counter()
            reference_generated = (
                _build_generated_state(
                    config=config,
                    generator=generator,
                    view_name="reference",
                    frame=config.reference,
                    prepared_view=reference_view,
                    output_root=reference_root,
                )
            )
            reference_timings[
                "generation_time_ref_sec"
            ] = (
                time.perf_counter()
                - stage_started_at
            )

            print(
                "[Batch reference 3/4] Self-alignment"
            )
            print(
                "[Batch reference 4/4] Mask/depth evaluation"
            )
            stage_started_at = time.perf_counter()
            (reference_state,) = (
                _self_align_generated_states(
                    config=config,
                    generated_states=(
                        reference_generated,
                    ),
                    output_root=reference_root,
                )
            )
            reference_timings[
                "source_anchor_time_sec"
            ] = (
                time.perf_counter()
                - stage_started_at
            )
            reference_timings[
                "_shared_reference_time_sec"
            ] = (
                time.perf_counter()
                - reference_work_started_at
            )

        except Exception as error:
            reference_record.update(
                {
                    "status": "failed",
                    "error_type": (
                        type(error).__name__
                    ),
                    "error": str(error),
                }
            )
            _save_batch_summary(
                config=config,
                reference_record=reference_record,
                query_records=query_records,
            )
            if (
                config.storage_results_only
                and not config.storage_keep_failed_intermediates
            ):
                from result_storage import (
                    remove_pipeline_intermediate_directory,
                )

                removed = remove_pipeline_intermediate_directory(
                    path=reference_root,
                    output_root=config.output_root,
                )
                reference_record.update(
                    {
                        "output_root": None,
                        "intermediate_removed": removed,
                    }
                )
                _save_batch_summary(
                    config=config,
                    reference_record=reference_record,
                    query_records=query_records,
                )
            raise

        reference_record.update(
            {
                "status": "completed",
                "mask_source": (
                    _prepared_mask_source(
                        reference_view
                    )
                ),
                "gt_assisted": (
                    _prepared_mask_source(
                        reference_view
                    )
                    == "gt_fallback"
                ),
                "selected_candidate_index": (
                    reference_state.self_alignment.candidate_index
                ),
                "self_evaluation_path": str(
                    reference_state.self_evaluation.summary_path
                ),
            }
        )
        _save_batch_summary(
            config=config,
            reference_record=reference_record,
            query_records=query_records,
        )

        query_frames = _query_frames(config)

        for (
            query_number,
            query_frame,
        ) in enumerate(
            query_frames,
            start=1,
        ):
            if (
                query_frame.scene_id,
                query_frame.image_id,
                query_frame.instance_index,
            ) in resumed_completed_keys:
                print(
                    "\n[Batch query "
                    f"{query_number}/{len(query_frames)}] "
                    f"scene={query_frame.scene_id}, "
                    f"image={query_frame.image_id} "
                    "- skipped (resume: already completed)"
                )
                continue

            query_root = (
                config.output_root
                / "queries"
                / (
                    f"q{query_frame.scene_id:06d}_"
                    f"{query_frame.image_id:06d}_"
                    f"i{query_frame.instance_index:02d}"
                )
            )
            record: dict[str, Any] = {
                "scene_id": query_frame.scene_id,
                "image_id": query_frame.image_id,
                "instance_index": (
                    query_frame.instance_index
                ),
                "output_root": str(query_root),
                "status": "running",
            }
            query_records.append(record)
            _save_batch_summary(
                config=config,
                reference_record=reference_record,
                query_records=query_records,
            )

            print(
                "\n[Batch query "
                f"{query_number}/{len(query_frames)}] "
                f"scene={query_frame.scene_id}, "
                f"image={query_frame.image_id}"
            )
            query_pipeline_started_at = (
                time.perf_counter()
            )
            timing_values = dict(
                reference_timings
            )

            try:
                print(
                    "[1/8] Prepare query RGB-D and mask"
                )
                stage_started_at = (
                    time.perf_counter()
                )
                try:
                    query_view = _prepare_linemod_view(
                        config=config,
                        view_name="query",
                        frame=query_frame,
                        output_root=query_root,
                    )
                    record.update(
                        {
                            "reference_mask_source": (
                                _prepared_mask_source(
                                    reference_view
                                )
                            ),
                            "query_mask_source": (
                                _prepared_mask_source(
                                    query_view
                                )
                            ),
                            "gt_assisted": (
                                "gt_fallback"
                                in {
                                    _prepared_mask_source(
                                        reference_view
                                    ),
                                    _prepared_mask_source(
                                        query_view
                                    ),
                                }
                            ),
                        }
                    )
                finally:
                    release_sam3_processor()
                timing_values[
                    "segmentation_time_sec"
                ] = reference_timings.get(
                    "segmentation_time_sec",
                    0.0,
                ) + (
                    time.perf_counter()
                    - stage_started_at
                )

                stage_started_at = (
                    time.perf_counter()
                )
                (
                    (
                        query_dino,
                        query_surface,
                    ),
                ) = _extract_dino_inputs(
                    config=config,
                    prepared_views=(query_view,),
                    output_roots=(query_root,),
                )
                timing_values[
                    "dino_feature_time_sec"
                ] = reference_timings.get(
                    "dino_feature_time_sec",
                    0.0,
                ) + (
                    time.perf_counter()
                    - stage_started_at
                )

                print(
                    "[2/8] Generate query proxy and scale candidates"
                )
                stage_started_at = (
                    time.perf_counter()
                )
                query_generated = (
                    _build_generated_state(
                        config=config,
                        generator=generator,
                        view_name="query",
                        frame=query_frame,
                        prepared_view=query_view,
                        output_root=query_root,
                    )
                )
                timing_values[
                    "generation_time_query_sec"
                ] = (
                    time.perf_counter()
                    - stage_started_at
                )

                print(
                    "[3/8] Self-align query proxies"
                )
                print(
                    "[4/8] Evaluate query self poses"
                )
                stage_started_at = (
                    time.perf_counter()
                )
                (query_state,) = (
                    _self_align_generated_states(
                        config=config,
                        generated_states=(
                            query_generated,
                        ),
                        output_root=query_root,
                    )
                )
                print(
                    "[Batch pair scale refinement] "
                    f"query_image_id={query_frame.image_id}"
                )
                (
                    pair_reference_state,
                    query_state,
                ) = _apply_pair_scale_refinement(
                    config=config,
                    aligned_states=(
                        reference_state,
                        query_state,
                    ),
                    output_root=query_root,
                    reference_frame=config.reference,
                    query_frame=query_frame,
                    pair_visible_scale_pending=True,
                )

                timing_values[
                    "source_anchor_time_sec"
                ] = reference_timings.get(
                    "source_anchor_time_sec",
                    0.0,
                ) + (
                    time.perf_counter()
                    - stage_started_at
                )

                outcome = _run_aligned_pair(
                    config=config,
                    reference_state=(
                        pair_reference_state
                    ),
                    query_state=query_state,
                    reference_dino=reference_dino,
                    reference_surface=(
                        reference_surface
                    ),
                    query_dino=query_dino,
                    query_surface=query_surface,
                    output_root=query_root,
                    research_context=(
                        research_context
                    ),
                    timings=timing_values,
                    pipeline_started_at=(
                        query_pipeline_started_at
                    ),
                )

                record.update(
                    {
                        "status": "completed",
                        "final_status": (
                            outcome.final_status
                        ),
                        "final_summary_path": str(
                            outcome.summary_path
                        ),
                        "pose_path": (
                            str(outcome.pose_path)
                            if outcome.pose_path
                            is not None
                            else None
                        ),
                        "visualization_path": str(
                            outcome.visualization_path
                        )
                        if (
                            outcome.visualization_path
                            is not None
                        )
                        else None,
                        "visualization_error": (
                            outcome.visualization_error
                        ),
                        "research_summary_path": (
                            str(
                                outcome.research_summary_path
                            )
                            if (
                                outcome.research_summary_path
                                is not None
                            )
                            else None
                        ),
                    }
                )
                if config.storage_results_only:
                    _persist_compact_batch_result(
                        config=config,
                        query_frame=query_frame,
                        record=record,
                        outcome=outcome,
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
                    "[Batch query failed] "
                    f"image={query_frame.image_id}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                if config.storage_results_only:
                    try:
                        _persist_compact_batch_result(
                            config=config,
                            query_frame=query_frame,
                            record=record,
                            outcome=None,
                            error=error,
                        )
                    except (
                        Exception
                    ) as storage_error:
                        record[
                            "result_storage_error"
                        ] = f"{type(storage_error).__name__}: {storage_error}"

            finally:
                if (
                    config.storage_results_only
                    and (
                        record.get("status")
                        != "failed"
                        or not config.storage_keep_failed_intermediates
                    )
                ):
                    from result_storage import (
                        remove_pipeline_intermediate_directory,
                    )

                    removed = remove_pipeline_intermediate_directory(
                        path=query_root,
                        output_root=config.output_root,
                    )
                    record.update(
                        {
                            "output_root": None,
                            "intermediate_removed": removed,
                        }
                    )
                summary_path = _save_batch_summary(
                    config=config,
                    reference_record=(
                        reference_record
                    ),
                    query_records=query_records,
                )

        if config.storage_results_only:
            from result_storage import (
                remove_pipeline_intermediate_directory,
            )

            removed = remove_pipeline_intermediate_directory(
                path=reference_root,
                output_root=config.output_root,
            )
            reference_record.update(
                {
                    "output_root": None,
                    "self_evaluation_path": None,
                    "intermediate_removed": removed,
                }
            )
            summary_path = _save_batch_summary(
                config=config,
                reference_record=reference_record,
                query_records=query_records,
            )

    failed_count = sum(
        record["status"] == "failed"
        for record in query_records
    )
    completed_count = sum(
        record["status"] == "completed"
        for record in query_records
    )

    print(
        f"\n[Batch complete] completed={completed_count}, failed={failed_count}"
    )
    print(f"[Batch summary] {summary_path}")

    return 1 if failed_count else 0


def _remove_object_ids_argument(
    argv: Sequence[str],
) -> list[str]:
    cleaned: list[str] = []
    index = 0

    while index < len(argv):
        argument = argv[index]

        if (
            argument == "--object-ids"
            or argument.startswith(
                "--object-ids="
            )
        ):
            index += 1

            while index < len(argv) and not argv[
                index
            ].startswith("-"):
                index += 1

            continue

        cleaned.append(argument)
        index += 1

    return cleaned


def run_multi_object_sequence(
    *,
    argv: Sequence[str],
    object_ids: Sequence[int],
    output_root: Path | None,
) -> int:
    normalized_ids = tuple(object_ids)
    conflicting_options = tuple(
        option
        for option in (
            "--object-name",
            "--sam3-prompt",
            "--reference-scene-id",
            "--query-scene-id",
        )
        if any(
            argument == option
            or argument.startswith(f"{option}=")
            for argument in argv
        )
    )

    if conflicting_options:
        raise ValueError(
            "--object-ids cannot be combined with "
            + ", ".join(conflicting_options)
            + ". Multi-object mode derives these values "
            "for each LINEMOD object."
        )

    if len(set(normalized_ids)) != len(
        normalized_ids
    ):
        raise ValueError(
            "object_ids must not contain duplicates."
        )

    unsupported_ids = sorted(
        object_id
        for object_id in normalized_ids
        if object_id
        not in LINEMOD_OBJECT_METADATA
    )

    if unsupported_ids:
        raise ValueError(
            "Unsupported LINEMOD object ID(s): "
            + ", ".join(
                str(object_id)
                for object_id in unsupported_ids
            )
        )

    base_argv = _remove_object_ids_argument(argv)
    failed_ids: list[int] = []

    print(
        "[Object batch] Objects run sequentially in separate Python processes."
    )

    for sequence_index, object_id in enumerate(
        normalized_ids,
        start=1,
    ):
        object_name, sam3_prompt = (
            LINEMOD_OBJECT_METADATA[object_id]
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *base_argv,
            "--object-id",
            str(object_id),
            "--object-name",
            object_name,
            "--sam3-prompt",
            sam3_prompt,
            "--reference-scene-id",
            str(object_id),
            "--query-scene-id",
            str(object_id),
        ]

        if output_root is not None:
            command.extend(
                (
                    "--output-root",
                    str(
                        output_root
                        / f"object_{object_id:02d}"
                    ),
                )
            )

        print(
            "\n[Object batch "
            f"{sequence_index}/{len(normalized_ids)}] "
            f"object={object_id} ({object_name})"
        )
        completed = subprocess.run(
            command,
            check=False,
        )

        if completed.returncode != 0:
            failed_ids.append(object_id)
            print(
                "[Object batch warning] "
                f"object={object_id} exited with "
                f"code {completed.returncode}. "
                "Continuing with the next object.",
                file=sys.stderr,
            )

    completed_count = len(normalized_ids) - len(
        failed_ids
    )
    print(
        "\n[Object batch complete] "
        f"completed={completed_count}, "
        f"failed={len(failed_ids)}"
    )

    if failed_ids:
        print(
            "[Object batch failed IDs] "
            + ", ".join(
                str(object_id)
                for object_id in failed_ids
            ),
            file=sys.stderr,
        )

    return 1 if failed_ids else 0


def main(
    argv: Sequence[str] | None = None,
) -> int:
    materialized_argv = (
        list(argv)
        if argv is not None
        else sys.argv[1:]
    )
    args = parse_args(materialized_argv)
    object_ids = getattr(
        args,
        "object_ids",
        None,
    )

    if object_ids is not None:
        return run_multi_object_sequence(
            argv=materialized_argv,
            object_ids=object_ids,
            output_root=args.output_root,
        )

    config = build_config(args)

    if config.batch_query_image_ids is None:
        run_pipeline(config)
        return 0

    return run_batch_pipeline(config)


if __name__ == "__main__":
    raise SystemExit(main())
