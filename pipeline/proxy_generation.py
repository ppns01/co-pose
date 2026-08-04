from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from core.types import (
    AlignedProxyState,
    FrameSpec,
    GeneratedProxyState,
    PipelineConfig,
)
from pipeline.scale_refinement_dispatch import (
    _apply_post_coarse_proxy_refinement,
)
from pipeline.scale_refinement_shared import _candidate_by_index


def _generate_scaled_candidates(
    *,
    config: PipelineConfig,
    generator: Any,
    prepared_view: Any,
    view_name: str,
    output_root: Path,
) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    from generators.base import (
        MeshGenerationRequest,
    )
    from scale.depth_scale_initializer import (
        initialize_scale_from_depth,
    )
    from scale.mesh_normalizer import (
        normalize_generated_mesh,
    )
    from scale.mesh_scaler import (
        build_scaled_mesh_candidates,
    )

    mesh_result = generator.generate(
        MeshGenerationRequest(
            view_name=view_name,
            segmented_rgb_path=(
                prepared_view.segmented_rgb_path
            ),
            output_directory=(
                output_root / "generated"
            ),
            mask_bool_path=(
                prepared_view.segmentation.mask_bool_path
            ),
            mask_rgb_path=(
                prepared_view.segmentation.mask_rgb_path
            ),
        )
    )

    normalization_result = (
        normalize_generated_mesh(
            mesh_result=mesh_result,
            output_directory=(
                output_root / "normalized"
            ),
            quantile_low=(
                config.normalization_quantile_low
            ),
            quantile_high=(
                config.normalization_quantile_high
            ),
            sample_count=(
                config.normalization_sample_count
            ),
            random_seed=(
                config.normalization_random_seed
            ),
            normalization_tolerance=(
                config.normalization_tolerance
            ),
        )
    )

    scale_result = initialize_scale_from_depth(
        prepared_view=prepared_view,
        output_directory=(
            output_root / "scale_initialization"
        ),
        quantile_low=config.scale_quantile_low,
        quantile_high=config.scale_quantile_high,
        scale_multipliers=(
            config.scale_multipliers
        ),
        min_valid_points=(
            config.scale_minimum_valid_points
        ),
        max_points=config.scale_maximum_points,
        min_depth_m=config.scale_minimum_depth_m,
        max_depth_m=config.scale_maximum_depth_m,
        save_point_cloud=(
            config.scale_save_point_cloud
        ),
    )

    candidates = build_scaled_mesh_candidates(
        normalization_result=(
            normalization_result
        ),
        scale_result=scale_result,
        output_directory=(
            output_root / "scaled_candidates"
        ),
    )

    return (
        mesh_result,
        normalization_result,
        scale_result,
        candidates,
    )


def _self_align_generated_states(
    *,
    config: PipelineConfig,
    generated_states: Sequence[
        GeneratedProxyState
    ],
    output_root: Path,
) -> tuple[AlignedProxyState, ...]:
    from pose.alignment_evaluator import (
        evaluate_foundationpose_alignments,
        select_best_self_alignment,
    )
    from pose.alignment_scorer import (
        AlignmentScoreWeights,
    )
    from pose.foundationpose_process_pool import (
        FoundationPoseProcessJob,
        run_foundationpose_jobs,
    )
    from pose.mesh_renderer import (
        FoundationPoseMeshRenderer,
    )

    normalized_states = tuple(generated_states)

    if not normalized_states:
        raise ValueError(
            "At least one generated proxy state is required."
        )

    print(
        "==== [STAGE] Coarse self-pose search: "
        f"{sum(len(s.candidates) for s in normalized_states)} "
        "candidate(s) total, running FoundationPose register() "
        "per candidate ===="
    )

    jobs = tuple(
        FoundationPoseProcessJob(
            job_name=(
                f"self:{state.view_name}:{candidate.candidate_index:02d}"
            ),
            candidate=candidate,
            prepared_view=state.prepared_view,
        )
        for state in normalized_states
        for candidate in state.candidates
    )

    all_results = run_foundationpose_jobs(
        jobs=jobs,
        repository_path=(
            config.foundationpose_repository
        ),
        output_root=(
            output_root
            / "foundationpose"
            / "self"
        ),
        top_k=config.top_k,
        rotation_diversity_threshold_deg=(
            config.foundationpose_rotation_diversity_threshold_deg
        ),
        refine_iterations=(
            config.refine_iterations
        ),
        device=config.device,
        worker_count=(
            config.foundationpose_workers
        ),
        debug=config.foundationpose_debug,
    )

    result_groups: list[tuple[Any, ...]] = []
    cursor = 0

    for state in normalized_states:
        next_cursor = cursor + len(
            state.candidates
        )
        result_groups.append(
            tuple(all_results[cursor:next_cursor])
        )
        cursor = next_cursor

    if cursor != len(all_results):
        raise RuntimeError(
            "FoundationPose self result partition failed."
        )

    aligned_states: list[AlignedProxyState] = []

    with FoundationPoseMeshRenderer(
        foundationpose_repository_path=(
            config.foundationpose_repository
        ),
        device=config.device,
        render_batch_size=(
            config.renderer_batch_size
        ),
        maximum_texture_size=(
            config.renderer_maximum_texture_size
        ),
    ) as renderer:
        alignment_weights = AlignmentScoreWeights(
            mask=config.alignment_weight_mask,
            depth=config.alignment_weight_depth,
            free_space=(
                config.alignment_weight_free_space
            ),
            boundary=(
                config.alignment_weight_boundary
            ),
        )

        for state, results in zip(
            normalized_states,
            result_groups,
            strict=True,
        ):
            evaluation = evaluate_foundationpose_alignments(
                prepared_view=(
                    state.prepared_view
                ),
                candidate_results=results,
                renderer=renderer,
                output_directory=(
                    output_root
                    / "self_evaluation"
                    / state.view_name
                ),
                weights=alignment_weights,
                depth_trim_quantile=(
                    config.alignment_depth_trim_quantile
                ),
                min_depth_overlap_pixels=(
                    config.alignment_minimum_depth_overlap_pixels
                ),
                free_space_absolute_tolerance_m=(
                    config.alignment_free_space_absolute_tolerance_m
                ),
                free_space_relative_tolerance=(
                    config.alignment_free_space_relative_tolerance
                ),
            )

            self_alignment = (
                select_best_self_alignment(
                    evaluation
                )
            )
            selected_candidate = _candidate_by_index(
                state.candidates,
                self_alignment.candidate_index,
            )

            aligned_states.append(
                AlignedProxyState(
                    generated=state,
                    self_results=results,
                    self_evaluation=evaluation,
                    self_alignment=self_alignment,
                    selected_candidate=(
                        selected_candidate
                    ),
                )
            )

    coarse_aligned_states = tuple(aligned_states)

    return _apply_post_coarse_proxy_refinement(
        config=config,
        coarse_aligned_states=coarse_aligned_states,
        output_root=output_root,
    )



def _build_generated_state(
    *,
    config: PipelineConfig,
    generator: Any,
    view_name: str,
    frame: FrameSpec,
    prepared_view: Any,
    output_root: Path,
) -> GeneratedProxyState:
    (
        mesh_result,
        normalization_result,
        scale_result,
        candidates,
    ) = _generate_scaled_candidates(
        config=config,
        generator=generator,
        prepared_view=prepared_view,
        view_name=view_name,
        output_root=(
            output_root / "proxies" / view_name
        ),
    )

    return GeneratedProxyState(
        view_name=view_name,
        frame=frame,
        prepared_view=prepared_view,
        mesh_result=mesh_result,
        normalization_result=normalization_result,
        scale_result=scale_result,
        candidates=candidates,
    )
