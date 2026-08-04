from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from core.types import AlignedProxyState, PipelineConfig
from pipeline.scale_refinement_shared import (
    _candidate_by_index,
    _visible_scale_enabled_for_view,
    _visible_scale_loss_improved,
)


def _refine_aligned_states_visible_scale_independent(
    *,
    config: PipelineConfig,
    aligned_states: Sequence[AlignedProxyState],
    output_root: Path,
) -> tuple[AlignedProxyState, ...]:
    """
    Self visible 대응점으로 scale을 보정하고 좁은 범위에서 재정합한다.

    추정이 gate를 통과하지 못한 view는 coarse self 결과를
    그대로 유지한다.
    """

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
    from scale.local_scale_refiner import (
        build_visible_scale_candidates,
        estimate_visible_scale_from_self_alignment,
    )

    normalized_states = tuple(aligned_states)

    if not normalized_states:
        raise ValueError(
            "Visible scale을 보정할 self state가 없습니다."
        )

    enabled_state_entries: list[
        tuple[int, AlignedProxyState]
    ] = []
    for state_index, state in enumerate(
        normalized_states
    ):
        if _visible_scale_enabled_for_view(
            config=config,
            view_name=state.generated.view_name,
        ):
            enabled_state_entries.append(
                (state_index, state)
            )
            continue

        print(
            "[Visible scale disabled for view] "
            f"{state.generated.view_name}; "
            "keeping coarse self result"
        )

    if not enabled_state_entries:
        return normalized_states

    refinement_entries: list[
        tuple[
            int,
            AlignedProxyState,
            Any,
            tuple[Any, ...],
        ]
    ] = []

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
        for (
            state_index,
            state,
        ) in enabled_state_entries:
            refinement_root = (
                output_root
                / "visible_scale_refinement"
                / state.generated.view_name
            )
            estimate = estimate_visible_scale_from_self_alignment(
                prepared_view=(
                    state.generated.prepared_view
                ),
                self_alignment=(
                    state.self_alignment
                ),
                renderer=renderer,
                output_directory=(
                    refinement_root / "estimate"
                ),
                mask_erosion_kernel_size=(
                    config.visible_scale_mask_erosion_kernel_size
                ),
                minimum_correspondences=(
                    config.visible_scale_minimum_correspondences
                ),
                minimum_spatial_coverage=(
                    config.visible_scale_minimum_spatial_coverage
                ),
                depth_absolute_tolerance_m=(
                    config.visible_scale_depth_absolute_tolerance_m
                ),
                depth_relative_tolerance=(
                    config.visible_scale_depth_relative_tolerance
                ),
                irls_iterations=(
                    config.visible_scale_irls_iterations
                ),
                huber_delta_m=(
                    config.visible_scale_huber_delta_m
                ),
                maximum_abs_log_correction=(
                    config.visible_scale_maximum_abs_log_correction
                ),
                coverage_grid_size=(
                    config.visible_scale_coverage_grid_size
                ),
            )

            if not estimate.valid:
                print(
                    "[Visible scale skip] "
                    f"{state.generated.view_name}: "
                    f"{estimate.rejection_reason}; "
                    f"correspondences="
                    f"{estimate.correspondence_count}, "
                    f"coverage="
                    f"{estimate.spatial_coverage:.3f}"
                )
                continue

            candidates = build_visible_scale_candidates(
                normalization_result=(
                    state.generated.normalization_result
                ),
                original_scale_result=(
                    state.generated.scale_result
                ),
                estimate=estimate,
                output_directory=(
                    refinement_root
                    / "scaled_candidates"
                ),
                local_scale_multipliers=(
                    config.visible_scale_local_multipliers
                ),
            )
            refinement_entries.append(
                (
                    state_index,
                    state,
                    estimate,
                    candidates,
                )
            )

            print(
                "[Visible scale] "
                f"{state.generated.view_name}: "
                f"s0={estimate.initial_scale_m:.6f} m, "
                f"s_hat="
                f"{estimate.absolute_scale_m:.6f} m, "
                f"delta_log="
                f"{estimate.log_scale_correction:+.6f}"
            )

    if not refinement_entries:
        return normalized_states

    jobs = tuple(
        FoundationPoseProcessJob(
            job_name=(
                f"visible_scale:"
                f"{state.generated.view_name}:"
                f"{candidate.candidate_index:02d}"
            ),
            candidate=candidate,
            prepared_view=(
                state.generated.prepared_view
            ),
        )
        for _, state, _, candidates in refinement_entries
        for candidate in candidates
    )
    all_results = run_foundationpose_jobs(
        jobs=jobs,
        repository_path=(
            config.foundationpose_repository
        ),
        output_root=(
            output_root
            / "foundationpose"
            / "self_visible_scale"
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

    for _, _, _, candidates in refinement_entries:
        next_cursor = cursor + len(candidates)
        result_groups.append(
            tuple(all_results[cursor:next_cursor])
        )
        cursor = next_cursor

    if cursor != len(all_results):
        raise RuntimeError(
            "Visible scale FoundationPose result partition failed."
        )

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
    refined_states = list(normalized_states)

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
        for (
            entry,
            results,
        ) in zip(
            refinement_entries,
            result_groups,
            strict=True,
        ):
            (
                state_index,
                state,
                _,
                candidates,
            ) = entry
            evaluation = evaluate_foundationpose_alignments(
                prepared_view=(
                    state.generated.prepared_view
                ),
                candidate_results=results,
                renderer=renderer,
                output_directory=(
                    output_root
                    / "self_evaluation_visible_scale"
                    / state.generated.view_name
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
                candidates,
                self_alignment.candidate_index,
            )

            coarse_loss = state.self_alignment.alignment_loss
            refined_loss = (
                self_alignment.alignment_loss
            )
            accepted = _visible_scale_loss_improved(
                coarse_loss=coarse_loss,
                refined_loss=refined_loss,
                minimum_improvement_ratio=(
                    config.visible_scale_minimum_loss_improvement_ratio
                ),
            )

            decision_path = (
                output_root
                / "visible_scale_refinement"
                / state.generated.view_name
                / "selection.json"
            )
            decision_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            with decision_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "accepted": accepted,
                        "selected_source": (
                            "visible_scale_refined"
                            if accepted
                            else "coarse_fallback"
                        ),
                        "coarse_loss": coarse_loss,
                        "refined_loss": refined_loss,
                        "minimum_loss_improvement_ratio": (
                            config.visible_scale_minimum_loss_improvement_ratio
                        ),
                        "required_maximum_refined_loss": (
                            (
                                coarse_loss
                                * (
                                    1.0
                                    - config.visible_scale_minimum_loss_improvement_ratio
                                )
                            )
                            if coarse_loss
                            is not None
                            else None
                        ),
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
                file.write("\n")

            if not accepted:
                print(
                    "[Visible scale fallback] "
                    f"{state.generated.view_name}: "
                    f"coarse_loss={coarse_loss}, "
                    f"refined_loss={refined_loss}, "
                    "keeping coarse self result"
                )
                continue

            print(
                "[Visible scale accepted] "
                f"{state.generated.view_name}: "
                f"coarse_loss={coarse_loss}, "
                f"refined_loss={refined_loss}"
            )

            refined_states[state_index] = (
                AlignedProxyState(
                    generated=state.generated,
                    self_results=results,
                    self_evaluation=evaluation,
                    self_alignment=self_alignment,
                    selected_candidate=(
                        selected_candidate
                    ),
                )
            )

    return tuple(refined_states)


