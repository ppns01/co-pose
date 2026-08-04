from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from core.types import AlignedProxyState, PipelineConfig
from pipeline.scale_refinement_shared import _visible_scale_loss_improved


def _refine_aligned_states_independent_axis_scale(
    *,
    config: PipelineConfig,
    aligned_states: Sequence[AlignedProxyState],
    output_root: Path,
) -> tuple[AlignedProxyState, ...]:
    """Fit camera-frame Su/Sv/Sd independently, then rerun FP once."""
    from pose.alignment_evaluator import (
        evaluate_foundationpose_alignments,
        select_best_self_alignment,
    )
    from pose.alignment_scorer import AlignmentScoreWeights
    from pose.foundationpose_process_pool import (
        FoundationPoseProcessJob,
        run_foundationpose_jobs,
    )
    from pose.mesh_renderer import FoundationPoseMeshRenderer
    from scale.independent_axis_scale_refiner import (
        fit_independent_axis_scale,
    )

    states = tuple(aligned_states)
    if len(states) != 2:
        raise ValueError(
            "Independent axis fitting requires Reference and Query: "
            f"count={len(states)}"
        )
    state_by_name = {
        state.generated.view_name: state
        for state in states
    }
    if set(state_by_name) != {"reference", "query"}:
        raise ValueError(
            "Independent axis fitting requires one reference and one query"
        )

    source_scale_by_name = {
        view_name: float(state_by_name[view_name].selected_candidate.scale_m)
        for view_name in ("reference", "query")
    }
    pair_shared_scale_verified = math.isclose(
        source_scale_by_name["reference"],
        source_scale_by_name["query"],
        rel_tol=1e-6,
        abs_tol=1e-9,
    )

    print(
        "==== [STAGE] Independent camera-frame Su/Sv/Sd refinement: "
        "A0/B0 define image-horizontal/image-vertical/depth axes; "
        "no shared D and no G0 ===="
    )
    print(
        "[Independent axis input scale] "
        f"reference={source_scale_by_name['reference']:.6f} m, "
        f"query={source_scale_by_name['query']:.6f} m, "
        f"pair_shared_verified={pair_shared_scale_verified}"
    )
    weights = AlignmentScoreWeights(
        mask=config.alignment_weight_mask,
        depth=config.alignment_weight_depth,
        free_space=config.alignment_weight_free_space,
        boundary=config.alignment_weight_boundary,
    )
    searches: dict[str, Any] = {}
    with FoundationPoseMeshRenderer(
        foundationpose_repository_path=config.foundationpose_repository,
        device=config.device,
        render_batch_size=config.renderer_batch_size,
        maximum_texture_size=config.renderer_maximum_texture_size,
    ) as renderer:
        for view_name in ("reference", "query"):
            state = state_by_name[view_name]
            search = fit_independent_axis_scale(
                view_name=view_name,
                source_candidate=state.selected_candidate,
                prepared_view=state.generated.prepared_view,
                fixed_pose_camera_from_proxy=(
                    state.self_alignment.pose_camera_from_proxy
                ),
                renderer=renderer,
                output_directory=(
                    output_root / "axis_scale_refinement" / view_name
                ),
                weights=weights,
                depth_trim_quantile=config.alignment_depth_trim_quantile,
                minimum_depth_overlap_pixels=(
                    config.alignment_minimum_depth_overlap_pixels
                ),
                free_space_absolute_tolerance_m=(
                    config.alignment_free_space_absolute_tolerance_m
                ),
                free_space_relative_tolerance=(
                    config.alignment_free_space_relative_tolerance
                ),
                quantile_low=config.normalization_quantile_low,
                quantile_high=config.normalization_quantile_high,
                sample_count=config.normalization_sample_count,
                random_seed=config.normalization_random_seed,
                minimum_factor=config.axis_scale_minimum_factor,
                maximum_factor=config.axis_scale_maximum_factor,
                grid_step_count=config.axis_scale_grid_step_count,
                scale_penalty_weight=config.axis_scale_penalty_weight,
                uncertainty_scale_penalty_weight=(
                    config.axis_scale_uncertainty_penalty_weight
                ),
                depth_quantile_low=config.scale_quantile_low,
                depth_quantile_high=config.scale_quantile_high,
                depth_minimum_valid_points=(
                    config.scale_minimum_valid_points
                ),
                minimum_loss_improvement_ratio=(
                    config.axis_scale_minimum_loss_improvement_ratio
                ),
                coarse_shortlist_count=(
                    config.axis_scale_coarse_shortlist_count
                ),
                fine_factor_radius=(
                    config.axis_scale_fine_factor_radius
                ),
                fine_grid_step_count=(
                    config.axis_scale_fine_grid_step_count
                ),
            )
            searches[view_name] = search
            print(
                "[Independent camera-axis search] "
                f"view={view_name} factors={search.selected_scale_factors} "
                f"depth_observability={search.depth_observability:.6f} "
                f"fixed_loss={search.baseline_score.total_loss:.6f}->"
                f"{search.selected_score.total_loss:.6f} "
                f"applied={search.applied}"
            )

    applied_names = tuple(
        view_name
        for view_name in ("reference", "query")
        if searches[view_name].applied
    )
    results_by_name: dict[str, tuple[Any, ...]] = {}
    evaluations_by_name: dict[str, Any] = {}
    alignments_by_name: dict[str, Any] = {}
    if applied_names:
        jobs = tuple(
            FoundationPoseProcessJob(
                job_name=f"axis_scale:{view_name}",
                candidate=searches[view_name].selected_candidate,
                prepared_view=state_by_name[view_name].generated.prepared_view,
            )
            for view_name in applied_names
        )
        all_results = run_foundationpose_jobs(
            jobs=jobs,
            repository_path=config.foundationpose_repository,
            output_root=(
                output_root / "foundationpose" / "self_axis_scale"
            ),
            top_k=config.top_k,
            rotation_diversity_threshold_deg=(
                config.foundationpose_rotation_diversity_threshold_deg
            ),
            refine_iterations=config.refine_iterations,
            device=config.device,
            worker_count=config.foundationpose_workers,
            debug=config.foundationpose_debug,
        )
        if len(all_results) != len(applied_names):
            raise RuntimeError(
                "Independent axis FoundationPose result count mismatch"
            )
        with FoundationPoseMeshRenderer(
            foundationpose_repository_path=config.foundationpose_repository,
            device=config.device,
            render_batch_size=config.renderer_batch_size,
            maximum_texture_size=config.renderer_maximum_texture_size,
        ) as renderer:
            for view_name, result in zip(
                applied_names,
                all_results,
                strict=True,
            ):
                evaluation = evaluate_foundationpose_alignments(
                    prepared_view=(
                        state_by_name[view_name].generated.prepared_view
                    ),
                    candidate_results=(result,),
                    renderer=renderer,
                    output_directory=(
                        output_root
                        / "self_evaluation_axis_scale"
                        / view_name
                    ),
                    weights=weights,
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
                results_by_name[view_name] = (result,)
                evaluations_by_name[view_name] = evaluation
                alignments_by_name[view_name] = (
                    select_best_self_alignment(evaluation)
                )

    refined_by_name: dict[str, AlignedProxyState] = {}
    for view_name in ("reference", "query"):
        state = state_by_name[view_name]
        search = searches[view_name]
        final_alignment = alignments_by_name.get(view_name)
        final_loss = (
            final_alignment.alignment_loss
            if final_alignment is not None
            else None
        )
        accepted = bool(
            search.applied
            and _visible_scale_loss_improved(
                coarse_loss=search.baseline_score.total_loss,
                refined_loss=final_loss,
                minimum_improvement_ratio=(
                    config.axis_scale_minimum_loss_improvement_ratio
                ),
            )
        )
        applied_factors = (
            search.selected_scale_factors
            if accepted
            else (1.0, 1.0, 1.0)
        )
        applied_dimensions = (
            search.selected_dimensions_m
            if accepted
            else search.source_dimensions_m
        )
        selection_path = (
            output_root
            / "axis_scale_refinement"
            / view_name
            / "selection.json"
        )
        selection_path.write_text(
            json.dumps(
                {
                    "method": "independent_camera_axis_scale",
                    "view": view_name,
                    "axis_convention": (
                        "u=image horizontal, v=image vertical, d=camera depth"
                    ),
                    "accepted": accepted,
                    "selected_source": (
                        "axis_scaled" if accepted else "pre_axis_fallback"
                    ),
                    "requested_scale_factors_xyz": list(
                        search.selected_scale_factors
                    ),
                    "requested_scale_factors_uvd": list(
                        search.selected_scale_factors
                    ),
                    "applied_scale_factors_xyz": list(
                        applied_factors
                    ),
                    "applied_scale_factors_uvd": list(
                        applied_factors
                    ),
                    # Backward-compatible compact field: always the factors
                    # used by the final state, never a rejected proposal.
                    "scale_factors_xyz": list(
                        applied_factors
                    ),
                    "source_object_scale_m": (
                        search.source_candidate.scale_m
                    ),
                    "pair_shared_scale_verified": (
                        pair_shared_scale_verified
                    ),
                    "depth_observability": search.depth_observability,
                    "axis_penalty_weights_uvd": list(
                        search.axis_penalty_weights
                    ),
                    "source_dimensions_m": list(
                        search.source_dimensions_m
                    ),
                    "requested_dimensions_m": list(
                        search.selected_dimensions_m
                    ),
                    "applied_dimensions_m": list(
                        applied_dimensions
                    ),
                    "selected_dimensions_m": list(
                        applied_dimensions
                    ),
                    "fixed_pose_baseline_loss": (
                        search.baseline_score.total_loss
                    ),
                    "fixed_pose_selected_loss": (
                        search.selected_score.total_loss
                    ),
                    "final_foundationpose_loss": final_loss,
                    "minimum_loss_improvement_ratio": (
                        config.axis_scale_minimum_loss_improvement_ratio
                    ),
                    "search_path": str(search.summary_path),
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if accepted:
            refined_by_name[view_name] = AlignedProxyState(
                generated=state.generated,
                self_results=results_by_name[view_name],
                self_evaluation=evaluations_by_name[view_name],
                self_alignment=final_alignment,
                selected_candidate=search.selected_candidate,
            )
        else:
            refined_by_name[view_name] = state
        print(
            "[Independent axis final] "
            f"view={view_name} accepted={accepted} "
            f"requested_factors={search.selected_scale_factors} "
            f"applied_factors={applied_factors} "
            f"final_loss={final_loss}"
        )

    return tuple(
        refined_by_name[state.generated.view_name]
        for state in states
    )


