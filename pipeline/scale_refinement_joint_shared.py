from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from core.types import AlignedProxyState, PipelineConfig
from pipeline.scale_refinement_shared import _candidate_by_index


def _refine_aligned_states_joint_shared_scale(
    *,
    config: PipelineConfig,
    aligned_states: Sequence[AlignedProxyState],
    output_root: Path,
) -> tuple[AlignedProxyState, ...]:
    """
    Reference와 Query를 동시에 설명하는 하나의 physical scale을 선택한다.

    후보 bank:
        Reference visible estimate 주변 3개
        Query visible estimate 주변 3개
        두 estimate의 기하평균 1개

    선택:
        각 view에서 candidate별 best hypothesis loss를 구한다.
        view별 minimum loss로 정규화한 뒤 worst normalized loss를
        최소화한다. 동률이면 mean normalized loss와 기하평균
        중심 거리를 순서대로 사용한다.
    """
    from pose.alignment_evaluator import (
        evaluate_foundationpose_alignments,
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
    from pose.relative_pose_builder import (
        select_self_alignment,
    )
    from scale.local_scale_refiner import (
        build_absolute_scale_candidates,
        estimate_visible_scale_from_self_alignment,
    )

    states = tuple(aligned_states)

    if len(states) != 2:
        raise ValueError(
            "joint_shared scale은 정확히 Reference와 Query "
            f"두 state를 요구합니다: count={len(states)}"
        )

    print(
        "==== [STAGE] Joint-shared S* candidate search: "
        "coarse self-pose 결과를 기반으로 reference/query가 공유할 "
        "metric scale 후보를 만들고 재평가합니다 ===="
    )

    state_by_name: dict[
        str, AlignedProxyState
    ] = {}
    state_index_by_name: dict[str, int] = {}

    for state_index, state in enumerate(states):
        view_name = state.generated.view_name

        if view_name in state_by_name:
            raise ValueError(
                f"중복 view state입니다: {view_name}"
            )

        state_by_name[view_name] = state
        state_index_by_name[view_name] = (
            state_index
        )

    required_views = {"reference", "query"}

    if set(state_by_name) != required_views:
        raise ValueError(
            "joint_shared scale에 필요한 view가 없습니다: "
            f"found={sorted(state_by_name)}"
        )

    estimates: dict[str, Any] = {}

    with FoundationPoseMeshRenderer(
        foundationpose_repository_path=(
            config.foundationpose_repository
        ),
        device=config.device,
        render_batch_size=config.renderer_batch_size,
        maximum_texture_size=(
            config.renderer_maximum_texture_size
        ),
    ) as renderer:
        for view_name in ("reference", "query"):
            state = state_by_name[view_name]

            estimate_root = (
                output_root
                / "visible_scale_refinement"
                / view_name
                / "estimate"
            )

            estimate = estimate_visible_scale_from_self_alignment(
                prepared_view=(
                    state.generated.prepared_view
                ),
                self_alignment=state.self_alignment,
                renderer=renderer,
                output_directory=estimate_root,
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

            if (
                not estimate.valid
                or estimate.absolute_scale_m
                is None
            ):
                print(
                    "[Joint shared scale fallback] "
                    f"{view_name}: "
                    f"{estimate.rejection_reason}; "
                    "keeping coarse states"
                )
                return states

            estimates[view_name] = estimate

            print(
                "[Joint shared scale estimate] "
                f"{view_name}: "
                f"s0={estimate.initial_scale_m:.6f} m, "
                f"s_hat={estimate.absolute_scale_m:.6f} m, "
                f"correspondences="
                f"{estimate.correspondence_count}, "
                f"coverage={estimate.spatial_coverage:.3f}"
            )

    reference_center = float(
        estimates["reference"].absolute_scale_m
    )
    query_center = float(
        estimates["query"].absolute_scale_m
    )

    geometric_center = math.sqrt(
        reference_center * query_center
    )

    raw_scale_bank = [
        reference_center * multiplier
        for multiplier in config.visible_scale_local_multipliers
    ]

    raw_scale_bank.append(geometric_center)

    raw_scale_bank.extend(
        query_center * multiplier
        for multiplier in config.visible_scale_local_multipliers
    )

    shared_scales: list[float] = []

    for scale_m in sorted(raw_scale_bank):
        value = float(scale_m)

        if (
            not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                f"잘못된 joint scale candidate: {value}"
            )

        if not shared_scales or not math.isclose(
            value,
            shared_scales[-1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            shared_scales.append(value)

    if not shared_scales:
        raise RuntimeError(
            "Joint shared scale bank가 비어 있습니다."
        )

    print(
        "[Joint shared scale candidate bank] "
        + ", ".join(
            f"{value:.6f}"
            for value in shared_scales
        )
    )

    candidates_by_name: dict[
        str,
        tuple[Any, ...],
    ] = {}

    for view_name in ("reference", "query"):
        state = state_by_name[view_name]

        candidates_by_name[view_name] = (
            build_absolute_scale_candidates(
                normalization_result=(
                    state.generated.normalization_result
                ),
                original_scale_result=(
                    state.generated.scale_result
                ),
                absolute_scales_m=shared_scales,
                output_directory=(
                    output_root
                    / "visible_scale_refinement"
                    / "joint_shared"
                    / view_name
                    / "scaled_candidates"
                ),
            )
        )

    jobs = tuple(
        FoundationPoseProcessJob(
            job_name=(
                f"joint_shared_scale:{view_name}:{candidate.candidate_index:02d}"
            ),
            candidate=candidate,
            prepared_view=(
                state_by_name[
                    view_name
                ].generated.prepared_view
            ),
        )
        for view_name in ("reference", "query")
        for candidate in candidates_by_name[
            view_name
        ]
    )

    all_results = run_foundationpose_jobs(
        jobs=jobs,
        repository_path=(
            config.foundationpose_repository
        ),
        output_root=(
            output_root
            / "foundationpose"
            / "self_joint_shared_scale"
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

    results_by_name: dict[
        str,
        tuple[Any, ...],
    ] = {}

    cursor = 0

    for view_name in ("reference", "query"):
        candidate_count = len(
            candidates_by_name[view_name]
        )
        next_cursor = cursor + candidate_count

        results_by_name[view_name] = tuple(
            all_results[cursor:next_cursor]
        )

        cursor = next_cursor

    if cursor != len(all_results):
        raise RuntimeError(
            "Joint shared scale FoundationPose 결과 partition에 실패했습니다."
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

    evaluations_by_name: dict[str, Any] = {}
    best_by_name: dict[
        str,
        dict[int, Any],
    ] = {}

    with FoundationPoseMeshRenderer(
        foundationpose_repository_path=(
            config.foundationpose_repository
        ),
        device=config.device,
        render_batch_size=config.renderer_batch_size,
        maximum_texture_size=(
            config.renderer_maximum_texture_size
        ),
    ) as renderer:
        for view_name in ("reference", "query"):
            state = state_by_name[view_name]

            evaluation = evaluate_foundationpose_alignments(
                prepared_view=(
                    state.generated.prepared_view
                ),
                candidate_results=(
                    results_by_name[view_name]
                ),
                renderer=renderer,
                output_directory=(
                    output_root
                    / "self_evaluation_joint_shared_scale"
                    / view_name
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

            evaluations_by_name[view_name] = (
                evaluation
            )

            candidate_best: dict[int, Any] = {}

            # evaluations는 전체 loss 오름차순이므로,
            # candidate별 첫 항목이 그 candidate의 best hypothesis다.
            for item in evaluation.evaluations:
                candidate_index = item.candidate_result.candidate_index

                if (
                    candidate_index
                    not in candidate_best
                ):
                    candidate_best[
                        candidate_index
                    ] = item

            expected_indices = set(
                range(len(shared_scales))
            )

            if (
                set(candidate_best)
                != expected_indices
            ):
                raise RuntimeError(
                    "Joint candidate 평가 index가 "
                    f"일치하지 않습니다: "
                    f"view={view_name}, "
                    f"found={sorted(candidate_best)}, "
                    f"expected={sorted(expected_indices)}"
                )

            best_by_name[view_name] = (
                candidate_best
            )

    minimum_loss_by_name: dict[str, float] = {}

    for view_name in ("reference", "query"):
        losses = [
            float(
                best_by_name[view_name][
                    candidate_index
                ].alignment_score.total_loss
            )
            for candidate_index in range(
                len(shared_scales)
            )
        ]

        if any(
            not math.isfinite(loss) or loss < 0.0
            for loss in losses
        ):
            raise RuntimeError(
                f"유효하지 않은 alignment loss: {view_name}"
            )

        minimum_loss_by_name[view_name] = min(
            losses
        )

    candidate_records: list[dict[str, Any]] = []

    for candidate_index, scale_m in enumerate(
        shared_scales
    ):
        reference_loss = float(
            best_by_name["reference"][
                candidate_index
            ].alignment_score.total_loss
        )

        query_loss = float(
            best_by_name["query"][
                candidate_index
            ].alignment_score.total_loss
        )

        reference_denominator = max(
            minimum_loss_by_name["reference"],
            1e-12,
        )

        query_denominator = max(
            minimum_loss_by_name["query"],
            1e-12,
        )

        reference_normalized = (
            reference_loss / reference_denominator
        )

        query_normalized = (
            query_loss / query_denominator
        )

        joint_worst = max(
            reference_normalized,
            query_normalized,
        )

        joint_mean = (
            reference_normalized
            + query_normalized
        ) / 2.0

        candidate_records.append(
            {
                "candidate_index": candidate_index,
                "scale_m": scale_m,
                "reference_loss": reference_loss,
                "query_loss": query_loss,
                "reference_normalized_loss": (
                    reference_normalized
                ),
                "query_normalized_loss": (
                    query_normalized
                ),
                "joint_worst_normalized_loss": (
                    joint_worst
                ),
                "joint_mean_normalized_loss": (
                    joint_mean
                ),
                "log_distance_from_geometric_mean": (
                    abs(
                        math.log(
                            scale_m
                            / geometric_center
                        )
                    )
                ),
                "reference_hypothesis_rank": (
                    best_by_name["reference"][
                        candidate_index
                    ].hypothesis.rank
                ),
                "query_hypothesis_rank": (
                    best_by_name["query"][
                        candidate_index
                    ].hypothesis.rank
                ),
            }
        )

    selected_record = min(
        candidate_records,
        key=lambda record: (
            record["joint_worst_normalized_loss"],
            record["joint_mean_normalized_loss"],
            record[
                "log_distance_from_geometric_mean"
            ],
            record["candidate_index"],
        ),
    )

    selected_candidate_index = int(
        selected_record["candidate_index"]
    )

    selected_scale_m = float(
        selected_record["scale_m"]
    )

    selection_root = (
        output_root
        / "visible_scale_refinement"
        / "joint_shared"
    )

    selection_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_path = (
        selection_root / "selection.json"
    )

    selection_payload = {
        "policy": "joint_shared_minimax",
        "forced_diagnostic_selection": True,
        "reference_visible_estimate_m": (
            reference_center
        ),
        "query_visible_estimate_m": (
            query_center
        ),
        "geometric_mean_m": (geometric_center),
        "scale_candidates_m": (shared_scales),
        "selection_order": [
            "minimum joint_worst_normalized_loss",
            "minimum joint_mean_normalized_loss",
            "minimum log distance from geometric mean",
            "minimum candidate index",
        ],
        "selected_candidate_index": (
            selected_candidate_index
        ),
        "selected_scale_m": (selected_scale_m),
        "selected_record": (selected_record),
        "candidates": candidate_records,
    }

    with selection_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selection_payload,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    refined_states = list(states)

    for view_name in ("reference", "query"):
        state = state_by_name[view_name]

        selected_evaluation = best_by_name[
            view_name
        ][selected_candidate_index]

        selected_alignment = select_self_alignment(
            result=(
                selected_evaluation.candidate_result
            ),
            hypothesis_rank=(
                selected_evaluation.hypothesis.rank
            ),
            alignment_loss=(
                selected_evaluation.alignment_score.total_loss
            ),
        )

        selected_candidate = _candidate_by_index(
            candidates_by_name[view_name],
            selected_candidate_index,
        )

        refined_states[
            state_index_by_name[view_name]
        ] = AlignedProxyState(
            generated=state.generated,
            self_results=(
                results_by_name[view_name]
            ),
            self_evaluation=(
                evaluations_by_name[view_name]
            ),
            self_alignment=selected_alignment,
            selected_candidate=selected_candidate,
        )

        view_selection_path = (
            output_root
            / "visible_scale_refinement"
            / view_name
            / "selection.json"
        )

        view_selection_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with view_selection_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "accepted": True,
                    "selected_source": (
                        "joint_shared_scale"
                    ),
                    "policy": (
                        "joint_shared_minimax"
                    ),
                    "selected_candidate_index": (
                        selected_candidate_index
                    ),
                    "selected_scale_m": (
                        selected_scale_m
                    ),
                    "alignment_loss": float(
                        selected_evaluation.alignment_score.total_loss
                    ),
                    "joint_selection_path": str(
                        selection_path
                    ),
                },
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")

    print(
        "[Joint shared scale selected] "
        f"candidate={selected_candidate_index}, "
        f"scale={selected_scale_m:.6f} m, "
        f"reference_loss="
        f"{selected_record['reference_loss']:.6f}, "
        f"query_loss="
        f"{selected_record['query_loss']:.6f}, "
        f"worst_normalized="
        f"{selected_record['joint_worst_normalized_loss']:.6f}, "
        f"mean_normalized="
        f"{selected_record['joint_mean_normalized_loss']:.6f}"
    )

    return tuple(refined_states)
