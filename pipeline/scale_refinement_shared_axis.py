from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from core.types import AlignedProxyState, PipelineConfig


def _shared_axis_scale_gt_report(
    *,
    config: PipelineConfig,
    reference_frame: Any,
    query_frame: Any,
    reference_pose_camera_from_proxy: np.ndarray,
    query_pose_camera_from_proxy: np.ndarray,
    g0_proxy_pose_query_from_reference: np.ndarray,
) -> dict[str, Any]:
    """Diagnostic-only GT comparison for H0 = B0 @ G0 @ inv(A0).

    Never gates the pipeline -- GT is unavailable at real deployment time.
    Logs, for research purposes, what the relative pose would have been
    using the pre shared-D-correction self-poses (A0/B0) together with
    the G0 dGeDi result, as a baseline against the actual final pose that
    _run_self_mesh_final_dgedi later produces from the corrected A1/B1.
    """
    if reference_frame is None or query_frame is None:
        return {"available": False, "reason": "No reference/query frame given."}
    try:
        import numpy as np

        from evaluation.relative_pose_evaluator import (
            BOPFrameGTSpec,
            build_ground_truth_relative_pose,
            load_bop_linemod_absolute_gt_pose,
        )

        reference_gt = load_bop_linemod_absolute_gt_pose(
            BOPFrameGTSpec(
                dataset_root=config.dataset_root,
                split=config.split,
                scene_id=reference_frame.scene_id,
                image_id=reference_frame.image_id,
                object_id=config.object_id,
                instance_index=reference_frame.instance_index,
            )
        )
        query_gt = load_bop_linemod_absolute_gt_pose(
            BOPFrameGTSpec(
                dataset_root=config.dataset_root,
                split=config.split,
                scene_id=query_frame.scene_id,
                image_id=query_frame.image_id,
                object_id=config.object_id,
                instance_index=query_frame.instance_index,
            )
        )
        ground_truth = build_ground_truth_relative_pose(
            reference_absolute_pose=reference_gt,
            query_absolute_pose=query_gt,
        )

        def _error(h: np.ndarray) -> dict[str, Any]:
            delta = h @ np.linalg.inv(ground_truth)
            cosine = float(
                np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            )
            # Translation error is measured as object-centre displacement,
            # not from delta[:3, 3]: a relative pose is anchored at the
            # reference camera origin, which sits far from the object, so
            # any rotation error would leak into the translation number.
            predicted_query_absolute = np.asarray(h, dtype=np.float64) @ (
                np.asarray(reference_gt, dtype=np.float64)
            )
            translation_error_m = float(
                np.linalg.norm(
                    predicted_query_absolute[:3, 3]
                    - np.asarray(query_gt, dtype=np.float64)[:3, 3]
                )
            )
            return {
                "rotation_error_deg": math.degrees(math.acos(cosine)),
                "translation_error_cm": translation_error_m * 100.0,
                "translation_error_definition": (
                    "object-centre displacement of "
                    "H @ T_reference_from_object_gt versus "
                    "T_query_from_object_gt"
                ),
            }

        h0 = query_pose_camera_from_proxy @ (
            g0_proxy_pose_query_from_reference
            @ np.linalg.inv(reference_pose_camera_from_proxy)
        )
        return {
            "available": True,
            "note": (
                "H0 = B0 @ G0 @ inv(A0), using the pre shared-D-correction "
                "self-poses A0/B0. Diagnostic baseline only; not used by "
                "the pipeline and not compared against the actual final "
                "pose (which uses corrected A1/B1 and a separately-run "
                "G, computed once downstream by "
                "_run_self_mesh_final_dgedi)."
            ),
            "h0_vs_gt": _error(h0),
        }
    except Exception as error:
        return {
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
        }


def _refine_aligned_states_shared_axis_scale(
    *,
    config: PipelineConfig,
    aligned_states: Sequence[AlignedProxyState],
    output_root: Path,
    reference_frame: Any = None,
    query_frame: Any = None,
) -> tuple[AlignedProxyState, ...]:
    """Search one D shared by both proxies via dGeDi's own R0 (Cr=D,
    Cq=R0 D R0^T). This function runs dGeDi exactly once (G0), solely to
    derive R0 for the D* search, and returns the corrected
    AlignedProxyStates (self_alignment=A1/B1 after re-fitting
    FoundationPose on the D*-corrected proxies). The final registration
    (G) is computed exactly once, downstream, by the unmodified
    _run_self_mesh_final_dgedi -- this function never computes a second
    dGeDi pass and never gates on one.

    A genuine G0 failure raises RuntimeError. A D* search that declines
    to move (search.applied is False, i.e. D*=identity) is a legitimate
    outcome, not a failure: fit_shared_axis_scale already returns the
    original, unmodified source candidates in that case, so the pipeline
    just continues with those.
    """
    import numpy as np

    from pose.alignment_evaluator import (
        evaluate_foundationpose_alignments,
        select_best_self_alignment,
    )
    from pose.alignment_scorer import AlignmentScoreWeights
    from pose.dgedi_runner import run_dgedi_registration
    from pose.foundationpose_process_pool import (
        FoundationPoseProcessJob,
        run_foundationpose_jobs,
    )
    from pose.mesh_renderer import FoundationPoseMeshRenderer
    from scale.shared_axis_scale_refiner import fit_shared_axis_scale

    states = tuple(aligned_states)
    if len(states) != 2:
        raise ValueError(
            "Shared axis-scale fitting requires Reference and Query: "
            f"count={len(states)}"
        )
    state_by_name = {
        state.generated.view_name: state for state in states
    }
    if set(state_by_name) != {"reference", "query"}:
        raise ValueError(
            "Shared axis-scale fitting requires one reference and one query"
        )
    reference_state = state_by_name["reference"]
    query_state = state_by_name["query"]

    print(
        "==== [STAGE] Shared D* refinement: A0/B0-based dGeDi G0 defines "
        "R0; Cr=D / Cq=R0 D R0^T jointly fit both proxies. The corrected "
        "A1/B1 self-poses are returned for exactly one downstream final "
        "dGeDi registration (no G1 here, no gate) ===="
    )

    dgedi_root = output_root / "mesh_registration" / "dgedi"

    def _run_dgedi(
        *,
        reference_alignment: Any,
        query_alignment: Any,
        stage_name: str,
    ) -> tuple[Any, dict[str, Any]]:
        result = run_dgedi_registration(
            repository_path=config.dgedi_repository,
            python_executable=config.dgedi_python,
            config_path=config.dgedi_config,
            reference_self_alignment=reference_alignment,
            query_self_alignment=query_alignment,
            reference_camera_matrix=(
                reference_state.generated.prepared_view.view.camera_matrix
            ),
            query_camera_matrix=(
                query_state.generated.prepared_view.view.camera_matrix
            ),
            reference_mask_bool=(
                reference_state.generated.prepared_view.segmentation.mask_bool
            ),
            query_mask_bool=(
                query_state.generated.prepared_view.segmentation.mask_bool
            ),
            reference_depth_m=(
                reference_state.generated.prepared_view.view.depth_m
            ),
            query_depth_m=(
                query_state.generated.prepared_view.view.depth_m
            ),
            output_directory=dgedi_root / stage_name,
            mode=config.dgedi_mode,
            device=config.dgedi_device,
            sample_count=config.dgedi_sample_count,
            ransac_threshold=config.dgedi_ransac_threshold,
            icp_threshold=config.dgedi_icp_threshold,
            maximum_surface_depth_residual_m=(
                config.dgedi_maximum_surface_depth_residual_m
            ),
            minimum_visible_depth_pixels=(
                config.dgedi_minimum_visible_depth_pixels
            ),
            minimum_pair_point_count_ratio=(
                config.dgedi_minimum_pair_point_count_ratio
            ),
            minimum_pair_diameter_ratio=(
                config.dgedi_minimum_pair_diameter_ratio
            ),
            registration_candidate_count=(
                config.dgedi_registration_candidate_count
            ),
        )
        metadata = json.loads(
            Path(result.metadata_path).read_text(encoding="utf-8")
        )
        return result, metadata

    try:
        g0_result, g0_metadata = _run_dgedi(
            reference_alignment=reference_state.self_alignment,
            query_alignment=query_state.self_alignment,
            stage_name="g0_early",
        )
    except Exception as error:
        raise RuntimeError(
            f"Shared-D refinement failed: G0 (A0/B0-based dGeDi) failed: "
            f"{type(error).__name__}: {error}"
        ) from error

    rotation_query_from_reference = np.ascontiguousarray(
        np.asarray(
            g0_result.proxy_pose_query_from_reference, dtype=np.float64
        )[:3, :3]
    )
    g0_icp_fitness = float(g0_metadata["icp"]["fitness"])
    g0_icp_rmse = float(g0_metadata["icp"]["inlier_rmse_m"])

    weights = AlignmentScoreWeights(
        mask=config.alignment_weight_mask,
        depth=config.alignment_weight_depth,
        free_space=config.alignment_weight_free_space,
        boundary=config.alignment_weight_boundary,
    )
    with FoundationPoseMeshRenderer(
        foundationpose_repository_path=config.foundationpose_repository,
        device=config.device,
        render_batch_size=config.renderer_batch_size,
        maximum_texture_size=config.renderer_maximum_texture_size,
    ) as renderer:
        search = fit_shared_axis_scale(
            reference_source_candidate=reference_state.selected_candidate,
            reference_prepared_view=reference_state.generated.prepared_view,
            reference_fixed_pose_camera_from_proxy=(
                reference_state.self_alignment.pose_camera_from_proxy
            ),
            query_source_candidate=query_state.selected_candidate,
            query_prepared_view=query_state.generated.prepared_view,
            query_fixed_pose_camera_from_proxy=(
                query_state.self_alignment.pose_camera_from_proxy
            ),
            rotation_query_from_reference=rotation_query_from_reference,
            renderer=renderer,
            output_directory=(
                output_root / "shared_axis_scale_refinement"
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
            minimum_loss_improvement_ratio=(
                config.axis_scale_minimum_loss_improvement_ratio
            ),
        )
    print(
        "[Shared axis-scale search] "
        f"factors={search.selected_scale_factors} applied={search.applied} "
        f"joint_loss={search.baseline_joint_loss:.6f}->"
        f"{search.selected_joint_loss:.6f}"
    )
    # search.applied is False (D*=identity) is a legitimate outcome, not a
    # failure -- fit_shared_axis_scale already returns the original,
    # unmodified source candidates below in that case, so we just continue.

    jobs = (
        FoundationPoseProcessJob(
            job_name="shared_axis_scale:reference",
            candidate=search.reference_selected_candidate,
            prepared_view=reference_state.generated.prepared_view,
        ),
        FoundationPoseProcessJob(
            job_name="shared_axis_scale:query",
            candidate=search.query_selected_candidate,
            prepared_view=query_state.generated.prepared_view,
        ),
    )
    all_results = run_foundationpose_jobs(
        jobs=jobs,
        repository_path=config.foundationpose_repository,
        output_root=(
            output_root / "foundationpose" / "shared_axis_scale"
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
    if len(all_results) != 2:
        raise RuntimeError(
            "Shared axis-scale FoundationPose result count mismatch"
        )

    refined_alignment_by_name: dict[str, Any] = {}
    refined_evaluation_by_name: dict[str, Any] = {}
    with FoundationPoseMeshRenderer(
        foundationpose_repository_path=config.foundationpose_repository,
        device=config.device,
        render_batch_size=config.renderer_batch_size,
        maximum_texture_size=config.renderer_maximum_texture_size,
    ) as renderer:
        for view_name, result in zip(
            ("reference", "query"), all_results, strict=True
        ):
            evaluation = evaluate_foundationpose_alignments(
                prepared_view=(
                    state_by_name[view_name].generated.prepared_view
                ),
                candidate_results=(result,),
                renderer=renderer,
                output_directory=(
                    output_root
                    / "self_evaluation_shared_axis_scale"
                    / view_name
                ),
                weights=weights,
                depth_trim_quantile=config.alignment_depth_trim_quantile,
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
            refined_evaluation_by_name[view_name] = evaluation
            refined_alignment_by_name[view_name] = (
                select_best_self_alignment(evaluation)
            )

    refined_by_name = {
        "reference": AlignedProxyState(
            generated=reference_state.generated,
            self_results=(all_results[0],),
            self_evaluation=refined_evaluation_by_name["reference"],
            self_alignment=refined_alignment_by_name["reference"],
            selected_candidate=search.reference_selected_candidate,
        ),
        "query": AlignedProxyState(
            generated=query_state.generated,
            self_results=(all_results[1],),
            self_evaluation=refined_evaluation_by_name["query"],
            self_alignment=refined_alignment_by_name["query"],
            selected_candidate=search.query_selected_candidate,
        ),
    }

    gt_report = _shared_axis_scale_gt_report(
        config=config,
        reference_frame=reference_frame,
        query_frame=query_frame,
        reference_pose_camera_from_proxy=np.asarray(
            reference_state.self_alignment.pose_camera_from_proxy,
            dtype=np.float64,
        ),
        query_pose_camera_from_proxy=np.asarray(
            query_state.self_alignment.pose_camera_from_proxy,
            dtype=np.float64,
        ),
        g0_proxy_pose_query_from_reference=(
            g0_result.proxy_pose_query_from_reference
        ),
    )

    summary_directory = output_root / "shared_axis_scale_refinement"
    summary_directory.mkdir(parents=True, exist_ok=True)
    summary_path = summary_directory / "g0_diagnostic_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "note": (
                    "G0 is diagnostic-only here: it derives R0 for the D* "
                    "search. The final registration (G) is computed "
                    "exactly once, downstream, by "
                    "_run_self_mesh_final_dgedi; its own metadata is "
                    "persisted separately as dgedi_registration.json "
                    "under its own output_directory."
                ),
                "g0_icp_fitness": g0_icp_fitness,
                "g0_icp_inlier_rmse_m": g0_icp_rmse,
                "rotation_query_from_reference_g0": (
                    rotation_query_from_reference.tolist()
                ),
                "shared_d_search_applied": search.applied,
                "selected_scale_factors_reference_uvd": list(
                    search.selected_scale_factors
                ),
                "baseline_joint_loss": search.baseline_joint_loss,
                "selected_joint_loss": search.selected_joint_loss,
                "ground_truth_comparison": gt_report,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "[Shared axis-scale G0 diagnostic] "
        f"icp_fitness={g0_icp_fitness:.6f} icp_rmse={g0_icp_rmse:.6f} "
        f"summary={summary_path}"
    )

    return tuple(
        refined_by_name[state.generated.view_name] for state in states
    )


