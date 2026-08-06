from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from core.types import AlignedProxyState, PairPipelineOutcome, PipelineConfig


def _load_compact_axis_scale_selection(
    output_root: Path,
) -> dict[str, Any] | None:
    """Load compact per-view camera-axis Su/Sv/Sd decisions.

    Supports both independent and shared axis scale refinement methods.
    """
    # Try independent axis scale refinement first (legacy)
    independent_fields = (
        "accepted",
        "selected_source",
        "requested_scale_factors_xyz",
        "applied_scale_factors_xyz",
        "scale_factors_xyz",
        "requested_scale_factors_uvd",
        "applied_scale_factors_uvd",
        "depth_observability",
        "axis_penalty_weights_uvd",
        "source_object_scale_m",
        "pair_shared_scale_verified",
        "source_dimensions_m",
        "requested_dimensions_m",
        "applied_dimensions_m",
        "selected_dimensions_m",
        "fixed_pose_baseline_loss",
        "fixed_pose_selected_loss",
        "final_foundationpose_loss",
    )
    views: dict[str, Any] = {}

    # Try independent axis scale refinement
    for view_name in ("reference", "query"):
        selection_path = (
            Path(output_root)
            / "axis_scale_refinement"
            / view_name
            / "selection.json"
        )
        if selection_path.is_file():
            try:
                payload = json.loads(
                    selection_path.read_text(encoding="utf-8")
                )
                if isinstance(payload, dict):
                    views[view_name] = {
                        key: payload[key]
                        for key in independent_fields
                        if key in payload
                    }
            except (OSError, json.JSONDecodeError):
                pass

    if views:
        return {
            "method": "independent_camera_axis_scale",
            "shared_axes": False,
            "shared_dimensions": False,
            "axis_convention": (
                "u=image horizontal, v=image vertical, d=camera depth"
            ),
            "views": views,
        }

    # Try shared axis scale refinement
    views = {}
    for view_name in ("reference", "query"):
        candidate_path = (
            Path(output_root)
            / "shared_axis_scale_refinement"
            / view_name
            / "selected_candidate"
            / "shared_axis_scale_candidate.json"
        )
        if not candidate_path.is_file():
            continue
        try:
            payload = json.loads(
                candidate_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue

        # Extract relevant fields from shared axis scale format
        view_data: dict[str, Any] = {
            "method": payload.get("method", "shared_dgedi_axis_scale"),
            "source_candidate_index": payload.get("source_candidate_index"),
            "source_object_scale_m": payload.get("source_object_scale_m"),
            "scale_factors_uvd": payload.get("scale_factors_reference_uvd"),
            "source_dimensions_m": payload.get("source_dimensions_m"),
            "scaled_mesh_path": payload.get("scaled_mesh_path"),
        }
        views[view_name] = view_data

    if not views:
        return None

    # Load shared rotation if available
    search_path = (
        Path(output_root)
        / "shared_axis_scale_refinement"
        / "shared_axis_scale_search.json"
    )
    rotation_query_from_reference = None
    if search_path.is_file():
        try:
            search_data = json.loads(search_path.read_text(encoding="utf-8"))
            if isinstance(search_data, dict):
                rotation_query_from_reference = search_data.get(
                    "rotation_query_from_reference"
                )
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "method": "shared_dgedi_axis_scale",
        "shared_axes": True,
        "shared_dimensions": False,
        "axis_convention": (
            "u=image horizontal, v=image vertical, d=camera depth"
        ),
        "rotation_query_from_reference": rotation_query_from_reference,
        "views": views,
    }


def _run_self_mesh_final_dgedi(
    *,
    config: PipelineConfig,
    reference_state: AlignedProxyState,
    query_state: AlignedProxyState,
    output_root: Path,
    timing_values: dict[str, Any],
) -> PairPipelineOutcome:
    """Run the single final visible/depth-consistent dGeDi registration."""
    import numpy as np

    from pose.alignment_scorer import AlignmentScoreWeights
    from pose.dgedi_observation_validator import (
        validate_dgedi_against_observations,
    )
    from pose.dgedi_runner import run_dgedi_registration
    from pose.independent_pose_paths import save_independent_pose_path

    reference_view = reference_state.generated.prepared_view
    query_view = query_state.generated.prepared_view
    dgedi_started_at = time.perf_counter()
    print(
        "==== [STAGE] Final visible-depth proxy-surface dGeDi G: "
        "one registration after A1/B1; no G0 and no shared D ===="
    )
    try:
        dgedi_result = run_dgedi_registration(
            repository_path=config.dgedi_repository,
            python_executable=config.dgedi_python,
            config_path=config.dgedi_config,
            reference_self_alignment=reference_state.self_alignment,
            query_self_alignment=query_state.self_alignment,
            reference_camera_matrix=reference_view.view.camera_matrix,
            query_camera_matrix=query_view.view.camera_matrix,
            reference_mask_bool=reference_view.segmentation.mask_bool,
            query_mask_bool=query_view.segmentation.mask_bool,
            reference_depth_m=reference_view.view.depth_m,
            query_depth_m=query_view.view.depth_m,
            output_directory=(
                output_root / "mesh_registration" / "dgedi" / "final"
            ),
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
    except Exception as error:
        timing_values["dgedi_registration_time_sec"] = (
            time.perf_counter() - dgedi_started_at
        )
        failure_root = output_root / "method_results" / "self_mesh"
        failure_root.mkdir(parents=True, exist_ok=True)
        failure_path = failure_root / "dgedi_attempt_failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "status": "DGEDI_EXECUTION_FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "reason": (
                        "The final visible-depth proxy-surface registration "
                        "did not produce a pose. No fallback was substituted."
                    ),
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        summary_path = failure_root / "final_selection.json"
        summary_path.write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "method": "self_mesh",
                    "pose_accepted": False,
                    "relative_pose_query_from_reference": None,
                    "reason": (
                        "Final dGeDi execution failed; no relative pose exists."
                    ),
                    "sources": {
                        "mesh_registration_backend": (
                            "single_final_visible_depth_dual_proxy_dgedi"
                        ),
                        "dgedi_attempt_failure_path": str(failure_path),
                        "dgedi_registration_time_sec": timing_values[
                            "dgedi_registration_time_sec"
                        ],
                    },
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[Final dGeDi execution failed] {error}")
        print(f"[Final summary] {summary_path}")
        return PairPipelineOutcome(
            final_status="FAILED",
            summary_path=summary_path,
            pose_path=None,
            visualization_path=None,
            pose_produced=False,
            quality_gate_mode="not_run",
            quality_gate_passed=None,
        )

    timing_values["dgedi_registration_time_sec"] = (
        time.perf_counter() - dgedi_started_at
    )
    validation_root = (
        output_root
        / "method_results"
        / "self_mesh"
        / "observation_validation"
        / "final"
    )
    validation = validate_dgedi_against_observations(
        relative_pose_query_from_reference=(
            dgedi_result.relative_pose_query_from_reference
        ),
        reference_mesh_path=dgedi_result.reference_self_aligned_mesh_path,
        query_mesh_path=dgedi_result.query_self_aligned_mesh_path,
        reference_camera_k=np.asarray(
            reference_view.view.camera_matrix, dtype=np.float64
        ),
        query_camera_k=np.asarray(
            query_view.view.camera_matrix, dtype=np.float64
        ),
        reference_mask_bool=np.asarray(
            reference_view.segmentation.mask_bool, dtype=bool
        ),
        query_mask_bool=np.asarray(
            query_view.segmentation.mask_bool, dtype=bool
        ),
        reference_depth_m=np.asarray(
            reference_view.view.depth_m, dtype=np.float32
        ),
        query_depth_m=np.asarray(
            query_view.view.depth_m, dtype=np.float32
        ),
        object_scale_m=float(
            math.sqrt(
                reference_state.selected_candidate.scale_m
                * query_state.selected_candidate.scale_m
            )
        ),
        weights=AlignmentScoreWeights(
            mask=config.alignment_weight_mask,
            depth=config.alignment_weight_depth,
            free_space=config.alignment_weight_free_space,
            boundary=config.alignment_weight_boundary,
        ),
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
        minimum_mask_iou=config.dgedi_validation_minimum_mask_iou,
        baseline_mask_iou_drop=(
            config.dgedi_validation_baseline_mask_iou_drop
        ),
        maximum_depth_residual_normalized=(
            config.dgedi_validation_maximum_depth_residual_normalized
        ),
        baseline_depth_residual_margin=(
            config.dgedi_validation_baseline_depth_residual_margin
        ),
        maximum_total_loss=config.dgedi_validation_maximum_total_loss,
        baseline_total_loss_margin=(
            config.dgedi_validation_baseline_total_loss_margin
        ),
        output_directory=validation_root,
        diagnostic_only=True,
    )

    visualization_path: Path | None = None
    try:
        from evaluation.mesh_on_photo_visualizer import render_mesh_on_photo
        from evaluation.relative_pose_evaluator import (
            BOPFrameGTSpec,
            load_bop_linemod_absolute_gt_pose,
        )
        import json

        # Load reference GT pose
        reference_gt_pose = None
        try:
            reference_frame_spec = BOPFrameGTSpec(
                dataset_root=config.dataset_root,
                split=config.split,
                scene_id=config.reference.scene_id,
                image_id=config.reference.image_id,
                object_id=config.object_id,
                instance_index=config.reference.instance_index,
            )
            reference_gt_pose = load_bop_linemod_absolute_gt_pose(reference_frame_spec)
            print(f"[GT pose loaded] reference object {config.object_id}")
            print(f"[DEBUG] GT pose shape: {reference_gt_pose.shape}")
        except Exception as gt_error:
            # GT not available, fallback to predicted reference
            print(f"[GT loading failed] {type(gt_error).__name__}: {gt_error}")
            import traceback
            traceback.print_exc()

        # Get reference A1 from metadata
        reference_self_pose_A1 = None
        try:
            metadata_path = dgedi_result.metadata_path
            print(f"[DEBUG] dgedi_result.metadata_path = {metadata_path}")
            if metadata_path is None:
                raise ValueError("metadata_path is None")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            reference_self_pose_A1 = np.array(metadata['reference_pose_camera_from_proxy'], dtype=np.float64)
            print(f"[A1 loaded] from {metadata_path}")
            print(f"[DEBUG] A1 shape: {reference_self_pose_A1.shape}")
        except Exception as a1_error:
            print(f"[A1 loading failed] {type(a1_error).__name__}: {a1_error}")
            import traceback
            traceback.print_exc()

        visualization_filename = f"mesh_on_photo_obj{config.object_id}.png"
        visualization_path = render_mesh_on_photo(
            reference_mesh_path=(
                dgedi_result.reference_self_aligned_mesh_path
            ),
            query_mesh_path=dgedi_result.query_self_aligned_mesh_path,
            reference_camera_k=np.asarray(
                reference_view.view.camera_matrix, dtype=np.float64
            ),
            query_camera_k=np.asarray(
                query_view.view.camera_matrix, dtype=np.float64
            ),
            reference_rgb=np.asarray(reference_view.view.rgb),
            query_rgb=np.asarray(query_view.view.rgb),
            reference_mask_bool=np.asarray(
                reference_view.segmentation.mask_bool, dtype=bool
            ),
            query_mask_bool=np.asarray(
                query_view.segmentation.mask_bool, dtype=bool
            ),
            output_path=(
                output_root / "visualizations" / visualization_filename
            ),
            title=(
                f"Object {config.object_id}: Reference (GT) vs Query (Predicted)"
            ),
            reference_gt_pose=reference_gt_pose,
            reference_self_pose_A1=reference_self_pose_A1,
        )
        print(f"[mesh-on-photo visualization] {visualization_path}")
    except Exception as error:
        print(
            "[mesh-on-photo visualization failed; continuing] "
            f"{type(error).__name__}: {error}"
        )

    axis_scale_source = _load_compact_axis_scale_selection(output_root)
    sources: dict[str, Any] = {
        "mesh_registration_backend": (
            "single_final_visible_depth_dual_proxy_dgedi"
        ),
        "coordinate_frame_contract": (
            "A1=T_Cr_from_Pr, B1=T_Cq_from_Pq, "
            "G=T_Pq_from_Pr, H=B1@G@inv(A1)"
        ),
        "dgedi_proxy_pose_path": str(dgedi_result.proxy_pose_path),
        "dgedi_relative_pose_path": str(dgedi_result.relative_pose_path),
        "dgedi_metadata_path": str(dgedi_result.metadata_path),
        "dgedi_proxy_pose_query_from_reference": (
            dgedi_result.proxy_pose_query_from_reference.tolist()
        ),
        "dgedi_reference_proxy_surface_cloud_path": str(
            dgedi_result.reference_registration_cloud_path
        ),
        "dgedi_query_proxy_surface_cloud_path": str(
            dgedi_result.query_registration_cloud_path
        ),
        "reference_self_pose_A1": (
            reference_state.self_alignment.pose_camera_from_proxy.tolist()
        ),
        "query_self_pose_B1": (
            query_state.self_alignment.pose_camera_from_proxy.tolist()
        ),
        "dgedi_registration_time_sec": timing_values[
            "dgedi_registration_time_sec"
        ],
        "dgedi_observation_validation": {
            "diagnostic_only": True,
            "legacy_gate_would_accept": validation.accepted,
            "summary_path": str(validation.summary_path),
            "reference_render_path": str(validation.reference_render_path),
            "query_render_path": str(validation.query_render_path),
        },
    }
    if axis_scale_source is not None:
        sources["axis_scale_refinement"] = axis_scale_source

    independent_result = save_independent_pose_path(
        method="self_mesh",
        relative_pose_query_from_reference=(
            dgedi_result.relative_pose_query_from_reference
        ),
        output_directory=output_root / "method_results" / "self_mesh",
        composition=(
            "H = B1 @ G @ inv(A1), where A1/B1 are final FoundationPose "
            "self-alignments after independent camera-axis scale correction "
            "and G is one visible-depth-consistent dual-proxy dGeDi result"
        ),
        selection_policy=(
            "single final dual-proxy estimate; observation validation is "
            "diagnostic-only and no G0/shared-D/fallback pose is used"
        ),
        sources=sources,
    )
    print("[Independent pose path] self_mesh+camera_axis_scale+final_dGeDi")
    print(f"[dGeDi local proxy pose G] {dgedi_result.proxy_pose_path}")
    print(f"[dGeDi relative pose] {dgedi_result.relative_pose_path}")
    print(f"[Final summary] {independent_result.summary_path}")
    print(f"[Final pose] {independent_result.pose_path}")
    print(independent_result.relative_pose_query_from_reference)
    return PairPipelineOutcome(
        final_status="COMPLETED",
        summary_path=independent_result.summary_path,
        pose_path=independent_result.pose_path,
        pose_produced=True,
        visualization_path=visualization_path,
        quality_gate_mode="diagnostic",
        quality_gate_passed=bool(validation.accepted),
    )


def _run_aligned_pair(
    *,
    config: PipelineConfig,
    reference_state: AlignedProxyState,
    query_state: AlignedProxyState,
    reference_dino: Any | None,
    reference_surface: Any | None,
    query_dino: Any | None,
    query_surface: Any | None,
    output_root: Path,
    research_context: Any,
    timings: dict[str, Any] | None = None,
    pipeline_started_at: float | None = None,
) -> PairPipelineOutcome:
    timing_values = dict(timings or {})
    aligned_pair_started_at = time.perf_counter()

    if pipeline_started_at is None:
        pipeline_started_at = (
            aligned_pair_started_at
        )

    for _diag_state in (
        reference_state,
        query_state,
    ):
        _diag_alignment = (
            _diag_state.self_alignment
        )
        try:
            import numpy as np
            import open3d as _diag_o3d
            from pose.dgedi_runner import (
                _diameter as _diag_diameter,
            )

            _diag_mesh = _diag_o3d.io.read_triangle_mesh(
                str(
                    _diag_alignment.scaled_mesh_path
                )
            )
            _diag_diam = _diag_diameter(
                np.asarray(_diag_mesh.vertices)
            )
        except Exception as _diag_error:
            _diag_diam = f"<error: {_diag_error}>"

        print(
            "[_run_aligned_pair received self_alignment] "
            f"view={_diag_alignment.proxy_view} "
            f"candidate_index={_diag_alignment.candidate_index} "
            f"scale_m={_diag_alignment.scale_m} "
            f"scaled_mesh_path={_diag_alignment.scaled_mesh_path} "
            f"actual_mesh_diameter_m={_diag_diam}"
        )

    reference_view = (
        reference_state.generated.prepared_view
    )
    query_view = (
        query_state.generated.prepared_view
    )
    timing_values[
        "dgedi_registration_time_sec"
    ] = 0.0

    if config.pose_path == "self_mesh":
        return _run_self_mesh_final_dgedi(
            config=config,
            reference_state=reference_state,
            query_state=query_state,
            output_root=output_root,
            timing_values=timing_values,
        )
