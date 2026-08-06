from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.types import FrameSpec, PipelineConfig
from pipeline.config_cli import _batch_query_output_label
from pipeline.config_defaults import (
    DEFAULT_INSTANTMESH_CONFIG,
    LINEMOD_OBJECT_METADATA,
    PROJECT_ROOT,
    _default_dinov3_weights_path,
    _default_instantmesh_python,
)


def build_config(
    args: argparse.Namespace,
) -> PipelineConfig:
    metadata = LINEMOD_OBJECT_METADATA.get(
        args.object_id
    )
    default_object_name = (
        metadata[0]
        if metadata is not None
        else f"object_{args.object_id:02d}"
    )
    default_sam3_prompt = (
        metadata[1]
        if metadata is not None
        else default_object_name.replace("_", " ")
    )
    object_name = (
        args.object_name.strip()
        if args.object_name is not None
        else default_object_name
    )

    sam3_prompt = (
        args.sam3_prompt.strip()
        if args.sam3_prompt is not None
        else (
            object_name.replace("_", " ")
            if args.object_name is not None
            else default_sam3_prompt
        )
    )

    reference = FrameSpec(
        scene_id=args.reference_scene_id,
        image_id=args.reference_image_id,
        instance_index=(
            args.reference_instance_index
        ),
    )

    batch_query_image_ids = (
        tuple(args.query_image_ids)
        if args.query_image_ids is not None
        else None
    )

    if batch_query_image_ids is not None and len(
        set(batch_query_image_ids)
    ) != len(batch_query_image_ids):
        raise ValueError(
            "query_image_ids must not contain duplicates."
        )

    if batch_query_image_ids is not None and any(
        image_id < 0
        for image_id in batch_query_image_ids
    ):
        raise ValueError(
            "query_image_ids must be non-negative."
        )

    query_image_id = (
        batch_query_image_ids[0]
        if batch_query_image_ids is not None
        else (
            args.query_image_id
            if args.query_image_id is not None
            else 1
        )
    )

    query = FrameSpec(
        scene_id=args.query_scene_id,
        image_id=query_image_id,
        instance_index=(
            args.query_instance_index
        ),
    )

    instantmesh_repository = args.instantmesh_repository.expanduser().resolve()

    instantmesh_config = (
        args.instantmesh_config
        if args.instantmesh_config is not None
        else DEFAULT_INSTANTMESH_CONFIG
    )

    dinov3_repository = args.dinov3_repository.expanduser().resolve()

    dinov3_checkpoint = (
        args.dinov3_checkpoint
        if args.dinov3_checkpoint is not None
        else _default_dinov3_weights_path(
            dinov3_repository,
            args.dinov3_model,
        )
    )

    if args.output_root is None:
        if batch_query_image_ids is None:
            output_name = (
                f"object_{args.object_id:02d}"
                f"_r{reference.scene_id:06d}_"
                f"{reference.image_id:06d}"
                f"_q{query.scene_id:06d}_"
                f"{query.image_id:06d}"
            )
        else:
            query_label = (
                _batch_query_output_label(
                    batch_query_image_ids
                )
            )
            output_name = (
                f"batch_object_{args.object_id:02d}"
                f"_r{reference.scene_id:06d}_"
                f"{reference.image_id:06d}"
                f"_ri{reference.instance_index:02d}"
                f"_qs{query.scene_id:06d}"
                f"_qi{query.instance_index:02d}"
                f"_q{query_label}"
            )

        if args.pose_path != "combined":
            output_name = (
                f"{output_name}_{args.pose_path}"
            )

        output_root = (
            PROJECT_ROOT / "outputs" / output_name
        )
    else:
        output_root = args.output_root

    instantmesh_python = (
        args.instantmesh_python
        if args.instantmesh_python is not None
        else _default_instantmesh_python()
    )
    sam3_device = (
        args.sam3_device.strip()
        if args.sam3_device is not None
        else args.device.strip()
    )
    dgedi_python = (
        args.dgedi_python
        if args.dgedi_python is not None
        else Path(sys.executable)
    )

    return PipelineConfig(
        source_config_path=(
            args.config_path.expanduser().resolve()
        ),
        random_seed=args.random_seed,
        pose_path=args.pose_path,
        storage_results_only=args.storage_results_only,
        storage_keep_failed_intermediates=(
            args.storage_keep_failed_intermediates
        ),
        resume=args.resume,
        linemod_all_object_ids=tuple(
            args.linemod_all_object_ids
        ),
        linemod_all_query_stride=(
            args.linemod_all_query_stride
        ),
        linemod_all_maximum_queries_per_object=(
            args.linemod_all_maximum_queries_per_object
        ),
        linemod_all_continue_on_error=(
            args.linemod_all_continue_on_error
        ),
        dataset_root=(
            args.dataset_root.expanduser().resolve()
        ),
        split=args.split.strip(),
        object_id=args.object_id,
        object_name=object_name,
        sam3_prompt=sam3_prompt,
        sam3_repository=(
            args.sam3_repository.expanduser().resolve()
        ),
        sam3_checkpoint=(
            args.sam3_checkpoint.expanduser().resolve()
        ),
        sam3_bpe=(
            args.sam3_bpe.expanduser().resolve()
        ),
        sam3_device=sam3_device,
        sam3_use_amp=args.sam3_use_amp,
        sam3_confidence_threshold=(
            args.sam3_confidence_threshold
        ),
        reference=reference,
        query=query,
        mask_type=args.mask_type,
        gt_mask_fallback_on_sam3_failure=(
            args.gt_mask_fallback_on_sam3_failure
        ),
        evaluation_enabled=(
            args.evaluation_enabled
        ),
        instantmesh_repository=(
            instantmesh_repository
        ),
        instantmesh_python=(
            instantmesh_python.expanduser().resolve()
        ),
        instantmesh_config=(
            instantmesh_config.expanduser().resolve()
        ),
        instantmesh_offline=(
            args.instantmesh_offline
        ),
        instantmesh_diffusion_steps=(
            args.instantmesh_diffusion_steps
        ),
        instantmesh_view_count=(
            args.instantmesh_view_count
        ),
        instantmesh_model_scale=(
            args.instantmesh_model_scale
        ),
        instantmesh_render_distance=(
            args.instantmesh_render_distance
        ),
        instantmesh_use_rembg=(
            args.instantmesh_use_rembg
        ),
        instantmesh_export_texture_map=(
            args.instantmesh_export_texture_map
        ),
        instantmesh_save_video=(
            args.instantmesh_save_video
        ),
        foundationpose_repository=(
            args.foundationpose_repository.expanduser().resolve()
        ),
        foundationpose_debug=(
            args.foundationpose_debug
        ),
        renderer_batch_size=(
            args.renderer_batch_size
        ),
        renderer_maximum_texture_size=(
            args.renderer_maximum_texture_size
        ),
        dino_enabled=args.dino_enabled,
        dinov3_repository=dinov3_repository,
        dinov3_checkpoint=(
            dinov3_checkpoint.expanduser().resolve()
        ),
        dinov3_model=args.dinov3_model,
        dinov3_target_long_side=(
            args.dinov3_target_long_side
        ),
        dinov3_use_amp=args.dinov3_use_amp,
        dinov3_save_dtype=(
            args.dinov3_save_dtype
        ),
        dinov3_maximum_surface_points=(
            args.dinov3_maximum_surface_points
        ),
        dinov3_feature_chunk_size=(
            args.dinov3_feature_chunk_size
        ),
        output_root=(
            output_root.expanduser().resolve()
        ),
        device=args.device,
        top_k=args.top_k,
        foundationpose_rotation_diversity_threshold_deg=(
            args.foundationpose_rotation_diversity_threshold_deg
        ),
        refine_iterations=(
            args.refine_iterations
        ),
        foundationpose_workers=(
            args.foundationpose_workers
        ),
        batch_query_image_ids=(
            batch_query_image_ids
        ),
        normalization_quantile_low=(
            args.normalization_quantile_low
        ),
        normalization_quantile_high=(
            args.normalization_quantile_high
        ),
        normalization_sample_count=(
            args.normalization_sample_count
        ),
        normalization_random_seed=(
            args.normalization_random_seed
        ),
        normalization_tolerance=(
            args.normalization_tolerance
        ),
        axis_scale_minimum_factor=(
            args.axis_scale_minimum_factor
        ),
        axis_scale_maximum_factor=(
            args.axis_scale_maximum_factor
        ),
        axis_scale_grid_step_count=(
            args.axis_scale_grid_step_count
        ),
        axis_scale_penalty_weight=(
            args.axis_scale_penalty_weight
        ),
        axis_scale_uncertainty_penalty_weight=(
            args.axis_scale_uncertainty_penalty_weight
        ),
        axis_scale_minimum_loss_improvement_ratio=(
            args.axis_scale_minimum_loss_improvement_ratio
        ),
        axis_scale_coarse_shortlist_count=(
            args.axis_scale_coarse_shortlist_count
        ),
        axis_scale_fine_factor_radius=(
            args.axis_scale_fine_factor_radius
        ),
        axis_scale_fine_grid_step_count=(
            args.axis_scale_fine_grid_step_count
        ),
        scale_quantile_low=(
            args.scale_quantile_low
        ),
        scale_quantile_high=(
            args.scale_quantile_high
        ),
        scale_multipliers=tuple(
            args.scale_multipliers
        ),
        scale_minimum_valid_points=(
            args.scale_minimum_valid_points
        ),
        scale_maximum_points=(
            args.scale_maximum_points
        ),
        scale_minimum_depth_m=(
            args.scale_minimum_depth_m
        ),
        scale_maximum_depth_m=(
            args.scale_maximum_depth_m
        ),
        scale_save_point_cloud=(
            args.scale_save_point_cloud
        ),
        visible_scale_refinement_enabled=(
            args.visible_scale_refinement_enabled
        ),
        enable_shared_axis_scale_refinement=(
            args.enable_shared_axis_scale_refinement
        ),
        visible_scale_refinement_reference_enabled=(
            args.visible_scale_refinement_reference_enabled
        ),
        visible_scale_refinement_query_enabled=(
            args.visible_scale_refinement_query_enabled
        ),
        visible_scale_minimum_loss_improvement_ratio=(
            args.visible_scale_minimum_loss_improvement_ratio
        ),
        visible_scale_policy=args.visible_scale_policy,
        visible_scale_local_multipliers=tuple(
            args.visible_scale_local_multipliers
        ),
        visible_scale_mask_erosion_kernel_size=(
            args.visible_scale_mask_erosion_kernel_size
        ),
        visible_scale_minimum_correspondences=(
            args.visible_scale_minimum_correspondences
        ),
        visible_scale_minimum_spatial_coverage=(
            args.visible_scale_minimum_spatial_coverage
        ),
        visible_scale_depth_absolute_tolerance_m=(
            args.visible_scale_depth_absolute_tolerance_m
        ),
        visible_scale_depth_relative_tolerance=(
            args.visible_scale_depth_relative_tolerance
        ),
        visible_scale_irls_iterations=(
            args.visible_scale_irls_iterations
        ),
        visible_scale_huber_delta_m=(
            args.visible_scale_huber_delta_m
        ),
        visible_scale_maximum_abs_log_correction=(
            args.visible_scale_maximum_abs_log_correction
        ),
        visible_scale_coverage_grid_size=(
            args.visible_scale_coverage_grid_size
        ),
        dgedi_repository=(
            args.dgedi_repository.expanduser().resolve()
        ),
        dgedi_python=dgedi_python.expanduser().resolve(),
        dgedi_config=args.dgedi_config.expanduser().resolve(),
        dgedi_mode=args.dgedi_mode,
        dgedi_device=args.dgedi_device,
        dgedi_sample_count=args.dgedi_sample_count,
        dgedi_ransac_threshold=args.dgedi_ransac_threshold,
        dgedi_icp_threshold=args.dgedi_icp_threshold,
        dgedi_maximum_surface_depth_residual_m=(
            args.dgedi_maximum_surface_depth_residual_m
        ),
        dgedi_minimum_visible_depth_pixels=(
            args.dgedi_minimum_visible_depth_pixels
        ),
        dgedi_minimum_pair_point_count_ratio=(
            args.dgedi_minimum_pair_point_count_ratio
        ),
        dgedi_minimum_pair_diameter_ratio=(
            args.dgedi_minimum_pair_diameter_ratio
        ),
        dgedi_registration_candidate_count=(
            args.dgedi_registration_candidate_count
        ),
        dgedi_confidence_weight_correspondence=(
            args.dgedi_confidence_weight_correspondence
        ),
        dgedi_confidence_weight_inlier=(
            args.dgedi_confidence_weight_inlier
        ),
        dgedi_confidence_weight_rmse=(
            args.dgedi_confidence_weight_rmse
        ),
        dgedi_confidence_weight_mask=(
            args.dgedi_confidence_weight_mask
        ),
        dgedi_confidence_weight_depth=(
            args.dgedi_confidence_weight_depth
        ),
        dgedi_confidence_weight_free_space=(
            args.dgedi_confidence_weight_free_space
        ),
        dgedi_confidence_target_correspondence_fraction=(
            args.dgedi_confidence_target_correspondence_fraction
        ),
        dgedi_confidence_good_inlier_fitness=(
            args.dgedi_confidence_good_inlier_fitness
        ),
        dgedi_confidence_maximum_normalized_rmse=(
            args.dgedi_confidence_maximum_normalized_rmse
        ),
        dgedi_confidence_maximum_normalized_depth_residual=(
            args.dgedi_confidence_maximum_normalized_depth_residual
        ),
        dgedi_selection_rotation_penalty_weight=(
            args.dgedi_selection_rotation_penalty_weight
        ),
        dgedi_selection_uncertainty_penalty_weight=(
            args.dgedi_selection_uncertainty_penalty_weight
        ),
        dgedi_selection_deformation_penalty_weight=(
            args.dgedi_selection_deformation_penalty_weight
        ),
        dgedi_selection_near_tie_margin=(
            args.dgedi_selection_near_tie_margin
        ),
        dgedi_validation_minimum_mask_iou=(
            args.dgedi_validation_minimum_mask_iou
        ),
        dgedi_validation_baseline_mask_iou_drop=(
            args.dgedi_validation_baseline_mask_iou_drop
        ),
        dgedi_validation_maximum_depth_residual_normalized=(
            args.dgedi_validation_maximum_depth_residual_normalized
        ),
        dgedi_validation_baseline_depth_residual_margin=(
            args.dgedi_validation_baseline_depth_residual_margin
        ),
        dgedi_validation_maximum_total_loss=(
            args.dgedi_validation_maximum_total_loss
        ),
        dgedi_validation_baseline_total_loss_margin=(
            args.dgedi_validation_baseline_total_loss_margin
        ),
        alignment_weight_mask=(
            args.alignment_weight_mask
        ),
        alignment_weight_depth=(
            args.alignment_weight_depth
        ),
        alignment_weight_free_space=(
            args.alignment_weight_free_space
        ),
        alignment_weight_boundary=(
            args.alignment_weight_boundary
        ),
        alignment_depth_trim_quantile=(
            args.alignment_depth_trim_quantile
        ),
        alignment_minimum_depth_overlap_pixels=(
            args.alignment_minimum_depth_overlap_pixels
        ),
        alignment_free_space_absolute_tolerance_m=(
            args.alignment_free_space_absolute_tolerance_m
        ),
        alignment_free_space_relative_tolerance=(
            args.alignment_free_space_relative_tolerance
        ),
        dino_depth_absolute_tolerance_m=(
            args.dino_depth_absolute_tolerance_m
        ),
        dino_depth_relative_tolerance=(
            args.dino_depth_relative_tolerance
        ),
        dino_minimum_matched_points=(
            args.dino_minimum_matched_points
        ),
        dino_minimum_coverage=(
            args.dino_minimum_coverage
        ),
        dino_coverage_weight=(
            args.dino_coverage_weight
        ),
    )


def _require_directory(
    path: Path,
    description: str,
) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{description} directory not found: {path}"
        )


def _require_file(
    path: Path,
    description: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} file not found: {path}"
        )


def _query_frames(
    config: PipelineConfig,
) -> tuple[FrameSpec, ...]:
    if config.batch_query_image_ids is None:
        return (config.query,)

    return tuple(
        FrameSpec(
            scene_id=config.query.scene_id,
            image_id=image_id,
            instance_index=(
                config.query.instance_index
            ),
        )
        for image_id in config.batch_query_image_ids
    )


def _pose_path_uses_dino(
    config: PipelineConfig,
) -> bool:
    return (
        config.dino_enabled
        and config.pose_path
        in {
            "combined",
            "self_cross",
        }
    )
