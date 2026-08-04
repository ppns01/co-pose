from __future__ import annotations

import math

from core.types import PipelineConfig
from pipeline.config_build import (
    _pose_path_uses_dino,
    _query_frames,
    _require_directory,
    _require_file,
)
from pipeline.config_defaults import (
    LINEMOD_OBJECT_METADATA,
    POSE_PATH_CHOICES,
)


def validate_config(
    config: PipelineConfig,
) -> None:
    if config.pose_path not in POSE_PATH_CHOICES:
        raise ValueError(
            f"Unsupported pose_path: {config.pose_path}"
        )

    _require_file(
        config.source_config_path,
        "Pipeline source config",
    )

    _require_directory(
        config.dataset_root,
        "LINEMOD dataset root",
    )

    _require_directory(
        config.sam3_repository,
        "SAM3 repository",
    )

    _require_file(
        config.sam3_checkpoint,
        "SAM3 checkpoint",
    )

    _require_file(
        config.sam3_bpe,
        "SAM3 BPE vocabulary",
    )

    _require_directory(
        config.instantmesh_repository,
        "InstantMesh repository",
    )

    _require_file(
        config.instantmesh_repository / "run.py",
        "InstantMesh run.py",
    )

    _require_file(
        config.instantmesh_config,
        "InstantMesh config",
    )

    _require_file(
        config.instantmesh_python,
        "InstantMesh Python",
    )

    _require_directory(
        config.foundationpose_repository,
        "FoundationPose repository",
    )

    _require_file(
        (
            config.foundationpose_repository
            / "estimater.py"
        ),
        "FoundationPose estimater.py",
    )

    if _pose_path_uses_dino(config):
        if config.dinov3_checkpoint.is_dir():
            for file_name in (
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
            ):
                _require_file(
                    (
                        config.dinov3_checkpoint
                        / file_name
                    ),
                    (
                        f"Hugging Face DINOv3 {file_name}"
                    ),
                )

        elif config.dinov3_checkpoint.is_file():
            _require_directory(
                config.dinov3_repository,
                "DINOv3 repository",
            )
            _require_file(
                (
                    config.dinov3_repository
                    / "hubconf.py"
                ),
                "DINOv3 hubconf.py",
            )

        else:
            raise FileNotFoundError(
                "DINOv3 weights file or Hugging Face "
                "model directory not found: "
                f"{config.dinov3_checkpoint}"
            )

        from features.dinov3_extractor import (
            SUPPORTED_DINOV3_MODELS,
        )

        if (
            config.dinov3_model
            not in SUPPORTED_DINOV3_MODELS
        ):
            raise ValueError(
                f"Unsupported DINOv3 model: {config.dinov3_model}"
            )

        if not config.device.startswith("cuda"):
            raise ValueError(
                f"DINOv3 requires a CUDA device: {config.device}"
            )

    if config.object_id <= 0:
        raise ValueError(
            "object_id must be greater than zero."
        )

    if not config.object_name:
        raise ValueError(
            "object_name must not be empty."
        )

    if not config.sam3_prompt:
        raise ValueError(
            "sam3_prompt must not be empty."
        )

    if not config.split:
        raise ValueError(
            "split must not be empty."
        )

    if (
        isinstance(config.random_seed, bool)
        or config.random_seed < 0
    ):
        raise ValueError(
            "random_seed must be a non-negative integer."
        )

    if not config.sam3_device:
        raise ValueError(
            "sam3_device must not be empty."
        )

    if not math.isfinite(
        config.sam3_confidence_threshold
    ) or not (
        0.0
        <= config.sam3_confidence_threshold
        < 1.0
    ):
        raise ValueError(
            "sam3_confidence_threshold must be finite and in [0, 1)."
        )

    if config.instantmesh_view_count not in (
        4,
        6,
    ):
        raise ValueError(
            "instantmesh_view_count must be 4 or 6."
        )

    if config.instantmesh_diffusion_steps < 1:
        raise ValueError(
            "instantmesh_diffusion_steps must be at least 1."
        )

    if (
        config.instantmesh_model_scale <= 0.0
        or config.instantmesh_render_distance
        <= 0.0
    ):
        raise ValueError(
            "InstantMesh scale and distance must be positive."
        )

    if config.dinov3_target_long_side < 64:
        raise ValueError(
            "dinov3_target_long_side must be at least 64."
        )

    if config.dinov3_save_dtype != "float16":
        raise ValueError(
            "dinov3_save_dtype must be float16."
        )

    if (
        config.dinov3_maximum_surface_points < 1
        or config.dinov3_feature_chunk_size < 1
    ):
        raise ValueError(
            "DINOv3 point and chunk sizes must be positive."
        )

    if (
        config.renderer_batch_size < 1
        or config.renderer_maximum_texture_size
        < 1
    ):
        raise ValueError(
            "Renderer batch and texture sizes must be positive."
        )

    if config.foundationpose_debug < 0:
        raise ValueError(
            "foundationpose_debug must be non-negative."
        )

    if (
        not 0.0
        <= config.normalization_quantile_low
        < config.normalization_quantile_high
        <= 1.0
    ):
        raise ValueError(
            "Mesh normalization quantiles are invalid."
        )

    if (
        not 0.0
        <= config.scale_quantile_low
        < config.scale_quantile_high
        <= 1.0
    ):
        raise ValueError(
            "Depth scale quantiles are invalid."
        )

    if not config.scale_multipliers or any(
        multiplier <= 0.0
        for multiplier in config.scale_multipliers
    ):
        raise ValueError(
            "scale_multipliers must contain positive values."
        )

    if (
        config.normalization_sample_count < 1
        or config.normalization_random_seed < 0
        or config.normalization_tolerance <= 0.0
    ):
        raise ValueError(
            "Mesh normalization settings are invalid."
        )

    if not (
        math.isfinite(config.axis_scale_minimum_factor)
        and math.isfinite(config.axis_scale_maximum_factor)
        and math.isfinite(config.axis_scale_penalty_weight)
        and math.isfinite(
            config.axis_scale_uncertainty_penalty_weight
        )
        and math.isfinite(
            config.axis_scale_minimum_loss_improvement_ratio
        )
        and math.isfinite(config.axis_scale_fine_factor_radius)
        and 0.0 < config.axis_scale_minimum_factor <= 1.0
        <= config.axis_scale_maximum_factor
        and config.axis_scale_grid_step_count >= 2
        and config.axis_scale_penalty_weight >= 0.0
        and config.axis_scale_uncertainty_penalty_weight >= 0.0
        and 0.0
        <= config.axis_scale_minimum_loss_improvement_ratio
        < 1.0
        and config.axis_scale_coarse_shortlist_count >= 1
        and config.axis_scale_fine_factor_radius > 0.0
        and config.axis_scale_fine_grid_step_count >= 2
    ):
        raise ValueError(
            "Confidence-aware shared-dimension settings are invalid."
        )

    if (
        config.scale_minimum_valid_points < 1
        or (
            config.scale_maximum_points
            is not None
            and config.scale_maximum_points < 1
        )
        or config.scale_minimum_depth_m < 0.0
        or (
            config.scale_maximum_depth_m
            is not None
            and config.scale_maximum_depth_m
            <= config.scale_minimum_depth_m
        )
    ):
        raise ValueError(
            "Depth scale settings are invalid."
        )

    if (
        not 0.0
        <= config.visible_scale_minimum_loss_improvement_ratio
        < 1.0
    ):
        raise ValueError(
            "visible scale minimum loss improvement ratio must be in [0, 1)."
        )

    if (
        not 0.0
        < config.alignment_depth_trim_quantile
        <= 1.0
        or config.alignment_minimum_depth_overlap_pixels
        < 1
        or config.alignment_free_space_absolute_tolerance_m
        < 0.0
        or config.alignment_free_space_relative_tolerance
        < 0.0
    ):
        raise ValueError(
            "Alignment evidence settings are invalid."
        )

    if (
        config.dino_depth_absolute_tolerance_m
        < 0.0
        or config.dino_depth_relative_tolerance
        < 0.0
        or config.dino_minimum_matched_points < 1
        or not 0.0
        <= config.dino_minimum_coverage
        <= 1.0
        or config.dino_coverage_weight < 0.0
    ):
        raise ValueError(
            "DINO evidence settings are invalid."
        )

    weight_groups = {
        "alignment": (
            config.alignment_weight_mask,
            config.alignment_weight_depth,
            config.alignment_weight_free_space,
            config.alignment_weight_boundary,
        ),
    }

    for (
        group_name,
        values,
    ) in weight_groups.items():
        if (
            any(
                not math.isfinite(value)
                or value < 0.0
                for value in values
            )
            or sum(values) <= 0.0
        ):
            raise ValueError(
                f"{group_name} weights are invalid: {values}"
            )

    if (
        isinstance(config.top_k, bool)
        or config.top_k < 1
    ):
        raise ValueError(
            "top_k must be at least 1."
        )
    if not (
        math.isfinite(
            config.foundationpose_rotation_diversity_threshold_deg
        )
        and 0.0
        <= config.foundationpose_rotation_diversity_threshold_deg
        <= 180.0
    ):
        raise ValueError(
            "foundationpose_rotation_diversity_threshold_deg must be "
            "in [0, 180]."
        )

    if (
        isinstance(config.refine_iterations, bool)
        or config.refine_iterations < 1
    ):
        raise ValueError(
            "refine_iterations must be at least 1."
        )

    if (
        isinstance(
            config.foundationpose_workers,
            bool,
        )
        or config.foundationpose_workers < 1
    ):
        raise ValueError(
            "foundationpose_workers must be at least 1."
        )

    if (
        not config.linemod_all_object_ids
        or len(set(config.linemod_all_object_ids))
        != len(config.linemod_all_object_ids)
        or any(
            object_id
            not in LINEMOD_OBJECT_METADATA
            for object_id in config.linemod_all_object_ids
        )
    ):
        raise ValueError(
            "linemod_all.object_ids must be a non-empty unique subset "
            "of configured LINEMOD object IDs."
        )
    if config.linemod_all_query_stride < 1:
        raise ValueError(
            "linemod_all.query_stride must be at least 1."
        )
    if (
        config.linemod_all_maximum_queries_per_object
        is not None
        and config.linemod_all_maximum_queries_per_object
        < 1
    ):
        raise ValueError(
            "linemod_all.maximum_queries_per_object must be null or >= 1."
        )
    if config.visible_scale_policy not in {
        "independent",
        "joint_shared",
    }:
        raise ValueError(
            "scale.visible_refinement.policy must be independent or joint_shared."
        )
    if config.pose_path == "self_mesh" and not (
        config.visible_scale_refinement_enabled
        and config.visible_scale_policy == "joint_shared"
        and config.visible_scale_refinement_reference_enabled
        and config.visible_scale_refinement_query_enabled
    ):
        raise ValueError(
            "self_mesh requires enabled pair-shared S*: set "
            "scale.visible_refinement.enabled/reference_enabled/"
            "query_enabled=true and policy=joint_shared."
        )
    if (
        not config.visible_scale_local_multipliers
        or any(
            not math.isfinite(value)
            or value <= 0.0
            for value in config.visible_scale_local_multipliers
        )
    ):
        raise ValueError(
            "scale.visible_refinement.local_scale_multipliers must "
            "contain positive finite values."
        )
    positive_visible_integers = (
        config.visible_scale_mask_erosion_kernel_size,
        config.visible_scale_minimum_correspondences,
        config.visible_scale_irls_iterations,
        config.visible_scale_coverage_grid_size,
    )
    if any(
        value < 1
        for value in positive_visible_integers
    ):
        raise ValueError(
            "Visible-scale integer parameters must be at least 1."
        )
    if (
        config.visible_scale_mask_erosion_kernel_size
        % 2
        == 0
    ):
        raise ValueError(
            "scale.visible_refinement.mask_erosion_kernel_size must be odd."
        )
    if not (
        0.0
        <= config.visible_scale_minimum_spatial_coverage
        <= 1.0
        and config.visible_scale_depth_absolute_tolerance_m
        > 0.0
        and config.visible_scale_depth_relative_tolerance
        >= 0.0
        and config.visible_scale_huber_delta_m
        > 0.0
        and config.visible_scale_maximum_abs_log_correction
        > 0.0
    ):
        raise ValueError(
            "Visible-scale numeric parameters are invalid."
        )
    if config.pose_path == "self_mesh":
        _require_directory(
            config.dgedi_repository,
            "dGeDi repository",
        )
        _require_file(
            config.dgedi_python, "dGeDi Python"
        )
        _require_file(
            config.dgedi_config, "dGeDi config"
        )
    if config.dgedi_mode not in {
        "single_scale",
        "multi_scale",
    }:
        raise ValueError(
            "dgedi.mode must be single_scale or multi_scale."
        )
    if not config.dgedi_device.strip():
        raise ValueError(
            "dgedi.device must not be empty."
        )
    if not (
        config.dgedi_sample_count >= 256
        and config.dgedi_ransac_threshold > 0.0
        and config.dgedi_icp_threshold > 0.0
        and config.dgedi_maximum_surface_depth_residual_m
        > 0.0
        and config.dgedi_minimum_visible_depth_pixels
        >= 256
        and 0.0
        < config.dgedi_minimum_pair_point_count_ratio
        <= 1.0
        and 0.0
        < config.dgedi_minimum_pair_diameter_ratio
        <= 1.0
    ):
        raise ValueError(
            "dGeDi numeric parameters are invalid."
        )
    if not (
        0.0
        <= config.dgedi_validation_minimum_mask_iou
        <= 1.0
        and config.dgedi_validation_baseline_mask_iou_drop
        >= 0.0
        and config.dgedi_validation_maximum_depth_residual_normalized
        >= 0.0
        and config.dgedi_validation_baseline_depth_residual_margin
        >= 0.0
        and config.dgedi_validation_maximum_total_loss
        >= 0.0
        and config.dgedi_validation_baseline_total_loss_margin
        >= 0.0
    ):
        raise ValueError(
            "dGeDi observation-validation parameters are invalid."
        )

    dgedi_confidence_weights = (
        config.dgedi_confidence_weight_correspondence,
        config.dgedi_confidence_weight_inlier,
        config.dgedi_confidence_weight_rmse,
        config.dgedi_confidence_weight_mask,
        config.dgedi_confidence_weight_depth,
        config.dgedi_confidence_weight_free_space,
    )
    if (
        any(
            not math.isfinite(value) or value < 0.0
            for value in dgedi_confidence_weights
        )
        or sum(dgedi_confidence_weights) <= 0.0
        or not 0.0
        < config.dgedi_confidence_target_correspondence_fraction
        <= 1.0
        or not 0.0
        < config.dgedi_confidence_good_inlier_fitness
        <= 1.0
        or not math.isfinite(
            config.dgedi_confidence_maximum_normalized_rmse
        )
        or config.dgedi_confidence_maximum_normalized_rmse <= 0.0
        or (
            not math.isfinite(
                config.dgedi_confidence_maximum_normalized_depth_residual
            )
            or
            config.dgedi_confidence_maximum_normalized_depth_residual
            <= 0.0
        )
    ):
        raise ValueError(
            "dGeDi continuous-confidence parameters are invalid."
        )
    dgedi_selection_values = (
        config.dgedi_selection_rotation_penalty_weight,
        config.dgedi_selection_uncertainty_penalty_weight,
        config.dgedi_selection_deformation_penalty_weight,
    )
    if (
        any(
            not math.isfinite(value) or value < 0.0
            for value in dgedi_selection_values
        )
        or not math.isfinite(config.dgedi_selection_near_tie_margin)
        or config.dgedi_selection_near_tie_margin <= 0.0
    ):
        raise ValueError(
            "dGeDi continuous final-selection parameters are invalid."
        )

    query_frames = _query_frames(config)

    if (
        config.batch_query_image_ids is not None
        and not config.batch_query_image_ids
    ):
        raise ValueError(
            "query_image_ids must not be empty."
        )

    if (
        config.batch_query_image_ids is not None
        and len(set(config.batch_query_image_ids))
        != len(config.batch_query_image_ids)
    ):
        raise ValueError(
            "query_image_ids must not contain duplicates."
        )

    if (
        config.batch_query_image_ids is not None
        and config.query != query_frames[0]
    ):
        raise ValueError(
            "config.query must match the first batch query."
        )

    for query_frame in query_frames:
        same_input = (
            config.reference.scene_id
            == query_frame.scene_id
            and config.reference.image_id
            == query_frame.image_id
            and config.reference.instance_index
            == query_frame.instance_index
        )

        if same_input:
            raise ValueError(
                "Reference and Query must be different inputs: "
                f"scene={query_frame.scene_id}, "
                f"image={query_frame.image_id}, "
                f"instance={query_frame.instance_index}."
            )

