from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from core.types import PipelineConfig
from pipeline.config_build import _pose_path_uses_dino
from pipeline.config_defaults import PROJECT_ROOT


def _save_run_config(
    config: PipelineConfig,
) -> Path:
    config.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = (
        config.output_root
        / "pipeline_config.json"
    )

    payload = {
        "source_config_path": str(
            config.source_config_path
        ),
        "pose_path": config.pose_path,
        "mode": (
            "multi_query"
            if config.batch_query_image_ids
            is not None
            else "single_query"
        ),
        "storage": {
            "results_only": config.storage_results_only,
            "keep_failed_intermediates": (
                config.storage_keep_failed_intermediates
            ),
        },
        "linemod_all": {
            "object_ids": list(
                config.linemod_all_object_ids
            ),
            "query_stride": (
                config.linemod_all_query_stride
            ),
            "maximum_queries_per_object": (
                config.linemod_all_maximum_queries_per_object
            ),
            "continue_on_error": (
                config.linemod_all_continue_on_error
            ),
        },
        "dataset_root": str(config.dataset_root),
        "split": config.split,
        "object_id": config.object_id,
        "object_name": config.object_name,
        "segmentation_source": "sam3",
        "segmentation": {
            "primary_source": "sam3",
            "gt_mask_fallback_on_sam3_failure": (
                config.gt_mask_fallback_on_sam3_failure
            ),
        },
        "evaluation": {
            "enabled": config.evaluation_enabled,
        },
        "sam3_prompt": config.sam3_prompt,
        "reference": {
            "scene_id": (
                config.reference.scene_id
            ),
            "image_id": (
                config.reference.image_id
            ),
            "instance_index": (
                config.reference.instance_index
            ),
        },
        "query": {
            "scene_id": config.query.scene_id,
            "image_id": config.query.image_id,
            "instance_index": (
                config.query.instance_index
            ),
        },
        "query_image_ids": (
            list(config.batch_query_image_ids)
            if config.batch_query_image_ids
            is not None
            else None
        ),
        "mask_type": config.mask_type,
        "instantmesh_repository": str(
            config.instantmesh_repository
        ),
        "instantmesh_python": str(
            config.instantmesh_python
        ),
        "instantmesh_config": str(
            config.instantmesh_config
        ),
        "instantmesh_offline": (
            config.instantmesh_offline
        ),
        "foundationpose_repository": str(
            config.foundationpose_repository
        ),
        "output_root": str(config.output_root),
        "device": config.device,
        "top_k": config.top_k,
        "foundationpose_rotation_diversity_threshold_deg": (
            config.foundationpose_rotation_diversity_threshold_deg
        ),
        "refine_iterations": (
            config.refine_iterations
        ),
        "foundationpose_workers": (
            config.foundationpose_workers
        ),
        "dino_enabled": config.dino_enabled,
        "dinov3_repository": str(
            config.dinov3_repository
        ),
        "dinov3_checkpoint": str(
            config.dinov3_checkpoint
        ),
        "dinov3_model": config.dinov3_model,
        "dinov3_backend": (
            (
                "huggingface_transformers"
                if config.dinov3_checkpoint.is_dir()
                else "pytorch_hub"
            )
            if config.dino_enabled
            else None
        ),
        "generator": "instantmesh",
        "random_seed": config.random_seed,
        "random_seed_scope": (
            "instantmesh_and_dinov3_surface"
        ),
        "sam3": {
            "repository": str(
                config.sam3_repository
            ),
            "checkpoint": str(
                config.sam3_checkpoint
            ),
            "bpe": str(config.sam3_bpe),
            "prompt": config.sam3_prompt,
            "device": config.sam3_device,
            "use_amp": config.sam3_use_amp,
            "confidence_threshold": (
                config.sam3_confidence_threshold
            ),
        },
        "instantmesh": {
            "repository": str(
                config.instantmesh_repository
            ),
            "python": str(
                config.instantmesh_python
            ),
            "config": str(
                config.instantmesh_config
            ),
            "offline": config.instantmesh_offline,
            "diffusion_steps": (
                config.instantmesh_diffusion_steps
            ),
            "view_count": (
                config.instantmesh_view_count
            ),
            "model_scale": (
                config.instantmesh_model_scale
            ),
            "render_distance": (
                config.instantmesh_render_distance
            ),
            "use_rembg": (
                config.instantmesh_use_rembg
            ),
            "export_texture_map": (
                config.instantmesh_export_texture_map
            ),
            "save_video": (
                config.instantmesh_save_video
            ),
        },
        "dinov3": {
            "enabled": config.dino_enabled,
            "repository": str(
                config.dinov3_repository
            ),
            "checkpoint": str(
                config.dinov3_checkpoint
            ),
            "model": config.dinov3_model,
            "target_long_side": (
                config.dinov3_target_long_side
            ),
            "use_amp": config.dinov3_use_amp,
            "save_dtype": (
                config.dinov3_save_dtype
            ),
            "maximum_surface_points": (
                config.dinov3_maximum_surface_points
            ),
            "feature_chunk_size": (
                config.dinov3_feature_chunk_size
            ),
        },
        "foundationpose": {
            "repository": str(
                config.foundationpose_repository
            ),
            "top_k": config.top_k,
            "rotation_diversity_threshold_deg": (
                config.foundationpose_rotation_diversity_threshold_deg
            ),
            "refine_iterations": (
                config.refine_iterations
            ),
            "workers": (
                config.foundationpose_workers
            ),
            "debug": config.foundationpose_debug,
            "renderer": {
                "batch_size": (
                    config.renderer_batch_size
                ),
                "maximum_texture_size": (
                    config.renderer_maximum_texture_size
                ),
            },
        },
        "scale": {
            "normalization": {
                "quantile_low": (
                    config.normalization_quantile_low
                ),
                "quantile_high": (
                    config.normalization_quantile_high
                ),
                "sample_count": (
                    config.normalization_sample_count
                ),
                "random_seed": (
                    config.normalization_random_seed
                ),
                "tolerance": (
                    config.normalization_tolerance
                ),
            },
            "axis_refinement": {
                "minimum_factor": (
                    config.axis_scale_minimum_factor
                ),
                "maximum_factor": (
                    config.axis_scale_maximum_factor
                ),
                "grid_step_count": (
                    config.axis_scale_grid_step_count
                ),
                "scale_penalty_weight": (
                    config.axis_scale_penalty_weight
                ),
                "uncertainty_scale_penalty_weight": (
                    config.axis_scale_uncertainty_penalty_weight
                ),
                "minimum_loss_improvement_ratio": (
                    config.axis_scale_minimum_loss_improvement_ratio
                ),
                "coarse_shortlist_count": (
                    config.axis_scale_coarse_shortlist_count
                ),
                "fine_factor_radius": (
                    config.axis_scale_fine_factor_radius
                ),
                "fine_grid_step_count": (
                    config.axis_scale_fine_grid_step_count
                ),
            },
            "depth": {
                "quantile_low": (
                    config.scale_quantile_low
                ),
                "quantile_high": (
                    config.scale_quantile_high
                ),
                "multipliers": list(
                    config.scale_multipliers
                ),
                "minimum_valid_points": (
                    config.scale_minimum_valid_points
                ),
                "maximum_points": (
                    config.scale_maximum_points
                ),
                "minimum_depth_m": (
                    config.scale_minimum_depth_m
                ),
                "maximum_depth_m": (
                    config.scale_maximum_depth_m
                ),
                "save_point_cloud": (
                    config.scale_save_point_cloud
                ),
            },
            "visible_refinement": {
                "enabled": (
                    config.visible_scale_refinement_enabled
                ),
                "reference_enabled": (
                    config.visible_scale_refinement_reference_enabled
                ),
                "query_enabled": (
                    config.visible_scale_refinement_query_enabled
                ),
                "minimum_loss_improvement_ratio": (
                    config.visible_scale_minimum_loss_improvement_ratio
                ),
                "policy": config.visible_scale_policy,
                "local_scale_multipliers": list(
                    config.visible_scale_local_multipliers
                ),
                "mask_erosion_kernel_size": (
                    config.visible_scale_mask_erosion_kernel_size
                ),
                "minimum_correspondences": (
                    config.visible_scale_minimum_correspondences
                ),
                "minimum_spatial_coverage": (
                    config.visible_scale_minimum_spatial_coverage
                ),
                "depth_absolute_tolerance_m": (
                    config.visible_scale_depth_absolute_tolerance_m
                ),
                "depth_relative_tolerance": (
                    config.visible_scale_depth_relative_tolerance
                ),
                "irls_iterations": (
                    config.visible_scale_irls_iterations
                ),
                "huber_delta_m": (
                    config.visible_scale_huber_delta_m
                ),
                "maximum_abs_log_correction": (
                    config.visible_scale_maximum_abs_log_correction
                ),
                "coverage_grid_size": (
                    config.visible_scale_coverage_grid_size
                ),
            },
        },
        "dgedi": {
            "repository": str(
                config.dgedi_repository
            ),
            "python": str(config.dgedi_python),
            "config": str(config.dgedi_config),
            "mode": config.dgedi_mode,
            "device": config.dgedi_device,
            "sample_count": config.dgedi_sample_count,
            "ransac_threshold": config.dgedi_ransac_threshold,
            "icp_threshold": config.dgedi_icp_threshold,
            "maximum_surface_depth_residual_m": (
                config.dgedi_maximum_surface_depth_residual_m
            ),
            "minimum_visible_depth_pixels": (
                config.dgedi_minimum_visible_depth_pixels
            ),
            "minimum_pair_point_count_ratio": (
                config.dgedi_minimum_pair_point_count_ratio
            ),
            "minimum_pair_diameter_ratio": (
                config.dgedi_minimum_pair_diameter_ratio
            ),
            "registration_candidate_count": (
                config.dgedi_registration_candidate_count
            ),
            "confidence": {
                "weights": {
                    "correspondence": (
                        config.dgedi_confidence_weight_correspondence
                    ),
                    "inlier": config.dgedi_confidence_weight_inlier,
                    "rmse": config.dgedi_confidence_weight_rmse,
                    "mask": config.dgedi_confidence_weight_mask,
                    "depth": config.dgedi_confidence_weight_depth,
                    "free_space": (
                        config.dgedi_confidence_weight_free_space
                    ),
                },
                "target_correspondence_fraction": (
                    config.dgedi_confidence_target_correspondence_fraction
                ),
                "good_inlier_fitness": (
                    config.dgedi_confidence_good_inlier_fitness
                ),
                "maximum_normalized_rmse": (
                    config.dgedi_confidence_maximum_normalized_rmse
                ),
                "maximum_normalized_depth_residual": (
                    config.dgedi_confidence_maximum_normalized_depth_residual
                ),
            },
            "final_selection": {
                "rotation_penalty_weight": (
                    config.dgedi_selection_rotation_penalty_weight
                ),
                "uncertainty_penalty_weight": (
                    config.dgedi_selection_uncertainty_penalty_weight
                ),
                "deformation_penalty_weight": (
                    config.dgedi_selection_deformation_penalty_weight
                ),
                "near_tie_margin": (
                    config.dgedi_selection_near_tie_margin
                ),
            },
            "observation_validation": {
                "minimum_mask_iou": (
                    config.dgedi_validation_minimum_mask_iou
                ),
                "baseline_mask_iou_drop": (
                    config.dgedi_validation_baseline_mask_iou_drop
                ),
                "maximum_depth_residual_normalized": (
                    config.dgedi_validation_maximum_depth_residual_normalized
                ),
                "baseline_depth_residual_margin": (
                    config.dgedi_validation_baseline_depth_residual_margin
                ),
                "maximum_total_loss": (
                    config.dgedi_validation_maximum_total_loss
                ),
                "baseline_total_loss_margin": (
                    config.dgedi_validation_baseline_total_loss_margin
                ),
            },
        },
        "alignment": {
            "weights": {
                "mask": (
                    config.alignment_weight_mask
                ),
                "depth": (
                    config.alignment_weight_depth
                ),
                "free_space": (
                    config.alignment_weight_free_space
                ),
                "boundary": (
                    config.alignment_weight_boundary
                ),
            },
            "depth_trim_quantile": (
                config.alignment_depth_trim_quantile
            ),
            "minimum_depth_overlap_pixels": (
                config.alignment_minimum_depth_overlap_pixels
            ),
            "free_space_absolute_tolerance_m": (
                config.alignment_free_space_absolute_tolerance_m
            ),
            "free_space_relative_tolerance": (
                config.alignment_free_space_relative_tolerance
            ),
            "dino": {
                "depth_absolute_tolerance_m": (
                    config.dino_depth_absolute_tolerance_m
                ),
                "depth_relative_tolerance": (
                    config.dino_depth_relative_tolerance
                ),
                "minimum_matched_points": (
                    config.dino_minimum_matched_points
                ),
                "minimum_coverage": (
                    config.dino_minimum_coverage
                ),
                "coverage_weight": (
                    config.dino_coverage_weight
                ),
            },
        },
    }

    with config_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return config_path


def _initialize_research_run(
    *,
    config: PipelineConfig,
    config_path: Path,
) -> Any:
    from evaluation.research_result_logger import (
        initialize_research_run,
    )

    return initialize_research_run(
        output_root=config.output_root,
        config_path=config_path,
        project_root=PROJECT_ROOT,
        instantmesh_repository=(
            config.instantmesh_repository
        ),
        foundationpose_repository=(
            config.foundationpose_repository
        ),
        random_seed=config.random_seed,
    )


def _print_runtime_config(
    config: PipelineConfig,
) -> None:
    print("[Runtime configuration]")
    print(
        f"  config: {config.source_config_path}"
    )
    print(f"  pose_path: {config.pose_path}")
    print(
        "  results_only: "
        f"{config.storage_results_only} "
        "(keep_failed_intermediates="
        f"{config.storage_keep_failed_intermediates})"
    )
    print(f"  resume: {config.resume}")
    print(
        f"  dataset_root: {config.dataset_root}"
    )
    print(
        "  reference: "
        f"scene={config.reference.scene_id}, "
        f"image={config.reference.image_id}, "
        f"instance={config.reference.instance_index}"
    )
    if config.batch_query_image_ids is None:
        print(
            "  query: "
            f"scene={config.query.scene_id}, "
            f"image={config.query.image_id}, "
            f"instance={config.query.instance_index}"
        )
    else:
        print(
            "  queries: "
            f"scene={config.query.scene_id}, "
            "images="
            f"{list(config.batch_query_image_ids)}, "
            f"instance={config.query.instance_index}"
        )
    print(
        f"  object: id={config.object_id}, name={config.object_name}"
    )
    print("  segmentation_source: sam3")
    print(
        f"  gt_mask_fallback_on_sam3_failure: {config.gt_mask_fallback_on_sam3_failure}"
    )
    print(
        f"  evaluation_enabled: {config.evaluation_enabled}"
    )
    print(f"  sam3_prompt: {config.sam3_prompt}")
    print(
        f"  sam3_confidence_threshold: {config.sam3_confidence_threshold}"
    )
    print(f"  mask_type: {config.mask_type}")
    print(
        f"  instantmesh_repository: {config.instantmesh_repository}"
    )
    print(
        f"  instantmesh_config: {config.instantmesh_config}"
    )
    print(
        f"  instantmesh_python: {config.instantmesh_python}"
    )
    print(
        f"  foundationpose_repository: {config.foundationpose_repository}"
    )
    print(
        f"  foundationpose_workers: {config.foundationpose_workers}"
    )
    print(
        "  foundationpose_rotation_diversity_threshold_deg: "
        f"{config.foundationpose_rotation_diversity_threshold_deg}"
    )
    if config.pose_path == "self_mesh":
        print(
            f"  visible_scale_policy: {config.visible_scale_policy}"
        )
        print(
            "  independent_camera_axis_scale: "
            f"range=[{config.axis_scale_minimum_factor}, "
            f"{config.axis_scale_maximum_factor}], "
            f"coarse_grid={config.axis_scale_grid_step_count}^3, "
            f"fine=top{config.axis_scale_coarse_shortlist_count}"
            f"±{config.axis_scale_fine_factor_radius}"
            f"@{config.axis_scale_fine_grid_step_count}^3, "
            "depth observability adds a soft Sd prior"
        )
        print(
            "  dgedi: "
            f"mode={config.dgedi_mode}, "
            f"device={config.dgedi_device}, "
            f"samples={config.dgedi_sample_count}, "
            "surface_depth_residual_m="
            f"{config.dgedi_maximum_surface_depth_residual_m}"
        )
    effective_dino = _pose_path_uses_dino(config)
    print(
        f"  dino_enabled: {config.dino_enabled} (effective={effective_dino})"
    )
    if effective_dino:
        print(
            f"  dinov3_repository: {config.dinov3_repository}"
        )
        print(
            f"  dinov3_checkpoint: {config.dinov3_checkpoint}"
        )
        print(
            f"  dinov3_model: {config.dinov3_model}"
        )
        print(
            "  dinov3_backend: "
            + (
                "huggingface_transformers"
                if config.dinov3_checkpoint.is_dir()
                else "pytorch_hub"
            )
        )
    print(f"  output_root: {config.output_root}")

    if (
        config.instantmesh_python.resolve()
        == Path(sys.executable).resolve()
    ):
        print(
            "[Warning] instantmesh_clean was not "
            "auto-detected; using the current Python "
            "for InstantMesh."
        )

