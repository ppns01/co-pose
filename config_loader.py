from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_CONFIG_VERSION = 1


FIELD_MAP: dict[tuple[str, ...], str] = {
    ("experiment", "random_seed"): "random_seed",
    ("experiment", "device"): "device",
    ("experiment", "output_root"): "output_root",
    ("experiment", "pose_path"): "pose_path",
    (
        "segmentation",
        "gt_mask_fallback_on_sam3_failure",
    ): "gt_mask_fallback_on_sam3_failure",
    ("storage", "results_only"): "storage_results_only",
    (
        "storage",
        "keep_failed_intermediates",
    ): "storage_keep_failed_intermediates",
    ("storage", "resume"): "resume",
    (
        "linemod_all",
        "object_ids",
    ): "linemod_all_object_ids",
    (
        "linemod_all",
        "query_stride",
    ): "linemod_all_query_stride",
    (
        "linemod_all",
        "maximum_queries_per_object",
    ): "linemod_all_maximum_queries_per_object",
    (
        "linemod_all",
        "continue_on_error",
    ): "linemod_all_continue_on_error",
    ("data", "dataset_root"): "dataset_root",
    ("data", "split"): "split",
    ("data", "object_id"): "object_id",
    ("data", "object_name"): "object_name",
    ("data", "mask_type"): "mask_type",
    (
        "data",
        "reference",
        "scene_id",
    ): "reference_scene_id",
    (
        "data",
        "reference",
        "image_id",
    ): "reference_image_id",
    (
        "data",
        "reference",
        "instance_index",
    ): "reference_instance_index",
    (
        "data",
        "query",
        "scene_id",
    ): "query_scene_id",
    (
        "data",
        "query",
        "image_id",
    ): "query_image_id",
    (
        "data",
        "query",
        "image_ids",
    ): "query_image_ids",
    (
        "data",
        "query",
        "instance_index",
    ): "query_instance_index",
    ("sam3", "repository"): "sam3_repository",
    ("sam3", "checkpoint"): "sam3_checkpoint",
    ("sam3", "bpe"): "sam3_bpe",
    ("sam3", "prompt"): "sam3_prompt",
    ("sam3", "device"): "sam3_device",
    ("sam3", "use_amp"): "sam3_use_amp",
    (
        "sam3",
        "confidence_threshold",
    ): "sam3_confidence_threshold",
    (
        "instantmesh",
        "repository",
    ): "instantmesh_repository",
    ("instantmesh", "python"): "instantmesh_python",
    ("instantmesh", "config"): "instantmesh_config",
    ("instantmesh", "offline"): "instantmesh_offline",
    (
        "instantmesh",
        "diffusion_steps",
    ): "instantmesh_diffusion_steps",
    (
        "instantmesh",
        "view_count",
    ): "instantmesh_view_count",
    (
        "instantmesh",
        "model_scale",
    ): "instantmesh_model_scale",
    (
        "instantmesh",
        "render_distance",
    ): "instantmesh_render_distance",
    (
        "instantmesh",
        "use_rembg",
    ): "instantmesh_use_rembg",
    (
        "instantmesh",
        "export_texture_map",
    ): "instantmesh_export_texture_map",
    (
        "instantmesh",
        "save_video",
    ): "instantmesh_save_video",
    (
        "dinov3",
        "enabled",
    ): "dino_enabled",
    (
        "dinov3",
        "repository",
    ): "dinov3_repository",
    ("dinov3", "checkpoint"): "dinov3_checkpoint",
    ("dinov3", "model"): "dinov3_model",
    (
        "dinov3",
        "target_long_side",
    ): "dinov3_target_long_side",
    ("dinov3", "use_amp"): "dinov3_use_amp",
    ("dinov3", "save_dtype"): "dinov3_save_dtype",
    (
        "dinov3",
        "maximum_surface_points",
    ): "dinov3_maximum_surface_points",
    (
        "dinov3",
        "feature_chunk_size",
    ): "dinov3_feature_chunk_size",
    (
        "foundationpose",
        "repository",
    ): "foundationpose_repository",
    ("foundationpose", "top_k"): "top_k",
    (
        "foundationpose",
        "rotation_diversity_threshold_deg",
    ): "foundationpose_rotation_diversity_threshold_deg",
    (
        "foundationpose",
        "refine_iterations",
    ): "refine_iterations",
    (
        "foundationpose",
        "workers",
    ): "foundationpose_workers",
    (
        "foundationpose",
        "debug",
    ): "foundationpose_debug",
    (
        "foundationpose",
        "renderer",
        "batch_size",
    ): "renderer_batch_size",
    (
        "foundationpose",
        "renderer",
        "maximum_texture_size",
    ): "renderer_maximum_texture_size",
    (
        "scale",
        "normalization",
        "quantile_low",
    ): "normalization_quantile_low",
    (
        "scale",
        "normalization",
        "quantile_high",
    ): "normalization_quantile_high",
    (
        "scale",
        "normalization",
        "sample_count",
    ): "normalization_sample_count",
    (
        "scale",
        "normalization",
        "random_seed",
    ): "normalization_random_seed",
    (
        "scale",
        "normalization",
        "tolerance",
    ): "normalization_tolerance",
    (
        "scale",
        "axis_refinement",
        "minimum_factor",
    ): "axis_scale_minimum_factor",
    (
        "scale",
        "axis_refinement",
        "maximum_factor",
    ): "axis_scale_maximum_factor",
    (
        "scale",
        "axis_refinement",
        "grid_step_count",
    ): "axis_scale_grid_step_count",
    (
        "scale",
        "axis_refinement",
        "scale_penalty_weight",
    ): "axis_scale_penalty_weight",
    (
        "scale",
        "axis_refinement",
        "uncertainty_scale_penalty_weight",
    ): "axis_scale_uncertainty_penalty_weight",
    (
        "scale",
        "axis_refinement",
        "minimum_loss_improvement_ratio",
    ): "axis_scale_minimum_loss_improvement_ratio",
    (
        "scale",
        "axis_refinement",
        "coarse_shortlist_count",
    ): "axis_scale_coarse_shortlist_count",
    (
        "scale",
        "axis_refinement",
        "fine_factor_radius",
    ): "axis_scale_fine_factor_radius",
    (
        "scale",
        "axis_refinement",
        "fine_grid_step_count",
    ): "axis_scale_fine_grid_step_count",
    (
        "scale",
        "shared_axis_refinement",
        "enabled",
    ): "enable_shared_axis_scale_refinement",
    (
        "scale",
        "depth",
        "quantile_low",
    ): "scale_quantile_low",
    (
        "scale",
        "depth",
        "quantile_high",
    ): "scale_quantile_high",
    (
        "scale",
        "depth",
        "multipliers",
    ): "scale_multipliers",
    (
        "scale",
        "depth",
        "minimum_valid_points",
    ): "scale_minimum_valid_points",
    (
        "scale",
        "depth",
        "maximum_points",
    ): "scale_maximum_points",
    (
        "scale",
        "depth",
        "minimum_depth_m",
    ): "scale_minimum_depth_m",
    (
        "scale",
        "depth",
        "maximum_depth_m",
    ): "scale_maximum_depth_m",
    (
        "scale",
        "depth",
        "save_point_cloud",
    ): "scale_save_point_cloud",
    (
        "scale",
        "visible_refinement",
        "enabled",
    ): "visible_scale_refinement_enabled",
    (
        "scale",
        "visible_refinement",
        "reference_enabled",
    ): "visible_scale_refinement_reference_enabled",
    (
        "scale",
        "visible_refinement",
        "query_enabled",
    ): "visible_scale_refinement_query_enabled",
    (
        "scale",
        "visible_refinement",
        "minimum_loss_improvement_ratio",
    ): "visible_scale_minimum_loss_improvement_ratio",
    (
        "scale",
        "visible_refinement",
        "policy",
    ): "visible_scale_policy",
    (
        "scale",
        "visible_refinement",
        "local_scale_multipliers",
    ): "visible_scale_local_multipliers",
    (
        "scale",
        "visible_refinement",
        "mask_erosion_kernel_size",
    ): "visible_scale_mask_erosion_kernel_size",
    (
        "scale",
        "visible_refinement",
        "minimum_correspondences",
    ): "visible_scale_minimum_correspondences",
    (
        "scale",
        "visible_refinement",
        "minimum_spatial_coverage",
    ): "visible_scale_minimum_spatial_coverage",
    (
        "scale",
        "visible_refinement",
        "depth_absolute_tolerance_m",
    ): "visible_scale_depth_absolute_tolerance_m",
    (
        "scale",
        "visible_refinement",
        "depth_relative_tolerance",
    ): "visible_scale_depth_relative_tolerance",
    (
        "scale",
        "visible_refinement",
        "irls_iterations",
    ): "visible_scale_irls_iterations",
    (
        "scale",
        "visible_refinement",
        "huber_delta_m",
    ): "visible_scale_huber_delta_m",
    (
        "scale",
        "visible_refinement",
        "maximum_abs_log_correction",
    ): "visible_scale_maximum_abs_log_correction",
    (
        "scale",
        "visible_refinement",
        "coverage_grid_size",
    ): "visible_scale_coverage_grid_size",
    ("dgedi", "repository"): "dgedi_repository",
    ("dgedi", "python"): "dgedi_python",
    ("dgedi", "config"): "dgedi_config",
    ("dgedi", "mode"): "dgedi_mode",
    ("dgedi", "device"): "dgedi_device",
    ("dgedi", "sample_count"): "dgedi_sample_count",
    (
        "dgedi",
        "ransac_threshold",
    ): "dgedi_ransac_threshold",
    (
        "dgedi",
        "icp_threshold",
    ): "dgedi_icp_threshold",
    (
        "dgedi",
        "maximum_surface_depth_residual_m",
    ): "dgedi_maximum_surface_depth_residual_m",
    (
        "dgedi",
        "minimum_visible_depth_pixels",
    ): "dgedi_minimum_visible_depth_pixels",
    (
        "dgedi",
        "minimum_pair_point_count_ratio",
    ): "dgedi_minimum_pair_point_count_ratio",
    (
        "dgedi",
        "minimum_pair_diameter_ratio",
    ): "dgedi_minimum_pair_diameter_ratio",
    (
        "dgedi",
        "confidence",
        "weights",
        "correspondence",
    ): "dgedi_confidence_weight_correspondence",
    (
        "dgedi",
        "confidence",
        "weights",
        "inlier",
    ): "dgedi_confidence_weight_inlier",
    (
        "dgedi",
        "confidence",
        "weights",
        "rmse",
    ): "dgedi_confidence_weight_rmse",
    (
        "dgedi",
        "confidence",
        "weights",
        "mask",
    ): "dgedi_confidence_weight_mask",
    (
        "dgedi",
        "confidence",
        "weights",
        "depth",
    ): "dgedi_confidence_weight_depth",
    (
        "dgedi",
        "confidence",
        "weights",
        "free_space",
    ): "dgedi_confidence_weight_free_space",
    (
        "dgedi",
        "confidence",
        "target_correspondence_fraction",
    ): "dgedi_confidence_target_correspondence_fraction",
    (
        "dgedi",
        "confidence",
        "good_inlier_fitness",
    ): "dgedi_confidence_good_inlier_fitness",
    (
        "dgedi",
        "confidence",
        "maximum_normalized_rmse",
    ): "dgedi_confidence_maximum_normalized_rmse",
    (
        "dgedi",
        "confidence",
        "maximum_normalized_depth_residual",
    ): "dgedi_confidence_maximum_normalized_depth_residual",
    (
        "dgedi",
        "final_selection",
        "rotation_penalty_weight",
    ): "dgedi_selection_rotation_penalty_weight",
    (
        "dgedi",
        "final_selection",
        "uncertainty_penalty_weight",
    ): "dgedi_selection_uncertainty_penalty_weight",
    (
        "dgedi",
        "final_selection",
        "deformation_penalty_weight",
    ): "dgedi_selection_deformation_penalty_weight",
    (
        "dgedi",
        "final_selection",
        "near_tie_margin",
    ): "dgedi_selection_near_tie_margin",
    (
        "dgedi",
        "observation_validation",
        "minimum_mask_iou",
    ): "dgedi_validation_minimum_mask_iou",
    (
        "dgedi",
        "observation_validation",
        "baseline_mask_iou_drop",
    ): "dgedi_validation_baseline_mask_iou_drop",
    (
        "dgedi",
        "observation_validation",
        "maximum_depth_residual_normalized",
    ): "dgedi_validation_maximum_depth_residual_normalized",
    (
        "dgedi",
        "observation_validation",
        "baseline_depth_residual_margin",
    ): "dgedi_validation_baseline_depth_residual_margin",
    (
        "dgedi",
        "observation_validation",
        "maximum_total_loss",
    ): "dgedi_validation_maximum_total_loss",
    (
        "dgedi",
        "observation_validation",
        "baseline_total_loss_margin",
    ): "dgedi_validation_baseline_total_loss_margin",
    (
        "alignment",
        "weights",
        "mask",
    ): "alignment_weight_mask",
    (
        "alignment",
        "weights",
        "depth",
    ): "alignment_weight_depth",
    (
        "alignment",
        "weights",
        "free_space",
    ): "alignment_weight_free_space",
    (
        "alignment",
        "weights",
        "boundary",
    ): "alignment_weight_boundary",
    (
        "alignment",
        "depth_trim_quantile",
    ): "alignment_depth_trim_quantile",
    (
        "alignment",
        "minimum_depth_overlap_pixels",
    ): "alignment_minimum_depth_overlap_pixels",
    (
        "alignment",
        "free_space_absolute_tolerance_m",
    ): "alignment_free_space_absolute_tolerance_m",
    (
        "alignment",
        "free_space_relative_tolerance",
    ): "alignment_free_space_relative_tolerance",
    (
        "alignment",
        "dino",
        "depth_absolute_tolerance_m",
    ): "dino_depth_absolute_tolerance_m",
    (
        "alignment",
        "dino",
        "depth_relative_tolerance",
    ): "dino_depth_relative_tolerance",
    (
        "alignment",
        "dino",
        "minimum_matched_points",
    ): "dino_minimum_matched_points",
    (
        "alignment",
        "dino",
        "minimum_coverage",
    ): "dino_minimum_coverage",
    (
        "alignment",
        "dino",
        "coverage_weight",
    ): "dino_coverage_weight",
}


PATH_FIELDS = {
    "dataset_root",
    "output_root",
    "sam3_repository",
    "sam3_checkpoint",
    "sam3_bpe",
    "instantmesh_repository",
    "instantmesh_python",
    "instantmesh_config",
    "dinov3_repository",
    "dinov3_checkpoint",
    "foundationpose_repository",
    "dgedi_repository",
    "dgedi_python",
    "dgedi_config",
}

OPTIONAL_PATH_FIELDS = {
    "output_root",
    "instantmesh_python",
    "dinov3_checkpoint",
    "dgedi_python",
}

INTEGER_FIELDS = {
    "random_seed",
    "object_id",
    "reference_scene_id",
    "reference_image_id",
    "reference_instance_index",
    "query_scene_id",
    "query_image_id",
    "query_instance_index",
    "instantmesh_diffusion_steps",
    "instantmesh_view_count",
    "dinov3_target_long_side",
    "dinov3_maximum_surface_points",
    "dinov3_feature_chunk_size",
    "top_k",
    "refine_iterations",
    "foundationpose_workers",
    "foundationpose_debug",
    "renderer_batch_size",
    "renderer_maximum_texture_size",
    "normalization_sample_count",
    "normalization_random_seed",
    "axis_scale_grid_step_count",
    "axis_scale_coarse_shortlist_count",
    "axis_scale_fine_grid_step_count",
    "scale_minimum_valid_points",
    "scale_maximum_points",
    "alignment_minimum_depth_overlap_pixels",
    "dino_minimum_matched_points",
    "linemod_all_query_stride",
    "linemod_all_maximum_queries_per_object",
    "visible_scale_mask_erosion_kernel_size",
    "visible_scale_minimum_correspondences",
    "visible_scale_irls_iterations",
    "visible_scale_coverage_grid_size",
    "dgedi_sample_count",
    "dgedi_minimum_visible_depth_pixels",
}

OPTIONAL_INTEGER_FIELDS = {
    "query_image_id",
    "scale_maximum_points",
    "linemod_all_maximum_queries_per_object",
}

FLOAT_FIELDS = {
    "sam3_confidence_threshold",
    "instantmesh_model_scale",
    "instantmesh_render_distance",
    "foundationpose_rotation_diversity_threshold_deg",
    "normalization_quantile_low",
    "normalization_quantile_high",
    "normalization_tolerance",
    "axis_scale_minimum_factor",
    "axis_scale_maximum_factor",
    "axis_scale_penalty_weight",
    "axis_scale_uncertainty_penalty_weight",
    "axis_scale_minimum_loss_improvement_ratio",
    "axis_scale_fine_factor_radius",
    "scale_quantile_low",
    "scale_quantile_high",
    "scale_minimum_depth_m",
    "scale_maximum_depth_m",
    "alignment_weight_mask",
    "alignment_weight_depth",
    "alignment_weight_free_space",
    "alignment_weight_boundary",
    "alignment_depth_trim_quantile",
    "alignment_free_space_absolute_tolerance_m",
    "alignment_free_space_relative_tolerance",
    "dino_depth_absolute_tolerance_m",
    "dino_depth_relative_tolerance",
    "dino_minimum_coverage",
    "dino_coverage_weight",
    "visible_scale_minimum_loss_improvement_ratio",
    "visible_scale_minimum_spatial_coverage",
    "visible_scale_depth_absolute_tolerance_m",
    "visible_scale_depth_relative_tolerance",
    "visible_scale_huber_delta_m",
    "visible_scale_maximum_abs_log_correction",
    "dgedi_ransac_threshold",
    "dgedi_icp_threshold",
    "dgedi_maximum_surface_depth_residual_m",
    "dgedi_minimum_pair_point_count_ratio",
    "dgedi_minimum_pair_diameter_ratio",
    "dgedi_confidence_weight_correspondence",
    "dgedi_confidence_weight_inlier",
    "dgedi_confidence_weight_rmse",
    "dgedi_confidence_weight_mask",
    "dgedi_confidence_weight_depth",
    "dgedi_confidence_weight_free_space",
    "dgedi_confidence_target_correspondence_fraction",
    "dgedi_confidence_good_inlier_fitness",
    "dgedi_confidence_maximum_normalized_rmse",
    "dgedi_confidence_maximum_normalized_depth_residual",
    "dgedi_selection_rotation_penalty_weight",
    "dgedi_selection_uncertainty_penalty_weight",
    "dgedi_selection_deformation_penalty_weight",
    "dgedi_selection_near_tie_margin",
    "dgedi_validation_minimum_mask_iou",
    "dgedi_validation_baseline_mask_iou_drop",
    "dgedi_validation_maximum_depth_residual_normalized",
    "dgedi_validation_baseline_depth_residual_margin",
    "dgedi_validation_maximum_total_loss",
    "dgedi_validation_baseline_total_loss_margin",
}

OPTIONAL_FLOAT_FIELDS = {
    "scale_maximum_depth_m",
}

BOOLEAN_FIELDS = {
    "gt_mask_fallback_on_sam3_failure",
    "sam3_use_amp",
    "instantmesh_offline",
    "instantmesh_use_rembg",
    "instantmesh_export_texture_map",
    "instantmesh_save_video",
    "dino_enabled",
    "dinov3_use_amp",
    "scale_save_point_cloud",
    "visible_scale_refinement_enabled",
    "visible_scale_refinement_reference_enabled",
    "visible_scale_refinement_query_enabled",
    "enable_shared_axis_scale_refinement",
    "storage_results_only",
    "storage_keep_failed_intermediates",
    "resume",
    "linemod_all_continue_on_error",
}

STRING_FIELDS = {
    "device",
    "pose_path",
    "split",
    "object_name",
    "mask_type",
    "sam3_prompt",
    "sam3_device",
    "dinov3_model",
    "dinov3_save_dtype",
    "visible_scale_policy",
    "dgedi_mode",
    "dgedi_device",
}

OPTIONAL_STRING_FIELDS = {
    "sam3_device",
}

SEQUENCE_FIELDS = {
    "query_image_ids",
    "scale_multipliers",
    "visible_scale_local_multipliers",
    "linemod_all_object_ids",
}

OPTIONAL_SEQUENCE_FIELDS = {
    "query_image_ids",
}


def _walk_config(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> dict[tuple[str, ...], Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "Pipeline config root must be a mapping."
        )

    leaves: dict[tuple[str, ...], Any] = {}

    for raw_key, child in value.items():
        if not isinstance(raw_key, str):
            raise TypeError(
                "Pipeline config keys must be strings: "
                f"{raw_key!r}"
            )

        child_path = (*path, raw_key)

        if isinstance(child, Mapping):
            leaves.update(
                _walk_config(
                    child,
                    path=child_path,
                )
            )
        else:
            leaves[child_path] = child

    return leaves


def _require_int(
    value: Any,
    *,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{field_name} must be an integer: {value!r}"
        )

    return value


def _require_float(
    value: Any,
    *,
    field_name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{field_name} must be numeric: {value!r}"
        )

    return float(value)


def _normalize_value(
    *,
    destination: str,
    value: Any,
    project_root: Path,
) -> Any:
    if destination in PATH_FIELDS:
        if (
            value is None
            and destination in OPTIONAL_PATH_FIELDS
        ):
            return None

        if (
            destination in {"instantmesh_python", "dgedi_python"}
            and isinstance(value, str)
            and value.strip().lower() == "auto"
        ):
            return None

        if not isinstance(value, (str, Path)):
            raise TypeError(
                f"{destination} must be a path: {value!r}"
            )

        path = Path(value).expanduser()

        if not path.is_absolute():
            path = project_root / path

        return path.resolve()

    if destination in INTEGER_FIELDS:
        if (
            value is None
            and destination
            in OPTIONAL_INTEGER_FIELDS
        ):
            return None

        return _require_int(
            value,
            field_name=destination,
        )

    if destination in FLOAT_FIELDS:
        if (
            value is None
            and destination
            in OPTIONAL_FLOAT_FIELDS
        ):
            return None

        return _require_float(
            value,
            field_name=destination,
        )

    if destination in BOOLEAN_FIELDS:
        if not isinstance(value, bool):
            raise TypeError(
                f"{destination} must be boolean: {value!r}"
            )

        return value

    if destination in STRING_FIELDS:
        if (
            value is None
            and destination
            in OPTIONAL_STRING_FIELDS
        ):
            return None

        if not isinstance(value, str):
            raise TypeError(
                f"{destination} must be a string: {value!r}"
            )

        return value

    if destination in SEQUENCE_FIELDS:
        if (
            value is None
            and destination
            in OPTIONAL_SEQUENCE_FIELDS
        ):
            return None

        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
        ):
            raise TypeError(
                f"{destination} must be a sequence: {value!r}"
            )

        if destination in {
            "query_image_ids",
            "linemod_all_object_ids",
        }:
            return tuple(
                _require_int(
                    item,
                    field_name=destination,
                )
                for item in value
            )

        return tuple(
            _require_float(
                item,
                field_name=destination,
            )
            for item in value
        )

    raise KeyError(
        f"No value normalizer for {destination}."
    )


def load_pipeline_config_file(
    config_path: Path,
    *,
    project_root: Path,
    require_complete: bool,
) -> dict[str, Any]:
    resolved_path = (
        Path(config_path)
        .expanduser()
        .resolve()
    )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            "Pipeline config file not found: "
            f"{resolved_path}"
        )

    with resolved_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        payload = yaml.safe_load(file)

    if payload is None:
        payload = {}

    if not isinstance(payload, Mapping):
        raise TypeError(
            "Pipeline config root must be a mapping: "
            f"{resolved_path}"
        )

    version = payload.get(
        "version",
        SUPPORTED_CONFIG_VERSION,
    )

    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != SUPPORTED_CONFIG_VERSION
    ):
        raise ValueError(
            "Unsupported pipeline config version: "
            f"{version!r}. Expected "
            f"{SUPPORTED_CONFIG_VERSION}."
        )

    payload_without_version = dict(payload)
    payload_without_version.pop("version", None)
    leaves = _walk_config(
        payload_without_version
    )
    unknown_paths = sorted(
        path
        for path in leaves
        if path not in FIELD_MAP
    )

    if unknown_paths:
        formatted_paths = ", ".join(
            ".".join(path)
            for path in unknown_paths
        )
        raise ValueError(
            "Unknown pipeline config key(s): "
            f"{formatted_paths}"
        )

    result = {
        FIELD_MAP[path]: _normalize_value(
            destination=FIELD_MAP[path],
            value=value,
            project_root=project_root,
        )
        for path, value in leaves.items()
    }

    if require_complete:
        missing_destinations = sorted(
            set(FIELD_MAP.values()) - set(result)
        )

        if missing_destinations:
            raise ValueError(
                "Default pipeline config is incomplete; "
                "missing field(s): "
                + ", ".join(missing_destinations)
            )

    if (
        result.get("query_image_id") is not None
        and result.get("query_image_ids") is not None
    ):
        raise ValueError(
            "Config may set only one of "
            "data.query.image_id or "
            "data.query.image_ids."
        )

    return result
