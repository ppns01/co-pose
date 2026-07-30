from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PIPELINE_CONFIG = (
    PROJECT_ROOT / "configs" / "pipeline.yaml"
)
DEFAULT_INSTANTMESH_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "instantmesh_16gb"
    / "instant-mesh-large.yaml"
)
DEFAULT_DINOV3_REPOSITORY = (
    PROJECT_ROOT / "dinov3"
)
DINOV3_CHECKPOINT_FILENAMES = {
    "dinov3_vits16": (
        "dinov3_vits16_pretrain_lvd1689m-"
        "08c60483.pth"
    ),
    "dinov3_vits16plus": (
        "dinov3_vits16plus_pretrain_lvd1689m-"
        "4057cbaa.pth"
    ),
    "dinov3_vitb16": (
        "dinov3_vitb16_pretrain_lvd1689m-"
        "73cec8be.pth"
    ),
    "dinov3_vitl16": (
        "dinov3_vitl16_pretrain_lvd1689m-"
        "8aa4cbdd.pth"
    ),
    "dinov3_vith16plus": (
        "dinov3_vith16plus_pretrain_lvd1689m-"
        "7c1da9a5.pth"
    ),
}
DINOV3_HF_MODEL_DIRECTORIES = {
    "dinov3_vits16": (
        "dinov3-vits16-"
        "pretrain-lvd1689m"
    ),
    "dinov3_vits16plus": (
        "dinov3-vits16plus-"
        "pretrain-lvd1689m"
    ),
    "dinov3_vitb16": (
        "dinov3-vitb16-"
        "pretrain-lvd1689m"
    ),
    "dinov3_vitl16": (
        "dinov3-vitl16-"
        "pretrain-lvd1689m"
    ),
    "dinov3_vith16plus": (
        "dinov3-vith16plus-"
        "pretrain-lvd1689m"
    ),
}
LINEMOD_OBJECT_METADATA: dict[int, tuple[str, str]] = {
    1: ("ape", "brown toy"),
    2: (
        "benchvise",
        "a metallic bench vise tool with a screw handle",
    ),
    3: ("bowl", "a white bowl"),
    4: ("camera", "camera"),
    5: ("can", "white watering can"),
    6: ("cat", "pink cat figurine"),
    7: ("cup", "cup"),
    8: ("driller", "power drill"),
    9: ("duck", "a small yellow rubber duck"),
    10: ("eggbox", "plastic case"),
    11: (
        "glue",
        "a small white glue bottle with a nozzle",
    ),
    12: (
        "holepuncher",
        "blue paper hole punch",
    ),
    13: ("iron", "clothes iron"),
    14: ("lamp", "white desk lamp"),
    15: ("phone", "phone handset"),
}
POSE_PATH_CHOICES = (
    "combined",
    "self_mesh",
    "self_cross",
    "cross_mesh",
)


def _default_dinov3_weights_path(
    repository_path: Path,
    model_name: str,
) -> Path:
    weights_directory = (
        repository_path / "weights"
    )
    huggingface_directory = (
        weights_directory
        / DINOV3_HF_MODEL_DIRECTORIES[
            model_name
        ]
    )
    legacy_checkpoint = (
        weights_directory
        / DINOV3_CHECKPOINT_FILENAMES[
            model_name
        ]
    )

    if huggingface_directory.is_dir():
        return huggingface_directory

    if legacy_checkpoint.is_file():
        return legacy_checkpoint

    return huggingface_directory


def _default_instantmesh_python() -> Path:
    candidates: list[Path] = []

    conda_prefix = os.environ.get(
        "CONDA_PREFIX"
    )

    if conda_prefix:
        candidates.append(
            Path(conda_prefix)
            .expanduser()
            .resolve()
            .parent
            / "instantmesh_clean"
            / "bin"
            / "python"
        )

    candidates.extend(
        (
            (
                Path.home()
                / "miniforge3"
                / "envs"
                / "instantmesh_clean"
                / "bin"
                / "python"
            ),
            (
                Path.home()
                / "miniconda3"
                / "envs"
                / "instantmesh_clean"
                / "bin"
                / "python"
            ),
        )
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return Path(sys.executable)


def _default_bufferx_python() -> Path:
    candidates: list[Path] = []

    conda_prefix = os.environ.get(
        "CONDA_PREFIX"
    )

    if conda_prefix:
        candidates.append(
            Path(conda_prefix)
            .expanduser()
            .resolve()
            .parent
            / "bufferx"
            / "bin"
            / "python"
        )

    candidates.extend(
        (
            (
                Path.home()
                / "miniforge3"
                / "envs"
                / "bufferx"
                / "bin"
                / "python"
            ),
            (
                Path.home()
                / "miniconda3"
                / "envs"
                / "bufferx"
                / "bin"
                / "python"
            ),
        )
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return (
        Path.home()
        / "miniforge3"
        / "envs"
        / "bufferx"
        / "bin"
        / "python"
    )


@dataclass(frozen=True)
class FrameSpec:
    scene_id: int
    image_id: int
    instance_index: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "scene_id",
            "image_id",
            "instance_index",
        ):
            value = getattr(
                self,
                field_name,
            )

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer: "
                    f"{value}"
                )


@dataclass(frozen=True)
class PipelineConfig:
    source_config_path: Path
    random_seed: int
    pose_path: str
    dataset_root: Path
    split: str
    object_id: int
    object_name: str
    sam3_prompt: str
    sam3_repository: Path
    sam3_checkpoint: Path
    sam3_bpe: Path
    sam3_device: str
    sam3_use_amp: bool
    sam3_confidence_threshold: float
    reference: FrameSpec
    query: FrameSpec
    mask_type: str
    instantmesh_repository: Path
    instantmesh_python: Path
    instantmesh_config: Path
    instantmesh_offline: bool
    instantmesh_diffusion_steps: int
    instantmesh_view_count: int
    instantmesh_model_scale: float
    instantmesh_render_distance: float
    instantmesh_use_rembg: bool
    instantmesh_export_texture_map: bool
    instantmesh_save_video: bool
    foundationpose_repository: Path
    foundationpose_debug: int
    renderer_batch_size: int
    renderer_maximum_texture_size: int
    bufferx_enabled: bool
    bufferx_repository: Path
    bufferx_python: Path
    bufferx_pose_estimator: str
    bufferx_sample_count: int
    bufferx_icp_refine: bool
    dino_enabled: bool
    dinov3_repository: Path
    dinov3_checkpoint: Path
    dinov3_model: str
    dinov3_target_long_side: int
    dinov3_use_amp: bool
    dinov3_save_dtype: str
    dinov3_maximum_surface_points: int
    dinov3_feature_chunk_size: int
    output_root: Path
    device: str
    top_k: int
    refine_iterations: int
    foundationpose_workers: int
    batch_query_image_ids: tuple[int, ...] | None
    normalization_quantile_low: float
    normalization_quantile_high: float
    normalization_sample_count: int
    normalization_random_seed: int
    normalization_tolerance: float
    scale_quantile_low: float
    scale_quantile_high: float
    scale_multipliers: tuple[float, ...]
    scale_minimum_valid_points: int
    scale_maximum_points: int | None
    scale_minimum_depth_m: float
    scale_maximum_depth_m: float | None
    scale_save_point_cloud: bool
    visible_scale_refinement_enabled: bool
    visible_scale_refinement_reference_enabled: bool
    visible_scale_refinement_query_enabled: bool
    visible_scale_minimum_loss_improvement_ratio: float
    alignment_weight_mask: float
    alignment_weight_depth: float
    alignment_weight_free_space: float
    alignment_weight_boundary: float
    alignment_depth_trim_quantile: float
    alignment_minimum_depth_overlap_pixels: int
    alignment_free_space_absolute_tolerance_m: float
    alignment_free_space_relative_tolerance: float
    dino_depth_absolute_tolerance_m: float
    dino_depth_relative_tolerance: float
    dino_minimum_matched_points: int
    dino_minimum_coverage: float
    dino_coverage_weight: float
    consistency_weight_rotation: float
    consistency_weight_translation: float
    consistency_weight_scale: float
    consistency_rotation_threshold_deg: float
    consistency_translation_threshold_ratio: float
    consistency_maximum_scale_log_difference: (
        float | None
    )
    consistency_translation_normalizer_m: float | None
    selection_weight_self_alignment: float
    selection_weight_cross_alignment: float
    selection_weight_dino: float
    selection_weight_path_evidence: float
    selection_weight_consistency: float


@dataclass(frozen=True)
class GeneratedProxyState:
    view_name: str
    frame: FrameSpec
    prepared_view: Any
    mesh_result: Any
    normalization_result: Any
    scale_result: Any
    candidates: tuple[Any, ...]


@dataclass(frozen=True)
class AlignedProxyState:
    generated: GeneratedProxyState
    self_results: tuple[Any, ...]
    self_evaluation: Any
    self_alignment: Any
    selected_candidate: Any


@dataclass(frozen=True)
class PairPipelineOutcome:
    final_status: str
    summary_path: Path
    pose_path: Path | None
    visualization_path: Path | None
    visualization_error: str | None = None
    research_summary_path: Path | None = None


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    from config_loader import (
        load_pipeline_config_file,
    )

    materialized_argv = (
        list(argv)
        if argv is not None
        else None
    )

    bootstrap_parser = argparse.ArgumentParser(
        add_help=False,
    )
    bootstrap_parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=DEFAULT_PIPELINE_CONFIG,
    )
    bootstrap_args, _ = (
        bootstrap_parser.parse_known_args(
            materialized_argv
        )
    )
    selected_config_path = (
        bootstrap_args.config_path
        .expanduser()
        .resolve()
    )
    default_config_path = (
        DEFAULT_PIPELINE_CONFIG.resolve()
    )

    merged_values = load_pipeline_config_file(
        default_config_path,
        project_root=PROJECT_ROOT,
        require_complete=True,
    )

    if selected_config_path != default_config_path:
        override_values = load_pipeline_config_file(
            selected_config_path,
            project_root=PROJECT_ROOT,
            require_complete=False,
        )

        if (
            override_values.get(
                "query_image_ids"
            )
            is not None
        ):
            merged_values[
                "query_image_id"
            ] = None

        if (
            override_values.get(
                "query_image_id"
            )
            is not None
        ):
            merged_values[
                "query_image_ids"
            ] = None

        if (
            "object_id" in override_values
            and override_values["object_id"]
            != merged_values["object_id"]
        ):
            if "object_name" not in override_values:
                merged_values["object_name"] = None

            if "sam3_prompt" not in override_values:
                merged_values["sam3_prompt"] = None

        if (
            (
                "dinov3_model" in override_values
                or "dinov3_repository"
                in override_values
            )
            and "dinov3_checkpoint"
            not in override_values
        ):
            merged_values["dinov3_checkpoint"] = None

        merged_values.update(
            override_values
        )

    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description=(
            "LINEMOD Reference/Query RGB-D에서 두 proxy mesh를 "
            "생성하고 양방향 상대 pose를 선택합니다."
        )
    )

    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help=(
            "Pipeline YAML file. Values override "
            "configs/pipeline.yaml."
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "BOP LINEMOD root. 기본값: "
            "<project>/datasets"
        ),
    )

    parser.add_argument(
        "--pose-path",
        choices=POSE_PATH_CHOICES,
        help=(
            "Pose experiment path. Method-specific main "
            "files set this automatically."
        ),
    )

    parser.add_argument(
        "--split",
        help="BOP split 이름. 기본값: test",
    )

    object_group = (
        parser.add_mutually_exclusive_group()
    )
    object_group.add_argument(
        "--object-id",
        type=int,
        help="BOP LINEMOD object ID. Default: 8",
    )
    object_group.add_argument(
        "--object-ids",
        type=int,
        nargs="+",
        help=(
            "Run multiple LINEMOD objects sequentially in "
            "separate Python processes. Each object's scene "
            "ID is set to its object ID, and every object "
            "uses the same query image option(s)."
        ),
    )

    parser.add_argument(
        "--object-name",
        help=(
            "Object name. Default: driller for object 8, "
            "otherwise object_<ID>."
        ),
    )

    parser.add_argument(
        "--sam3-prompt",
        help=(
            "SAM3 text prompt. Default: 'power drill' "
            "for object 8, otherwise object name."
        ),
    )

    parser.add_argument(
        "--reference-scene-id",
        type=int,
        help="Reference scene ID. Default: 8",
    )

    parser.add_argument(
        "--reference-image-id",
        type=int,
        help="Reference image ID. Default: 0",
    )

    parser.add_argument(
        "--reference-instance-index",
        type=int,
    )

    parser.add_argument(
        "--query-scene-id",
        type=int,
        help="Query scene ID. Default: 8",
    )

    query_image_group = (
        parser.add_mutually_exclusive_group()
    )

    query_image_group.add_argument(
        "--query-image-id",
        type=int,
        help="Query image ID. Default: 1",
    )

    query_image_group.add_argument(
        "--query-image-ids",
        type=int,
        nargs="+",
        help=(
            "Multiple query image IDs. Reference preparation "
            "and self-alignment are reused once."
        ),
    )

    parser.add_argument(
        "--query-instance-index",
        type=int,
    )

    parser.add_argument(
        "--mask-type",
        choices=(
            "mask_visib",
            "mask",
        ),
    )

    parser.add_argument(
        "--instantmesh-repository",
        type=Path,
    )

    parser.add_argument(
        "--instantmesh-python",
        type=Path,
        help=(
            "InstantMesh environment Python executable. "
            "By default, auto-detects the sibling "
            "instantmesh_clean conda environment."
        ),
    )

    parser.add_argument(
        "--instantmesh-config",
        type=Path,
        help=(
            "InstantMesh YAML. Default: project 16 GB "
            "proxy config with grid_res=64."
        ),
    )

    offline_group = (
        parser.add_mutually_exclusive_group()
    )
    offline_group.add_argument(
        "--instantmesh-offline",
        action="store_true",
        help=(
            "Hugging Face 자동 다운로드를 차단합니다. "
            "모델이 캐시에 있을 때만 사용하세요."
        ),
    )
    offline_group.add_argument(
        "--instantmesh-online",
        "--no-instantmesh-offline",
        dest="instantmesh_offline",
        action="store_false",
        help=(
            "Allow InstantMesh network access when "
            "its cache is missing."
        ),
    )

    parser.add_argument(
        "--foundationpose-repository",
        type=Path,
    )

    dino_group = (
        parser.add_mutually_exclusive_group()
    )
    dino_group.add_argument(
        "--enable-dino",
        dest="dino_enabled",
        action="store_true",
        help="Enable DINOv3 appearance evaluation.",
    )
    dino_group.add_argument(
        "--disable-dino",
        dest="dino_enabled",
        action="store_false",
        help=(
            "DINOv3 양방향 appearance 평가를 "
            "비활성화합니다."
        ),
    )

    parser.add_argument(
        "--dinov3-repository",
        type=Path,
    )

    parser.add_argument(
        "--dinov3-checkpoint",
        type=Path,
        help=(
            "DINOv3 .pth 파일 또는 공식 HF 모델 "
            "디렉터리. 기본값: "
            "<dinov3-repository>/weights/"
            "<dinov3-...-pretrain-lvd1689m>"
        ),
    )

    parser.add_argument(
        "--dinov3-model",
        choices=tuple(
            DINOV3_CHECKPOINT_FILENAMES
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
    )

    parser.add_argument(
        "--device",
    )

    parser.add_argument(
        "--top-k",
        type=int,
    )

    parser.add_argument(
        "--refine-iterations",
        type=int,
    )

    parser.add_argument(
        "--foundationpose-workers",
        type=int,
        help=(
            "Independent FoundationPose process count. "
            "Default: 1 (sequential)."
        ),
    )

    visible_scale_group = (
        parser.add_mutually_exclusive_group()
    )
    visible_scale_group.add_argument(
        "--enable-visible-scale-refinement",
        dest="visible_scale_refinement_enabled",
        action="store_true",
        help=(
            "Self pose의 visible proxy/RGB-D 대응점으로 "
            "등방성 scale을 보정한 뒤 FoundationPose를 "
            "좁은 scale 범위에서 다시 실행합니다."
        ),
    )
    visible_scale_group.add_argument(
        "--disable-visible-scale-refinement",
        dest="visible_scale_refinement_enabled",
        action="store_false",
        help=(
            "Visible correspondence scale 보정을 "
            "비활성화하고 기존 coarse scale 결과를 "
            "그대로 사용합니다."
        ),
    )

    explicit_values = vars(
        parser.parse_args(materialized_argv)
    )
    explicit_values.pop(
        "config_path",
        None,
    )

    if "query_image_id" in explicit_values:
        merged_values["query_image_ids"] = None

    if "query_image_ids" in explicit_values:
        merged_values["query_image_id"] = None

    if (
        "object_id" in explicit_values
        and explicit_values["object_id"]
        != merged_values["object_id"]
    ):
        if "object_name" not in explicit_values:
            merged_values["object_name"] = None

        if "sam3_prompt" not in explicit_values:
            merged_values["sam3_prompt"] = None

    if (
        (
            "dinov3_model" in explicit_values
            or "dinov3_repository"
            in explicit_values
        )
        and "dinov3_checkpoint"
        not in explicit_values
    ):
        merged_values["dinov3_checkpoint"] = None

    merged_values.update(explicit_values)
    merged_values["config_path"] = (
        selected_config_path
    )

    return argparse.Namespace(
        **merged_values
    )


def _batch_query_output_label(
    image_ids: tuple[int, ...],
) -> str:
    joined_ids = "-".join(
        f"{image_id:06d}"
        for image_id in image_ids
    )

    if len(joined_ids) <= 96:
        return joined_ids

    digest = hashlib.sha256(
        joined_ids.encode("ascii")
    ).hexdigest()[:12]

    return (
        f"{image_ids[0]:06d}-"
        f"{image_ids[-1]:06d}-"
        f"n{len(image_ids):04d}-"
        f"{digest}"
    )


def build_config(
    args: argparse.Namespace,
) -> PipelineConfig:
    object_name = (
        args.object_name.strip()
        if args.object_name is not None
        else f"object_{args.object_id:02d}"
    )

    sam3_prompt = (
        args.sam3_prompt.strip()
        if args.sam3_prompt is not None
        else object_name.replace("_", " ")
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

    if (
        batch_query_image_ids is not None
        and len(set(batch_query_image_ids))
        != len(batch_query_image_ids)
    ):
        raise ValueError(
            "query_image_ids must not contain duplicates."
        )

    if (
        batch_query_image_ids is not None
        and any(
            image_id < 0
            for image_id in batch_query_image_ids
        )
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

    instantmesh_repository = (
        args.instantmesh_repository
        .expanduser()
        .resolve()
    )

    instantmesh_config = (
        args.instantmesh_config
        if args.instantmesh_config is not None
        else DEFAULT_INSTANTMESH_CONFIG
    )

    dinov3_repository = (
        args.dinov3_repository
        .expanduser()
        .resolve()
    )

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
            PROJECT_ROOT
            / "outputs"
            / output_name
        )
    else:
        output_root = args.output_root

    instantmesh_python = (
        args.instantmesh_python
        if args.instantmesh_python is not None
        else _default_instantmesh_python()
    )
    bufferx_python = (
        args.bufferx_python
        if args.bufferx_python is not None
        else _default_bufferx_python()
    )
    sam3_device = (
        args.sam3_device.strip()
        if args.sam3_device is not None
        else args.device.strip()
    )

    return PipelineConfig(
        source_config_path=(
            args.config_path
            .expanduser()
            .resolve()
        ),
        random_seed=args.random_seed,
        pose_path=args.pose_path,
        dataset_root=(
            args.dataset_root
            .expanduser()
            .resolve()
        ),
        split=args.split.strip(),
        object_id=args.object_id,
        object_name=object_name,
        sam3_prompt=sam3_prompt,
        sam3_repository=(
            args.sam3_repository
            .expanduser()
            .resolve()
        ),
        sam3_checkpoint=(
            args.sam3_checkpoint
            .expanduser()
            .resolve()
        ),
        sam3_bpe=(
            args.sam3_bpe
            .expanduser()
            .resolve()
        ),
        sam3_device=sam3_device,
        sam3_use_amp=args.sam3_use_amp,
        sam3_confidence_threshold=(
            args.sam3_confidence_threshold
        ),
        reference=reference,
        query=query,
        mask_type=args.mask_type,
        instantmesh_repository=(
            instantmesh_repository
        ),
        instantmesh_python=(
            instantmesh_python
            .expanduser()
            .resolve()
        ),
        instantmesh_config=(
            instantmesh_config
            .expanduser()
            .resolve()
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
            args.foundationpose_repository
            .expanduser()
            .resolve()
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
        bufferx_enabled=args.bufferx_enabled,
        bufferx_repository=(
            args.bufferx_repository
            .expanduser()
            .resolve()
        ),
        bufferx_python=(
            bufferx_python
            .expanduser()
            .resolve()
        ),
        bufferx_pose_estimator=(
            args.bufferx_pose_estimator
        ),
        bufferx_sample_count=(
            args.bufferx_sample_count
        ),
        bufferx_icp_refine=(
            args.bufferx_icp_refine
        ),
        dino_enabled=args.dino_enabled,
        dinov3_repository=dinov3_repository,
        dinov3_checkpoint=(
            dinov3_checkpoint
            .expanduser()
            .resolve()
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
            output_root
            .expanduser()
            .resolve()
        ),
        device=args.device,
        top_k=args.top_k,
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
        visible_scale_refinement_reference_enabled=(
            args
            .visible_scale_refinement_reference_enabled
        ),
        visible_scale_refinement_query_enabled=(
            args
            .visible_scale_refinement_query_enabled
        ),
        visible_scale_minimum_loss_improvement_ratio=(
            args
            .visible_scale_minimum_loss_improvement_ratio
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
            args
            .alignment_free_space_absolute_tolerance_m
        ),
        alignment_free_space_relative_tolerance=(
            args
            .alignment_free_space_relative_tolerance
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
        consistency_weight_rotation=(
            args.consistency_weight_rotation
        ),
        consistency_weight_translation=(
            args.consistency_weight_translation
        ),
        consistency_weight_scale=(
            args.consistency_weight_scale
        ),
        consistency_rotation_threshold_deg=(
            args.consistency_rotation_threshold_deg
        ),
        consistency_translation_threshold_ratio=(
            args
            .consistency_translation_threshold_ratio
        ),
        consistency_maximum_scale_log_difference=(
            args
            .consistency_maximum_scale_log_difference
        ),
        consistency_translation_normalizer_m=(
            args
            .consistency_translation_normalizer_m
        ),
        selection_weight_self_alignment=(
            args.selection_weight_self_alignment
        ),
        selection_weight_cross_alignment=(
            args.selection_weight_cross_alignment
        ),
        selection_weight_dino=(
            args.selection_weight_dino
        ),
        selection_weight_path_evidence=(
            args.selection_weight_path_evidence
        ),
        selection_weight_consistency=(
            args.selection_weight_consistency
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


def _pose_path_uses_bufferx(
    config: PipelineConfig,
) -> bool:
    return (
        config.bufferx_enabled
        and config.pose_path
        in {
            "combined",
            "self_mesh",
            "cross_mesh",
        }
    )


def _pose_path_uses_dino(
    config: PipelineConfig,
) -> bool:
    return (
        config.dino_enabled
        and config.pose_path
        in {"combined", "self_cross"}
    )


def validate_config(
    config: PipelineConfig,
) -> None:
    if config.pose_path not in POSE_PATH_CHOICES:
        raise ValueError(
            "Unsupported pose_path: "
            f"{config.pose_path}"
        )

    if (
        config.pose_path
        in {"self_mesh", "cross_mesh"}
        and not config.bufferx_enabled
    ):
        raise ValueError(
            f"{config.pose_path} requires "
            "bufferx.enabled=true."
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

    if _pose_path_uses_bufferx(config):
        _require_directory(
            config.bufferx_repository,
            "BUFFER-X repository",
        )
        _require_file(
            (
                config.bufferx_repository
                / "models"
                / "BUFFERX.py"
            ),
            "BUFFER-X model implementation",
        )
        _require_file(
            (
                config.bufferx_repository
                / "snapshot"
                / "threedmatch"
                / "Desc"
                / "best.pth"
            ),
            "BUFFER-X descriptor checkpoint",
        )
        _require_file(
            (
                config.bufferx_repository
                / "snapshot"
                / "threedmatch"
                / "Pose"
                / "best.pth"
            ),
            "BUFFER-X pose checkpoint",
        )
        _require_file(
            config.bufferx_python,
            "BUFFER-X Python",
        )

        if (
            config.bufferx_pose_estimator
            not in {"ransac", "kiss_matcher"}
        ):
            raise ValueError(
                "bufferx_pose_estimator must be "
                "'ransac' or 'kiss_matcher'."
            )

        if config.bufferx_sample_count < 2000:
            raise ValueError(
                "bufferx_sample_count must be at least "
                "2000."
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
                        "Hugging Face DINOv3 "
                        f"{file_name}"
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
                "Unsupported DINOv3 model: "
                f"{config.dinov3_model}"
            )

        if not config.device.startswith("cuda"):
            raise ValueError(
                "DINOv3 requires a CUDA device: "
                f"{config.device}"
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

    if (
        not math.isfinite(
            config.sam3_confidence_threshold
        )
        or not (
            0.0
            <= config.sam3_confidence_threshold
            < 1.0
        )
    ):
        raise ValueError(
            "sam3_confidence_threshold must be "
            "finite and in [0, 1)."
        )

    if config.instantmesh_view_count not in (4, 6):
        raise ValueError(
            "instantmesh_view_count must be 4 or 6."
        )

    if config.instantmesh_diffusion_steps < 1:
        raise ValueError(
            "instantmesh_diffusion_steps must be at least 1."
        )

    if (
        config.instantmesh_model_scale <= 0.0
        or config.instantmesh_render_distance <= 0.0
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
        or config.renderer_maximum_texture_size < 1
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

    if (
        not config.scale_multipliers
        or any(
            multiplier <= 0.0
            for multiplier in config.scale_multipliers
        )
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

    if (
        config.scale_minimum_valid_points < 1
        or (
            config.scale_maximum_points is not None
            and config.scale_maximum_points < 1
        )
        or config.scale_minimum_depth_m < 0.0
        or (
            config.scale_maximum_depth_m is not None
            and config.scale_maximum_depth_m
            <= config.scale_minimum_depth_m
        )
    ):
        raise ValueError(
            "Depth scale settings are invalid."
        )

    if (
        not 0.0
        <= config
        .visible_scale_minimum_loss_improvement_ratio
        < 1.0
    ):
        raise ValueError(
            "visible scale minimum loss improvement ratio "
            "must be in [0, 1)."
        )

    if (
        not 0.0
        < config.alignment_depth_trim_quantile
        <= 1.0
        or config
        .alignment_minimum_depth_overlap_pixels
        < 1
        or config
        .alignment_free_space_absolute_tolerance_m
        < 0.0
        or config
        .alignment_free_space_relative_tolerance
        < 0.0
    ):
        raise ValueError(
            "Alignment evidence settings are invalid."
        )

    if (
        config.dino_depth_absolute_tolerance_m
        < 0.0
        or config.dino_depth_relative_tolerance < 0.0
        or config.dino_minimum_matched_points < 1
        or not 0.0
        <= config.dino_minimum_coverage
        <= 1.0
        or config.dino_coverage_weight < 0.0
    ):
        raise ValueError(
            "DINO evidence settings are invalid."
        )

    if (
        config.consistency_maximum_scale_log_difference
        is not None
        and config
        .consistency_maximum_scale_log_difference
        < 0.0
    ):
        raise ValueError(
            "maximum_scale_log_difference must be "
            "non-negative or null."
        )

    if (
        config.consistency_translation_normalizer_m
        is not None
        and config
        .consistency_translation_normalizer_m
        <= 0.0
    ):
        raise ValueError(
            "translation_normalizer_m must be positive "
            "or null."
        )

    weight_groups = {
        "alignment": (
            config.alignment_weight_mask,
            config.alignment_weight_depth,
            config.alignment_weight_free_space,
            config.alignment_weight_boundary,
        ),
        "consistency": (
            config.consistency_weight_rotation,
            config.consistency_weight_translation,
            config.consistency_weight_scale,
        ),
        "candidate selection": (
            config.selection_weight_self_alignment,
            config.selection_weight_cross_alignment,
            config.selection_weight_dino,
        ),
        "pair selection": (
            config.selection_weight_path_evidence,
            config.selection_weight_consistency,
        ),
    }

    for group_name, values in weight_groups.items():
        if (
            any(
                not math.isfinite(value)
                or value < 0.0
                for value in values
            )
            or sum(values) <= 0.0
        ):
            raise ValueError(
                f"{group_name} weights are invalid: "
                f"{values}"
            )

    if (
        config.consistency_rotation_threshold_deg
        <= 0.0
        or config
        .consistency_translation_threshold_ratio
        <= 0.0
    ):
        raise ValueError(
            "Consistency thresholds must be positive."
        )

    if (
        isinstance(config.top_k, bool)
        or config.top_k < 1
    ):
        raise ValueError(
            "top_k must be at least 1."
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
        "dataset_root": str(config.dataset_root),
        "split": config.split,
        "object_id": config.object_id,
        "object_name": config.object_name,
        "segmentation_source": "sam3",
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
                config
                .instantmesh_export_texture_map
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
                config
                .dinov3_maximum_surface_points
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
                    config
                    .renderer_maximum_texture_size
                ),
            },
        },
        "bufferx": {
            "enabled": config.bufferx_enabled,
            "repository": str(
                config.bufferx_repository
            ),
            "python": str(
                config.bufferx_python
            ),
            "pose_estimator": (
                config.bufferx_pose_estimator
            ),
            "sample_count": (
                config.bufferx_sample_count
            ),
            "icp_refine": (
                config.bufferx_icp_refine
            ),
            "result_scope": (
                "independent_mesh_to_mesh_comparison"
            ),
        },
        "scale": {
            "normalization": {
                "quantile_low": (
                    config
                    .normalization_quantile_low
                ),
                "quantile_high": (
                    config
                    .normalization_quantile_high
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
                    config
                    .scale_minimum_valid_points
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
                    config
                    .visible_scale_refinement_enabled
                ),
                "reference_enabled": (
                    config
                    .visible_scale_refinement_reference_enabled
                ),
                "query_enabled": (
                    config
                    .visible_scale_refinement_query_enabled
                ),
                "minimum_loss_improvement_ratio": (
                    config
                    .visible_scale_minimum_loss_improvement_ratio
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
                    config
                    .alignment_weight_free_space
                ),
                "boundary": (
                    config.alignment_weight_boundary
                ),
            },
            "depth_trim_quantile": (
                config.alignment_depth_trim_quantile
            ),
            "minimum_depth_overlap_pixels": (
                config
                .alignment_minimum_depth_overlap_pixels
            ),
            "free_space_absolute_tolerance_m": (
                config
                .alignment_free_space_absolute_tolerance_m
            ),
            "free_space_relative_tolerance": (
                config
                .alignment_free_space_relative_tolerance
            ),
            "dino": {
                "depth_absolute_tolerance_m": (
                    config
                    .dino_depth_absolute_tolerance_m
                ),
                "depth_relative_tolerance": (
                    config
                    .dino_depth_relative_tolerance
                ),
                "minimum_matched_points": (
                    config
                    .dino_minimum_matched_points
                ),
                "minimum_coverage": (
                    config.dino_minimum_coverage
                ),
                "coverage_weight": (
                    config.dino_coverage_weight
                ),
            },
        },
        "consistency": {
            "weights": {
                "rotation": (
                    config
                    .consistency_weight_rotation
                ),
                "translation": (
                    config
                    .consistency_weight_translation
                ),
                "scale": (
                    config.consistency_weight_scale
                ),
            },
            "rotation_threshold_deg": (
                config
                .consistency_rotation_threshold_deg
            ),
            "translation_threshold_ratio": (
                config
                .consistency_translation_threshold_ratio
            ),
            "maximum_scale_log_difference": (
                config
                .consistency_maximum_scale_log_difference
            ),
            "translation_normalizer_m": (
                config
                .consistency_translation_normalizer_m
            ),
        },
        "selection": {
            "candidate_weights": {
                "self_alignment": (
                    config
                    .selection_weight_self_alignment
                ),
                "cross_alignment": (
                    config
                    .selection_weight_cross_alignment
                ),
                "dino": (
                    config.selection_weight_dino
                ),
            },
            "pair_weights": {
                "path_evidence": (
                    config
                    .selection_weight_path_evidence
                ),
                "consistency": (
                    config
                    .selection_weight_consistency
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
        "  config: "
        f"{config.source_config_path}"
    )
    print(f"  pose_path: {config.pose_path}")
    print(f"  dataset_root: {config.dataset_root}")
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
        "  object: "
        f"id={config.object_id}, "
        f"name={config.object_name}"
    )
    print("  segmentation_source: sam3")
    print(f"  sam3_prompt: {config.sam3_prompt}")
    print(
        "  sam3_confidence_threshold: "
        f"{config.sam3_confidence_threshold}"
    )
    print(f"  mask_type: {config.mask_type}")
    print(
        "  instantmesh_repository: "
        f"{config.instantmesh_repository}"
    )
    print(
        "  instantmesh_config: "
        f"{config.instantmesh_config}"
    )
    print(
        "  instantmesh_python: "
        f"{config.instantmesh_python}"
    )
    print(
        "  foundationpose_repository: "
        f"{config.foundationpose_repository}"
    )
    print(
        "  foundationpose_workers: "
        f"{config.foundationpose_workers}"
    )
    effective_bufferx = (
        _pose_path_uses_bufferx(config)
    )
    print(
        "  bufferx_enabled: "
        f"{config.bufferx_enabled} "
        f"(effective={effective_bufferx})"
    )
    if effective_bufferx:
        print(
            "  bufferx_repository: "
            f"{config.bufferx_repository}"
        )
        print(
            "  bufferx_python: "
            f"{config.bufferx_python}"
        )
        print(
            "  bufferx_pose_estimator: "
            f"{config.bufferx_pose_estimator}"
        )
    effective_dino = _pose_path_uses_dino(
        config
    )
    print(
        "  dino_enabled: "
        f"{config.dino_enabled} "
        f"(effective={effective_dino})"
    )
    if effective_dino:
        print(
            "  dinov3_repository: "
            f"{config.dinov3_repository}"
        )
        print(
            "  dinov3_checkpoint: "
            f"{config.dinov3_checkpoint}"
        )
        print(
            "  dinov3_model: "
            f"{config.dinov3_model}"
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


def _prepare_linemod_view(
    *,
    config: PipelineConfig,
    view_name: str,
    frame: FrameSpec,
    output_root: Path | None = None,
) -> Any:
    from dataset_io.linemod_loader import (
        load_linemod_view,
    )
    from input_selector import build_view_input
    from mask_provider import (
        generate_sam3_segmentation,
    )
    from preprocessing.mask_processing import (
        prepare_masked_view,
    )

    selected_input = build_view_input(
        dataset_root=config.dataset_root,
        split=config.split,
        scene_id=frame.scene_id,
        image_id=frame.image_id,
        object_id=config.object_id,
        instance_index=frame.instance_index,
        mask_type=config.mask_type,
    )

    loaded_view = load_linemod_view(
        dataset_root=config.dataset_root,
        view_name=view_name,
        scene_id=frame.scene_id,
        image_id=frame.image_id,
        object_name=config.object_name,
        object_id=config.object_id,
        split=config.split,
    )

    if (
        selected_input.rgb_path.resolve()
        != loaded_view.source.rgb_path.resolve()
        or selected_input.depth_path.resolve()
        != loaded_view.source.depth_path.resolve()
    ):
        raise RuntimeError(
            "Input selector and LINEMOD loader resolved "
            f"different files for {view_name}."
        )

    resolved_output_root = (
        config.output_root
        if output_root is None
        else output_root
    )

    view_output = (
        resolved_output_root
        / "views"
        / view_name
    )

    segmentation = (
        generate_sam3_segmentation(
            view=loaded_view,
            output_directory=(
                view_output
                / "segmentation"
            ),
            text_prompt=config.sam3_prompt,
            repository_path=(
                config.sam3_repository
            ),
            checkpoint_path=(
                config.sam3_checkpoint
            ),
            bpe_path=config.sam3_bpe,
            device=config.sam3_device,
            use_amp=config.sam3_use_amp,
            confidence_threshold=(
                config.sam3_confidence_threshold
            ),
        )
    )

    return prepare_masked_view(
        view=loaded_view,
        segmentation=segmentation,
        output_directory=(
            view_output
            / "prepared"
        ),
    )


def _extract_dino_inputs(
    *,
    config: PipelineConfig,
    prepared_views: Sequence[Any],
    output_roots: Sequence[Path],
) -> tuple[
    tuple[Any | None, Any | None],
    ...,
]:
    """
    관측 RGB-D별 DINOv3 dense feature와
    mask/depth 표면 feature를 생성한다.

    반환 tuple의 각 원소는
    (DINOFeatureResult, ObservedSurfaceFeatureResult)이다.
    """

    views = tuple(prepared_views)
    roots = tuple(output_roots)

    if len(views) != len(roots):
        raise ValueError(
            "prepared_views와 output_roots 개수가 "
            "다릅니다."
        )

    if not _pose_path_uses_dino(config):
        return tuple(
            (None, None)
            for _ in views
        )

    from features.dinov3_extractor import (
        DINOv3Extractor,
    )
    from features.observed_surface_features import (
        build_observed_surface_features,
    )

    print(
        "[DINOv3] Extract dense features: "
        + ", ".join(
            view.view.source.name
            for view in views
        )
    )

    dense_results: list[Any] = []

    with DINOv3Extractor(
        repository_path=(
            config.dinov3_repository
        ),
        checkpoint_path=(
            config.dinov3_checkpoint
        ),
        model_name=config.dinov3_model,
        device=config.device,
        target_long_side=(
            config.dinov3_target_long_side
        ),
        use_amp=config.dinov3_use_amp,
        save_dtype=config.dinov3_save_dtype,
    ) as extractor:
        for prepared_view, output_root in zip(
            views,
            roots,
            strict=True,
        ):
            view_name = (
                prepared_view.view.source.name
            )
            feature_root = (
                Path(output_root)
                / "features"
                / "dinov3"
                / view_name
            )

            dense_results.append(
                extractor.extract(
                    view=prepared_view.view,
                    output_directory=(
                        feature_root / "dense"
                    ),
                )
            )

    results: list[
        tuple[Any, Any]
    ] = []

    for (
        prepared_view,
        dense_result,
        output_root,
    ) in zip(
        views,
        dense_results,
        roots,
        strict=True,
    ):
        view_name = (
            prepared_view.view.source.name
        )
        feature_root = (
            Path(output_root)
            / "features"
            / "dinov3"
            / view_name
        )

        surface_result = (
            build_observed_surface_features(
                prepared_view=prepared_view,
                dino_result=dense_result,
                output_directory=(
                    feature_root / "surface"
                ),
                maximum_point_count=(
                    config
                    .dinov3_maximum_surface_points
                ),
                random_seed=config.random_seed,
                device=config.device,
                feature_chunk_size=(
                    config.dinov3_feature_chunk_size
                ),
            )
        )

        results.append(
            (
                dense_result,
                surface_result,
            )
        )

    return tuple(results)


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
                prepared_view
                .segmented_rgb_path
            ),
            output_directory=(
                output_root
                / "generated"
            ),
            mask_bool_path=(
                prepared_view
                .segmentation
                .mask_bool_path
            ),
            mask_rgb_path=(
                prepared_view
                .segmentation
                .mask_rgb_path
            ),
        )
    )

    normalization_result = (
        normalize_generated_mesh(
            mesh_result=mesh_result,
            output_directory=(
                output_root
                / "normalized"
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
            output_root
            / "scale_initialization"
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
            output_root
            / "scaled_candidates"
        ),
    )

    return (
        mesh_result,
        normalization_result,
        scale_result,
        candidates,
    )


def _candidate_by_index(
    candidates: tuple[Any, ...],
    candidate_index: int,
) -> Any:
    matches = [
        candidate
        for candidate in candidates
        if candidate.candidate_index
        == candidate_index
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one scaled mesh candidate "
            f"for index {candidate_index}; "
            f"found {len(matches)}."
        )

    return matches[0]


def _self_align_generated_states(
    *,
    config: PipelineConfig,
    generated_states: Sequence[GeneratedProxyState],
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

    jobs = tuple(
        FoundationPoseProcessJob(
            job_name=(
                f"self:{state.view_name}:"
                f"{candidate.candidate_index:02d}"
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
        next_cursor = cursor + len(state.candidates)
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
            evaluation = (
                evaluate_foundationpose_alignments(
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
                        config
                        .alignment_depth_trim_quantile
                    ),
                    min_depth_overlap_pixels=(
                        config
                        .alignment_minimum_depth_overlap_pixels
                    ),
                    free_space_absolute_tolerance_m=(
                        config
                        .alignment_free_space_absolute_tolerance_m
                    ),
                    free_space_relative_tolerance=(
                        config
                        .alignment_free_space_relative_tolerance
                    ),
                )
            )

            self_alignment = (
                select_best_self_alignment(
                    evaluation
                )
            )
            selected_candidate = (
                _candidate_by_index(
                    state.candidates,
                    self_alignment.candidate_index,
                )
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

    coarse_aligned_states = tuple(
        aligned_states
    )

    if not config.visible_scale_refinement_enabled:
        return coarse_aligned_states

    return _refine_aligned_states_visible_scale(
        config=config,
        aligned_states=coarse_aligned_states,
        output_root=output_root,
    )


def _visible_scale_enabled_for_view(
    *,
    config: PipelineConfig,
    view_name: str,
) -> bool:
    if view_name == "reference":
        return (
            config
            .visible_scale_refinement_reference_enabled
        )

    if view_name == "query":
        return (
            config
            .visible_scale_refinement_query_enabled
        )

    raise ValueError(
        f"Unsupported visible-scale view: {view_name}"
    )


def _visible_scale_loss_improved(
    *,
    coarse_loss: float | None,
    refined_loss: float | None,
    minimum_improvement_ratio: float,
) -> bool:
    if (
        coarse_loss is None
        or refined_loss is None
        or not math.isfinite(coarse_loss)
        or not math.isfinite(refined_loss)
        or coarse_loss < 0.0
        or refined_loss < 0.0
    ):
        return False

    required_maximum_loss = (
        coarse_loss
        * (1.0 - minimum_improvement_ratio)
    )
    return refined_loss <= required_maximum_loss


def _refine_aligned_states_visible_scale(
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
        for state_index, state in enabled_state_entries:
            refinement_root = (
                output_root
                / "visible_scale_refinement"
                / state.generated.view_name
            )
            estimate = (
                estimate_visible_scale_from_self_alignment(
                    prepared_view=(
                        state.generated.prepared_view
                    ),
                    self_alignment=(
                        state.self_alignment
                    ),
                    renderer=renderer,
                    output_directory=(
                        refinement_root
                        / "estimate"
                    ),
                )
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

            candidates = (
                build_visible_scale_candidates(
                    normalization_result=(
                        state
                        .generated
                        .normalization_result
                    ),
                    original_scale_result=(
                        state.generated.scale_result
                    ),
                    estimate=estimate,
                    output_directory=(
                        refinement_root
                        / "scaled_candidates"
                    ),
                )
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
        for _, state, _, candidates
        in refinement_entries
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
            "Visible scale FoundationPose result "
            "partition failed."
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
            evaluation = (
                evaluate_foundationpose_alignments(
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
                        config
                        .alignment_depth_trim_quantile
                    ),
                    min_depth_overlap_pixels=(
                        config
                        .alignment_minimum_depth_overlap_pixels
                    ),
                    free_space_absolute_tolerance_m=(
                        config
                        .alignment_free_space_absolute_tolerance_m
                    ),
                    free_space_relative_tolerance=(
                        config
                        .alignment_free_space_relative_tolerance
                    ),
                )
            )
            self_alignment = (
                select_best_self_alignment(
                    evaluation
                )
            )
            selected_candidate = (
                _candidate_by_index(
                    candidates,
                    self_alignment.candidate_index,
                )
            )

            coarse_loss = (
                state.self_alignment.alignment_loss
            )
            refined_loss = (
                self_alignment.alignment_loss
            )
            accepted = (
                _visible_scale_loss_improved(
                    coarse_loss=coarse_loss,
                    refined_loss=refined_loss,
                    minimum_improvement_ratio=(
                        config
                        .visible_scale_minimum_loss_improvement_ratio
                    ),
                )
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
                            config
                            .visible_scale_minimum_loss_improvement_ratio
                        ),
                        "required_maximum_refined_loss": (
                            (
                                coarse_loss
                                * (
                                    1.0
                                    - config
                                    .visible_scale_minimum_loss_improvement_ratio
                                )
                            )
                            if coarse_loss is not None
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


def _save_visualization_report_best_effort(
    *,
    output_root: Path,
    reference_view: Any,
    query_view: Any,
    reference_mesh_result: Any,
    query_mesh_result: Any,
    reference_self_evaluation: Any,
    query_self_evaluation: Any,
    cross_evidence: Any,
    consistency_result: Any,
    final_result: Any,
) -> tuple[Path | None, str | None]:
    try:
        from utils.pipeline_visualizer import (
            save_pipeline_visualization_report,
        )

        report_path = (
            save_pipeline_visualization_report(
                output_root=output_root,
                reference_view=reference_view,
                query_view=query_view,
                reference_mesh_result=(
                    reference_mesh_result
                ),
                query_mesh_result=query_mesh_result,
                reference_self_evaluation=(
                    reference_self_evaluation
                ),
                query_self_evaluation=(
                    query_self_evaluation
                ),
                cross_evidence=cross_evidence,
                consistency_result=(
                    consistency_result
                ),
                final_result=final_result,
            )
        )
        return report_path, None

    except Exception as error:
        error_message = (
            f"{type(error).__name__}: {error}"
        )
        print(
            "[Visualization warning] Final pose files "
            "were saved, but the report failed: "
            f"{error_message}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return None, error_message


def _run_bufferx_best_effort(
    *,
    config: PipelineConfig,
    reference_state: AlignedProxyState,
    query_state: AlignedProxyState,
    output_root: Path,
) -> Any | None:
    if not _pose_path_uses_bufferx(config):
        return None

    from pose.bufferx_runner import (
        run_bufferx_registration,
    )

    bufferx_output = (
        output_root
        / "mesh_registration"
        / "bufferx"
    )

    print(
        "[Mesh registration] BUFFER-X on selected "
        "self-aligned proxies"
    )

    try:
        result = run_bufferx_registration(
            repository_path=(
                config.bufferx_repository
            ),
            python_executable=(
                config.bufferx_python
            ),
            reference_self_alignment=(
                reference_state.self_alignment
            ),
            query_self_alignment=(
                query_state.self_alignment
            ),
            output_directory=bufferx_output,
            pose_estimator=(
                config.bufferx_pose_estimator
            ),
            sample_count=(
                config.bufferx_sample_count
            ),
            icp_refine=(
                config.bufferx_icp_refine
            ),
            random_seed=config.random_seed,
        )

    except Exception as error:
        bufferx_is_required = (
            config.pose_path
            in {"self_mesh", "cross_mesh"}
        )
        bufferx_output.mkdir(
            parents=True,
            exist_ok=True,
        )
        failure_path = (
            bufferx_output
            / "bufferx_failure.json"
        )
        failure_payload = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "selection_effect": (
                (
                    "required pose path failed"
                    if bufferx_is_required
                    else (
                        "none; existing Cross "
                        "pipeline continued"
                    )
                )
            ),
        }

        with failure_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                failure_payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "[BUFFER-X warning] Independent mesh "
            "registration failed"
            + (
                "; selected pose path cannot continue: "
                if bufferx_is_required
                else "; Cross will continue: "
            )
            + f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        print(
            f"[BUFFER-X failure] {failure_path}",
            file=sys.stderr,
        )

        if bufferx_is_required:
            raise RuntimeError(
                f"{config.pose_path} requires "
                "BUFFER-X, but registration failed."
            ) from error

        return None

    print(
        "[BUFFER-X relative pose] "
        f"{result.relative_pose_path}"
    )
    print(
        result.relative_pose_query_from_reference
    )
    return result


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
    from evaluation.research_result_logger import (
        save_pair_research_results,
    )
    from pose.bidirectional_consistency import (
        ConsistencyThresholds,
        ConsistencyWeights,
        evaluate_bidirectional_consistency,
        save_bidirectional_consistency,
    )
    from pose.alignment_scorer import (
        AlignmentScoreWeights,
    )
    from pose.cross_alignment_evaluator import (
        evaluate_bidirectional_cross_evidence,
    )
    from pose.cross_alignment_runner import (
        combine_bidirectional_cross_alignment_results,
        finalize_cross_alignment,
    )
    from pose.final_candidate_selector import (
        CandidateScoreWeights,
        PairScoreWeights,
        save_final_selection,
        select_final_candidate,
    )
    from pose.foundationpose_process_pool import (
        FoundationPoseProcessJob,
        run_foundationpose_jobs,
    )
    from pose.mesh_renderer import (
        FoundationPoseMeshRenderer,
    )
    from pose.relative_pose_builder import (
        build_bidirectional_relative_candidates,
        save_relative_pose_candidates,
    )

    timing_values = dict(timings or {})
    aligned_pair_started_at = time.perf_counter()

    if pipeline_started_at is None:
        pipeline_started_at = (
            aligned_pair_started_at
        )

    reference_view = (
        reference_state.generated.prepared_view
    )
    query_view = query_state.generated.prepared_view

    bufferx_started_at = time.perf_counter()
    bufferx_result = _run_bufferx_best_effort(
        config=config,
        reference_state=reference_state,
        query_state=query_state,
        output_root=output_root,
    )
    timing_values[
        "bufferx_registration_time_sec"
    ] = (
        time.perf_counter()
        - bufferx_started_at
        if _pose_path_uses_bufferx(config)
        else 0.0
    )

    if config.pose_path == "self_mesh":
        if bufferx_result is None:
            raise RuntimeError(
                "self_mesh requires a successful "
                "BUFFER-X registration."
            )

        from pose.independent_pose_paths import (
            save_independent_pose_path,
        )

        independent_result = (
            save_independent_pose_path(
                method="self_mesh",
                relative_pose_query_from_reference=(
                    bufferx_result
                    .relative_pose_query_from_reference
                ),
                output_directory=(
                    output_root
                    / "method_results"
                    / "self_mesh"
                ),
                composition=(
                    "T_Cq_from_Pq @ T_Pq_from_Pr "
                    "@ inv(T_Cr_from_Pr)"
                ),
                sources={
                    "reference_self_pose": (
                        reference_state
                        .self_alignment
                        .pose_camera_from_proxy
                        .tolist()
                    ),
                    "query_self_pose": (
                        query_state
                        .self_alignment
                        .pose_camera_from_proxy
                        .tolist()
                    ),
                    "bufferx_proxy_pose_path": str(
                        bufferx_result
                        .proxy_pose_path
                    ),
                    "bufferx_metadata_path": str(
                        bufferx_result
                        .metadata_path
                    ),
                },
            )
        )
        print(
            "[Independent pose path] self_mesh"
        )
        print(
            f"[Final summary] "
            f"{independent_result.summary_path}"
        )
        print(
            f"[Final pose] "
            f"{independent_result.pose_path}"
        )
        print(
            independent_result
            .relative_pose_query_from_reference
        )
        return PairPipelineOutcome(
            final_status="COMPLETED",
            summary_path=(
                independent_result.summary_path
            ),
            pose_path=independent_result.pose_path,
            visualization_path=None,
        )

    print(
        "[5/8] Selected proxies -> opposite RGB-D "
        "cross-alignment"
    )

    cross_jobs = (
        FoundationPoseProcessJob(
            job_name="cross:reference_proxy_to_query",
            candidate=(
                reference_state.selected_candidate
            ),
            prepared_view=query_view,
        ),
        FoundationPoseProcessJob(
            job_name="cross:query_proxy_to_reference",
            candidate=query_state.selected_candidate,
            prepared_view=reference_view,
        ),
    )

    cross_alignment_started_at = (
        time.perf_counter()
    )

    cross_results = run_foundationpose_jobs(
        jobs=cross_jobs,
        repository_path=(
            config.foundationpose_repository
        ),
        output_root=(
            output_root
            / "foundationpose"
            / "cross"
        ),
        top_k=config.top_k,
        refine_iterations=(
            config.refine_iterations
        ),
        device=config.device,
        worker_count=(
            config.foundationpose_workers
        ),
        debug=config.foundationpose_debug,
    )

    cross_output_directory = (
        output_root / "cross_alignment"
    )

    reference_proxy_to_query = (
        finalize_cross_alignment(
            foundationpose_result=cross_results[0],
            self_alignment=(
                reference_state.self_alignment
            ),
            scaled_mesh_candidate=(
                reference_state.selected_candidate
            ),
            target_view=query_view,
            output_directory=(
                cross_output_directory
                / "reference_proxy_to_query"
            ),
        )
    )
    query_proxy_to_reference = (
        finalize_cross_alignment(
            foundationpose_result=cross_results[1],
            self_alignment=(
                query_state.self_alignment
            ),
            scaled_mesh_candidate=(
                query_state.selected_candidate
            ),
            target_view=reference_view,
            output_directory=(
                cross_output_directory
                / "query_proxy_to_reference"
            ),
        )
    )
    cross_alignment = (
        combine_bidirectional_cross_alignment_results(
            reference_proxy_to_query=(
                reference_proxy_to_query
            ),
            query_proxy_to_reference=(
                query_proxy_to_reference
            ),
            output_directory=(
                cross_output_directory
            ),
        )
    )
    timing_values["cross_alignment_time_sec"] = (
        time.perf_counter()
        - cross_alignment_started_at
    )

    if config.pose_path == "cross_mesh":
        if bufferx_result is None:
            raise RuntimeError(
                "cross_mesh requires a successful "
                "BUFFER-X registration."
            )

        from pose.alignment_evaluator import (
            evaluate_foundationpose_alignments,
        )
        from pose.independent_pose_paths import (
            compose_cross_mesh_relative_pose,
            save_independent_pose_path,
        )

        cross_mesh_evaluation_root = (
            output_root
            / "method_results"
            / "cross_mesh"
            / "cross_alignment_evaluation"
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

        with FoundationPoseMeshRenderer(
            foundationpose_repository_path=(
                config.foundationpose_repository
            ),
            device=config.device,
            render_batch_size=(
                config.renderer_batch_size
            ),
            maximum_texture_size=(
                config
                .renderer_maximum_texture_size
            ),
        ) as renderer:
            reference_cross_evaluation = (
                evaluate_foundationpose_alignments(
                    prepared_view=query_view,
                    candidate_results=(
                        cross_alignment
                        .reference_proxy_to_query
                        .foundationpose_result,
                    ),
                    renderer=renderer,
                    output_directory=(
                        cross_mesh_evaluation_root
                        / "reference_proxy_to_query"
                    ),
                    weights=alignment_weights,
                    depth_trim_quantile=(
                        config
                        .alignment_depth_trim_quantile
                    ),
                    min_depth_overlap_pixels=(
                        config
                        .alignment_minimum_depth_overlap_pixels
                    ),
                    free_space_absolute_tolerance_m=(
                        config
                        .alignment_free_space_absolute_tolerance_m
                    ),
                    free_space_relative_tolerance=(
                        config
                        .alignment_free_space_relative_tolerance
                    ),
                )
            )
            query_cross_evaluation = (
                evaluate_foundationpose_alignments(
                    prepared_view=reference_view,
                    candidate_results=(
                        cross_alignment
                        .query_proxy_to_reference
                        .foundationpose_result,
                    ),
                    renderer=renderer,
                    output_directory=(
                        cross_mesh_evaluation_root
                        / "query_proxy_to_reference"
                    ),
                    weights=alignment_weights,
                    depth_trim_quantile=(
                        config
                        .alignment_depth_trim_quantile
                    ),
                    min_depth_overlap_pixels=(
                        config
                        .alignment_minimum_depth_overlap_pixels
                    ),
                    free_space_absolute_tolerance_m=(
                        config
                        .alignment_free_space_absolute_tolerance_m
                    ),
                    free_space_relative_tolerance=(
                        config
                        .alignment_free_space_relative_tolerance
                    ),
                )
            )

        reference_best = (
            reference_cross_evaluation.best
        )
        query_best = query_cross_evaluation.best
        relative_pose = (
            compose_cross_mesh_relative_pose(
                reference_proxy_to_query_camera=(
                    reference_best
                    .hypothesis
                    .pose_cam_from_proxy
                ),
                query_proxy_to_reference_camera=(
                    query_best
                    .hypothesis
                    .pose_cam_from_proxy
                ),
                proxy_pose_query_from_reference=(
                    bufferx_result
                    .proxy_pose_query_from_reference
                ),
            )
        )
        independent_result = (
            save_independent_pose_path(
                method="cross_mesh",
                relative_pose_query_from_reference=(
                    relative_pose
                ),
                output_directory=(
                    output_root
                    / "method_results"
                    / "cross_mesh"
                ),
                composition=(
                    "T_Cq_from_Pr @ inv(T_Pq_from_Pr) "
                    "@ inv(T_Cr_from_Pq)"
                ),
                sources={
                    "reference_proxy_to_query": {
                        "candidate_index": (
                            reference_best
                            .candidate_result
                            .candidate_index
                        ),
                        "hypothesis_rank": (
                            reference_best
                            .hypothesis
                            .rank
                        ),
                        "foundationpose_score": (
                            reference_best
                            .hypothesis
                            .score
                        ),
                        "alignment_loss": (
                            reference_best
                            .alignment_score
                            .total_loss
                        ),
                        "pose": (
                            reference_best
                            .hypothesis
                            .pose_cam_from_proxy
                            .tolist()
                        ),
                    },
                    "query_proxy_to_reference": {
                        "candidate_index": (
                            query_best
                            .candidate_result
                            .candidate_index
                        ),
                        "hypothesis_rank": (
                            query_best
                            .hypothesis
                            .rank
                        ),
                        "foundationpose_score": (
                            query_best
                            .hypothesis
                            .score
                        ),
                        "alignment_loss": (
                            query_best
                            .alignment_score
                            .total_loss
                        ),
                        "pose": (
                            query_best
                            .hypothesis
                            .pose_cam_from_proxy
                            .tolist()
                        ),
                    },
                    "bufferx_proxy_pose_path": str(
                        bufferx_result
                        .proxy_pose_path
                    ),
                    "bufferx_metadata_path": str(
                        bufferx_result
                        .metadata_path
                    ),
                },
            )
        )
        print(
            "[Independent pose path] cross_mesh"
        )
        print(
            f"[Final summary] "
            f"{independent_result.summary_path}"
        )
        print(
            f"[Final pose] "
            f"{independent_result.pose_path}"
        )
        print(
            independent_result
            .relative_pose_query_from_reference
        )
        return PairPipelineOutcome(
            final_status="COMPLETED",
            summary_path=(
                independent_result.summary_path
            ),
            pose_path=independent_result.pose_path,
            visualization_path=None,
        )

    print(
        "[6/8] Build H_query_from_reference candidates"
    )

    relative_pose_started_at = (
        time.perf_counter()
    )
    relative_candidates = (
        build_bidirectional_relative_candidates(
            reference_self=(
                reference_state.self_alignment
            ),
            query_self=query_state.self_alignment,
            reference_proxy_to_query=(
                cross_alignment
                .reference_proxy_to_query
                .foundationpose_result
            ),
            query_proxy_to_reference=(
                cross_alignment
                .query_proxy_to_reference
                .foundationpose_result
            ),
        )
    )

    save_relative_pose_candidates(
        candidate_set=relative_candidates,
        output_directory=(
            output_root
            / "relative_pose_candidates"
        ),
    )
    timing_values["relative_pose_time_sec"] = (
        time.perf_counter()
        - relative_pose_started_at
    )

    print(
        "[7/8] Bidirectional consistency + "
        "cross mask/depth evaluation"
    )

    consistency_started_at = time.perf_counter()
    consistency_result = (
        evaluate_bidirectional_consistency(
            relative_candidates,
            weights=ConsistencyWeights(
                rotation=(
                    config
                    .consistency_weight_rotation
                ),
                translation=(
                    config
                    .consistency_weight_translation
                ),
                scale=(
                    config.consistency_weight_scale
                ),
            ),
            thresholds=ConsistencyThresholds(
                rotation_deg=(
                    config
                    .consistency_rotation_threshold_deg
                ),
                translation_ratio=(
                    config
                    .consistency_translation_threshold_ratio
                ),
                maximum_scale_log_difference=(
                    config
                    .consistency_maximum_scale_log_difference
                ),
            ),
            translation_normalizer_m=(
                config
                .consistency_translation_normalizer_m
            ),
        )
    )

    save_bidirectional_consistency(
        result=consistency_result,
        output_directory=(
            output_root / "consistency"
        ),
    )
    timing_values["consistency_time_sec"] = (
        time.perf_counter()
        - consistency_started_at
    )

    cross_evidence_started_at = (
        time.perf_counter()
    )
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
        cross_evidence = (
            evaluate_bidirectional_cross_evidence(
                cross_alignment=cross_alignment,
                relative_candidates=(
                    relative_candidates
                ),
                reference_view=reference_view,
                query_view=query_view,
                renderer=renderer,
                output_directory=(
                    output_root
                    / "cross_evidence"
                ),
                enable_dino=(
                    config.dino_enabled
                ),
                reference_surface=(
                    reference_surface
                ),
                query_surface=query_surface,
                reference_dino=reference_dino,
                query_dino=query_dino,
                dino_device=config.device,
                alignment_weights=alignment_weights,
                depth_trim_quantile=(
                    config.alignment_depth_trim_quantile
                ),
                min_depth_overlap_pixels=(
                    config
                    .alignment_minimum_depth_overlap_pixels
                ),
                free_space_absolute_tolerance_m=(
                    config
                    .alignment_free_space_absolute_tolerance_m
                ),
                free_space_relative_tolerance=(
                    config
                    .alignment_free_space_relative_tolerance
                ),
                dino_depth_absolute_tolerance_m=(
                    config
                    .dino_depth_absolute_tolerance_m
                ),
                dino_depth_relative_tolerance=(
                    config
                    .dino_depth_relative_tolerance
                ),
                dino_minimum_matched_points=(
                    config.dino_minimum_matched_points
                ),
                dino_minimum_coverage=(
                    config.dino_minimum_coverage
                ),
                dino_coverage_weight=(
                    config.dino_coverage_weight
                ),
                dino_feature_chunk_size=(
                    config.dinov3_feature_chunk_size
                ),
            )
        )
    timing_values["cross_evidence_time_sec"] = (
        time.perf_counter()
        - cross_evidence_started_at
    )

    print(
        "[8/8] Select final consistent pose and proxy path"
    )

    selection_started_at = time.perf_counter()
    final_result = select_final_candidate(
        consistency_result=consistency_result,
        reference_evidence=(
            cross_evidence
            .reference_proxy
            .evidences
        ),
        query_evidence=(
            cross_evidence
            .query_proxy
            .evidences
        ),
        candidate_weights=CandidateScoreWeights(
            self_alignment=(
                config.selection_weight_self_alignment
            ),
            cross_alignment=(
                config
                .selection_weight_cross_alignment
            ),
            dino=config.selection_weight_dino,
        ),
        pair_weights=PairScoreWeights(
            path_evidence=(
                config
                .selection_weight_path_evidence
            ),
            consistency=(
                config.selection_weight_consistency
            ),
        ),
    )

    summary_path, pose_path = (
        save_final_selection(
            result=final_result,
            output_directory=(
                output_root / "final"
            ),
        )
    )
    timing_values["selection_time_sec"] = (
        time.perf_counter()
        - selection_started_at
    )

    visualization_started_at = (
        time.perf_counter()
    )
    (
        visualization_path,
        visualization_error,
    ) = _save_visualization_report_best_effort(
        output_root=output_root,
        reference_view=reference_view,
        query_view=query_view,
        reference_mesh_result=(
            reference_state
            .generated
            .mesh_result
        ),
        query_mesh_result=(
            query_state.generated.mesh_result
        ),
        reference_self_evaluation=(
            reference_state.self_evaluation
        ),
        query_self_evaluation=(
            query_state.self_evaluation
        ),
        cross_evidence=cross_evidence,
        consistency_result=consistency_result,
        final_result=final_result,
    )
    timing_values["visualization_time_sec"] = (
        time.perf_counter()
        - visualization_started_at
    )
    timing_values["generation_time_sec"] = (
        timing_values.get(
            "generation_time_ref_sec",
            0.0,
        )
        + timing_values.get(
            "generation_time_query_sec",
            0.0,
        )
    )
    timing_values["foundationpose_time_sec"] = (
        timing_values.get(
            "source_anchor_time_sec",
            0.0,
        )
        + timing_values.get(
            "cross_alignment_time_sec",
            0.0,
        )
    )
    timing_values["scoring_time_sec"] = (
        timing_values.get(
            "dino_feature_time_sec",
            0.0,
        )
        + timing_values.get(
            "consistency_time_sec",
            0.0,
        )
        + timing_values.get(
            "cross_evidence_time_sec",
            0.0,
        )
        + timing_values.get(
            "selection_time_sec",
            0.0,
        )
    )
    observed_wall_time_sec = (
        time.perf_counter()
        - pipeline_started_at
    )
    shared_reference_time_sec = (
        timing_values.get(
            "_shared_reference_time_sec",
            0.0,
        )
    )
    timing_values["total_time_sec"] = (
        observed_wall_time_sec
    )
    timing_values["shared_reference_time_sec"] = (
        shared_reference_time_sec
    )
    timing_values[
        "standalone_equivalent_time_sec"
    ] = (
        observed_wall_time_sec
        + shared_reference_time_sec
    )
    timing_values["total_time_scope"] = (
        "batch_query_excluding_shared_reference"
        if shared_reference_time_sec > 0.0
        else "single_query_end_to_end"
    )

    research_result = save_pair_research_results(
        context=research_context,
        pair_output_root=output_root,
        dataset_root=config.dataset_root,
        split=config.split,
        object_id=config.object_id,
        object_name=config.object_name,
        reference_frame=(
            reference_state.generated.frame
        ),
        query_frame=query_state.generated.frame,
        mask_type="sam3",
        segmentation_mode="text_prompt",
        segmentation_model="sam3",
        reference_prepared_view=reference_view,
        query_prepared_view=query_view,
        reference_mesh_result=(
            reference_state
            .generated
            .mesh_result
        ),
        query_mesh_result=(
            query_state.generated.mesh_result
        ),
        reference_self_evaluation=(
            reference_state.self_evaluation
        ),
        query_self_evaluation=(
            query_state.self_evaluation
        ),
        reference_self_alignment=(
            reference_state.self_alignment
        ),
        query_self_alignment=(
            query_state.self_alignment
        ),
        cross_evidence=cross_evidence,
        final_result=final_result,
        timings=timing_values,
    )
    research_summary_path = (
        output_root
        / "research"
        / "research_result_paths.json"
    )

    print(f"[Final status] {final_result.status}")
    print(f"[Final summary] {summary_path}")
    print(
        "[Research CSV] "
        f"{research_result.pair_results_path}"
    )

    if visualization_path is not None:
        print(
            f"[Visualization] {visualization_path}"
        )

    if pose_path is not None:
        print(f"[Final pose] {pose_path}")
        print(
            final_result
            .selected_relative_pose_query_from_reference
        )

    return PairPipelineOutcome(
        final_status=final_result.status,
        summary_path=summary_path,
        pose_path=pose_path,
        visualization_path=visualization_path,
        visualization_error=visualization_error,
        research_summary_path=(
            research_summary_path
        ),
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
    ) = (
        _generate_scaled_candidates(
            config=config,
            generator=generator,
            prepared_view=prepared_view,
            view_name=view_name,
            output_root=(
                output_root
                / "proxies"
                / view_name
            ),
        )
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


def run_pipeline(
    config: PipelineConfig,
) -> Path | None:
    pipeline_started_at = time.perf_counter()
    validate_config(config)
    _print_runtime_config(config)

    if config.batch_query_image_ids is not None:
        raise ValueError(
            "run_pipeline only accepts single-query configs."
        )

    from generators.instantmesh_generator import (
        InstantMeshGenerator,
    )
    from mask_provider import (
        release_sam3_processor,
    )

    config_path = _save_run_config(config)
    research_context = _initialize_research_run(
        config=config,
        config_path=config_path,
    )
    timing_values: dict[str, Any] = {}

    print(f"[Config] {config_path}")
    print("[1/8] Prepare LINEMOD RGB-D and masks")

    segmentation_started_at = (
        time.perf_counter()
    )
    try:
        reference_view = _prepare_linemod_view(
            config=config,
            view_name="reference",
            frame=config.reference,
        )
        query_view = _prepare_linemod_view(
            config=config,
            view_name="query",
            frame=config.query,
        )
    finally:
        release_sam3_processor()
    timing_values["segmentation_time_sec"] = (
        time.perf_counter()
        - segmentation_started_at
    )

    dino_started_at = time.perf_counter()
    (
        (
            reference_dino,
            reference_surface,
        ),
        (
            query_dino,
            query_surface,
        ),
    ) = _extract_dino_inputs(
        config=config,
        prepared_views=(
            reference_view,
            query_view,
        ),
        output_roots=(
            config.output_root,
            config.output_root,
        ),
    )
    timing_values["dino_feature_time_sec"] = (
        time.perf_counter()
        - dino_started_at
    )

    print(
        "[2/8] Generate InstantMesh proxies and "
        "scale candidates"
    )

    with InstantMeshGenerator(
        repository_path=(
            config.instantmesh_repository
        ),
        python_executable=(
            config.instantmesh_python
        ),
        config_path=config.instantmesh_config,
        use_rembg=(
            config.instantmesh_use_rembg
        ),
        offline=config.instantmesh_offline,
        seed=config.random_seed,
        diffusion_steps=(
            config.instantmesh_diffusion_steps
        ),
        view_count=(
            config.instantmesh_view_count
        ),
        model_scale=(
            config.instantmesh_model_scale
        ),
        render_distance=(
            config.instantmesh_render_distance
        ),
        export_texture_map=(
            config.instantmesh_export_texture_map
        ),
        save_video=(
            config.instantmesh_save_video
        ),
    ) as generator:
        generation_started_at = (
            time.perf_counter()
        )
        reference_generated = (
            _build_generated_state(
                config=config,
                generator=generator,
                view_name="reference",
                frame=config.reference,
                prepared_view=reference_view,
                output_root=config.output_root,
            )
        )
        timing_values[
            "generation_time_ref_sec"
        ] = (
            time.perf_counter()
            - generation_started_at
        )

        generation_started_at = (
            time.perf_counter()
        )
        query_generated = (
            _build_generated_state(
                config=config,
                generator=generator,
                view_name="query",
                frame=config.query,
                prepared_view=query_view,
                output_root=config.output_root,
            )
        )
        timing_values[
            "generation_time_query_sec"
        ] = (
            time.perf_counter()
            - generation_started_at
        )

    print("[3/8] Self-align proxies to their RGB-D")
    print("[4/8] Evaluate self poses with mask + depth")

    source_anchor_started_at = (
        time.perf_counter()
    )
    reference_state, query_state = (
        _self_align_generated_states(
            config=config,
            generated_states=(
                reference_generated,
                query_generated,
            ),
            output_root=config.output_root,
        )
    )
    timing_values["source_anchor_time_sec"] = (
        time.perf_counter()
        - source_anchor_started_at
    )

    outcome = _run_aligned_pair(
        config=config,
        reference_state=reference_state,
        query_state=query_state,
        reference_dino=reference_dino,
        reference_surface=reference_surface,
        query_dino=query_dino,
        query_surface=query_surface,
        output_root=config.output_root,
        research_context=research_context,
        timings=timing_values,
        pipeline_started_at=pipeline_started_at,
    )

    return outcome.pose_path


def _save_batch_summary(
    *,
    config: PipelineConfig,
    reference_record: dict[str, Any],
    query_records: Sequence[dict[str, Any]],
) -> Path:
    summary_path = (
        config.output_root / "batch_summary.json"
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = list(query_records)
    payload = {
        "mode": "multi_query",
        "reference": reference_record,
        "requested_query_image_ids": list(
            config.batch_query_image_ids or ()
        ),
        "foundationpose_workers": (
            config.foundationpose_workers
        ),
        "completed_count": sum(
            record.get("status") == "completed"
            for record in records
        ),
        "failed_count": sum(
            record.get("status") == "failed"
            for record in records
        ),
        "queries": records,
    }

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return summary_path


def run_batch_pipeline(
    config: PipelineConfig,
) -> int:
    validate_config(config)
    _print_runtime_config(config)

    if config.batch_query_image_ids is None:
        raise ValueError(
            "run_batch_pipeline requires --query-image-ids."
        )

    from generators.instantmesh_generator import (
        InstantMeshGenerator,
    )
    from mask_provider import (
        release_sam3_processor,
    )

    config_path = _save_run_config(config)
    research_context = _initialize_research_run(
        config=config,
        config_path=config_path,
    )
    reference_timings: dict[str, Any] = {}
    reference_root = (
        config.output_root / "shared_reference"
    )
    reference_record: dict[str, Any] = {
        "scene_id": config.reference.scene_id,
        "image_id": config.reference.image_id,
        "instance_index": (
            config.reference.instance_index
        ),
        "output_root": str(reference_root),
        "status": "running",
    }
    query_records: list[dict[str, Any]] = []

    summary_path = _save_batch_summary(
        config=config,
        reference_record=reference_record,
        query_records=query_records,
    )

    print(f"[Config] {config_path}")
    print(f"[Batch summary] {summary_path}")
    print(
        "[Batch] Queries run sequentially; "
        "candidate workers apply inside each query."
    )

    with InstantMeshGenerator(
        repository_path=(
            config.instantmesh_repository
        ),
        python_executable=(
            config.instantmesh_python
        ),
        config_path=config.instantmesh_config,
        use_rembg=(
            config.instantmesh_use_rembg
        ),
        offline=config.instantmesh_offline,
        seed=config.random_seed,
        diffusion_steps=(
            config.instantmesh_diffusion_steps
        ),
        view_count=(
            config.instantmesh_view_count
        ),
        model_scale=(
            config.instantmesh_model_scale
        ),
        render_distance=(
            config.instantmesh_render_distance
        ),
        export_texture_map=(
            config.instantmesh_export_texture_map
        ),
        save_video=(
            config.instantmesh_save_video
        ),
    ) as generator:
        try:
            reference_work_started_at = (
                time.perf_counter()
            )
            print(
                "[Batch reference 1/4] Prepare RGB-D and mask"
            )
            stage_started_at = time.perf_counter()
            try:
                reference_view = _prepare_linemod_view(
                    config=config,
                    view_name="reference",
                    frame=config.reference,
                    output_root=reference_root,
                )
            finally:
                release_sam3_processor()
            reference_timings[
                "segmentation_time_sec"
            ] = (
                time.perf_counter()
                - stage_started_at
            )

            stage_started_at = time.perf_counter()
            (
                (
                    reference_dino,
                    reference_surface,
                ),
            ) = _extract_dino_inputs(
                config=config,
                prepared_views=(reference_view,),
                output_roots=(reference_root,),
            )
            reference_timings[
                "dino_feature_time_sec"
            ] = (
                time.perf_counter()
                - stage_started_at
            )

            print(
                "[Batch reference 2/4] Generate proxy "
                "and scale candidates"
            )
            stage_started_at = time.perf_counter()
            reference_generated = (
                _build_generated_state(
                    config=config,
                    generator=generator,
                    view_name="reference",
                    frame=config.reference,
                    prepared_view=reference_view,
                    output_root=reference_root,
                )
            )
            reference_timings[
                "generation_time_ref_sec"
            ] = (
                time.perf_counter()
                - stage_started_at
            )

            print(
                "[Batch reference 3/4] Self-alignment"
            )
            print(
                "[Batch reference 4/4] Mask/depth evaluation"
            )
            stage_started_at = time.perf_counter()
            (reference_state,) = (
                _self_align_generated_states(
                    config=config,
                    generated_states=(
                        reference_generated,
                    ),
                    output_root=reference_root,
                )
            )
            reference_timings[
                "source_anchor_time_sec"
            ] = (
                time.perf_counter()
                - stage_started_at
            )
            reference_timings[
                "_shared_reference_time_sec"
            ] = (
                time.perf_counter()
                - reference_work_started_at
            )

        except Exception as error:
            reference_record.update(
                {
                    "status": "failed",
                    "error_type": (
                        type(error).__name__
                    ),
                    "error": str(error),
                }
            )
            _save_batch_summary(
                config=config,
                reference_record=reference_record,
                query_records=query_records,
            )
            raise

        reference_record.update(
            {
                "status": "completed",
                "selected_candidate_index": (
                    reference_state
                    .self_alignment
                    .candidate_index
                ),
                "self_evaluation_path": str(
                    reference_state
                    .self_evaluation
                    .summary_path
                ),
            }
        )
        _save_batch_summary(
            config=config,
            reference_record=reference_record,
            query_records=query_records,
        )

        query_frames = _query_frames(config)

        for query_number, query_frame in enumerate(
            query_frames,
            start=1,
        ):
            query_root = (
                config.output_root
                / "queries"
                / (
                    f"q{query_frame.scene_id:06d}_"
                    f"{query_frame.image_id:06d}_"
                    f"i{query_frame.instance_index:02d}"
                )
            )
            record: dict[str, Any] = {
                "scene_id": query_frame.scene_id,
                "image_id": query_frame.image_id,
                "instance_index": (
                    query_frame.instance_index
                ),
                "output_root": str(query_root),
                "status": "running",
            }
            query_records.append(record)
            _save_batch_summary(
                config=config,
                reference_record=reference_record,
                query_records=query_records,
            )

            print(
                "\n[Batch query "
                f"{query_number}/{len(query_frames)}] "
                f"scene={query_frame.scene_id}, "
                f"image={query_frame.image_id}"
            )
            query_pipeline_started_at = (
                time.perf_counter()
            )
            timing_values = dict(
                reference_timings
            )

            try:
                print("[1/8] Prepare query RGB-D and mask")
                stage_started_at = (
                    time.perf_counter()
                )
                try:
                    query_view = _prepare_linemod_view(
                        config=config,
                        view_name="query",
                        frame=query_frame,
                        output_root=query_root,
                    )
                finally:
                    release_sam3_processor()
                timing_values[
                    "segmentation_time_sec"
                ] = (
                    reference_timings.get(
                        "segmentation_time_sec",
                        0.0,
                    )
                    + (
                        time.perf_counter()
                        - stage_started_at
                    )
                )

                stage_started_at = (
                    time.perf_counter()
                )
                (
                    (
                        query_dino,
                        query_surface,
                    ),
                ) = _extract_dino_inputs(
                    config=config,
                    prepared_views=(query_view,),
                    output_roots=(query_root,),
                )
                timing_values[
                    "dino_feature_time_sec"
                ] = (
                    reference_timings.get(
                        "dino_feature_time_sec",
                        0.0,
                    )
                    + (
                        time.perf_counter()
                        - stage_started_at
                    )
                )

                print(
                    "[2/8] Generate query proxy and "
                    "scale candidates"
                )
                stage_started_at = (
                    time.perf_counter()
                )
                query_generated = (
                    _build_generated_state(
                        config=config,
                        generator=generator,
                        view_name="query",
                        frame=query_frame,
                        prepared_view=query_view,
                        output_root=query_root,
                    )
                )
                timing_values[
                    "generation_time_query_sec"
                ] = (
                    time.perf_counter()
                    - stage_started_at
                )

                print(
                    "[3/8] Self-align query proxies"
                )
                print(
                    "[4/8] Evaluate query self poses"
                )
                stage_started_at = (
                    time.perf_counter()
                )
                (query_state,) = (
                    _self_align_generated_states(
                        config=config,
                        generated_states=(
                            query_generated,
                        ),
                        output_root=query_root,
                    )
                )
                timing_values[
                    "source_anchor_time_sec"
                ] = (
                    reference_timings.get(
                        "source_anchor_time_sec",
                        0.0,
                    )
                    + (
                        time.perf_counter()
                        - stage_started_at
                    )
                )

                outcome = _run_aligned_pair(
                    config=config,
                    reference_state=(
                        reference_state
                    ),
                    query_state=query_state,
                    reference_dino=reference_dino,
                    reference_surface=(
                        reference_surface
                    ),
                    query_dino=query_dino,
                    query_surface=query_surface,
                    output_root=query_root,
                    research_context=(
                        research_context
                    ),
                    timings=timing_values,
                    pipeline_started_at=(
                        query_pipeline_started_at
                    ),
                )

                record.update(
                    {
                        "status": "completed",
                        "final_status": (
                            outcome.final_status
                        ),
                        "final_summary_path": str(
                            outcome.summary_path
                        ),
                        "pose_path": (
                            str(outcome.pose_path)
                            if outcome.pose_path
                            is not None
                            else None
                        ),
                        "visualization_path": str(
                            outcome.visualization_path
                        )
                        if (
                            outcome.visualization_path
                            is not None
                        )
                        else None,
                        "visualization_error": (
                            outcome.visualization_error
                        ),
                        "research_summary_path": (
                            str(
                                outcome
                                .research_summary_path
                            )
                            if (
                                outcome
                                .research_summary_path
                                is not None
                            )
                            else None
                        ),
                    }
                )

            except Exception as error:
                record.update(
                    {
                        "status": "failed",
                        "error_type": (
                            type(error).__name__
                        ),
                        "error": str(error),
                    }
                )
                print(
                    "[Batch query failed] "
                    f"image={query_frame.image_id}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                traceback.print_exc()

            finally:
                summary_path = _save_batch_summary(
                    config=config,
                    reference_record=(
                        reference_record
                    ),
                    query_records=query_records,
                )

    failed_count = sum(
        record["status"] == "failed"
        for record in query_records
    )
    completed_count = sum(
        record["status"] == "completed"
        for record in query_records
    )

    print(
        "\n[Batch complete] "
        f"completed={completed_count}, "
        f"failed={failed_count}"
    )
    print(f"[Batch summary] {summary_path}")

    return 1 if failed_count else 0


def _remove_object_ids_argument(
    argv: Sequence[str],
) -> list[str]:
    cleaned: list[str] = []
    index = 0

    while index < len(argv):
        argument = argv[index]

        if (
            argument == "--object-ids"
            or argument.startswith(
                "--object-ids="
            )
        ):
            index += 1

            while (
                index < len(argv)
                and not argv[index].startswith("-")
            ):
                index += 1

            continue

        cleaned.append(argument)
        index += 1

    return cleaned


def run_multi_object_sequence(
    *,
    argv: Sequence[str],
    object_ids: Sequence[int],
    output_root: Path | None,
) -> int:
    normalized_ids = tuple(object_ids)
    conflicting_options = tuple(
        option
        for option in (
            "--object-name",
            "--sam3-prompt",
            "--reference-scene-id",
            "--query-scene-id",
        )
        if any(
            argument == option
            or argument.startswith(
                f"{option}="
            )
            for argument in argv
        )
    )

    if conflicting_options:
        raise ValueError(
            "--object-ids cannot be combined with "
            + ", ".join(conflicting_options)
            + ". Multi-object mode derives these values "
            "for each LINEMOD object."
        )

    if len(set(normalized_ids)) != len(
        normalized_ids
    ):
        raise ValueError(
            "object_ids must not contain duplicates."
        )

    unsupported_ids = sorted(
        object_id
        for object_id in normalized_ids
        if object_id not in LINEMOD_OBJECT_METADATA
    )

    if unsupported_ids:
        raise ValueError(
            "Unsupported LINEMOD object ID(s): "
            + ", ".join(
                str(object_id)
                for object_id in unsupported_ids
            )
        )

    base_argv = _remove_object_ids_argument(argv)
    failed_ids: list[int] = []

    print(
        "[Object batch] Objects run sequentially in "
        "separate Python processes."
    )

    for sequence_index, object_id in enumerate(
        normalized_ids,
        start=1,
    ):
        object_name, sam3_prompt = (
            LINEMOD_OBJECT_METADATA[object_id]
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *base_argv,
            "--object-id",
            str(object_id),
            "--object-name",
            object_name,
            "--sam3-prompt",
            sam3_prompt,
            "--reference-scene-id",
            str(object_id),
            "--query-scene-id",
            str(object_id),
        ]

        if output_root is not None:
            command.extend(
                (
                    "--output-root",
                    str(
                        output_root
                        / f"object_{object_id:02d}"
                    ),
                )
            )

        print(
            "\n[Object batch "
            f"{sequence_index}/{len(normalized_ids)}] "
            f"object={object_id} ({object_name})"
        )
        completed = subprocess.run(
            command,
            check=False,
        )

        if completed.returncode != 0:
            failed_ids.append(object_id)
            print(
                "[Object batch warning] "
                f"object={object_id} exited with "
                f"code {completed.returncode}. "
                "Continuing with the next object.",
                file=sys.stderr,
            )

    completed_count = (
        len(normalized_ids) - len(failed_ids)
    )
    print(
        "\n[Object batch complete] "
        f"completed={completed_count}, "
        f"failed={len(failed_ids)}"
    )

    if failed_ids:
        print(
            "[Object batch failed IDs] "
            + ", ".join(
                str(object_id)
                for object_id in failed_ids
            ),
            file=sys.stderr,
        )

    return 1 if failed_ids else 0


def main(
    argv: Sequence[str] | None = None,
) -> int:
    materialized_argv = (
        list(argv)
        if argv is not None
        else sys.argv[1:]
    )
    args = parse_args(materialized_argv)
    object_ids = getattr(
        args,
        "object_ids",
        None,
    )

    if object_ids is not None:
        return run_multi_object_sequence(
            argv=materialized_argv,
            object_ids=object_ids,
            output_root=args.output_root,
        )

    config = build_config(args)

    if config.batch_query_image_ids is None:
        run_pipeline(config)
        return 0

    return run_batch_pipeline(config)


if __name__ == "__main__":
    raise SystemExit(main())
