from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from typing import Any, Literal


ViewName = Literal[
    "reference",
    "query",
]

MeshGeneratorName = Literal[
    "instantmesh",
    "trellis",
    "sam3d",
]
@dataclass(frozen=True)
class ViewInput:
    name: ViewName
    rgb_path: Path
    depth_path: Path
    intrinsics_path: Path
    object_name: str
    scene_id: int
    image_id: int
    object_id: int | None

    def __post_init__(self) -> None:
        if self.name not in ("reference", "query"):
            raise ValueError(
                f"Invalid view name: {self.name}"
            )

        object_name = self.object_name.strip()

        if not object_name:
            raise ValueError(
                "Object name must not be empty."
            )

        object.__setattr__(
            self,
            "object_name",
            object_name,
        )

        for field_name in (
            "scene_id",
            "image_id",
        ):
            field_value = getattr(
                self,
                field_name,
            )

            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer: "
                    f"{field_value}"
                )

        if (
            self.object_id is not None
            and (
                isinstance(self.object_id, bool)
                or not isinstance(self.object_id, int)
                or self.object_id < 0
            )
        ):
            raise ValueError(
                "object_id must be None or a non-negative integer: "
                f"{self.object_id}"
            )
@dataclass(frozen=True)
class LoadedView:

    source: ViewInput
    rgb: NDArray[np.uint8]
    depth_m: NDArray[np.float32]
    camera_matrix: NDArray[np.float32]
    depth_scale_to_m: float
    def __post_init__(self)->None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"Invalid RGB image shape: {self.rgb.shape}")
        if self.rgb.dtype != np.uint8:
            raise ValueError(f"Invalid RGB image dtype: {self.rgb.dtype}")
        if self.depth_m.ndim != 2:
            raise ValueError(f"Invalid depth image shape: {self.depth_m.shape}")
        if (
    not np.isfinite(self.depth_scale_to_m)
    or self.depth_scale_to_m <= 0.0
):
            raise ValueError(
        "depth_scale_to_m은 유한한 양수여야 합니다: "
        f"{self.depth_scale_to_m}"
    )
        if self.rgb.shape[:2] != self.depth_m.shape:
            raise ValueError(f"RGB and depth image shapes do not match: {self.rgb.shape} vs {self.depth_m.shape}")
        if np.any(self.depth_m < 0.0):
            raise ValueError("Depth image contains negative values")
        if not np.isfinite(self.depth_m).all():
            raise ValueError("Depth image contains non-finite values")
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(f"Invalid camera matrix shape: {self.camera_matrix.shape}")
        if self.camera_matrix.dtype != np.float32:
            raise ValueError(f"Invalid camera matrix dtype: {self.camera_matrix.dtype}")
        if not np.isfinite(self.camera_matrix).all():
            raise ValueError("Camera matrix contains non-finite values")
        fx =float(self.camera_matrix
            [0, 0] )
        fy = float(self.camera_matrix[1, 1])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError(f"Invalid focal lengths: fx={fx}, fy={fy}")
        if(not np.isfinite(self.depth_scale_to_m) or self.depth_scale_to_m <= 0.0):
            raise ValueError(f"Invalid depth scale: {self.depth_scale_to_m}")
@dataclass(frozen=True)
class SegmentationResult:
    """
    SAM3 객체 분할 결과.

    mask_bool:
        (H, W) Boolean mask.

    mask_rgb:
        (H, W, 3) uint8 mask.
        객체=255, 배경=0.
    """

    mask_bool: NDArray[np.bool_]
    mask_rgb: NDArray[np.uint8]

    mask_bool_path: Path
    mask_rgb_path: Path

    score: float | None = None
    overlay_path: Path | None = None
    metadata_path: Path | None = None
    source: str = "sam3"

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Segmentation source must not be empty.")

        if self.mask_bool.ndim != 2:
            raise ValueError(
                "mask_bool shape은 (H, W)이어야 합니다: "
                f"{self.mask_bool.shape}"
            )

        if self.mask_bool.dtype != np.bool_:
            raise TypeError(
                "mask_bool dtype은 bool이어야 합니다: "
                f"{self.mask_bool.dtype}"
            )

        if not np.any(self.mask_bool):
            raise ValueError(
                "SAM3가 생성한 mask가 비어 있습니다."
            )

        expected_rgb_shape = (
            self.mask_bool.shape[0],
            self.mask_bool.shape[1],
            3,
        )

        if self.mask_rgb.shape != expected_rgb_shape:
            raise ValueError(
                "mask_rgb shape이 올바르지 않습니다: "
                f"expected={expected_rgb_shape}, "
                f"actual={self.mask_rgb.shape}"
            )

        if self.mask_rgb.dtype != np.uint8:
            raise TypeError(
                "mask_rgb dtype은 uint8이어야 합니다: "
                f"{self.mask_rgb.dtype}"
            )

        expected_mask_rgb = np.repeat(
            (
                self.mask_bool.astype(np.uint8)
                * np.uint8(255)
            )[:, :, None],
            repeats=3,
            axis=2,
        )

        if not np.array_equal(
            self.mask_rgb,
            expected_mask_rgb,
        ):
            raise ValueError(
                "mask_rgb가 mask_bool의 0/255 표현과 "
                "일치하지 않습니다."
            )

        if not self.mask_bool_path.is_file():
            raise FileNotFoundError(
                "Boolean mask 파일이 없습니다: "
                f"{self.mask_bool_path}"
            )

        if not self.mask_rgb_path.is_file():
            raise FileNotFoundError(
                "RGB mask 파일이 없습니다: "
                f"{self.mask_rgb_path}"
            )

        if self.score is not None:
            if not np.isfinite(self.score):
                raise ValueError(
                    "Mask score가 유한하지 않습니다."
                )

            if not 0.0 <= self.score <= 1.0:
                raise ValueError(
                    "Mask score는 0과 1 사이여야 합니다: "
                    f"{self.score}"
                )

        if (
            self.overlay_path is not None
            and not self.overlay_path.is_file()
        ):
            raise FileNotFoundError(
                "Overlay 파일이 없습니다: "
                f"{self.overlay_path}"
            )

        if (
            self.metadata_path is not None
            and not self.metadata_path.is_file()
        ):
            raise FileNotFoundError(
                "Metadata 파일이 없습니다: "
                f"{self.metadata_path}"
            )

    # 구형 코드의 attribute 접근을 임시 지원한다.
    @property
    def mask(self) -> NDArray[np.bool_]:
        return self.mask_bool

    @property
    def mask_path(self) -> Path:
        return self.mask_rgb_path

@dataclass(frozen=True)
class PreparedView:
    """
    SAM3 mask를 RGB와 Depth에 적용한 결과.

    segmented_rgb:
        3D 생성기 입력용 이미지.
        객체 영역은 원본 RGB를 유지하고,
        mask 외부 RGB는 0으로 설정한다.

        shape = (H, W, 3)
        dtype = uint8

    masked_depth_m:
        scale 초기화와 FoundationPose에 사용할 depth.
        mask 외부는 0으로 설정한다.

        shape = (H, W)
        dtype = float32
        단위 = meter
    """

    view: LoadedView
    segmentation: SegmentationResult

    segmented_rgb: NDArray[np.uint8]
    masked_depth_m: NDArray[np.float32]

    segmented_rgb_path: Path
    masked_depth_path: Path

    def __post_init__(self) -> None:
        height, width = self.view.depth_m.shape
        image_shape = (height, width)

        if self.segmentation.mask_bool.shape != image_shape:
            raise ValueError(
                "Mask와 입력 영상 해상도가 다릅니다: "
                f"Mask={self.segmentation.mask_bool.shape}, "
                f"Image={image_shape}"
            )

        expected_rgb_shape = (
            height,
            width,
            3,
        )

        if self.segmented_rgb.shape != expected_rgb_shape:
            raise ValueError(
                "Segmented RGB shape이 올바르지 않습니다: "
                f"expected={expected_rgb_shape}, "
                f"actual={self.segmented_rgb.shape}"
            )

        if self.segmented_rgb.dtype != np.uint8:
            raise TypeError(
                "Segmented RGB dtype은 uint8이어야 합니다: "
                f"{self.segmented_rgb.dtype}"
            )

        outside_mask = ~self.segmentation.mask_bool

        if np.any(
            self.segmented_rgb[outside_mask] != 0
        ):
            raise ValueError(
                "Mask 외부의 segmented RGB 값은 "
                "모두 0이어야 합니다."
            )

        if self.masked_depth_m.shape != image_shape:
            raise ValueError(
                "Masked depth shape이 올바르지 않습니다: "
                f"expected={image_shape}, "
                f"actual={self.masked_depth_m.shape}"
            )

        if self.masked_depth_m.dtype != np.float32:
            raise TypeError(
                "Masked depth dtype은 float32이어야 합니다: "
                f"{self.masked_depth_m.dtype}"
            )

        if not np.isfinite(
            self.masked_depth_m
        ).all():
            raise ValueError(
                "Masked depth에 NaN 또는 Inf가 있습니다."
            )

        if np.any(
            self.masked_depth_m[outside_mask] != 0.0
        ):
            raise ValueError(
                "Mask 외부의 depth는 0이어야 합니다."
            )

@dataclass(frozen=True)
class MeshGenerationResult:
    """
    InstantMesh, TRELLIS, SAM3D의 공통 출력.
    """

    generator_name: MeshGeneratorName
    output_dir: Path
    primary_output_path: Path
    artifact_paths: tuple[Path, ...] = ()
    metadata_path: Path | None = None

    def __post_init__(self) -> None:
        supported_generators = {
            "instantmesh",
            "trellis",
            "sam3d",
        }

        if self.generator_name not in supported_generators:
            raise ValueError(
                "지원하지 않는 mesh generator입니다: "
                f"{self.generator_name}"
            )

        supported_mesh_suffixes = {
            ".glb",
            ".gltf",
            ".obj",
            ".ply",
        }

        mesh_suffix = (
            self.primary_output_path
            .suffix
            .lower()
        )

        if mesh_suffix not in supported_mesh_suffixes:
            raise ValueError(
                "지원하지 않는 mesh 파일 형식입니다: "
                f"{self.primary_output_path}\n"
                f"지원 형식: "
                f"{sorted(supported_mesh_suffixes)}"
            )

    @property
    def mesh_path(self) -> Path:
        """대표 mesh 경로의 호환용 별칭."""

        return self.primary_output_path


@dataclass(frozen=True)
class ScaleInitializationResult:
    """
    Mask 내부 depth point cloud를 이용한
    초기 등방성 scale 탐색 결과.
    """

    visible_diagonal_m: float
    scale_candidates_m: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.visible_diagonal_m)
            or self.visible_diagonal_m <= 0.0
        ):
            raise ValueError(
                "visible_diagonal_m은 "
                "유한한 양수여야 합니다: "
                f"{self.visible_diagonal_m}"
            )

        if not self.scale_candidates_m:
            raise ValueError(
                "Scale 후보가 하나도 없습니다."
            )

        for scale_candidate in self.scale_candidates_m:
            if (
                not np.isfinite(scale_candidate)
                or scale_candidate <= 0.0
            ):
                raise ValueError(
                    "Scale 후보는 유한한 양수여야 합니다: "
                    f"{scale_candidate}"
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
                    f"{field_name} must be a non-negative integer: {value}"
                )


@dataclass(frozen=True)
class PipelineConfig:
    source_config_path: Path
    random_seed: int
    pose_path: str
    storage_results_only: bool
    storage_keep_failed_intermediates: bool
    resume: bool
    linemod_all_object_ids: tuple[int, ...]
    linemod_all_query_stride: int
    linemod_all_maximum_queries_per_object: (
        int | None
    )
    linemod_all_continue_on_error: bool
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
    gt_mask_fallback_on_sam3_failure: bool
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
    foundationpose_rotation_diversity_threshold_deg: float
    refine_iterations: int
    foundationpose_workers: int
    batch_query_image_ids: tuple[int, ...] | None
    normalization_quantile_low: float
    normalization_quantile_high: float
    normalization_sample_count: int
    normalization_random_seed: int
    normalization_tolerance: float
    axis_scale_minimum_factor: float
    axis_scale_maximum_factor: float
    axis_scale_grid_step_count: int
    axis_scale_penalty_weight: float
    axis_scale_uncertainty_penalty_weight: float
    axis_scale_minimum_loss_improvement_ratio: float
    axis_scale_coarse_shortlist_count: int
    axis_scale_fine_factor_radius: float
    axis_scale_fine_grid_step_count: int
    enable_shared_axis_scale_refinement: bool
    scale_quantile_low: float
    scale_quantile_high: float
    scale_multipliers: tuple[float, ...]
    scale_minimum_valid_points: int
    scale_maximum_points: int | None
    scale_minimum_depth_m: float
    scale_maximum_depth_m: float | None
    scale_save_point_cloud: bool
    visible_scale_refinement_enabled: bool
    visible_scale_refinement_reference_enabled: (
        bool
    )
    visible_scale_refinement_query_enabled: bool
    visible_scale_minimum_loss_improvement_ratio: float
    visible_scale_policy: str
    visible_scale_local_multipliers: tuple[
        float, ...
    ]
    visible_scale_mask_erosion_kernel_size: int
    visible_scale_minimum_correspondences: int
    visible_scale_minimum_spatial_coverage: float
    visible_scale_depth_absolute_tolerance_m: (
        float
    )
    visible_scale_depth_relative_tolerance: float
    visible_scale_irls_iterations: int
    visible_scale_huber_delta_m: float
    visible_scale_maximum_abs_log_correction: (
        float
    )
    visible_scale_coverage_grid_size: int
    dgedi_repository: Path
    dgedi_python: Path
    dgedi_config: Path
    dgedi_mode: str
    dgedi_device: str
    dgedi_sample_count: int
    dgedi_ransac_threshold: float
    dgedi_icp_threshold: float
    dgedi_maximum_surface_depth_residual_m: float
    dgedi_minimum_visible_depth_pixels: int
    dgedi_minimum_pair_point_count_ratio: float
    dgedi_minimum_pair_diameter_ratio: float
    dgedi_confidence_weight_correspondence: float
    dgedi_confidence_weight_inlier: float
    dgedi_confidence_weight_rmse: float
    dgedi_confidence_weight_mask: float
    dgedi_confidence_weight_depth: float
    dgedi_confidence_weight_free_space: float
    dgedi_confidence_target_correspondence_fraction: float
    dgedi_confidence_good_inlier_fitness: float
    dgedi_confidence_maximum_normalized_rmse: float
    dgedi_confidence_maximum_normalized_depth_residual: float
    dgedi_selection_rotation_penalty_weight: float
    dgedi_selection_uncertainty_penalty_weight: float
    dgedi_selection_deformation_penalty_weight: float
    dgedi_selection_near_tie_margin: float
    dgedi_validation_minimum_mask_iou: float
    dgedi_validation_baseline_mask_iou_drop: float
    dgedi_validation_maximum_depth_residual_normalized: float
    dgedi_validation_baseline_depth_residual_margin: float
    dgedi_validation_maximum_total_loss: float
    dgedi_validation_baseline_total_loss_margin: (
        float
    )
    alignment_weight_mask: float
    alignment_weight_depth: float
    alignment_weight_free_space: float
    alignment_weight_boundary: float
    alignment_depth_trim_quantile: float
    alignment_minimum_depth_overlap_pixels: int
    alignment_free_space_absolute_tolerance_m: (
        float
    )
    alignment_free_space_relative_tolerance: float
    dino_depth_absolute_tolerance_m: float
    dino_depth_relative_tolerance: float
    dino_minimum_matched_points: int
    dino_minimum_coverage: float
    dino_coverage_weight: float


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
    pose_accepted: bool
    visualization_path: Path | None
    visualization_error: str | None = None
    research_summary_path: Path | None = None

