from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from core.types import (
    PreparedView,
    ScaleInitializationResult,
)
from pose.alignment_evaluator import (
    AlignmentEvaluationResult,
    evaluate_foundationpose_alignments,
    select_best_self_alignment,
)
from pose.alignment_scorer import AlignmentScoreWeights
from pose.foundationpose_runner import (
    FoundationPoseCandidateResult,
    FoundationPoseRunner,
)
from pose.mesh_renderer import FoundationPoseMeshRenderer
from pose.relative_pose_builder import (
    SelfAlignmentSelection,
)
from scale.mesh_normalizer import (
    MeshNormalizationResult,
)
from scale.mesh_scaler import (
    ScaledMeshCandidate,
    build_scaled_mesh_candidates,
)


DEFAULT_LOCAL_SCALE_MULTIPLIERS = (
    float(np.exp(-0.05)),
    1.00,
    float(np.exp(0.05)),
)

DEFAULT_MASK_EROSION_KERNEL_SIZE = 3
DEFAULT_MINIMUM_CORRESPONDENCES = 500
DEFAULT_MINIMUM_SPATIAL_COVERAGE = 0.20
DEFAULT_DEPTH_ABSOLUTE_TOLERANCE_M = 0.03
DEFAULT_DEPTH_RELATIVE_TOLERANCE = 0.05
DEFAULT_IRLS_ITERATIONS = 3
DEFAULT_HUBER_DELTA_M = 0.01
DEFAULT_MAXIMUM_ABS_LOG_CORRECTION = 0.25
DEFAULT_COVERAGE_GRID_SIZE = 8


@dataclass(frozen=True)
class VisibleScaleEstimate:
    """Self-view의 visible 대응점으로 계산한 scale 보정 결과."""

    view_name: str
    valid: bool
    rejection_reason: str | None

    initial_scale_m: float
    absolute_scale_m: float | None
    multiplicative_correction: float | None
    log_scale_correction: float | None

    translation_camera_from_proxy_m: (
        np.ndarray | None
    )
    correspondence_count: int
    inlier_count: int
    spatial_coverage: float

    median_residual_m: float | None
    p90_residual_m: float | None
    median_abs_object_residual_m: (
        tuple[float, float, float] | None
    )
    object_residual_covariance_m2: (
        np.ndarray | None
    )

    metadata_path: Path

    def __post_init__(self) -> None:
        if self.view_name not in (
            "reference",
            "query",
        ):
            raise ValueError(
                "지원하지 않는 visible scale view입니다: "
                f"{self.view_name}"
            )

        if (
            not np.isfinite(self.initial_scale_m)
            or self.initial_scale_m <= 0.0
        ):
            raise ValueError(
                "initial_scale_m은 유한한 양수여야 합니다."
            )

        if self.valid:
            if self.rejection_reason is not None:
                raise ValueError(
                    "유효한 scale 추정에 rejection reason이 "
                    "설정되어 있습니다."
                )

            for value_name, value in (
                ("absolute_scale_m", self.absolute_scale_m),
                (
                    "multiplicative_correction",
                    self.multiplicative_correction,
                ),
            ):
                if (
                    value is None
                    or not np.isfinite(value)
                    or value <= 0.0
                ):
                    raise ValueError(
                        f"{value_name}은 유한한 양수여야 합니다."
                    )

            if (
                self.log_scale_correction is None
                or not np.isfinite(
                    self.log_scale_correction
                )
            ):
                raise ValueError(
                    "log_scale_correction이 유효하지 않습니다."
                )

        elif not self.rejection_reason:
            raise ValueError(
                "무효한 scale 추정에는 rejection reason이 "
                "필요합니다."
            )

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                "Visible scale metadata가 없습니다: "
                f"{self.metadata_path}"
            )


def _binary_erode(
    mask: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """추가 영상처리 의존성 없이 binary erosion을 수행한다."""

    mask = np.asarray(mask, dtype=np.bool_)

    if (
        kernel_size < 1
        or kernel_size % 2 == 0
    ):
        raise ValueError(
            "Erosion kernel size는 홀수 양수여야 합니다."
        )

    if kernel_size == 1:
        return mask.copy()

    radius = kernel_size // 2
    padded = np.pad(
        mask,
        pad_width=radius,
        mode="constant",
        constant_values=False,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (kernel_size, kernel_size),
    )

    return np.ascontiguousarray(
        np.all(windows, axis=(-2, -1)),
        dtype=np.bool_,
    )


def _backproject_depth_pixels(
    *,
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """선택된 depth pixel을 camera-frame 3D point로 변환한다."""

    rows, columns = np.nonzero(valid_mask)
    depths = depth_m[rows, columns].astype(
        np.float64,
        copy=False,
    )

    pixel_homogeneous = np.stack(
        (
            columns.astype(np.float64),
            rows.astype(np.float64),
            np.ones_like(depths),
        ),
        axis=1,
    )
    inverse_camera = np.linalg.inv(
        np.asarray(
            camera_matrix,
            dtype=np.float64,
        )
    )
    rays = (
        inverse_camera
        @ pixel_homogeneous.T
    ).T

    return np.ascontiguousarray(
        rays * depths[:, None],
        dtype=np.float64,
    )


def _spatial_grid_coverage(
    *,
    correspondence_mask: np.ndarray,
    support_mask: np.ndarray,
    grid_size: int,
) -> float:
    """Mask가 차지한 grid 중 대응점이 분포한 grid 비율."""

    if grid_size < 1:
        raise ValueError(
            "Coverage grid size는 양수여야 합니다."
        )

    height, width = support_mask.shape
    supported_cells = 0
    covered_cells = 0

    for row_index in range(grid_size):
        row_start = height * row_index // grid_size
        row_end = (
            height * (row_index + 1) // grid_size
        )

        for column_index in range(grid_size):
            column_start = (
                width * column_index // grid_size
            )
            column_end = (
                width
                * (column_index + 1)
                // grid_size
            )

            support_cell = support_mask[
                row_start:row_end,
                column_start:column_end,
            ]

            if not np.any(support_cell):
                continue

            supported_cells += 1

            if np.any(
                correspondence_mask[
                    row_start:row_end,
                    column_start:column_end,
                ]
            ):
                covered_cells += 1

    if supported_cells == 0:
        return 0.0

    return float(
        covered_cells / supported_cells
    )


def estimate_scale_translation_fixed_rotation(
    *,
    proxy_points_object_unscaled: np.ndarray,
    observed_points_camera: np.ndarray,
    rotation_object_to_camera: np.ndarray,
    irls_iterations: int = DEFAULT_IRLS_ITERATIONS,
    huber_delta_m: float = DEFAULT_HUBER_DELTA_M,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    고정된 회전에서 positive isotropic scale과 translation을 추정한다.

    Returns
    -------
    scale_m, translation_m, residuals_camera_m, final_weights
    """

    proxy_points = np.asarray(
        proxy_points_object_unscaled,
        dtype=np.float64,
    )
    observed_points = np.asarray(
        observed_points_camera,
        dtype=np.float64,
    )
    rotation = np.asarray(
        rotation_object_to_camera,
        dtype=np.float64,
    )

    if (
        proxy_points.ndim != 2
        or proxy_points.shape[1] != 3
        or observed_points.shape
        != proxy_points.shape
        or proxy_points.shape[0] < 3
    ):
        raise ValueError(
            "Scale fitting point shape이 올바르지 않습니다."
        )

    if rotation.shape != (3, 3):
        raise ValueError(
            "rotation_object_to_camera shape은 "
            "(3, 3)이어야 합니다."
        )

    if (
        irls_iterations < 1
        or not np.isfinite(huber_delta_m)
        or huber_delta_m <= 0.0
    ):
        raise ValueError(
            "IRLS 설정이 올바르지 않습니다."
        )

    if (
        not np.isfinite(proxy_points).all()
        or not np.isfinite(observed_points).all()
        or not np.isfinite(rotation).all()
    ):
        raise ValueError(
            "Scale fitting 입력에 NaN 또는 Inf가 있습니다."
        )

    weights = np.ones(
        proxy_points.shape[0],
        dtype=np.float64,
    )

    def fit_once(
        current_weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        weight_sum = float(
            np.sum(current_weights)
        )

        if weight_sum <= 0.0:
            raise ValueError(
                "Scale fitting weight 합이 0입니다."
            )

        proxy_center = np.sum(
            current_weights[:, None]
            * proxy_points,
            axis=0,
        ) / weight_sum
        observed_center = np.sum(
            current_weights[:, None]
            * observed_points,
            axis=0,
        ) / weight_sum

        centered_proxy = (
            proxy_points - proxy_center
        )
        centered_observed = (
            observed_points - observed_center
        )
        rotated_proxy = (
            rotation @ centered_proxy.T
        ).T

        denominator = float(
            np.sum(
                current_weights
                * np.sum(
                    rotated_proxy * rotated_proxy,
                    axis=1,
                )
            )
        )

        if denominator <= np.finfo(np.float64).eps:
            raise ValueError(
                "Scale fitting point 분산이 부족합니다."
            )

        numerator = float(
            np.sum(
                current_weights
                * np.sum(
                    centered_observed
                    * rotated_proxy,
                    axis=1,
                )
            )
        )
        scale_m = numerator / denominator

        if (
            not np.isfinite(scale_m)
            or scale_m <= 0.0
        ):
            raise ValueError(
                "추정 scale이 유한한 양수가 아닙니다."
            )

        translation_m = (
            observed_center
            - scale_m
            * (rotation @ proxy_center)
        )

        return scale_m, translation_m

    for _ in range(irls_iterations):
        scale_m, translation_m = fit_once(
            weights
        )
        predicted = (
            scale_m
            * (rotation @ proxy_points.T).T
            + translation_m
        )
        residual_norms = np.linalg.norm(
            observed_points - predicted,
            axis=1,
        )
        robust_weights = np.ones_like(
            residual_norms,
            dtype=np.float64,
        )
        outside = residual_norms > huber_delta_m
        robust_weights[outside] = (
            huber_delta_m
            / residual_norms[outside]
        )
        weights = robust_weights

    scale_m, translation_m = fit_once(weights)
    predicted = (
        scale_m
        * (rotation @ proxy_points.T).T
        + translation_m
    )
    residuals = observed_points - predicted

    return (
        float(scale_m),
        np.ascontiguousarray(
            translation_m,
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            residuals,
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            weights,
            dtype=np.float64,
        ),
    )


def _save_visible_scale_estimate(
    *,
    output_directory: Path,
    metadata: dict[str, object],
) -> Path:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    metadata_path = (
        output_directory
        / "visible_scale_estimate.json"
    )

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return metadata_path.resolve()


def estimate_visible_scale_from_self_alignment(
    *,
    prepared_view: PreparedView,
    self_alignment: SelfAlignmentSelection,
    renderer: FoundationPoseMeshRenderer,
    output_directory: Path,
    mask_erosion_kernel_size: int = (
        DEFAULT_MASK_EROSION_KERNEL_SIZE
    ),
    minimum_correspondences: int = (
        DEFAULT_MINIMUM_CORRESPONDENCES
    ),
    minimum_spatial_coverage: float = (
        DEFAULT_MINIMUM_SPATIAL_COVERAGE
    ),
    depth_absolute_tolerance_m: float = (
        DEFAULT_DEPTH_ABSOLUTE_TOLERANCE_M
    ),
    depth_relative_tolerance: float = (
        DEFAULT_DEPTH_RELATIVE_TOLERANCE
    ),
    irls_iterations: int = DEFAULT_IRLS_ITERATIONS,
    huber_delta_m: float = DEFAULT_HUBER_DELTA_M,
    maximum_abs_log_correction: float = (
        DEFAULT_MAXIMUM_ABS_LOG_CORRECTION
    ),
    coverage_grid_size: int = (
        DEFAULT_COVERAGE_GRID_SIZE
    ),
) -> VisibleScaleEstimate:
    """선택된 self pose에서 visible scale을 로컬 재추정한다."""

    view_name = prepared_view.view.source.name
    initial_scale_m = float(
        self_alignment.scale_m
    )
    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    def rejected(
        reason: str,
        *,
        correspondence_count: int = 0,
        spatial_coverage: float = 0.0,
    ) -> VisibleScaleEstimate:
        metadata_path = _save_visible_scale_estimate(
            output_directory=output_directory,
            metadata={
                "view_name": view_name,
                "valid": False,
                "rejection_reason": reason,
                "initial_scale_m": initial_scale_m,
                "correspondence_count": (
                    correspondence_count
                ),
                "spatial_coverage": (
                    spatial_coverage
                ),
                "scale_convention": (
                    "absolute metric scale for the normalized "
                    "unit-diagonal proxy"
                ),
            },
        )

        return VisibleScaleEstimate(
            view_name=view_name,
            valid=False,
            rejection_reason=reason,
            initial_scale_m=initial_scale_m,
            absolute_scale_m=None,
            multiplicative_correction=None,
            log_scale_correction=None,
            translation_camera_from_proxy_m=None,
            correspondence_count=correspondence_count,
            inlier_count=0,
            spatial_coverage=spatial_coverage,
            median_residual_m=None,
            p90_residual_m=None,
            median_abs_object_residual_m=None,
            object_residual_covariance_m2=None,
            metadata_path=metadata_path,
        )

    if minimum_correspondences < 3:
        raise ValueError(
            "minimum_correspondences는 3 이상이어야 합니다."
        )

    if not 0.0 <= minimum_spatial_coverage <= 1.0:
        raise ValueError(
            "minimum_spatial_coverage 범위가 잘못됐습니다."
        )

    pose = np.asarray(
        self_alignment.pose_camera_from_proxy,
        dtype=np.float64,
    )
    rotation = pose[:3, :3]
    initial_translation = pose[:3, 3]

    render_result = renderer.render(
        mesh_path=self_alignment.scaled_mesh_path,
        poses_camera_from_proxy=pose[None, ...],
        camera_matrix=(
            prepared_view.view.camera_matrix
        ),
        image_height=(
            prepared_view.view.rgb.shape[0]
        ),
        image_width=(
            prepared_view.view.rgb.shape[1]
        ),
        output_directory=None,
    )

    observed_depth = np.asarray(
        prepared_view.masked_depth_m,
        dtype=np.float64,
    )
    rendered_depth = np.asarray(
        render_result.rendered_depth_m[0],
        dtype=np.float64,
    )
    eroded_mask = _binary_erode(
        prepared_view.segmentation.mask_bool,
        mask_erosion_kernel_size,
    )

    valid = (
        eroded_mask
        & (observed_depth > 0.0)
        & np.isfinite(observed_depth)
        & (rendered_depth > 0.0)
        & np.isfinite(rendered_depth)
    )
    depth_difference = np.abs(
        observed_depth - rendered_depth
    )
    depth_tolerance = (
        depth_absolute_tolerance_m
        + depth_relative_tolerance
        * observed_depth
    )
    valid &= depth_difference <= depth_tolerance

    correspondence_count = int(
        np.count_nonzero(valid)
    )
    spatial_coverage = _spatial_grid_coverage(
        correspondence_mask=valid,
        support_mask=eroded_mask,
        grid_size=coverage_grid_size,
    )

    if correspondence_count < minimum_correspondences:
        return rejected(
            "insufficient_correspondences",
            correspondence_count=correspondence_count,
            spatial_coverage=spatial_coverage,
        )

    if spatial_coverage < minimum_spatial_coverage:
        return rejected(
            "insufficient_spatial_coverage",
            correspondence_count=correspondence_count,
            spatial_coverage=spatial_coverage,
        )

    observed_points = _backproject_depth_pixels(
        depth_m=observed_depth,
        valid_mask=valid,
        camera_matrix=(
            prepared_view.view.camera_matrix
        ),
    )
    rendered_points = _backproject_depth_pixels(
        depth_m=rendered_depth,
        valid_mask=valid,
        camera_matrix=(
            prepared_view.view.camera_matrix
        ),
    )
    scaled_proxy_points = (
        rotation.T
        @ (
            rendered_points
            - initial_translation
        ).T
    ).T
    proxy_points_unscaled = (
        scaled_proxy_points / initial_scale_m
    )

    try:
        (
            absolute_scale_m,
            translation_m,
            residuals_camera,
            final_weights,
        ) = estimate_scale_translation_fixed_rotation(
            proxy_points_object_unscaled=(
                proxy_points_unscaled
            ),
            observed_points_camera=observed_points,
            rotation_object_to_camera=rotation,
            irls_iterations=irls_iterations,
            huber_delta_m=huber_delta_m,
        )
    except ValueError as error:
        return rejected(
            f"scale_fit_failed:{error}",
            correspondence_count=correspondence_count,
            spatial_coverage=spatial_coverage,
        )

    multiplier = (
        absolute_scale_m / initial_scale_m
    )
    log_correction = float(np.log(multiplier))

    if (
        not np.isfinite(log_correction)
        or abs(log_correction)
        > maximum_abs_log_correction
    ):
        return rejected(
            "log_scale_correction_out_of_range",
            correspondence_count=correspondence_count,
            spatial_coverage=spatial_coverage,
        )

    residual_norms = np.linalg.norm(
        residuals_camera,
        axis=1,
    )
    residual_median = float(
        np.median(residual_norms)
    )
    residual_mad = float(
        np.median(
            np.abs(
                residual_norms - residual_median
            )
        )
    )
    inlier_threshold_m = max(
        2.0 * huber_delta_m,
        residual_median
        + 3.0 * 1.4826 * residual_mad,
    )
    inlier_mask = (
        residual_norms <= inlier_threshold_m
    )
    inlier_count = int(
        np.count_nonzero(inlier_mask)
    )

    residuals_object = (
        residuals_camera @ rotation
    )
    inlier_object_residuals = (
        residuals_object[inlier_mask]
    )

    if inlier_object_residuals.shape[0] < 3:
        return rejected(
            "insufficient_inliers",
            correspondence_count=correspondence_count,
            spatial_coverage=spatial_coverage,
        )

    median_abs_object = tuple(
        float(value)
        for value in np.median(
            np.abs(inlier_object_residuals),
            axis=0,
        )
    )
    covariance = np.cov(
        inlier_object_residuals,
        rowvar=False,
        aweights=final_weights[inlier_mask],
    )
    p90_residual_m = float(
        np.quantile(
            residual_norms[inlier_mask],
            0.90,
        )
    )

    metadata = {
        "view_name": view_name,
        "valid": True,
        "rejection_reason": None,
        "scale_convention": (
            "absolute metric scale for the normalized "
            "unit-diagonal proxy"
        ),
        "initial_scale_m": initial_scale_m,
        "absolute_scale_m": absolute_scale_m,
        "multiplicative_correction": multiplier,
        "log_scale_correction": log_correction,
        "translation_camera_from_proxy_m": (
            translation_m.tolist()
        ),
        "correspondence_count": correspondence_count,
        "inlier_count": inlier_count,
        "spatial_coverage": spatial_coverage,
        "median_residual_m": residual_median,
        "p90_residual_m": p90_residual_m,
        "median_abs_object_residual_m": list(
            median_abs_object
        ),
        "object_residual_covariance_m2": (
            covariance.tolist()
        ),
        "settings": {
            "mask_erosion_kernel_size": (
                mask_erosion_kernel_size
            ),
            "minimum_correspondences": (
                minimum_correspondences
            ),
            "minimum_spatial_coverage": (
                minimum_spatial_coverage
            ),
            "depth_absolute_tolerance_m": (
                depth_absolute_tolerance_m
            ),
            "depth_relative_tolerance": (
                depth_relative_tolerance
            ),
            "irls_iterations": irls_iterations,
            "huber_delta_m": huber_delta_m,
            "maximum_abs_log_correction": (
                maximum_abs_log_correction
            ),
        },
    }
    metadata_path = _save_visible_scale_estimate(
        output_directory=output_directory,
        metadata=metadata,
    )

    return VisibleScaleEstimate(
        view_name=view_name,
        valid=True,
        rejection_reason=None,
        initial_scale_m=initial_scale_m,
        absolute_scale_m=absolute_scale_m,
        multiplicative_correction=multiplier,
        log_scale_correction=log_correction,
        translation_camera_from_proxy_m=translation_m,
        correspondence_count=correspondence_count,
        inlier_count=inlier_count,
        spatial_coverage=spatial_coverage,
        median_residual_m=residual_median,
        p90_residual_m=p90_residual_m,
        median_abs_object_residual_m=(
            median_abs_object
        ),
        object_residual_covariance_m2=(
            np.ascontiguousarray(
                covariance,
                dtype=np.float64,
            )
        ),
        metadata_path=metadata_path,
    )


def build_visible_scale_candidates(
    *,
    normalization_result: MeshNormalizationResult,
    original_scale_result: ScaleInitializationResult,
    estimate: VisibleScaleEstimate,
    output_directory: Path,
    local_scale_multipliers: Sequence[float] = (
        DEFAULT_LOCAL_SCALE_MULTIPLIERS
    ),
) -> tuple[ScaledMeshCandidate, ...]:
    """유효한 visible scale 주변의 좁은 mesh 후보를 생성한다."""

    if (
        not estimate.valid
        or estimate.absolute_scale_m is None
    ):
        raise ValueError(
            "유효하지 않은 visible scale estimate로 "
            "후보를 생성할 수 없습니다."
        )

    multipliers = _validate_local_multipliers(
        local_scale_multipliers
    )
    local_scale_result = _build_local_scale_result(
        coarse_scale_m=estimate.absolute_scale_m,
        visible_diagonal_m=(
            original_scale_result.visible_diagonal_m
        ),
        multipliers=multipliers,
    )

    return build_scaled_mesh_candidates(
        normalization_result=normalization_result,
        scale_result=local_scale_result,
        output_directory=output_directory,
    )


@dataclass(frozen=True)
class LocalScaleRefinementResult:
    """
    Coarse self scale 주변의 local refinement 결과.

    refined_self_alignment:
        Local scale 후보 중 최저 mask/depth loss를 가진
        최종 self scale과 pose.

    selected_scaled_candidate:
        이후 cross-alignment에서 그대로 재사용할 mesh.
    """

    coarse_self_alignment: SelfAlignmentSelection

    local_scale_candidates: tuple[
        ScaledMeshCandidate,
        ...
    ]

    foundationpose_results: tuple[
        FoundationPoseCandidateResult,
        ...
    ]

    alignment_evaluation: AlignmentEvaluationResult

    refined_self_alignment: SelfAlignmentSelection
    selected_scaled_candidate: ScaledMeshCandidate

    metadata_path: Path

    def __post_init__(self) -> None:
        if not self.local_scale_candidates:
            raise ValueError(
                "Local scale 후보가 없습니다."
            )

        if not self.foundationpose_results:
            raise ValueError(
                "Local FoundationPose 결과가 없습니다."
            )

        if (
            self.refined_self_alignment.proxy_view
            != self.coarse_self_alignment.proxy_view
        ):
            raise ValueError(
                "Coarse와 refined self view가 다릅니다."
            )

        if (
            self.selected_scaled_candidate.candidate_index
            != self.refined_self_alignment.candidate_index
        ):
            raise ValueError(
                "선택된 mesh candidate와 refined self의 "
                "candidate index가 다릅니다."
            )

        if not np.isclose(
            self.selected_scaled_candidate.scale_m,
            self.refined_self_alignment.scale_m,
            atol=1e-8,
            rtol=1e-6,
        ):
            raise ValueError(
                "선택된 mesh와 refined self의 scale이 다릅니다."
            )

        selected_mesh_path = (
            self.selected_scaled_candidate
            .scaled_mesh_path
            .resolve()
        )

        self_mesh_path = (
            self.refined_self_alignment
            .scaled_mesh_path
            .resolve()
        )

        if selected_mesh_path != self_mesh_path:
            raise ValueError(
                "선택된 mesh와 refined self의 mesh 경로가 "
                "다릅니다."
            )

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                "Local scale refinement metadata가 없습니다: "
                f"{self.metadata_path}"
            )


def _validate_local_multipliers(
    multipliers: Sequence[float],
) -> tuple[float, ...]:
    """Local scale multiplier를 검증하고 중복 제거한다."""

    if not multipliers:
        raise ValueError(
            "Local scale multiplier가 없습니다."
        )

    normalized: list[float] = []

    for multiplier in multipliers:
        value = float(multiplier)

        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "Local scale multiplier는 유한한 "
                "양수여야 합니다: "
                f"{value}"
            )

        if value not in normalized:
            normalized.append(value)

    if 1.0 not in normalized:
        raise ValueError(
            "Local scale 후보에는 coarse scale을 유지하는 "
            "multiplier 1.0이 반드시 포함되어야 합니다."
        )

    return tuple(normalized)


def _build_local_scale_result(
    *,
    coarse_scale_m: float,
    visible_diagonal_m: float,
    multipliers: tuple[float, ...],
) -> ScaleInitializationResult:
    """Coarse scale 주변의 실제 local scale 값을 생성한다."""

    if (
        not np.isfinite(coarse_scale_m)
        or coarse_scale_m <= 0.0
    ):
        raise ValueError(
            "coarse_scale_m은 유한한 양수여야 합니다: "
            f"{coarse_scale_m}"
        )

    if (
        not np.isfinite(visible_diagonal_m)
        or visible_diagonal_m <= 0.0
    ):
        raise ValueError(
            "visible_diagonal_m은 유한한 양수여야 합니다: "
            f"{visible_diagonal_m}"
        )

    local_scales = tuple(
        float(coarse_scale_m * multiplier)
        for multiplier in multipliers
    )

    return ScaleInitializationResult(
        visible_diagonal_m=visible_diagonal_m,
        scale_candidates_m=local_scales,
    )


def _find_selected_candidate(
    candidates: tuple[
        ScaledMeshCandidate,
        ...
    ],
    self_alignment: SelfAlignmentSelection,
) -> ScaledMeshCandidate:
    """Refined self 결과에 대응하는 scaled mesh를 찾는다."""

    matching_candidates = [
        candidate
        for candidate in candidates
        if candidate.candidate_index
        == self_alignment.candidate_index
    ]

    if len(matching_candidates) != 1:
        raise RuntimeError(
            "Refined self에 대응하는 scaled mesh를 "
            "정확히 하나 찾지 못했습니다: "
            f"candidate_index="
            f"{self_alignment.candidate_index}, "
            f"matches={len(matching_candidates)}"
        )

    candidate = matching_candidates[0]

    if not np.isclose(
        candidate.scale_m,
        self_alignment.scale_m,
        atol=1e-8,
        rtol=1e-6,
    ):
        raise RuntimeError(
            "Refined self와 scaled mesh의 scale이 다릅니다."
        )

    return candidate


def _save_metadata(
    *,
    output_directory: Path,
    coarse_self: SelfAlignmentSelection,
    local_multipliers: tuple[float, ...],
    local_candidates: tuple[
        ScaledMeshCandidate,
        ...
    ],
    refined_self: SelfAlignmentSelection,
    selected_candidate: ScaledMeshCandidate,
    alignment_evaluation: AlignmentEvaluationResult,
) -> Path:
    """Local scale refinement 결과를 JSON으로 저장한다."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        output_directory
        / "local_scale_refinement.json"
    )

    metadata = {
        "view_name": coarse_self.proxy_view,
        "coarse": {
            "candidate_index": (
                coarse_self.candidate_index
            ),
            "hypothesis_rank": (
                coarse_self.hypothesis_rank
            ),
            "scale_m": coarse_self.scale_m,
            "alignment_loss": (
                coarse_self.alignment_loss
            ),
            "scaled_mesh_path": str(
                coarse_self.scaled_mesh_path
            ),
        },
        "local_scale_multipliers": list(
            local_multipliers
        ),
        "local_candidates": [
            {
                "candidate_index": (
                    candidate.candidate_index
                ),
                "scale_m": candidate.scale_m,
                "scaled_mesh_path": str(
                    candidate.scaled_mesh_path
                ),
                "metadata_path": str(
                    candidate.metadata_path
                ),
            }
            for candidate in local_candidates
        ],
        "refined": {
            "candidate_index": (
                refined_self.candidate_index
            ),
            "hypothesis_rank": (
                refined_self.hypothesis_rank
            ),
            "scale_m": refined_self.scale_m,
            "alignment_loss": (
                refined_self.alignment_loss
            ),
            "foundationpose_score": (
                refined_self.foundationpose_score
            ),
            "scaled_mesh_path": str(
                selected_candidate.scaled_mesh_path
            ),
            "pose_camera_from_proxy": (
                refined_self
                .pose_camera_from_proxy
                .tolist()
            ),
        },
        "alignment_evaluation_path": str(
            alignment_evaluation.summary_path
        ),
        "scale_policy": (
            "The refined scale and the exact same scaled "
            "mesh are fixed during cross-alignment."
        ),
    }

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Local scale refinement metadata가 "
            "저장되지 않았습니다: "
            f"{metadata_path}"
        )

    return metadata_path


def refine_self_scale_locally(
    *,
    normalization_result: MeshNormalizationResult,
    coarse_self_alignment: SelfAlignmentSelection,
    original_scale_result: ScaleInitializationResult,
    prepared_view: PreparedView,
    foundationpose_runner: FoundationPoseRunner,
    renderer: FoundationPoseMeshRenderer,
    output_directory: Path,
    local_scale_multipliers: Sequence[float] = (
        DEFAULT_LOCAL_SCALE_MULTIPLIERS
    ),
    alignment_weights: AlignmentScoreWeights = (
        AlignmentScoreWeights()
    ),
    depth_trim_quantile: float = 0.90,
    min_depth_overlap_pixels: int = 50,
    free_space_absolute_tolerance_m: float = 0.005,
    free_space_relative_tolerance: float = 0.02,
) -> LocalScaleRefinementResult:
    """
    Coarse self scale의 ±5% 주변을 다시 탐색한다.

    처리 순서
    ---------
    1. coarse scale × local multiplier
    2. normalized mesh에 local scale 적용
    3. 각 local scale에서 FoundationPose top-K 추정
    4. mask/depth 외부 평가
    5. 최저 loss의 scale과 self pose 선택

    주의
    ----
    foundationpose_runner는 coarse self 단계와 다른
    output_root를 사용하는 local-refinement 전용 runner가
    권장된다. 동일 candidate index 출력의 덮어쓰기를 방지한다.
    """

    view_name = prepared_view.view.source.name

    if (
        coarse_self_alignment.proxy_view
        != view_name
    ):
        raise ValueError(
            "Coarse self와 PreparedView의 view가 다릅니다: "
            f"self={coarse_self_alignment.proxy_view}, "
            f"view={view_name}"
        )

    normalized_multipliers = (
        _validate_local_multipliers(
            local_scale_multipliers
        )
    )

    local_scale_result = (
        _build_local_scale_result(
            coarse_scale_m=(
                coarse_self_alignment.scale_m
            ),
            visible_diagonal_m=(
                original_scale_result
                .visible_diagonal_m
            ),
            multipliers=normalized_multipliers,
        )
    )

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    local_candidates = (
        build_scaled_mesh_candidates(
            normalization_result=(
                normalization_result
            ),
            scale_result=local_scale_result,
            output_directory=(
                output_directory
                / "scaled_meshes"
            ),
        )
    )

    foundationpose_results = (
        foundationpose_runner.run_candidates(
            candidates=local_candidates,
            prepared_view=prepared_view,
        )
    )

    alignment_evaluation = (
        evaluate_foundationpose_alignments(
            prepared_view=prepared_view,
            candidate_results=(
                foundationpose_results
            ),
            renderer=renderer,
            output_directory=(
                output_directory
                / "alignment"
            ),
            weights=alignment_weights,
            depth_trim_quantile=(
                depth_trim_quantile
            ),
            min_depth_overlap_pixels=(
                min_depth_overlap_pixels
            ),
            free_space_absolute_tolerance_m=(
                free_space_absolute_tolerance_m
            ),
            free_space_relative_tolerance=(
                free_space_relative_tolerance
            ),
        )
    )

    refined_self_alignment = (
        select_best_self_alignment(
            alignment_evaluation
        )
    )

    selected_candidate = (
        _find_selected_candidate(
            candidates=local_candidates,
            self_alignment=(
                refined_self_alignment
            ),
        )
    )

    metadata_path = _save_metadata(
        output_directory=output_directory,
        coarse_self=coarse_self_alignment,
        local_multipliers=(
            normalized_multipliers
        ),
        local_candidates=local_candidates,
        refined_self=refined_self_alignment,
        selected_candidate=selected_candidate,
        alignment_evaluation=(
            alignment_evaluation
        ),
    )

    return LocalScaleRefinementResult(
        coarse_self_alignment=(
            coarse_self_alignment
        ),
        local_scale_candidates=local_candidates,
        foundationpose_results=(
            foundationpose_results
        ),
        alignment_evaluation=(
            alignment_evaluation
        ),
        refined_self_alignment=(
            refined_self_alignment
        ),
        selected_scaled_candidate=(
            selected_candidate
        ),
        metadata_path=metadata_path,
    )
