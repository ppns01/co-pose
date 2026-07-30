from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AlignmentScoreWeights:
    """Mask/depth 정합 점수 가중치."""

    mask: float = 0.30
    depth: float = 0.40
    free_space: float = 0.20
    boundary: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.mask,
            self.depth,
            self.free_space,
            self.boundary,
        )

        for value in values:
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "Score weight는 유한한 0 이상의 값이어야 합니다: "
                    f"{value}"
                )

        if sum(values) <= 0.0:
            raise ValueError(
                "Score weight의 합은 0보다 커야 합니다."
            )


@dataclass(frozen=True)
class AlignmentScoreResult:
    """
    하나의 pose 후보에 대한 mask/depth 정합 결과.

    모든 loss는 낮을수록 좋다.
    """

    total_loss: float

    mask_loss: float
    depth_loss: float
    free_space_loss: float
    boundary_loss: float

    mask_iou: float
    depth_residual_m: float
    depth_residual_normalized: float

    overlap_pixel_count: int
    valid_depth_overlap_count: int
    rendered_pixel_count: int
    free_space_violation_count: int

    def __post_init__(self) -> None:
        scalar_values = (
            self.total_loss,
            self.mask_loss,
            self.depth_loss,
            self.free_space_loss,
            self.boundary_loss,
            self.mask_iou,
            self.depth_residual_m,
            self.depth_residual_normalized,
        )

        for value in scalar_values:
            if not np.isfinite(value):
                raise ValueError(
                    "Alignment score에 NaN 또는 Inf가 있습니다: "
                    f"{value}"
                )

        bounded_values = (
            self.mask_loss,
            self.depth_loss,
            self.free_space_loss,
            self.boundary_loss,
            self.mask_iou,
        )

        for value in bounded_values:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "정규화된 score는 0과 1 사이여야 합니다: "
                    f"{value}"
                )


def _validate_mask(
    mask: NDArray[np.bool_],
    name: str,
) -> NDArray[np.bool_]:
    """Mask를 연속적인 boolean 배열로 검증한다."""

    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"{name} shape은 (H, W)이어야 합니다: "
            f"{mask.shape}"
        )

    if mask.dtype != np.bool_:
        raise TypeError(
            f"{name} dtype은 bool이어야 합니다: "
            f"{mask.dtype}"
        )

    return np.ascontiguousarray(
        mask,
        dtype=np.bool_,
    )


def _validate_depth(
    depth_m: NDArray[np.floating],
    expected_shape: tuple[int, int],
    name: str,
) -> NDArray[np.float32]:
    """Depth를 meter 단위 float32 배열로 검증한다."""

    depth_m = np.asarray(
        depth_m,
        dtype=np.float32,
    )

    if depth_m.shape != expected_shape:
        raise ValueError(
            f"{name} shape이 올바르지 않습니다: "
            f"expected={expected_shape}, "
            f"actual={depth_m.shape}"
        )

    if not np.isfinite(depth_m).all():
        raise ValueError(
            f"{name}에 NaN 또는 Inf가 있습니다."
        )

    if np.any(depth_m < 0.0):
        raise ValueError(
            f"{name}에는 음수 depth가 존재할 수 없습니다."
        )

    return np.ascontiguousarray(
        depth_m,
        dtype=np.float32,
    )


def _compute_mask_loss(
    observed_mask: NDArray[np.bool_],
    rendered_mask: NDArray[np.bool_],
) -> tuple[float, float, int]:
    """
    Mask IoU와 1-IoU loss를 계산한다.
    """

    intersection = int(
        np.count_nonzero(
            observed_mask & rendered_mask
        )
    )

    union = int(
        np.count_nonzero(
            observed_mask | rendered_mask
        )
    )

    if union == 0:
        return 1.0, 0.0, 0

    mask_iou = intersection / union
    mask_loss = 1.0 - mask_iou

    return (
        float(mask_loss),
        float(mask_iou),
        intersection,
    )


def _compute_trimmed_depth_loss(
    observed_depth_m: NDArray[np.float32],
    rendered_depth_m: NDArray[np.float32],
    observed_mask: NDArray[np.bool_],
    rendered_mask: NDArray[np.bool_],
    object_scale_m: float,
    trim_quantile: float,
    min_depth_overlap_pixels: int,
) -> tuple[float, float, float, int]:
    """
    Mask overlap 영역의 trimmed depth residual을 계산한다.

    큰 residual 일부를 제거하여 depth outlier 영향을 줄인다.
    """

    valid_overlap = (
        observed_mask
        & rendered_mask
        & (observed_depth_m > 0.0)
        & (rendered_depth_m > 0.0)
    )

    valid_count = int(
        np.count_nonzero(valid_overlap)
    )

    if valid_count < min_depth_overlap_pixels:
        return 1.0, object_scale_m, 1.0, valid_count

    residuals_m = np.abs(
        observed_depth_m[valid_overlap]
        - rendered_depth_m[valid_overlap]
    ).astype(
        np.float64,
        copy=False,
    )

    trim_threshold = float(
        np.quantile(
            residuals_m,
            trim_quantile,
        )
    )

    trimmed_residuals = residuals_m[
        residuals_m <= trim_threshold
    ]

    if trimmed_residuals.size == 0:
        return 1.0, object_scale_m, 1.0, valid_count

    depth_residual_m = float(
        np.mean(trimmed_residuals)
    )

    normalized_residual = (
        depth_residual_m / object_scale_m
    )

    depth_loss = float(
        np.clip(
            normalized_residual,
            0.0,
            1.0,
        )
    )

    return (
        depth_loss,
        depth_residual_m,
        float(normalized_residual),
        valid_count,
    )


def _compute_free_space_loss(
    observed_depth_m: NDArray[np.float32],
    rendered_depth_m: NDArray[np.float32],
    rendered_mask: NDArray[np.bool_],
    object_scale_m: float,
    absolute_tolerance_m: float,
    relative_tolerance: float,
) -> tuple[float, int, int]:
    """
    Rendered mesh가 관측 surface보다 앞에 나타나는 영역을 계산한다.

    rendered_depth < observed_depth - tolerance
    이면 free-space violation으로 처리한다.
    """

    rendered_valid = (
        rendered_mask
        & (rendered_depth_m > 0.0)
    )

    comparison_valid = (
        rendered_valid
        & (observed_depth_m > 0.0)
    )

    comparison_count = int(
        np.count_nonzero(comparison_valid)
    )

    if comparison_count == 0:
        return 1.0, 0, 0

    tolerance_m = max(
        absolute_tolerance_m,
        object_scale_m * relative_tolerance,
    )

    free_space_violation = (
        comparison_valid
        & (
            rendered_depth_m
            < observed_depth_m - tolerance_m
        )
    )

    violation_count = int(
        np.count_nonzero(
            free_space_violation
        )
    )

    free_space_loss = (
        violation_count / comparison_count
    )

    return (
        float(
            np.clip(
                free_space_loss,
                0.0,
                1.0,
            )
        ),
        violation_count,
        comparison_count,
    )


def _extract_boundary(
    mask: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Binary mask에서 1-pixel 내부 경계를 추출한다."""

    mask_uint8 = (
        mask.astype(np.uint8)
        * np.uint8(255)
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    eroded = cv2.erode(
        mask_uint8,
        kernel,
        iterations=1,
    )

    boundary = (
        mask_uint8 > eroded
    )

    return np.ascontiguousarray(
        boundary,
        dtype=np.bool_,
    )


def _distance_to_boundary(
    boundary: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """
    각 픽셀에서 가장 가까운 boundary까지의 거리를 계산한다.
    """

    distance_input = (
        ~boundary
    ).astype(
        np.uint8
    )

    distance_map = cv2.distanceTransform(
        distance_input,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )

    return np.ascontiguousarray(
        distance_map,
        dtype=np.float32,
    )


def _compute_boundary_loss(
    observed_mask: NDArray[np.bool_],
    rendered_mask: NDArray[np.bool_],
) -> float:
    """
    관측 mask와 rendered silhouette 경계 간
    symmetric boundary distance를 계산한다.
    """

    observed_boundary = _extract_boundary(
        observed_mask
    )

    rendered_boundary = _extract_boundary(
        rendered_mask
    )

    observed_count = int(
        np.count_nonzero(
            observed_boundary
        )
    )

    rendered_count = int(
        np.count_nonzero(
            rendered_boundary
        )
    )

    if observed_count == 0 or rendered_count == 0:
        return 1.0

    distance_to_observed = _distance_to_boundary(
        observed_boundary
    )

    distance_to_rendered = _distance_to_boundary(
        rendered_boundary
    )

    rendered_to_observed = float(
        np.mean(
            distance_to_observed[
                rendered_boundary
            ]
        )
    )

    observed_to_rendered = float(
        np.mean(
            distance_to_rendered[
                observed_boundary
            ]
        )
    )

    symmetric_distance_pixels = (
        rendered_to_observed
        + observed_to_rendered
    ) * 0.5

    height, width = observed_mask.shape

    image_diagonal_pixels = float(
        np.hypot(
            height,
            width,
        )
    )

    boundary_loss = (
        symmetric_distance_pixels
        / image_diagonal_pixels
    )

    return float(
        np.clip(
            boundary_loss,
            0.0,
            1.0,
        )
    )


def score_alignment(
    *,
    observed_mask: NDArray[np.bool_],
    observed_depth_m: NDArray[np.floating],
    rendered_mask: NDArray[np.bool_],
    rendered_depth_m: NDArray[np.floating],
    object_scale_m: float,
    weights: AlignmentScoreWeights = (
        AlignmentScoreWeights()
    ),
    depth_trim_quantile: float = 0.90,
    min_depth_overlap_pixels: int = 50,
    free_space_absolute_tolerance_m: float = 0.005,
    free_space_relative_tolerance: float = 0.02,
) -> AlignmentScoreResult:
    """
    FoundationPose 후보의 mask/depth 정합 loss를 계산한다.

    Parameters
    ----------
    observed_mask:
        SAM3 boolean mask.

    observed_depth_m:
        LINEMOD 관측 depth. 단위 meter.

    rendered_mask:
        후보 pose에서 렌더링한 silhouette.

    rendered_depth_m:
        후보 pose에서 렌더링한 depth. 단위 meter.

    object_scale_m:
        현재 scaled mesh의 등방성 scale.

    Returns
    -------
    AlignmentScoreResult
        낮을수록 좋은 정합 loss.
    """

    observed_mask = _validate_mask(
        observed_mask,
        "observed_mask",
    )

    rendered_mask = _validate_mask(
        rendered_mask,
        "rendered_mask",
    )

    if rendered_mask.shape != observed_mask.shape:
        raise ValueError(
            "Observed mask와 rendered mask의 "
            "해상도가 다릅니다: "
            f"observed={observed_mask.shape}, "
            f"rendered={rendered_mask.shape}"
        )

    if (
        not np.isfinite(object_scale_m)
        or object_scale_m <= 0.0
    ):
        raise ValueError(
            "object_scale_m은 유한한 양수여야 합니다: "
            f"{object_scale_m}"
        )

    if not 0.0 < depth_trim_quantile <= 1.0:
        raise ValueError(
            "depth_trim_quantile은 0보다 크고 "
            "1 이하여야 합니다: "
            f"{depth_trim_quantile}"
        )

    if min_depth_overlap_pixels < 1:
        raise ValueError(
            "min_depth_overlap_pixels는 "
            "1 이상이어야 합니다."
        )

    if free_space_absolute_tolerance_m < 0.0:
        raise ValueError(
            "free_space_absolute_tolerance_m은 "
            "0 이상이어야 합니다."
        )

    if free_space_relative_tolerance < 0.0:
        raise ValueError(
            "free_space_relative_tolerance은 "
            "0 이상이어야 합니다."
        )

    observed_depth_m = _validate_depth(
        depth_m=observed_depth_m,
        expected_shape=observed_mask.shape,
        name="observed_depth_m",
    )

    rendered_depth_m = _validate_depth(
        depth_m=rendered_depth_m,
        expected_shape=observed_mask.shape,
        name="rendered_depth_m",
    )

    (
        mask_loss,
        mask_iou,
        overlap_pixel_count,
    ) = _compute_mask_loss(
        observed_mask=observed_mask,
        rendered_mask=rendered_mask,
    )

    (
        depth_loss,
        depth_residual_m,
        depth_residual_normalized,
        valid_depth_overlap_count,
    ) = _compute_trimmed_depth_loss(
        observed_depth_m=observed_depth_m,
        rendered_depth_m=rendered_depth_m,
        observed_mask=observed_mask,
        rendered_mask=rendered_mask,
        object_scale_m=object_scale_m,
        trim_quantile=depth_trim_quantile,
        min_depth_overlap_pixels=(
            min_depth_overlap_pixels
        ),
    )

    (
        free_space_loss,
        free_space_violation_count,
        free_space_comparison_count,
    ) = _compute_free_space_loss(
        observed_depth_m=observed_depth_m,
        rendered_depth_m=rendered_depth_m,
        rendered_mask=rendered_mask,
        object_scale_m=object_scale_m,
        absolute_tolerance_m=(
            free_space_absolute_tolerance_m
        ),
        relative_tolerance=(
            free_space_relative_tolerance
        ),
    )

    boundary_loss = _compute_boundary_loss(
        observed_mask=observed_mask,
        rendered_mask=rendered_mask,
    )

    weight_sum = (
        weights.mask
        + weights.depth
        + weights.free_space
        + weights.boundary
    )

    total_loss = (
        weights.mask * mask_loss
        + weights.depth * depth_loss
        + weights.free_space * free_space_loss
        + weights.boundary * boundary_loss
    ) / weight_sum

    rendered_pixel_count = int(
        np.count_nonzero(
            rendered_mask
        )
    )

    return AlignmentScoreResult(
        total_loss=float(total_loss),
        mask_loss=mask_loss,
        depth_loss=depth_loss,
        free_space_loss=free_space_loss,
        boundary_loss=boundary_loss,
        mask_iou=mask_iou,
        depth_residual_m=depth_residual_m,
        depth_residual_normalized=(
            depth_residual_normalized
        ),
        overlap_pixel_count=overlap_pixel_count,
        valid_depth_overlap_count=(
            valid_depth_overlap_count
        ),
        rendered_pixel_count=rendered_pixel_count,
        free_space_violation_count=(
            free_space_violation_count
        ),
    )


def save_alignment_score(
    result: AlignmentScoreResult,
    output_path: Path,
    *,
    view_name: str,
    candidate_index: int,
    hypothesis_rank: int,
    scale_m: float,
) -> Path:
    """Alignment score를 JSON 파일로 저장한다."""

    output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "view_name": view_name,
        "candidate_index": candidate_index,
        "hypothesis_rank": hypothesis_rank,
        "scale_m": scale_m,
        "total_loss": result.total_loss,
        "mask_loss": result.mask_loss,
        "depth_loss": result.depth_loss,
        "free_space_loss": result.free_space_loss,
        "boundary_loss": result.boundary_loss,
        "mask_iou": result.mask_iou,
        "depth_residual_m": (
            result.depth_residual_m
        ),
        "depth_residual_normalized": (
            result.depth_residual_normalized
        ),
        "overlap_pixel_count": (
            result.overlap_pixel_count
        ),
        "valid_depth_overlap_count": (
            result.valid_depth_overlap_count
        ),
        "rendered_pixel_count": (
            result.rendered_pixel_count
        ),
        "free_space_violation_count": (
            result.free_space_violation_count
        ),
        "lower_is_better": True,
    }

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if not output_path.is_file():
        raise FileNotFoundError(
            "Alignment score가 저장되지 않았습니다: "
            f"{output_path}"
        )

    return output_path