from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from core.types import (
    PreparedView,
    ScaleInitializationResult,
)


DEFAULT_SCALE_MULTIPLIERS = (
    0.8,
    1.0,
    1.25,
    1.5,
    1.8,
)


def _validate_parameters(
    quantile_low: float,
    quantile_high: float,
    scale_multipliers: Sequence[float],
    min_valid_points: int,
    max_points: int | None,
    min_depth_m: float,
    max_depth_m: float | None,
) -> tuple[float, ...]:
    """Scale 초기화 파라미터를 검증한다."""

    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError(
            "quantile 범위가 올바르지 않습니다: "
            f"low={quantile_low}, high={quantile_high}"
        )

    if isinstance(min_valid_points, bool):
        raise TypeError(
            "min_valid_points는 정수여야 합니다."
        )

    if min_valid_points < 3:
        raise ValueError(
            "min_valid_points는 3 이상이어야 합니다: "
            f"{min_valid_points}"
        )

    if max_points is not None:
        if isinstance(max_points, bool):
            raise TypeError(
                "max_points는 정수 또는 None이어야 합니다."
            )

        if max_points < min_valid_points:
            raise ValueError(
                "max_points는 min_valid_points 이상이어야 합니다: "
                f"max_points={max_points}, "
                f"min_valid_points={min_valid_points}"
            )

    if (
        not np.isfinite(min_depth_m)
        or min_depth_m < 0.0
    ):
        raise ValueError(
            "min_depth_m은 유한한 0 이상의 값이어야 합니다: "
            f"{min_depth_m}"
        )

    if max_depth_m is not None:
        if (
            not np.isfinite(max_depth_m)
            or max_depth_m <= min_depth_m
        ):
            raise ValueError(
                "max_depth_m은 min_depth_m보다 큰 "
                "유한한 값이어야 합니다: "
                f"min={min_depth_m}, max={max_depth_m}"
            )

    normalized_multipliers: list[float] = []

    for multiplier in scale_multipliers:
        value = float(multiplier)

        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "Scale multiplier는 유한한 양수여야 합니다: "
                f"{value}"
            )

        normalized_multipliers.append(value)

    if not normalized_multipliers:
        raise ValueError(
            "Scale multiplier가 하나도 없습니다."
        )

    return tuple(normalized_multipliers)


def _build_valid_depth_mask(
    prepared_view: PreparedView,
    min_depth_m: float,
    max_depth_m: float | None,
) -> NDArray[np.bool_]:
    """Mask 내부의 유효한 depth 픽셀을 선택한다."""

    depth_m = prepared_view.masked_depth_m
    object_mask = prepared_view.segmentation.mask_bool

    valid_mask = (
        object_mask
        & np.isfinite(depth_m)
        & (depth_m > min_depth_m)
    )

    if max_depth_m is not None:
        valid_mask &= depth_m <= max_depth_m

    return np.ascontiguousarray(
        valid_mask,
        dtype=np.bool_,
    )


def _depth_to_point_cloud(
    depth_m: NDArray[np.float32],
    camera_matrix: NDArray[np.float32],
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """
    유효한 depth 픽셀을 카메라 좌표계 point cloud로 변환한다.

    좌표계:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth
    """

    if depth_m.shape != valid_mask.shape:
        raise ValueError(
            "Depth와 valid mask의 shape이 다릅니다: "
            f"depth={depth_m.shape}, "
            f"mask={valid_mask.shape}"
        )

    if camera_matrix.shape != (3, 3):
        raise ValueError(
            "Camera intrinsic은 (3, 3)이어야 합니다: "
            f"{camera_matrix.shape}"
        )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(
            "fx와 fy는 양수여야 합니다: "
            f"fx={fx}, fy={fy}"
        )

    pixel_v, pixel_u = np.nonzero(valid_mask)

    if pixel_u.size == 0:
        raise ValueError(
            "유효한 depth 픽셀이 없습니다."
        )

    z = depth_m[
        pixel_v,
        pixel_u,
    ].astype(
        np.float64,
        copy=False,
    )

    u = pixel_u.astype(
        np.float64,
        copy=False,
    )

    v = pixel_v.astype(
        np.float64,
        copy=False,
    )

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack(
        (x, y, z),
        axis=1,
    )

    if not np.isfinite(points).all():
        raise ValueError(
            "생성된 point cloud에 NaN 또는 Inf가 있습니다."
        )

    return np.ascontiguousarray(
        points,
        dtype=np.float32,
    )


def _subsample_points(
    points: NDArray[np.float32],
    max_points: int | None,
) -> NDArray[np.float32]:
    """
    point 수가 많을 경우 결정론적으로 일부만 선택한다.

    동일 입력에서는 항상 동일한 point가 선택된다.
    """

    if max_points is None:
        return points

    point_count = points.shape[0]

    if point_count <= max_points:
        return points

    random_generator = np.random.default_rng(
        seed=0
    )

    selected_indices = random_generator.choice(
        point_count,
        size=max_points,
        replace=False,
    )

    selected_indices.sort()

    return np.ascontiguousarray(
        points[selected_indices],
        dtype=np.float32,
    )


def _compute_robust_diagonal(
    points: NDArray[np.float32],
    quantile_low: float,
    quantile_high: float,
) -> float:
    """Point cloud의 robust bounding-box diagonal을 계산한다."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "Point cloud shape은 (N, 3)이어야 합니다: "
            f"{points.shape}"
        )

    lower_bound = np.quantile(
        points,
        quantile_low,
        axis=0,
    )

    upper_bound = np.quantile(
        points,
        quantile_high,
        axis=0,
    )

    extent = upper_bound - lower_bound

    visible_diagonal_m = float(
        np.linalg.norm(extent)
    )

    if (
        not np.isfinite(visible_diagonal_m)
        or visible_diagonal_m <= 0.0
    ):
        raise ValueError(
            "유효한 visible diagonal을 계산하지 못했습니다: "
            f"{visible_diagonal_m}"
        )

    return visible_diagonal_m


def _build_scale_candidates(
    visible_diagonal_m: float,
    scale_multipliers: tuple[float, ...],
) -> tuple[float, ...]:
    """Visible diagonal을 중심으로 scale bank를 생성한다."""

    scale_candidates = tuple(
        float(
            visible_diagonal_m * multiplier
        )
        for multiplier in scale_multipliers
    )

    return scale_candidates


def _save_scale_metadata(
    output_directory: Path,
    result: ScaleInitializationResult,
    quantile_low: float,
    quantile_high: float,
    scale_multipliers: tuple[float, ...],
    valid_point_count: int,
    used_point_count: int,
) -> Path:
    """Scale 초기화 결과를 JSON으로 저장한다."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        output_directory
        / "scale_initialization.json"
    )

    metadata = {
        "visible_diagonal_m": (
            result.visible_diagonal_m
        ),
        "scale_candidates_m": list(
            result.scale_candidates_m
        ),
        "scale_multipliers": list(
            scale_multipliers
        ),
        "quantile_low": quantile_low,
        "quantile_high": quantile_high,
        "valid_point_count": valid_point_count,
        "used_point_count": used_point_count,
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
            "Scale metadata가 저장되지 않았습니다: "
            f"{metadata_path}"
        )

    return metadata_path


def initialize_scale_from_depth(
    prepared_view: PreparedView,
    output_directory: Path | None = None,
    *,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    scale_multipliers: Sequence[float] = (
        DEFAULT_SCALE_MULTIPLIERS
    ),
    min_valid_points: int = 100,
    max_points: int | None = 200_000,
    min_depth_m: float = 0.0,
    max_depth_m: float | None = None,
    save_point_cloud: bool = False,
) -> ScaleInitializationResult:
    """
    Mask 내부 depth에서 초기 등방성 scale bank를 생성한다.

    Parameters
    ----------
    prepared_view:
        SAM3 mask가 적용된 RGB-D 입력.

    output_directory:
        지정하면 scale_initialization.json을 저장한다.

    quantile_low, quantile_high:
        Robust bounding box 계산에 사용할 quantile.

    scale_multipliers:
        Visible diagonal에 곱할 scale 후보 비율.

    min_valid_points:
        Scale 계산에 필요한 최소 point 수.

    max_points:
        계산에 사용할 최대 point 수.
        None이면 전체 point를 사용한다.

    min_depth_m, max_depth_m:
        사용할 depth 범위.

    save_point_cloud:
        True이면 사용한 point cloud를 NPY로 저장한다.

    Returns
    -------
    ScaleInitializationResult
        visible diagonal과 초기 scale 후보.
    """

    normalized_multipliers = _validate_parameters(
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        scale_multipliers=scale_multipliers,
        min_valid_points=min_valid_points,
        max_points=max_points,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )

    valid_depth_mask = _build_valid_depth_mask(
        prepared_view=prepared_view,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )

    valid_point_count = int(
        np.count_nonzero(valid_depth_mask)
    )

    if valid_point_count < min_valid_points:
        raise ValueError(
            "Scale 초기화에 필요한 유효 depth point가 "
            "부족합니다: "
            f"actual={valid_point_count}, "
            f"required={min_valid_points}"
        )

    point_cloud_m = _depth_to_point_cloud(
        depth_m=prepared_view.masked_depth_m,
        camera_matrix=(
            prepared_view.view.camera_matrix
        ),
        valid_mask=valid_depth_mask,
    )

    sampled_point_cloud_m = _subsample_points(
        points=point_cloud_m,
        max_points=max_points,
    )

    visible_diagonal_m = _compute_robust_diagonal(
        points=sampled_point_cloud_m,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )

    scale_candidates_m = _build_scale_candidates(
        visible_diagonal_m=visible_diagonal_m,
        scale_multipliers=normalized_multipliers,
    )

    result = ScaleInitializationResult(
        visible_diagonal_m=visible_diagonal_m,
        scale_candidates_m=scale_candidates_m,
    )

    if output_directory is not None:
        output_directory = (
            Path(output_directory)
            .expanduser()
            .resolve()
        )

        _save_scale_metadata(
            output_directory=output_directory,
            result=result,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            scale_multipliers=normalized_multipliers,
            valid_point_count=valid_point_count,
            used_point_count=(
                sampled_point_cloud_m.shape[0]
            ),
        )

        if save_point_cloud:
            point_cloud_path = (
                output_directory
                / "masked_point_cloud_m.npy"
            )

            np.save(
                point_cloud_path,
                sampled_point_cloud_m,
                allow_pickle=False,
            )

    return result