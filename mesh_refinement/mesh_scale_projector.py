from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleReprojectionResult:
    reprojected_points_camera: np.ndarray
    scale_before_reprojection_m: float
    scale_after_reprojection_m: float
    compensation_beta: float


def reproject_to_target_scale(
    *,
    points_camera: np.ndarray,
    target_scale_m: float,
    diameter_fn,
) -> ScaleReprojectionResult:
    """
    국소 depth 보정 이후 mesh를 동일한 S* robust scale로 되돌린다.

    diameter_fn:
        S*를 만들 때 쓴 것과 같은 robust size 통계량 함수
        (예: pose.dgedi_runner._diameter). points -> float.
    """

    if not np.isfinite(target_scale_m) or target_scale_m <= 0.0:
        raise ValueError(f"target_scale_m은 유한한 양수여야 합니다: {target_scale_m}")

    scale_before = float(diameter_fn(points_camera))

    if scale_before <= 0.0:
        raise ValueError(f"보정된 mesh의 scale이 유효하지 않습니다: {scale_before}")

    centroid = points_camera.mean(axis=0)
    beta = target_scale_m / scale_before

    reprojected = centroid + beta * (points_camera - centroid)
    scale_after = float(diameter_fn(reprojected))

    return ScaleReprojectionResult(
        reprojected_points_camera=reprojected,
        scale_before_reprojection_m=scale_before,
        scale_after_reprojection_m=scale_after,
        compensation_beta=float(beta),
    )
