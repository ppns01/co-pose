from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core.types import ViewName
from pose.foundationpose_runner import (
    FoundationPoseCandidateResult,
    FoundationPoseHypothesis,
)


@dataclass(frozen=True)
class SelfAlignmentSelection:
    """
    Proxy를 자기 RGB-D 관측에 정렬한 최종 결과.

    Reference proxy:
        T_Cr_from_Pr

    Query proxy:
        T_Cq_from_Pq

    반드시 cross-alignment에서도 동일한 scaled mesh를 사용해야 한다.
    """

    proxy_view: ViewName

    candidate_index: int
    hypothesis_rank: int
    scale_m: float

    scaled_mesh_path: Path
    pose_camera_from_proxy: NDArray[np.float32]

    foundationpose_score: float
    alignment_loss: float | None = None

    def __post_init__(self) -> None:
        if self.proxy_view not in (
            "reference",
            "query",
        ):
            raise ValueError(
                "지원하지 않는 proxy view입니다: "
                f"{self.proxy_view}"
            )

        if self.candidate_index < 0:
            raise ValueError(
                "candidate_index는 0 이상이어야 합니다: "
                f"{self.candidate_index}"
            )

        if self.hypothesis_rank < 0:
            raise ValueError(
                "hypothesis_rank는 0 이상이어야 합니다: "
                f"{self.hypothesis_rank}"
            )

        if (
            not np.isfinite(self.scale_m)
            or self.scale_m <= 0.0
        ):
            raise ValueError(
                "scale_m은 유한한 양수여야 합니다: "
                f"{self.scale_m}"
            )

        if not self.scaled_mesh_path.is_file():
            raise FileNotFoundError(
                "Self-alignment에 사용한 scaled mesh가 "
                "없습니다: "
                f"{self.scaled_mesh_path}"
            )

        _validate_rigid_pose(
            self.pose_camera_from_proxy,
            "pose_camera_from_proxy",
        )

        if not np.isfinite(
            self.foundationpose_score
        ):
            raise ValueError(
                "FoundationPose score가 유한하지 않습니다."
            )

        if (
            self.alignment_loss is not None
            and not np.isfinite(self.alignment_loss)
        ):
            raise ValueError(
                "alignment_loss가 유한하지 않습니다."
            )


def _validate_rigid_pose(
    pose: NDArray[np.floating],
    name: str,
) -> NDArray[np.float32]:
    """4×4 SE(3) 변환 행렬을 검증한다."""

    pose_array = np.asarray(
        pose,
        dtype=np.float32,
    )

    if pose_array.shape != (4, 4):
        raise ValueError(
            f"{name} shape은 (4, 4)이어야 합니다: "
            f"{pose_array.shape}"
        )

    if not np.isfinite(pose_array).all():
        raise ValueError(
            f"{name}에 NaN 또는 Inf가 있습니다."
        )

    expected_last_row = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )

    if not np.allclose(
        pose_array[3],
        expected_last_row,
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name}의 마지막 행이 homogeneous transform "
            "형식이 아닙니다."
        )

    rotation = pose_array[:3, :3]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float32),
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name}의 회전 행렬이 직교 행렬이 아닙니다."
        )

    determinant = float(
        np.linalg.det(rotation)
    )

    if not np.isclose(
        determinant,
        1.0,
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name} 회전 행렬 determinant가 "
            f"1이 아닙니다: {determinant}"
        )

    return np.ascontiguousarray(
        pose_array,
        dtype=np.float32,
    )


def _find_hypothesis(
    result: FoundationPoseCandidateResult,
    hypothesis_rank: int,
) -> FoundationPoseHypothesis:
    """FoundationPose 결과에서 지정한 rank를 찾는다."""

    for hypothesis in result.hypotheses:
        if hypothesis.rank == hypothesis_rank:
            return hypothesis

    available_ranks = [
        hypothesis.rank
        for hypothesis in result.hypotheses
    ]

    raise KeyError(
        "요청한 FoundationPose hypothesis rank가 "
        "없습니다: "
        f"requested={hypothesis_rank}, "
        f"available={available_ranks}"
    )


def select_self_alignment(
    result: FoundationPoseCandidateResult,
    hypothesis_rank: int,
    *,
    alignment_loss: float | None = None,
) -> SelfAlignmentSelection:
    """
    FoundationPose self-alignment 결과에서 하나를 선택한다.

    Reference self 결과이면:
        proxy_view = reference

    Query self 결과이면:
        proxy_view = query
    """

    hypothesis = _find_hypothesis(
        result=result,
        hypothesis_rank=hypothesis_rank,
    )

    return SelfAlignmentSelection(
        proxy_view=result.view_name,
        candidate_index=result.candidate_index,
        hypothesis_rank=hypothesis.rank,
        scale_m=result.scale_m,
        scaled_mesh_path=(
            result.scaled_mesh_path.resolve()
        ),
        pose_camera_from_proxy=(
            hypothesis.pose_cam_from_proxy
        ),
        foundationpose_score=hypothesis.score,
        alignment_loss=alignment_loss,
    )
