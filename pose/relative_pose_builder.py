from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from core.types import ViewName
from pose.foundationpose_runner import (
    FoundationPoseCandidateResult,
    FoundationPoseHypothesis,
)


ProxyPathName = Literal[
    "reference_proxy",
    "query_proxy",
]


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


@dataclass(frozen=True)
class RelativePoseCandidate:
    """
    Proxy 좌표계가 소거된 상대 pose 후보.

    relative_pose_query_from_reference:
        Reference camera 좌표의 3D point를
        Query camera 좌표로 변환한다.

        x_query =
            relative_pose_query_from_reference
            @ x_reference
    """

    path_name: ProxyPathName

    scale_m: float
    scaled_mesh_path: Path

    self_candidate_index: int
    self_hypothesis_rank: int
    self_foundationpose_score: float

    cross_candidate_index: int
    cross_hypothesis_rank: int
    cross_foundationpose_score: float

    relative_pose_query_from_reference: (
        NDArray[np.float32]
    )

    self_alignment_loss: float | None = None

    def __post_init__(self) -> None:
        if self.path_name not in (
            "reference_proxy",
            "query_proxy",
        ):
            raise ValueError(
                "지원하지 않는 proxy path입니다: "
                f"{self.path_name}"
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
                "Relative pose에 사용한 scaled mesh가 "
                "없습니다: "
                f"{self.scaled_mesh_path}"
            )

        _validate_rigid_pose(
            self.relative_pose_query_from_reference,
            "relative_pose_query_from_reference",
        )


@dataclass(frozen=True)
class BidirectionalRelativePoseCandidates:
    """두 proxy 경로에서 생성된 상대 pose 후보 집합."""

    reference_proxy_candidates: tuple[
        RelativePoseCandidate,
        ...
    ]

    query_proxy_candidates: tuple[
        RelativePoseCandidate,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.reference_proxy_candidates:
            raise ValueError(
                "Reference proxy 상대 pose 후보가 없습니다."
            )

        if not self.query_proxy_candidates:
            raise ValueError(
                "Query proxy 상대 pose 후보가 없습니다."
            )

        for candidate in self.reference_proxy_candidates:
            if candidate.path_name != "reference_proxy":
                raise ValueError(
                    "Reference proxy 후보 집합에 다른 경로의 "
                    "후보가 포함되어 있습니다."
                )

        for candidate in self.query_proxy_candidates:
            if candidate.path_name != "query_proxy":
                raise ValueError(
                    "Query proxy 후보 집합에 다른 경로의 "
                    "후보가 포함되어 있습니다."
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


def _invert_rigid_pose(
    pose_target_from_source: NDArray[np.floating],
) -> NDArray[np.float32]:
    """
    SE(3) rigid transform을 안정적으로 역변환한다.

    T_target_from_source
    → T_source_from_target
    """

    pose = _validate_rigid_pose(
        pose_target_from_source,
        "pose_target_from_source",
    )

    rotation = pose[:3, :3].astype(
        np.float64,
        copy=False,
    )

    translation = pose[:3, 3].astype(
        np.float64,
        copy=False,
    )

    inverse_pose = np.eye(
        4,
        dtype=np.float64,
    )

    inverse_rotation = rotation.T

    inverse_pose[:3, :3] = inverse_rotation
    inverse_pose[:3, 3] = (
        -inverse_rotation @ translation
    )

    return _validate_rigid_pose(
        inverse_pose,
        "inverse_pose",
    )


def _compose_rigid_poses(
    left_pose: NDArray[np.floating],
    right_pose: NDArray[np.floating],
) -> NDArray[np.float32]:
    """두 SE(3) 변환을 행렬 곱 순서대로 합성한다."""

    left = _validate_rigid_pose(
        left_pose,
        "left_pose",
    ).astype(
        np.float64,
        copy=False,
    )

    right = _validate_rigid_pose(
        right_pose,
        "right_pose",
    ).astype(
        np.float64,
        copy=False,
    )

    composed_pose = left @ right

    return _validate_rigid_pose(
        composed_pose,
        "composed_pose",
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


def _validate_same_proxy_mesh(
    self_alignment: SelfAlignmentSelection,
    cross_result: FoundationPoseCandidateResult,
) -> None:
    """
    Self와 cross alignment가 정확히 같은 scaled proxy를
    사용했는지 검사한다.

    이 조건이 깨지면 proxy local frame이 소거되지 않는다.
    """

    self_mesh_path = (
        self_alignment
        .scaled_mesh_path
        .expanduser()
        .resolve()
    )

    cross_mesh_path = (
        cross_result
        .scaled_mesh_path
        .expanduser()
        .resolve()
    )

    if self_mesh_path != cross_mesh_path:
        raise ValueError(
            "Self와 cross alignment에 서로 다른 mesh가 "
            "사용되었습니다.\n"
            f"Self : {self_mesh_path}\n"
            f"Cross: {cross_mesh_path}\n"
            "동일한 proxy local frame을 유지해야 합니다."
        )

    if self_alignment.candidate_index != (
        cross_result.candidate_index
    ):
        raise ValueError(
            "Self와 cross alignment의 scale candidate "
            "index가 다릅니다: "
            f"self={self_alignment.candidate_index}, "
            f"cross={cross_result.candidate_index}"
        )

    if not np.isclose(
        self_alignment.scale_m,
        cross_result.scale_m,
        atol=1e-8,
        rtol=1e-6,
    ):
        raise ValueError(
            "Self와 cross alignment의 scale이 다릅니다: "
            f"self={self_alignment.scale_m}, "
            f"cross={cross_result.scale_m}"
        )


def build_reference_proxy_candidates(
    reference_self: SelfAlignmentSelection,
    reference_proxy_to_query: (
        FoundationPoseCandidateResult
    ),
) -> tuple[RelativePoseCandidate, ...]:
    """
    Reference proxy 경로의 상대 pose 후보를 계산한다.

    H_q_from_r^(reference)
        = T_Cq_from_Pr
          @ inverse(T_Cr_from_Pr)
    """

    if reference_self.proxy_view != "reference":
        raise ValueError(
            "Reference proxy 경로에는 Reference self "
            "alignment가 필요합니다."
        )

    if reference_proxy_to_query.view_name != "query":
        raise ValueError(
            "Reference proxy cross 결과의 target view는 "
            "query여야 합니다: "
            f"{reference_proxy_to_query.view_name}"
        )

    _validate_same_proxy_mesh(
        self_alignment=reference_self,
        cross_result=reference_proxy_to_query,
    )

    inverse_reference_self_pose = (
        _invert_rigid_pose(
            reference_self.pose_camera_from_proxy
        )
    )

    candidates: list[
        RelativePoseCandidate
    ] = []

    for cross_hypothesis in (
        reference_proxy_to_query.hypotheses
    ):
        relative_pose = _compose_rigid_poses(
            cross_hypothesis.pose_cam_from_proxy,
            inverse_reference_self_pose,
        )

        candidates.append(
            RelativePoseCandidate(
                path_name="reference_proxy",
                scale_m=reference_self.scale_m,
                scaled_mesh_path=(
                    reference_self.scaled_mesh_path
                ),
                self_candidate_index=(
                    reference_self.candidate_index
                ),
                self_hypothesis_rank=(
                    reference_self.hypothesis_rank
                ),
                self_foundationpose_score=(
                    reference_self
                    .foundationpose_score
                ),
                cross_candidate_index=(
                    reference_proxy_to_query
                    .candidate_index
                ),
                cross_hypothesis_rank=(
                    cross_hypothesis.rank
                ),
                cross_foundationpose_score=(
                    cross_hypothesis.score
                ),
                relative_pose_query_from_reference=(
                    relative_pose
                ),
                self_alignment_loss=(
                    reference_self.alignment_loss
                ),
            )
        )

    if not candidates:
        raise RuntimeError(
            "Reference proxy 상대 pose 후보가 "
            "생성되지 않았습니다."
        )

    return tuple(candidates)


def build_query_proxy_candidates(
    query_self: SelfAlignmentSelection,
    query_proxy_to_reference: (
        FoundationPoseCandidateResult
    ),
) -> tuple[RelativePoseCandidate, ...]:
    """
    Query proxy 경로의 상대 pose 후보를 계산한다.

    H_q_from_r^(query)
        = T_Cq_from_Pq
          @ inverse(T_Cr_from_Pq)
    """

    if query_self.proxy_view != "query":
        raise ValueError(
            "Query proxy 경로에는 Query self "
            "alignment가 필요합니다."
        )

    if query_proxy_to_reference.view_name != "reference":
        raise ValueError(
            "Query proxy cross 결과의 target view는 "
            "reference여야 합니다: "
            f"{query_proxy_to_reference.view_name}"
        )

    _validate_same_proxy_mesh(
        self_alignment=query_self,
        cross_result=query_proxy_to_reference,
    )

    candidates: list[
        RelativePoseCandidate
    ] = []

    for cross_hypothesis in (
        query_proxy_to_reference.hypotheses
    ):
        inverse_cross_pose = _invert_rigid_pose(
            cross_hypothesis.pose_cam_from_proxy
        )

        relative_pose = _compose_rigid_poses(
            query_self.pose_camera_from_proxy,
            inverse_cross_pose,
        )

        candidates.append(
            RelativePoseCandidate(
                path_name="query_proxy",
                scale_m=query_self.scale_m,
                scaled_mesh_path=(
                    query_self.scaled_mesh_path
                ),
                self_candidate_index=(
                    query_self.candidate_index
                ),
                self_hypothesis_rank=(
                    query_self.hypothesis_rank
                ),
                self_foundationpose_score=(
                    query_self.foundationpose_score
                ),
                cross_candidate_index=(
                    query_proxy_to_reference
                    .candidate_index
                ),
                cross_hypothesis_rank=(
                    cross_hypothesis.rank
                ),
                cross_foundationpose_score=(
                    cross_hypothesis.score
                ),
                relative_pose_query_from_reference=(
                    relative_pose
                ),
                self_alignment_loss=(
                    query_self.alignment_loss
                ),
            )
        )

    if not candidates:
        raise RuntimeError(
            "Query proxy 상대 pose 후보가 "
            "생성되지 않았습니다."
        )

    return tuple(candidates)


def build_bidirectional_relative_candidates(
    *,
    reference_self: SelfAlignmentSelection,
    query_self: SelfAlignmentSelection,
    reference_proxy_to_query: (
        FoundationPoseCandidateResult
    ),
    query_proxy_to_reference: (
        FoundationPoseCandidateResult
    ),
) -> BidirectionalRelativePoseCandidates:
    """Reference proxy와 Query proxy의 상대 pose 후보를 생성한다."""

    reference_candidates = (
        build_reference_proxy_candidates(
            reference_self=reference_self,
            reference_proxy_to_query=(
                reference_proxy_to_query
            ),
        )
    )

    query_candidates = (
        build_query_proxy_candidates(
            query_self=query_self,
            query_proxy_to_reference=(
                query_proxy_to_reference
            ),
        )
    )

    return BidirectionalRelativePoseCandidates(
        reference_proxy_candidates=(
            reference_candidates
        ),
        query_proxy_candidates=query_candidates,
    )


def _candidate_to_dict(
    candidate: RelativePoseCandidate,
) -> dict[str, object]:
    """Relative pose 후보를 JSON 저장 형식으로 변환한다."""

    return {
        "path_name": candidate.path_name,
        "scale_m": candidate.scale_m,
        "scaled_mesh_path": str(
            candidate.scaled_mesh_path
        ),
        "self_candidate_index": (
            candidate.self_candidate_index
        ),
        "self_hypothesis_rank": (
            candidate.self_hypothesis_rank
        ),
        "self_foundationpose_score": (
            candidate.self_foundationpose_score
        ),
        "self_alignment_loss": (
            candidate.self_alignment_loss
        ),
        "cross_candidate_index": (
            candidate.cross_candidate_index
        ),
        "cross_hypothesis_rank": (
            candidate.cross_hypothesis_rank
        ),
        "cross_foundationpose_score": (
            candidate.cross_foundationpose_score
        ),
        "relative_pose_query_from_reference": (
            candidate
            .relative_pose_query_from_reference
            .tolist()
        ),
    }


def save_relative_pose_candidates(
    candidate_set: BidirectionalRelativePoseCandidates,
    output_directory: Path,
) -> tuple[Path, Path, Path]:
    """
    양방향 상대 pose 후보를 NPY와 JSON으로 저장한다.

    생성 파일
    ---------
    reference_proxy_relative_poses.npy
    query_proxy_relative_poses.npy
    relative_pose_candidates.json
    """

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_poses = np.stack(
        [
            candidate.relative_pose_query_from_reference
            for candidate in (
                candidate_set.reference_proxy_candidates
            )
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    query_poses = np.stack(
        [
            candidate.relative_pose_query_from_reference
            for candidate in (
                candidate_set.query_proxy_candidates
            )
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    reference_poses_path = (
        output_directory
        / "reference_proxy_relative_poses.npy"
    )

    query_poses_path = (
        output_directory
        / "query_proxy_relative_poses.npy"
    )

    metadata_path = (
        output_directory
        / "relative_pose_candidates.json"
    )

    np.save(
        reference_poses_path,
        reference_poses,
        allow_pickle=False,
    )

    np.save(
        query_poses_path,
        query_poses,
        allow_pickle=False,
    )

    metadata = {
        "pose_convention": (
            "T_query_camera_from_reference_camera"
        ),
        "translation_unit": "meter",
        "reference_proxy_candidates": [
            _candidate_to_dict(candidate)
            for candidate in (
                candidate_set.reference_proxy_candidates
            )
        ],
        "query_proxy_candidates": [
            _candidate_to_dict(candidate)
            for candidate in (
                candidate_set.query_proxy_candidates
            )
        ],
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

    return (
        reference_poses_path,
        query_poses_path,
        metadata_path,
    )