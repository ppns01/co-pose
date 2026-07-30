from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pose.relative_pose_builder import (
    BidirectionalRelativePoseCandidates,
    RelativePoseCandidate,
)


@dataclass(frozen=True)
class ConsistencyWeights:
    """
    양방향 relative pose consistency 가중치.

    rotation:
        정규화된 회전 차이 가중치.

    translation:
        객체 크기로 정규화한 이동 차이 가중치.

    scale:
        두 proxy의 scale 차이 가중치.
        Scale은 보조 신뢰도이므로 기본 가중치를 낮게 둔다.
    """

    rotation: float = 1.0
    translation: float = 1.0
    scale: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.rotation,
            self.translation,
            self.scale,
        )

        for value in values:
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "Consistency weight는 유한한 "
                    "0 이상의 값이어야 합니다: "
                    f"{value}"
                )

        if sum(values) <= 0.0:
            raise ValueError(
                "Consistency weight의 합은 "
                "0보다 커야 합니다."
            )


@dataclass(frozen=True)
class ConsistencyThresholds:
    """
    양방향 pose 일치 여부를 판단하는 초기 기준.

    rotation_deg:
        두 상대 회전 차이의 최대 허용값.

    translation_ratio:
        객체 크기 대비 이동 차이의 최대 허용 비율.

    maximum_scale_log_difference:
        Scale 차이를 hard gate로 사용할 경우 지정한다.
        V1에서는 scale을 보조 지표로만 사용하므로 기본값은 None.
    """

    rotation_deg: float = 15.0
    translation_ratio: float = 0.10
    maximum_scale_log_difference: float | None = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.rotation_deg)
            or self.rotation_deg <= 0.0
        ):
            raise ValueError(
                "rotation_deg는 유한한 양수여야 합니다: "
                f"{self.rotation_deg}"
            )

        if (
            not np.isfinite(self.translation_ratio)
            or self.translation_ratio <= 0.0
        ):
            raise ValueError(
                "translation_ratio는 유한한 "
                "양수여야 합니다: "
                f"{self.translation_ratio}"
            )

        if self.maximum_scale_log_difference is not None:
            if (
                not np.isfinite(
                    self.maximum_scale_log_difference
                )
                or self.maximum_scale_log_difference < 0.0
            ):
                raise ValueError(
                    "maximum_scale_log_difference는 "
                    "유한한 0 이상의 값이어야 합니다."
                )


@dataclass(frozen=True)
class BidirectionalConsistencyPair:
    """
    Reference proxy 후보 하나와 Query proxy 후보 하나의
    일치도 결과.

    두 후보 pose는 모두 다음 규약을 사용한다.

        T_query_camera_from_reference_camera
    """

    reference_candidate_index: int
    query_candidate_index: int

    reference_candidate: RelativePoseCandidate
    query_candidate: RelativePoseCandidate

    rotation_difference_deg: float

    translation_difference_m: float
    translation_normalizer_m: float
    translation_difference_normalized: float

    scale_log_difference: float

    normalized_rotation_loss: float
    consistency_loss: float

    passes_rotation_gate: bool
    passes_translation_gate: bool
    passes_scale_gate: bool
    passes_hard_gate: bool

    def __post_init__(self) -> None:
        if self.reference_candidate_index < 0:
            raise ValueError(
                "reference_candidate_index는 "
                "0 이상이어야 합니다."
            )

        if self.query_candidate_index < 0:
            raise ValueError(
                "query_candidate_index는 "
                "0 이상이어야 합니다."
            )

        scalar_values = (
            self.rotation_difference_deg,
            self.translation_difference_m,
            self.translation_normalizer_m,
            self.translation_difference_normalized,
            self.scale_log_difference,
            self.normalized_rotation_loss,
            self.consistency_loss,
        )

        for value in scalar_values:
            if not np.isfinite(value):
                raise ValueError(
                    "Consistency 결과에 "
                    "NaN 또는 Inf가 있습니다."
                )

        if self.rotation_difference_deg < 0.0:
            raise ValueError(
                "회전 차이는 0 이상이어야 합니다."
            )

        if self.translation_difference_m < 0.0:
            raise ValueError(
                "이동 차이는 0 이상이어야 합니다."
            )

        if self.translation_normalizer_m <= 0.0:
            raise ValueError(
                "Translation normalizer는 "
                "양수여야 합니다."
            )

        if self.scale_log_difference < 0.0:
            raise ValueError(
                "Scale 차이는 0 이상이어야 합니다."
            )

        expected_hard_gate = (
            self.passes_rotation_gate
            and self.passes_translation_gate
            and self.passes_scale_gate
        )

        if self.passes_hard_gate != expected_hard_gate:
            raise ValueError(
                "passes_hard_gate 값이 개별 gate와 "
                "일치하지 않습니다."
            )


@dataclass(frozen=True)
class BidirectionalConsistencyResult:
    """모든 양방향 relative pose 조합의 평가 결과."""

    pairs: tuple[BidirectionalConsistencyPair, ...]
    accepted_pairs: tuple[
        BidirectionalConsistencyPair,
        ...
    ]

    reference_candidate_count: int
    query_candidate_count: int

    @property
    def best_accepted_pair(
        self,
    ) -> BidirectionalConsistencyPair | None:
        """Hard gate를 통과한 후보 중 최저 loss 후보."""

        if not self.accepted_pairs:
            return None

        return self.accepted_pairs[0]

    @property
    def best_overall_pair(
        self,
    ) -> BidirectionalConsistencyPair:
        """Gate 통과 여부와 관계없이 최저 loss 후보."""

        if not self.pairs:
            raise RuntimeError(
                "Consistency pair가 없습니다."
            )

        return self.pairs[0]

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError(
                "Consistency pair가 하나도 없습니다."
            )

        expected_pair_count = (
            self.reference_candidate_count
            * self.query_candidate_count
        )

        if len(self.pairs) != expected_pair_count:
            raise ValueError(
                "Consistency pair 개수가 올바르지 않습니다: "
                f"expected={expected_pair_count}, "
                f"actual={len(self.pairs)}"
            )

        for pair in self.accepted_pairs:
            if not pair.passes_hard_gate:
                raise ValueError(
                    "accepted_pairs에 hard gate를 "
                    "통과하지 않은 후보가 있습니다."
                )


def _validate_pose(
    pose: NDArray[np.floating],
    name: str,
) -> NDArray[np.float64]:
    """4×4 SE(3) 상대 pose를 검증한다."""

    pose_array = np.asarray(
        pose,
        dtype=np.float64,
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

    if not np.allclose(
        pose_array[3],
        np.array(
            [0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        ),
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name}의 마지막 행이 올바르지 않습니다."
        )

    rotation = pose_array[:3, :3]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        atol=1e-3,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name}의 회전 행렬이 직교하지 않습니다."
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
            f"{name} 회전 determinant가 "
            f"1이 아닙니다: {determinant}"
        )

    return np.ascontiguousarray(
        pose_array,
        dtype=np.float64,
    )


def _compute_rotation_difference_deg(
    reference_pose: NDArray[np.float64],
    query_pose: NDArray[np.float64],
) -> float:
    """
    두 상대 pose 회전의 SO(3) geodesic 차이를 계산한다.

    V1에서는 객체 대칭을 고려하지 않는 raw rotation
    difference를 사용한다.
    """

    reference_rotation = (
        reference_pose[:3, :3]
    )

    query_rotation = (
        query_pose[:3, :3]
    )

    relative_rotation = (
        reference_rotation
        @ query_rotation.T
    )

    cosine_value = (
        np.trace(relative_rotation) - 1.0
    ) * 0.5

    cosine_value = float(
        np.clip(
            cosine_value,
            -1.0,
            1.0,
        )
    )

    angle_rad = float(
        np.arccos(cosine_value)
    )

    return float(
        np.degrees(angle_rad)
    )


def _compute_translation_difference_m(
    reference_pose: NDArray[np.float64],
    query_pose: NDArray[np.float64],
) -> float:
    """두 상대 pose의 translation 차이를 계산한다."""

    reference_translation = (
        reference_pose[:3, 3]
    )

    query_translation = (
        query_pose[:3, 3]
    )

    return float(
        np.linalg.norm(
            reference_translation
            - query_translation
        )
    )


def _resolve_translation_normalizer_m(
    reference_candidate: RelativePoseCandidate,
    query_candidate: RelativePoseCandidate,
    explicit_normalizer_m: float | None,
) -> float:
    """
    Translation 차이를 정규화할 객체 크기를 결정한다.

    explicit_normalizer_m이 없으면 두 proxy scale의 평균을
    사용한다.

        (s_reference + s_query) / 2
    """

    if explicit_normalizer_m is not None:
        if (
            not np.isfinite(explicit_normalizer_m)
            or explicit_normalizer_m <= 0.0
        ):
            raise ValueError(
                "explicit_normalizer_m은 "
                "유한한 양수여야 합니다."
            )

        return float(explicit_normalizer_m)

    normalizer_m = (
        reference_candidate.scale_m
        + query_candidate.scale_m
    ) * 0.5

    if (
        not np.isfinite(normalizer_m)
        or normalizer_m <= 0.0
    ):
        raise ValueError(
            "두 proxy scale로부터 유효한 "
            "translation normalizer를 만들 수 없습니다."
        )

    return float(normalizer_m)


def _compute_scale_log_difference(
    reference_scale_m: float,
    query_scale_m: float,
) -> float:
    """
    두 독립 proxy scale의 log-ratio 차이를 계산한다.

        abs(log(s_reference / s_query))
    """

    if reference_scale_m <= 0.0:
        raise ValueError(
            "Reference scale은 양수여야 합니다."
        )

    if query_scale_m <= 0.0:
        raise ValueError(
            "Query scale은 양수여야 합니다."
        )

    return float(
        abs(
            np.log(
                reference_scale_m
                / query_scale_m
            )
        )
    )


def _evaluate_pair(
    *,
    reference_candidate_index: int,
    query_candidate_index: int,
    reference_candidate: RelativePoseCandidate,
    query_candidate: RelativePoseCandidate,
    weights: ConsistencyWeights,
    thresholds: ConsistencyThresholds,
    translation_normalizer_m: float | None,
) -> BidirectionalConsistencyPair:
    """상대 pose 후보 한 쌍의 일치도를 평가한다."""

    reference_pose = _validate_pose(
        reference_candidate
        .relative_pose_query_from_reference,
        "reference_proxy_relative_pose",
    )

    query_pose = _validate_pose(
        query_candidate
        .relative_pose_query_from_reference,
        "query_proxy_relative_pose",
    )

    rotation_difference_deg = (
        _compute_rotation_difference_deg(
            reference_pose=reference_pose,
            query_pose=query_pose,
        )
    )

    translation_difference_m = (
        _compute_translation_difference_m(
            reference_pose=reference_pose,
            query_pose=query_pose,
        )
    )

    resolved_normalizer_m = (
        _resolve_translation_normalizer_m(
            reference_candidate=reference_candidate,
            query_candidate=query_candidate,
            explicit_normalizer_m=(
                translation_normalizer_m
            ),
        )
    )

    translation_difference_normalized = (
        translation_difference_m
        / resolved_normalizer_m
    )

    scale_log_difference = (
        _compute_scale_log_difference(
            reference_scale_m=(
                reference_candidate.scale_m
            ),
            query_scale_m=(
                query_candidate.scale_m
            ),
        )
    )

    normalized_rotation_loss = (
        rotation_difference_deg
        / thresholds.rotation_deg
    )

    weight_sum = (
        weights.rotation
        + weights.translation
        + weights.scale
    )

    consistency_loss = (
        weights.rotation
        * normalized_rotation_loss
        + weights.translation
        * translation_difference_normalized
        + weights.scale
        * scale_log_difference
    ) / weight_sum

    passes_rotation_gate = (
        rotation_difference_deg
        < thresholds.rotation_deg
    )

    passes_translation_gate = (
        translation_difference_normalized
        < thresholds.translation_ratio
    )

    if (
        thresholds.maximum_scale_log_difference
        is None
    ):
        passes_scale_gate = True
    else:
        passes_scale_gate = (
            scale_log_difference
            <= thresholds.maximum_scale_log_difference
        )

    passes_hard_gate = (
        passes_rotation_gate
        and passes_translation_gate
        and passes_scale_gate
    )

    return BidirectionalConsistencyPair(
        reference_candidate_index=(
            reference_candidate_index
        ),
        query_candidate_index=(
            query_candidate_index
        ),
        reference_candidate=reference_candidate,
        query_candidate=query_candidate,
        rotation_difference_deg=(
            rotation_difference_deg
        ),
        translation_difference_m=(
            translation_difference_m
        ),
        translation_normalizer_m=(
            resolved_normalizer_m
        ),
        translation_difference_normalized=float(
            translation_difference_normalized
        ),
        scale_log_difference=(
            scale_log_difference
        ),
        normalized_rotation_loss=float(
            normalized_rotation_loss
        ),
        consistency_loss=float(
            consistency_loss
        ),
        passes_rotation_gate=(
            passes_rotation_gate
        ),
        passes_translation_gate=(
            passes_translation_gate
        ),
        passes_scale_gate=passes_scale_gate,
        passes_hard_gate=passes_hard_gate,
    )


def evaluate_bidirectional_consistency(
    candidate_set: BidirectionalRelativePoseCandidates,
    *,
    weights: ConsistencyWeights = ConsistencyWeights(),
    thresholds: ConsistencyThresholds = (
        ConsistencyThresholds()
    ),
    translation_normalizer_m: float | None = None,
) -> BidirectionalConsistencyResult:
    """
    Reference proxy 후보와 Query proxy 후보의 모든 조합을
    비교한다.

    Parameters
    ----------
    candidate_set:
        두 proxy 경로에서 생성된 상대 pose 후보.

    weights:
        회전·이동·scale soft loss 가중치.

    thresholds:
        초기 hard gate 기준.

    translation_normalizer_m:
        Translation 정규화 기준을 직접 지정할 때 사용한다.
        None이면 두 proxy scale 평균을 사용한다.

    Returns
    -------
    BidirectionalConsistencyResult
        전체 조합과 hard gate 통과 조합.
        각 집합은 consistency loss 오름차순으로 정렬된다.
    """

    reference_candidates = (
        candidate_set.reference_proxy_candidates
    )

    query_candidates = (
        candidate_set.query_proxy_candidates
    )

    pairs: list[
        BidirectionalConsistencyPair
    ] = []

    for reference_index, reference_candidate in enumerate(
        reference_candidates
    ):
        for query_index, query_candidate in enumerate(
            query_candidates
        ):
            pair = _evaluate_pair(
                reference_candidate_index=(
                    reference_index
                ),
                query_candidate_index=query_index,
                reference_candidate=(
                    reference_candidate
                ),
                query_candidate=query_candidate,
                weights=weights,
                thresholds=thresholds,
                translation_normalizer_m=(
                    translation_normalizer_m
                ),
            )

            pairs.append(pair)

    sorted_pairs = tuple(
        sorted(
            pairs,
            key=lambda pair: (
                pair.consistency_loss,
                pair.rotation_difference_deg,
                pair.translation_difference_normalized,
                pair.reference_candidate_index,
                pair.query_candidate_index,
            ),
        )
    )

    accepted_pairs = tuple(
        pair
        for pair in sorted_pairs
        if pair.passes_hard_gate
    )

    return BidirectionalConsistencyResult(
        pairs=sorted_pairs,
        accepted_pairs=accepted_pairs,
        reference_candidate_count=len(
            reference_candidates
        ),
        query_candidate_count=len(
            query_candidates
        ),
    )


def _candidate_summary(
    candidate: RelativePoseCandidate,
) -> dict[str, object]:
    """JSON 저장용 relative pose 후보 요약."""

    return {
        "path_name": candidate.path_name,
        "scale_m": candidate.scale_m,
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


def _pair_to_dict(
    pair: BidirectionalConsistencyPair,
) -> dict[str, object]:
    """Consistency pair를 JSON 저장 형식으로 변환한다."""

    return {
        "reference_candidate_index": (
            pair.reference_candidate_index
        ),
        "query_candidate_index": (
            pair.query_candidate_index
        ),
        "rotation_difference_deg": (
            pair.rotation_difference_deg
        ),
        "translation_difference_m": (
            pair.translation_difference_m
        ),
        "translation_normalizer_m": (
            pair.translation_normalizer_m
        ),
        "translation_difference_normalized": (
            pair.translation_difference_normalized
        ),
        "scale_log_difference": (
            pair.scale_log_difference
        ),
        "normalized_rotation_loss": (
            pair.normalized_rotation_loss
        ),
        "consistency_loss": (
            pair.consistency_loss
        ),
        "passes_rotation_gate": (
            pair.passes_rotation_gate
        ),
        "passes_translation_gate": (
            pair.passes_translation_gate
        ),
        "passes_scale_gate": (
            pair.passes_scale_gate
        ),
        "passes_hard_gate": (
            pair.passes_hard_gate
        ),
        "reference_candidate": (
            _candidate_summary(
                pair.reference_candidate
            )
        ),
        "query_candidate": (
            _candidate_summary(
                pair.query_candidate
            )
        ),
    }


def save_bidirectional_consistency(
    result: BidirectionalConsistencyResult,
    output_directory: Path,
    *,
    weights: ConsistencyWeights = ConsistencyWeights(),
    thresholds: ConsistencyThresholds = (
        ConsistencyThresholds()
    ),
) -> tuple[Path, Path]:
    """
    Consistency 결과를 JSON과 NPY matrix로 저장한다.

    생성 파일
    ---------
    bidirectional_consistency.json
    consistency_loss_matrix.npy
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

    metadata_path = (
        output_directory
        / "bidirectional_consistency.json"
    )

    loss_matrix_path = (
        output_directory
        / "consistency_loss_matrix.npy"
    )

    loss_matrix = np.full(
        (
            result.reference_candidate_count,
            result.query_candidate_count,
        ),
        fill_value=np.inf,
        dtype=np.float32,
    )

    for pair in result.pairs:
        loss_matrix[
            pair.reference_candidate_index,
            pair.query_candidate_index,
        ] = np.float32(
            pair.consistency_loss
        )

    np.save(
        loss_matrix_path,
        loss_matrix,
        allow_pickle=False,
    )

    best_accepted_pair = (
        result.best_accepted_pair
    )

    metadata = {
        "pose_convention": (
            "T_query_camera_from_reference_camera"
        ),
        "translation_unit": "meter",
        "reference_candidate_count": (
            result.reference_candidate_count
        ),
        "query_candidate_count": (
            result.query_candidate_count
        ),
        "total_pair_count": len(result.pairs),
        "accepted_pair_count": len(
            result.accepted_pairs
        ),
        "weights": {
            "rotation": weights.rotation,
            "translation": weights.translation,
            "scale": weights.scale,
        },
        "thresholds": {
            "rotation_deg": (
                thresholds.rotation_deg
            ),
            "translation_ratio": (
                thresholds.translation_ratio
            ),
            "maximum_scale_log_difference": (
                thresholds
                .maximum_scale_log_difference
            ),
        },
        "best_accepted_pair": (
            None
            if best_accepted_pair is None
            else _pair_to_dict(
                best_accepted_pair
            )
        ),
        "best_overall_pair": (
            _pair_to_dict(
                result.best_overall_pair
            )
        ),
        "pairs": [
            _pair_to_dict(pair)
            for pair in result.pairs
        ],
        "lower_is_better": True,
        "symmetry_aware": False,
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
            "Consistency JSON이 저장되지 않았습니다: "
            f"{metadata_path}"
        )

    if not loss_matrix_path.is_file():
        raise FileNotFoundError(
            "Consistency matrix가 저장되지 않았습니다: "
            f"{loss_matrix_path}"
        )

    return (
        metadata_path,
        loss_matrix_path,
    )