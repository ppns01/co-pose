from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from features.dino_pose_scorer import (
    BidirectionalDINOScore,
)
from pose.alignment_scorer import (
    AlignmentScoreResult,
)
from pose.bidirectional_consistency import (
    BidirectionalConsistencyPair,
    BidirectionalConsistencyResult,
)
from pose.relative_pose_builder import (
    ProxyPathName,
    RelativePoseCandidate,
)


SelectionStatus = Literal[
    "CONSISTENT",
    "REJECT",
]


@dataclass(frozen=True)
class CandidateScoreWeights:
    """
    Proxy 경로 하나의 점수 가중치.

    cross_alignment:
        반대편 RGB-D를 설명하는 mask/depth 점수.
        가장 중요한 항목이다.

    self_alignment:
        Proxy가 자기 RGB-D에 맞는 정도.

    dino:
        Depth가 일치하는 surface에서 계산한
        DINO appearance consistency.
    """

    self_alignment: float = 0.20
    cross_alignment: float = 0.60
    dino: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.self_alignment,
            self.cross_alignment,
            self.dino,
        )

        for value in values:
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "Candidate score weight는 유한한 "
                    "0 이상의 값이어야 합니다: "
                    f"{value}"
                )

        if sum(values) <= 0.0:
            raise ValueError(
                "Candidate score weight의 합은 "
                "0보다 커야 합니다."
            )


@dataclass(frozen=True)
class PairScoreWeights:
    """
    양방향 후보 쌍의 최종 점수 가중치.

    path_evidence:
        두 proxy 경로의 self/cross/DINO 설명력.

    consistency:
        두 경로가 산출한 상대 pose·scale 일치도.
    """

    path_evidence: float = 0.75
    consistency: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.path_evidence,
            self.consistency,
        )

        for value in values:
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "Pair score weight는 유한한 "
                    "0 이상의 값이어야 합니다: "
                    f"{value}"
                )

        if sum(values) <= 0.0:
            raise ValueError(
                "Pair score weight의 합은 "
                "0보다 커야 합니다."
            )


@dataclass(frozen=True)
class CandidateEvidence:
    """
    상대 pose 후보 하나에 대응하는 외부 관측 점수.

    candidate_index:
        Reference 또는 Query proxy 후보 집합 내부 index.

    cross_alignment:
        해당 proxy를 반대편 RGB-D에 렌더링한
        mask/depth alignment score.

    dino_score:
        해당 상대 pose를 양방향 DINO로 평가한 결과.
        DINO support가 없으면 available=False일 수 있다.
    """

    path_name: ProxyPathName
    candidate_index: int

    cross_alignment: AlignmentScoreResult
    dino_score: BidirectionalDINOScore | None = None

    def __post_init__(self) -> None:
        if self.path_name not in (
            "reference_proxy",
            "query_proxy",
        ):
            raise ValueError(
                "지원하지 않는 proxy path입니다: "
                f"{self.path_name}"
            )

        if self.candidate_index < 0:
            raise ValueError(
                "candidate_index는 0 이상이어야 합니다: "
                f"{self.candidate_index}"
            )


@dataclass(frozen=True)
class CandidateCompositeScore:
    """Proxy 경로 하나의 통합 점수."""

    path_name: ProxyPathName
    candidate_index: int

    candidate: RelativePoseCandidate

    self_alignment_loss: float | None
    cross_alignment_loss: float
    dino_loss: float | None
    dino_available: bool

    active_weight_sum: float
    path_loss: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.cross_alignment_loss <= 1.0:
            raise ValueError(
                "cross_alignment_loss는 "
                "[0, 1] 범위여야 합니다."
            )

        if self.self_alignment_loss is not None:
            if not 0.0 <= self.self_alignment_loss <= 1.0:
                raise ValueError(
                    "self_alignment_loss는 "
                    "[0, 1] 범위여야 합니다."
                )

        if self.dino_loss is not None:
            if not 0.0 <= self.dino_loss <= 1.0:
                raise ValueError(
                    "dino_loss는 [0, 1] 범위여야 합니다."
                )

        if self.active_weight_sum <= 0.0:
            raise ValueError(
                "active_weight_sum은 양수여야 합니다."
            )

        if not 0.0 <= self.path_loss <= 1.0:
            raise ValueError(
                "path_loss는 [0, 1] 범위여야 합니다: "
                f"{self.path_loss}"
            )


@dataclass(frozen=True)
class FinalPairScore:
    """양방향 후보 조합 하나의 최종 점수."""

    consistency_pair: BidirectionalConsistencyPair

    reference_score: CandidateCompositeScore
    query_score: CandidateCompositeScore

    mean_path_loss: float
    normalized_consistency_loss: float
    final_pair_loss: float

    def __post_init__(self) -> None:
        bounded_values = (
            self.mean_path_loss,
            self.normalized_consistency_loss,
            self.final_pair_loss,
        )

        for value in bounded_values:
            if not np.isfinite(value):
                raise ValueError(
                    "Final pair score에 NaN 또는 Inf가 있습니다."
                )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Final pair score는 [0, 1] 범위여야 합니다: "
                    f"{value}"
                )


@dataclass(frozen=True)
class FinalSelectionResult:
    """
    최종 relative pose 선택 결과.

    CONSISTENT:
        양방향 hard gate를 통과한 후보 조합에서
        최종 proxy 경로 하나를 선택했다.

    REJECT:
        양방향 일치 조건을 만족한 조합이 없다.
    """

    status: SelectionStatus

    selected_path_name: ProxyPathName | None
    selected_candidate: RelativePoseCandidate | None

    selected_relative_pose_query_from_reference: (
        NDArray[np.float32] | None
    )

    selected_scale_m: float | None

    final_loss: float
    confidence: float
    score_margin: float | None

    best_pair_score: FinalPairScore | None
    evaluated_pair_scores: tuple[FinalPairScore, ...]

    def __post_init__(self) -> None:
        if self.status == "CONSISTENT":
            if self.selected_path_name is None:
                raise ValueError(
                    "CONSISTENT 결과에는 selected_path가 필요합니다."
                )

            if self.selected_candidate is None:
                raise ValueError(
                    "CONSISTENT 결과에는 selected_candidate가 필요합니다."
                )

            if (
                self.selected_relative_pose_query_from_reference
                is None
            ):
                raise ValueError(
                    "CONSISTENT 결과에는 relative pose가 필요합니다."
                )

            if self.selected_scale_m is None:
                raise ValueError(
                    "CONSISTENT 결과에는 scale이 필요합니다."
                )

            if self.best_pair_score is None:
                raise ValueError(
                    "CONSISTENT 결과에는 best pair가 필요합니다."
                )

        elif self.status == "REJECT":
            if self.selected_candidate is not None:
                raise ValueError(
                    "REJECT 결과에는 후보를 선택할 수 없습니다."
                )

            if (
                self.selected_relative_pose_query_from_reference
                is not None
            ):
                raise ValueError(
                    "REJECT 결과에는 relative pose가 없어야 합니다."
                )

        else:
            raise ValueError(
                f"지원하지 않는 status입니다: {self.status}"
            )

        if not 0.0 <= self.final_loss <= 1.0:
            raise ValueError(
                "final_loss는 [0, 1] 범위여야 합니다."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence는 [0, 1] 범위여야 합니다."
            )


def _build_evidence_map(
    evidence_items: Sequence[CandidateEvidence],
    expected_path_name: ProxyPathName,
) -> dict[int, CandidateEvidence]:
    """후보 index별 evidence lookup을 생성한다."""

    evidence_map: dict[int, CandidateEvidence] = {}

    for evidence in evidence_items:
        if evidence.path_name != expected_path_name:
            raise ValueError(
                "Candidate evidence의 path가 잘못되었습니다: "
                f"expected={expected_path_name}, "
                f"actual={evidence.path_name}"
            )

        if evidence.candidate_index in evidence_map:
            raise ValueError(
                "중복된 candidate evidence index입니다: "
                f"path={expected_path_name}, "
                f"index={evidence.candidate_index}"
            )

        evidence_map[
            evidence.candidate_index
        ] = evidence

    return evidence_map


def _resolve_dino_loss(
    dino_score: BidirectionalDINOScore | None,
) -> tuple[float | None, bool]:
    """유효한 양방향 DINO loss를 반환한다."""

    if dino_score is None:
        return None, False

    if not dino_score.available:
        return None, False

    if dino_score.combined_loss is None:
        raise ValueError(
            "DINO available=True인데 combined_loss가 없습니다."
        )

    dino_loss = float(
        np.clip(
            dino_score.combined_loss,
            0.0,
            1.0,
        )
    )

    return dino_loss, True


def _score_candidate(
    *,
    candidate_index: int,
    candidate: RelativePoseCandidate,
    evidence: CandidateEvidence,
    weights: CandidateScoreWeights,
) -> CandidateCompositeScore:
    """Proxy 경로 후보 하나의 통합 loss를 계산한다."""

    if evidence.candidate_index != candidate_index:
        raise ValueError(
            "Candidate와 evidence index가 다릅니다."
        )

    if evidence.path_name != candidate.path_name:
        raise ValueError(
            "Candidate와 evidence path가 다릅니다: "
            f"candidate={candidate.path_name}, "
            f"evidence={evidence.path_name}"
        )

    cross_alignment_loss = float(
        np.clip(
            evidence.cross_alignment.total_loss,
            0.0,
            1.0,
        )
    )

    self_alignment_loss = (
        candidate.self_alignment_loss
    )

    if self_alignment_loss is not None:
        self_alignment_loss = float(
            np.clip(
                self_alignment_loss,
                0.0,
                1.0,
            )
        )

    dino_loss, dino_available = (
        _resolve_dino_loss(
            evidence.dino_score
        )
    )

    weighted_loss_sum = 0.0
    active_weight_sum = 0.0

    # Cross alignment는 필수 항목이다.
    weighted_loss_sum += (
        weights.cross_alignment
        * cross_alignment_loss
    )

    active_weight_sum += weights.cross_alignment

    if self_alignment_loss is not None:
        weighted_loss_sum += (
            weights.self_alignment
            * self_alignment_loss
        )

        active_weight_sum += (
            weights.self_alignment
        )

    if dino_available:
        assert dino_loss is not None

        weighted_loss_sum += (
            weights.dino
            * dino_loss
        )

        active_weight_sum += weights.dino

    if active_weight_sum <= 0.0:
        raise RuntimeError(
            "Candidate score의 활성 weight가 없습니다."
        )

    path_loss = (
        weighted_loss_sum
        / active_weight_sum
    )

    return CandidateCompositeScore(
        path_name=candidate.path_name,
        candidate_index=candidate_index,
        candidate=candidate,
        self_alignment_loss=(
            self_alignment_loss
        ),
        cross_alignment_loss=(
            cross_alignment_loss
        ),
        dino_loss=dino_loss,
        dino_available=dino_available,
        active_weight_sum=(
            active_weight_sum
        ),
        path_loss=float(
            np.clip(
                path_loss,
                0.0,
                1.0,
            )
        ),
    )


def _score_pair(
    *,
    pair: BidirectionalConsistencyPair,
    reference_evidence: CandidateEvidence,
    query_evidence: CandidateEvidence,
    candidate_weights: CandidateScoreWeights,
    pair_weights: PairScoreWeights,
) -> FinalPairScore:
    """양방향 후보 조합의 최종 loss를 계산한다."""

    reference_score = _score_candidate(
        candidate_index=(
            pair.reference_candidate_index
        ),
        candidate=pair.reference_candidate,
        evidence=reference_evidence,
        weights=candidate_weights,
    )

    query_score = _score_candidate(
        candidate_index=(
            pair.query_candidate_index
        ),
        candidate=pair.query_candidate,
        evidence=query_evidence,
        weights=candidate_weights,
    )

    mean_path_loss = (
        reference_score.path_loss
        + query_score.path_loss
    ) * 0.5

    # Consistency loss는 threshold 기준으로 정규화되어 있으므로
    # 최종 결합 전 [0, 1] 범위로 제한한다.
    normalized_consistency_loss = float(
        np.clip(
            pair.consistency_loss,
            0.0,
            1.0,
        )
    )

    weight_sum = (
        pair_weights.path_evidence
        + pair_weights.consistency
    )

    final_pair_loss = (
        pair_weights.path_evidence
        * mean_path_loss
        + pair_weights.consistency
        * normalized_consistency_loss
    ) / weight_sum

    return FinalPairScore(
        consistency_pair=pair,
        reference_score=reference_score,
        query_score=query_score,
        mean_path_loss=float(mean_path_loss),
        normalized_consistency_loss=(
            normalized_consistency_loss
        ),
        final_pair_loss=float(
            np.clip(
                final_pair_loss,
                0.0,
                1.0,
            )
        ),
    )


def _select_path_from_pair(
    pair_score: FinalPairScore,
) -> CandidateCompositeScore:
    """
    최종 pair 내부에서 더 신뢰도가 높은 proxy 경로를 선택한다.

    우선순위:
    1. path loss
    2. cross alignment loss
    3. FoundationPose cross score
    """

    reference_score = pair_score.reference_score
    query_score = pair_score.query_score

    reference_key = (
        reference_score.path_loss,
        reference_score.cross_alignment_loss,
        -reference_score.candidate.cross_foundationpose_score,
    )

    query_key = (
        query_score.path_loss,
        query_score.cross_alignment_loss,
        -query_score.candidate.cross_foundationpose_score,
    )

    if reference_key <= query_key:
        return reference_score

    return query_score


def _compute_confidence(
    *,
    best_loss: float,
    second_best_loss: float | None,
) -> tuple[float, float | None]:
    """
    최종 loss와 1·2위 score margin으로 상대 confidence를 계산한다.

    절대 확률이 아니라 후보 간 상대 신뢰도이다.
    """

    base_confidence = float(
        np.exp(-best_loss)
    )

    if second_best_loss is None:
        return (
            float(
                np.clip(
                    base_confidence,
                    0.0,
                    1.0,
                )
            ),
            None,
        )

    score_margin = max(
        0.0,
        second_best_loss - best_loss,
    )

    margin_factor = (
        1.0 - np.exp(-5.0 * score_margin)
    )

    confidence = (
        base_confidence
        * (
            0.5
            + 0.5 * margin_factor
        )
    )

    return (
        float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        ),
        float(score_margin),
    )


def select_final_candidate(
    *,
    consistency_result: BidirectionalConsistencyResult,
    reference_evidence: Sequence[CandidateEvidence],
    query_evidence: Sequence[CandidateEvidence],
    candidate_weights: CandidateScoreWeights = (
        CandidateScoreWeights()
    ),
    pair_weights: PairScoreWeights = (
        PairScoreWeights()
    ),
) -> FinalSelectionResult:
    """
    Mask/depth, DINO, 양방향 consistency를 결합해
    최종 relative pose와 proxy 경로를 선택한다.

    V1 정책
    -------
    Hard gate를 통과한 양방향 후보 조합만 최종 선택 대상이다.

    통과한 조합이 없으면:
        REJECT

    통과한 조합이 있으면:
        최저 final pair loss 조합 선택
        → 조합 내부에서 더 좋은 proxy 경로 선택
    """

    reference_evidence_map = _build_evidence_map(
        evidence_items=reference_evidence,
        expected_path_name="reference_proxy",
    )

    query_evidence_map = _build_evidence_map(
        evidence_items=query_evidence,
        expected_path_name="query_proxy",
    )

    evaluated_pair_scores: list[
        FinalPairScore
    ] = []

    for pair in consistency_result.pairs:
        reference_item = reference_evidence_map.get(
            pair.reference_candidate_index
        )

        query_item = query_evidence_map.get(
            pair.query_candidate_index
        )

        if reference_item is None:
            raise KeyError(
                "Reference candidate evidence가 없습니다: "
                f"{pair.reference_candidate_index}"
            )

        if query_item is None:
            raise KeyError(
                "Query candidate evidence가 없습니다: "
                f"{pair.query_candidate_index}"
            )

        pair_score = _score_pair(
            pair=pair,
            reference_evidence=reference_item,
            query_evidence=query_item,
            candidate_weights=candidate_weights,
            pair_weights=pair_weights,
        )

        evaluated_pair_scores.append(
            pair_score
        )

    sorted_pair_scores = tuple(
        sorted(
            evaluated_pair_scores,
            key=lambda item: (
                not item.consistency_pair.passes_hard_gate,
                item.final_pair_loss,
                item.mean_path_loss,
                item.normalized_consistency_loss,
            ),
        )
    )

    accepted_pair_scores = tuple(
        item
        for item in sorted_pair_scores
        if item.consistency_pair.passes_hard_gate
    )

    if not accepted_pair_scores:
        best_overall = sorted_pair_scores[0]

        return FinalSelectionResult(
            status="REJECT",
            selected_path_name=None,
            selected_candidate=None,
            selected_relative_pose_query_from_reference=None,
            selected_scale_m=None,
            final_loss=best_overall.final_pair_loss,
            confidence=0.0,
            score_margin=None,
            best_pair_score=None,
            evaluated_pair_scores=sorted_pair_scores,
        )

    best_pair_score = accepted_pair_scores[0]

    second_best_loss = (
        accepted_pair_scores[1].final_pair_loss
        if len(accepted_pair_scores) > 1
        else None
    )

    confidence, score_margin = (
        _compute_confidence(
            best_loss=best_pair_score.final_pair_loss,
            second_best_loss=second_best_loss,
        )
    )

    selected_path_score = (
        _select_path_from_pair(
            best_pair_score
        )
    )

    selected_candidate = (
        selected_path_score.candidate
    )

    selected_pose = np.ascontiguousarray(
        selected_candidate
        .relative_pose_query_from_reference,
        dtype=np.float32,
    )

    return FinalSelectionResult(
        status="CONSISTENT",
        selected_path_name=(
            selected_candidate.path_name
        ),
        selected_candidate=selected_candidate,
        selected_relative_pose_query_from_reference=(
            selected_pose
        ),
        selected_scale_m=(
            selected_candidate.scale_m
        ),
        final_loss=(
            best_pair_score.final_pair_loss
        ),
        confidence=confidence,
        score_margin=score_margin,
        best_pair_score=best_pair_score,
        evaluated_pair_scores=sorted_pair_scores,
    )


def _candidate_score_to_dict(
    score: CandidateCompositeScore,
) -> dict[str, object]:
    """Candidate score를 JSON 형식으로 변환한다."""

    return {
        "path_name": score.path_name,
        "candidate_index": score.candidate_index,
        "path_loss": score.path_loss,
        "self_alignment_loss": (
            score.self_alignment_loss
        ),
        "cross_alignment_loss": (
            score.cross_alignment_loss
        ),
        "dino_available": (
            score.dino_available
        ),
        "dino_loss": score.dino_loss,
        "active_weight_sum": (
            score.active_weight_sum
        ),
        "scale_m": score.candidate.scale_m,
        "cross_hypothesis_rank": (
            score.candidate.cross_hypothesis_rank
        ),
        "cross_foundationpose_score": (
            score
            .candidate
            .cross_foundationpose_score
        ),
    }


def _pair_score_to_dict(
    pair_score: FinalPairScore,
) -> dict[str, object]:
    """Final pair score를 JSON 형식으로 변환한다."""

    pair = pair_score.consistency_pair

    return {
        "reference_candidate_index": (
            pair.reference_candidate_index
        ),
        "query_candidate_index": (
            pair.query_candidate_index
        ),
        "passes_hard_gate": (
            pair.passes_hard_gate
        ),
        "rotation_difference_deg": (
            pair.rotation_difference_deg
        ),
        "translation_difference_m": (
            pair.translation_difference_m
        ),
        "translation_difference_normalized": (
            pair.translation_difference_normalized
        ),
        "scale_log_difference": (
            pair.scale_log_difference
        ),
        "consistency_loss": (
            pair.consistency_loss
        ),
        "normalized_consistency_loss": (
            pair_score.normalized_consistency_loss
        ),
        "mean_path_loss": (
            pair_score.mean_path_loss
        ),
        "final_pair_loss": (
            pair_score.final_pair_loss
        ),
        "reference_score": (
            _candidate_score_to_dict(
                pair_score.reference_score
            )
        ),
        "query_score": (
            _candidate_score_to_dict(
                pair_score.query_score
            )
        ),
    }


def save_final_selection(
    result: FinalSelectionResult,
    output_directory: Path,
    *,
    candidate_weights: CandidateScoreWeights = (
        CandidateScoreWeights()
    ),
    pair_weights: PairScoreWeights = (
        PairScoreWeights()
    ),
) -> tuple[Path, Path | None]:
    """
    최종 선택 결과를 JSON과 NPY로 저장한다.

    생성 파일
    ---------
    final_selection.json

    CONSISTENT인 경우:
        final_relative_pose.npy
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
        / "final_selection.json"
    )

    pose_path: Path | None = None

    if (
        result.selected_relative_pose_query_from_reference
        is not None
    ):
        pose_path = (
            output_directory
            / "final_relative_pose.npy"
        )

        np.save(
            pose_path,
            result
            .selected_relative_pose_query_from_reference,
            allow_pickle=False,
        )

    metadata = {
        "status": result.status,
        "pose_convention": (
            "T_query_camera_from_reference_camera"
        ),
        "translation_unit": "meter",
        "selected_path_name": (
            result.selected_path_name
        ),
        "selected_scale_m": (
            result.selected_scale_m
        ),
        "final_loss": result.final_loss,
        "confidence": result.confidence,
        "score_margin": result.score_margin,
        "final_relative_pose_path": (
            None
            if pose_path is None
            else str(pose_path)
        ),
        "candidate_weights": {
            "self_alignment": (
                candidate_weights.self_alignment
            ),
            "cross_alignment": (
                candidate_weights.cross_alignment
            ),
            "dino": candidate_weights.dino,
        },
        "pair_weights": {
            "path_evidence": (
                pair_weights.path_evidence
            ),
            "consistency": (
                pair_weights.consistency
            ),
        },
        "best_pair": (
            None
            if result.best_pair_score is None
            else _pair_score_to_dict(
                result.best_pair_score
            )
        ),
        "evaluated_pairs": [
            _pair_score_to_dict(pair_score)
            for pair_score
            in result.evaluated_pair_scores
        ],
        "confidence_interpretation": (
            "relative candidate confidence, "
            "not a calibrated probability"
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

    return metadata_path, pose_path