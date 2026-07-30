from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from core.types import PreparedView
from features.dino_pose_scorer import (
    BidirectionalDINOScore,
    save_bidirectional_dino_score,
    score_bidirectional_dino,
)
from features.dinov3_extractor import (
    DINOFeatureResult,
)
from features.observed_surface_features import (
    ObservedSurfaceFeatureResult,
)
from pose.alignment_evaluator import (
    AlignmentEvaluationResult,
    evaluate_foundationpose_alignments,
)
from pose.alignment_scorer import (
    AlignmentScoreResult,
    AlignmentScoreWeights,
)
from pose.cross_alignment_runner import (
    BidirectionalCrossAlignmentResult,
)
from pose.final_candidate_selector import (
    CandidateEvidence,
)
from pose.mesh_renderer import (
    FoundationPoseMeshRenderer,
)
from pose.relative_pose_builder import (
    BidirectionalRelativePoseCandidates,
    ProxyPathName,
    RelativePoseCandidate,
)


@dataclass(frozen=True)
class CrossPathEvidenceResult:
    """
    한 proxy 경로의 cross 후보 평가 결과.

    reference_proxy:
        Reference proxy를 Query에 정합한 후보들의 evidence.

    query_proxy:
        Query proxy를 Reference에 정합한 후보들의 evidence.
    """

    path_name: ProxyPathName

    evidences: tuple[
        CandidateEvidence,
        ...
    ]

    alignment_evaluation: AlignmentEvaluationResult
    summary_path: Path

    def __post_init__(self) -> None:
        if self.path_name not in (
            "reference_proxy",
            "query_proxy",
        ):
            raise ValueError(
                "지원하지 않는 proxy path입니다: "
                f"{self.path_name}"
            )

        if not self.evidences:
            raise ValueError(
                "Cross candidate evidence가 없습니다."
            )

        expected_indices = list(
            range(len(self.evidences))
        )

        actual_indices = [
            evidence.candidate_index
            for evidence in self.evidences
        ]

        if actual_indices != expected_indices:
            raise ValueError(
                "Candidate evidence index가 연속적이지 않습니다: "
                f"expected={expected_indices}, "
                f"actual={actual_indices}"
            )

        for evidence in self.evidences:
            if evidence.path_name != self.path_name:
                raise ValueError(
                    "다른 proxy path의 evidence가 "
                    "포함되어 있습니다."
                )

        if not self.summary_path.is_file():
            raise FileNotFoundError(
                "Cross path evidence summary가 없습니다: "
                f"{self.summary_path}"
            )


@dataclass(frozen=True)
class BidirectionalCrossEvidenceResult:
    """
    Reference proxy 경로와 Query proxy 경로의
    최종 후보 evidence.
    """

    reference_proxy: CrossPathEvidenceResult
    query_proxy: CrossPathEvidenceResult

    summary_path: Path

    def __post_init__(self) -> None:
        if (
            self.reference_proxy.path_name
            != "reference_proxy"
        ):
            raise ValueError(
                "Reference proxy evidence가 잘못되었습니다."
            )

        if (
            self.query_proxy.path_name
            != "query_proxy"
        ):
            raise ValueError(
                "Query proxy evidence가 잘못되었습니다."
            )

        if not self.summary_path.is_file():
            raise FileNotFoundError(
                "Bidirectional evidence summary가 없습니다: "
                f"{self.summary_path}"
            )


def _validate_relative_candidates(
    candidates: Sequence[RelativePoseCandidate],
    expected_path_name: ProxyPathName,
) -> None:
    """Relative pose 후보 집합을 검증한다."""

    if not candidates:
        raise ValueError(
            f"{expected_path_name} 후보가 없습니다."
        )

    for candidate in candidates:
        if candidate.path_name != expected_path_name:
            raise ValueError(
                "Relative pose 후보의 path가 다릅니다: "
                f"expected={expected_path_name}, "
                f"actual={candidate.path_name}"
            )


def _build_alignment_lookup(
    evaluation_result: AlignmentEvaluationResult,
) -> dict[
    tuple[int, int],
    AlignmentScoreResult,
]:
    """
    (scale candidate index, hypothesis rank)를 key로 하는
    alignment score lookup을 생성한다.
    """

    lookup: dict[
        tuple[int, int],
        AlignmentScoreResult,
    ] = {}

    for evaluation in evaluation_result.evaluations:
        key = (
            evaluation
            .candidate_result
            .candidate_index,
            evaluation.hypothesis.rank,
        )

        if key in lookup:
            raise ValueError(
                "중복된 alignment 평가 key입니다: "
                f"{key}"
            )

        lookup[key] = (
            evaluation.alignment_score
        )

    return lookup


def _validate_dino_inputs(
    *,
    enable_dino: bool,
    reference_surface: (
        ObservedSurfaceFeatureResult | None
    ),
    query_surface: (
        ObservedSurfaceFeatureResult | None
    ),
    reference_dino: DINOFeatureResult | None,
    query_dino: DINOFeatureResult | None,
) -> None:
    """
    DINO 평가를 활성화할 경우 필요한 네 입력을 확인한다.
    """

    if not enable_dino:
        return

    missing_inputs: list[str] = []

    if reference_surface is None:
        missing_inputs.append(
            "reference_surface"
        )

    if query_surface is None:
        missing_inputs.append(
            "query_surface"
        )

    if reference_dino is None:
        missing_inputs.append(
            "reference_dino"
        )

    if query_dino is None:
        missing_inputs.append(
            "query_dino"
        )

    if missing_inputs:
        raise ValueError(
            "DINO 평가 입력이 부족합니다: "
            f"{missing_inputs}"
        )

    assert reference_surface is not None
    assert query_surface is not None
    assert reference_dino is not None
    assert query_dino is not None

    if reference_surface.view_name != "reference":
        raise ValueError(
            "reference_surface의 view가 "
            "reference가 아닙니다."
        )

    if query_surface.view_name != "query":
        raise ValueError(
            "query_surface의 view가 query가 아닙니다."
        )

    if reference_dino.view_name != "reference":
        raise ValueError(
            "reference_dino의 view가 "
            "reference가 아닙니다."
        )

    if query_dino.view_name != "query":
        raise ValueError(
            "query_dino의 view가 query가 아닙니다."
        )


def _score_dino_candidate(
    *,
    candidate: RelativePoseCandidate,
    candidate_index: int,
    path_name: ProxyPathName,
    reference_surface: (
        ObservedSurfaceFeatureResult
    ),
    query_surface: (
        ObservedSurfaceFeatureResult
    ),
    reference_view: PreparedView,
    query_view: PreparedView,
    reference_dino: DINOFeatureResult,
    query_dino: DINOFeatureResult,
    output_directory: Path,
    depth_absolute_tolerance_m: float,
    depth_relative_tolerance: float,
    minimum_matched_points: int,
    minimum_coverage: float,
    coverage_weight: float,
    device: str,
    feature_chunk_size: int,
) -> BidirectionalDINOScore:
    """Relative pose 후보 하나를 양방향 DINO로 평가한다."""

    dino_score = score_bidirectional_dino(
        relative_pose_query_from_reference=(
            candidate
            .relative_pose_query_from_reference
        ),
        reference_surface=reference_surface,
        query_surface=query_surface,
        reference_view=reference_view,
        query_view=query_view,
        reference_dino=reference_dino,
        query_dino=query_dino,
        depth_absolute_tolerance_m=(
            depth_absolute_tolerance_m
        ),
        depth_relative_tolerance=(
            depth_relative_tolerance
        ),
        minimum_matched_points=(
            minimum_matched_points
        ),
        minimum_coverage=minimum_coverage,
        coverage_weight=coverage_weight,
        device=device,
        feature_chunk_size=feature_chunk_size,
    )

    dino_output_path = (
        output_directory
        / "dino"
        / f"candidate_{candidate_index:02d}.json"
    )

    save_bidirectional_dino_score(
        result=dino_score,
        output_path=dino_output_path,
        candidate_name=(
            f"{path_name}_{candidate_index:02d}"
        ),
        relative_pose_query_from_reference=(
            candidate
            .relative_pose_query_from_reference
        ),
    )

    return dino_score


def _evidence_to_dict(
    *,
    evidence: CandidateEvidence,
    candidate: RelativePoseCandidate,
) -> dict[str, object]:
    """Candidate evidence를 JSON 저장 형식으로 변환한다."""

    dino_score = evidence.dino_score

    return {
        "path_name": evidence.path_name,
        "candidate_index": (
            evidence.candidate_index
        ),
        "cross_candidate_index": (
            candidate.cross_candidate_index
        ),
        "cross_hypothesis_rank": (
            candidate.cross_hypothesis_rank
        ),
        "scale_m": candidate.scale_m,
        "cross_foundationpose_score": (
            candidate.cross_foundationpose_score
        ),
        "cross_alignment": asdict(
            evidence.cross_alignment
        ),
        "dino": (
            None
            if dino_score is None
            else {
                "available": (
                    dino_score.available
                ),
                "both_directions_available": (
                    dino_score
                    .both_directions_available
                ),
                "combined_loss": (
                    dino_score.combined_loss
                ),
                "total_matched_surface_count": (
                    dino_score
                    .total_matched_surface_count
                ),
                "reference_to_query": asdict(
                    dino_score.reference_to_query
                ),
                "query_to_reference": asdict(
                    dino_score.query_to_reference
                ),
            }
        ),
    }


def _build_path_evidences(
    *,
    path_name: ProxyPathName,
    candidates: Sequence[RelativePoseCandidate],
    alignment_evaluation: AlignmentEvaluationResult,
    reference_view: PreparedView,
    query_view: PreparedView,
    output_directory: Path,
    enable_dino: bool,
    reference_surface: (
        ObservedSurfaceFeatureResult | None
    ),
    query_surface: (
        ObservedSurfaceFeatureResult | None
    ),
    reference_dino: DINOFeatureResult | None,
    query_dino: DINOFeatureResult | None,
    depth_absolute_tolerance_m: float,
    depth_relative_tolerance: float,
    minimum_matched_points: int,
    minimum_coverage: float,
    coverage_weight: float,
    dino_device: str,
    feature_chunk_size: int,
) -> CrossPathEvidenceResult:
    """한 proxy 경로의 후보별 evidence를 생성한다."""

    _validate_relative_candidates(
        candidates=candidates,
        expected_path_name=path_name,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    alignment_lookup = (
        _build_alignment_lookup(
            alignment_evaluation
        )
    )

    evidences: list[
        CandidateEvidence
    ] = []

    evidence_metadata: list[
        dict[str, object]
    ] = []

    for candidate_index, candidate in enumerate(
        candidates
    ):
        alignment_key = (
            candidate.cross_candidate_index,
            candidate.cross_hypothesis_rank,
        )

        cross_alignment_score = (
            alignment_lookup.get(
                alignment_key
            )
        )

        if cross_alignment_score is None:
            raise KeyError(
                "Relative pose 후보에 대응하는 "
                "cross alignment score가 없습니다: "
                f"path={path_name}, "
                f"key={alignment_key}"
            )

        dino_score: (
            BidirectionalDINOScore | None
        ) = None

        if enable_dino:
            assert reference_surface is not None
            assert query_surface is not None
            assert reference_dino is not None
            assert query_dino is not None

            dino_score = _score_dino_candidate(
                candidate=candidate,
                candidate_index=candidate_index,
                path_name=path_name,
                reference_surface=reference_surface,
                query_surface=query_surface,
                reference_view=reference_view,
                query_view=query_view,
                reference_dino=reference_dino,
                query_dino=query_dino,
                output_directory=output_directory,
                depth_absolute_tolerance_m=(
                    depth_absolute_tolerance_m
                ),
                depth_relative_tolerance=(
                    depth_relative_tolerance
                ),
                minimum_matched_points=(
                    minimum_matched_points
                ),
                minimum_coverage=(
                    minimum_coverage
                ),
                coverage_weight=coverage_weight,
                device=dino_device,
                feature_chunk_size=(
                    feature_chunk_size
                ),
            )

        evidence = CandidateEvidence(
            path_name=path_name,
            candidate_index=candidate_index,
            cross_alignment=(
                cross_alignment_score
            ),
            dino_score=dino_score,
        )

        evidences.append(evidence)

        evidence_metadata.append(
            _evidence_to_dict(
                evidence=evidence,
                candidate=candidate,
            )
        )

    summary_path = (
        output_directory
        / "cross_candidate_evidence.json"
    )

    metadata = {
        "path_name": path_name,
        "candidate_count": len(evidences),
        "dino_enabled": enable_dino,
        "alignment_evaluation_path": str(
            alignment_evaluation.summary_path
        ),
        "candidates": evidence_metadata,
    }

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return CrossPathEvidenceResult(
        path_name=path_name,
        evidences=tuple(evidences),
        alignment_evaluation=(
            alignment_evaluation
        ),
        summary_path=summary_path,
    )


def evaluate_bidirectional_cross_evidence(
    *,
    cross_alignment: (
        BidirectionalCrossAlignmentResult
    ),
    relative_candidates: (
        BidirectionalRelativePoseCandidates
    ),
    reference_view: PreparedView,
    query_view: PreparedView,
    renderer: FoundationPoseMeshRenderer,
    output_directory: Path,
    alignment_weights: AlignmentScoreWeights = (
        AlignmentScoreWeights()
    ),
    enable_dino: bool = True,
    reference_surface: (
        ObservedSurfaceFeatureResult | None
    ) = None,
    query_surface: (
        ObservedSurfaceFeatureResult | None
    ) = None,
    reference_dino: DINOFeatureResult | None = None,
    query_dino: DINOFeatureResult | None = None,
    depth_trim_quantile: float = 0.90,
    min_depth_overlap_pixels: int = 50,
    free_space_absolute_tolerance_m: float = 0.005,
    free_space_relative_tolerance: float = 0.02,
    dino_depth_absolute_tolerance_m: float = 0.005,
    dino_depth_relative_tolerance: float = 0.02,
    dino_minimum_matched_points: int = 50,
    dino_minimum_coverage: float = 0.05,
    dino_coverage_weight: float = 0.25,
    dino_device: str = "cuda:0",
    dino_feature_chunk_size: int = 8192,
) -> BidirectionalCrossEvidenceResult:
    """
    두 cross-alignment 경로의 top-K 후보를
    mask/depth 및 DINO로 평가한다.

    처리 순서
    ---------
    Reference proxy → Query:
        Query mask/depth로 평가

    Query proxy → Reference:
        Reference mask/depth로 평가

    각 상대 pose:
        Reference↔Query 양방향 DINO로 평가
    """

    if reference_view.view.source.name != "reference":
        raise ValueError(
            "reference_view 이름이 reference가 아닙니다."
        )

    if query_view.view.source.name != "query":
        raise ValueError(
            "query_view 이름이 query가 아닙니다."
        )

    _validate_dino_inputs(
        enable_dino=enable_dino,
        reference_surface=reference_surface,
        query_surface=query_surface,
        reference_dino=reference_dino,
        query_dino=query_dino,
    )

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Reference proxy를 Query에 정합한 후보는
    # Query의 mask/depth로 평가한다.
    reference_cross_alignment_evaluation = (
        evaluate_foundationpose_alignments(
            prepared_view=query_view,
            candidate_results=(
                cross_alignment
                .reference_proxy_to_query
                .foundationpose_result,
            ),
            renderer=renderer,
            output_directory=(
                output_directory
                / "reference_proxy"
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

    # Query proxy를 Reference에 정합한 후보는
    # Reference의 mask/depth로 평가한다.
    query_cross_alignment_evaluation = (
        evaluate_foundationpose_alignments(
            prepared_view=reference_view,
            candidate_results=(
                cross_alignment
                .query_proxy_to_reference
                .foundationpose_result,
            ),
            renderer=renderer,
            output_directory=(
                output_directory
                / "query_proxy"
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

    reference_path_result = (
        _build_path_evidences(
            path_name="reference_proxy",
            candidates=(
                relative_candidates
                .reference_proxy_candidates
            ),
            alignment_evaluation=(
                reference_cross_alignment_evaluation
            ),
            reference_view=reference_view,
            query_view=query_view,
            output_directory=(
                output_directory
                / "reference_proxy"
            ),
            enable_dino=enable_dino,
            reference_surface=reference_surface,
            query_surface=query_surface,
            reference_dino=reference_dino,
            query_dino=query_dino,
            depth_absolute_tolerance_m=(
                dino_depth_absolute_tolerance_m
            ),
            depth_relative_tolerance=(
                dino_depth_relative_tolerance
            ),
            minimum_matched_points=(
                dino_minimum_matched_points
            ),
            minimum_coverage=(
                dino_minimum_coverage
            ),
            coverage_weight=(
                dino_coverage_weight
            ),
            dino_device=dino_device,
            feature_chunk_size=(
                dino_feature_chunk_size
            ),
        )
    )

    query_path_result = (
        _build_path_evidences(
            path_name="query_proxy",
            candidates=(
                relative_candidates
                .query_proxy_candidates
            ),
            alignment_evaluation=(
                query_cross_alignment_evaluation
            ),
            reference_view=reference_view,
            query_view=query_view,
            output_directory=(
                output_directory
                / "query_proxy"
            ),
            enable_dino=enable_dino,
            reference_surface=reference_surface,
            query_surface=query_surface,
            reference_dino=reference_dino,
            query_dino=query_dino,
            depth_absolute_tolerance_m=(
                dino_depth_absolute_tolerance_m
            ),
            depth_relative_tolerance=(
                dino_depth_relative_tolerance
            ),
            minimum_matched_points=(
                dino_minimum_matched_points
            ),
            minimum_coverage=(
                dino_minimum_coverage
            ),
            coverage_weight=(
                dino_coverage_weight
            ),
            dino_device=dino_device,
            feature_chunk_size=(
                dino_feature_chunk_size
            ),
        )
    )

    summary_path = (
        output_directory
        / "bidirectional_cross_evidence.json"
    )

    metadata = {
        "dino_enabled": enable_dino,
        "reference_proxy": {
            "candidate_count": len(
                reference_path_result.evidences
            ),
            "summary_path": str(
                reference_path_result.summary_path
            ),
        },
        "query_proxy": {
            "candidate_count": len(
                query_path_result.evidences
            ),
            "summary_path": str(
                query_path_result.summary_path
            ),
        },
        "alignment_weights": {
            "mask": alignment_weights.mask,
            "depth": alignment_weights.depth,
            "free_space": (
                alignment_weights.free_space
            ),
            "boundary": (
                alignment_weights.boundary
            ),
        },
    }

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return BidirectionalCrossEvidenceResult(
        reference_proxy=reference_path_result,
        query_proxy=query_path_result,
        summary_path=summary_path,
    )