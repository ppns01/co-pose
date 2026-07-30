from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from core.types import PreparedView
from pose.alignment_scorer import (
    AlignmentScoreResult,
    AlignmentScoreWeights,
    save_alignment_score,
    score_alignment,
)
from pose.foundationpose_runner import (
    FoundationPoseCandidateResult,
    FoundationPoseHypothesis,
)
from pose.mesh_renderer import FoundationPoseMeshRenderer
from pose.relative_pose_builder import (
    SelfAlignmentSelection,
    select_self_alignment,
)


@dataclass(frozen=True)
class HypothesisAlignmentEvaluation:
    """
    FoundationPose pose 후보 하나의 외부 정합 평가 결과.
    """

    candidate_result: FoundationPoseCandidateResult
    hypothesis: FoundationPoseHypothesis
    alignment_score: AlignmentScoreResult

    render_directory: Path
    score_path: Path

    def __post_init__(self) -> None:
        if (
            self.candidate_result.candidate_index < 0
            or self.hypothesis.rank < 0
        ):
            raise ValueError(
                "Candidate index와 hypothesis rank는 "
                "0 이상이어야 합니다."
            )

        available_ranks = {
            item.rank
            for item in self.candidate_result.hypotheses
        }

        if self.hypothesis.rank not in available_ranks:
            raise ValueError(
                "Hypothesis가 candidate result에 "
                "포함되어 있지 않습니다."
            )

        if not self.render_directory.is_dir():
            raise FileNotFoundError(
                "렌더링 출력 폴더가 없습니다: "
                f"{self.render_directory}"
            )

        if not self.score_path.is_file():
            raise FileNotFoundError(
                "Alignment score 파일이 없습니다: "
                f"{self.score_path}"
            )


@dataclass(frozen=True)
class AlignmentEvaluationResult:
    """
    한 관측 영상에 대한 모든 scale·pose 후보 평가 결과.

    evaluations:
        alignment loss 오름차순으로 정렬된다.
    """

    view_name: str
    evaluations: tuple[
        HypothesisAlignmentEvaluation,
        ...
    ]
    summary_path: Path

    def __post_init__(self) -> None:
        if self.view_name not in (
            "reference",
            "query",
        ):
            raise ValueError(
                "지원하지 않는 view입니다: "
                f"{self.view_name}"
            )

        if not self.evaluations:
            raise ValueError(
                "Alignment 평가 결과가 없습니다."
            )

        for evaluation in self.evaluations:
            if (
                evaluation.candidate_result.view_name
                != self.view_name
            ):
                raise ValueError(
                    "평가 결과에 다른 view의 후보가 "
                    "포함되어 있습니다."
                )

        if not self.summary_path.is_file():
            raise FileNotFoundError(
                "Alignment summary 파일이 없습니다: "
                f"{self.summary_path}"
            )

    @property
    def best(self) -> HypothesisAlignmentEvaluation:
        """최저 alignment loss 후보를 반환한다."""

        return self.evaluations[0]


def _validate_candidate_results(
    prepared_view: PreparedView,
    candidate_results: Sequence[
        FoundationPoseCandidateResult
    ],
) -> None:
    """FoundationPose 결과와 대상 view를 검증한다."""

    if not candidate_results:
        raise ValueError(
            "평가할 FoundationPose 결과가 없습니다."
        )

    view_name = prepared_view.view.source.name

    seen_candidate_indices: set[int] = set()

    for result in candidate_results:
        if result.view_name != view_name:
            raise ValueError(
                "FoundationPose 결과와 관측 view가 다릅니다: "
                f"view={view_name}, "
                f"result={result.view_name}"
            )

        if result.candidate_index in seen_candidate_indices:
            raise ValueError(
                "중복된 scale candidate index입니다: "
                f"{result.candidate_index}"
            )

        seen_candidate_indices.add(
            result.candidate_index
        )

        if not result.hypotheses:
            raise ValueError(
                "FoundationPose hypothesis가 없습니다: "
                f"candidate={result.candidate_index}"
            )

        if not result.scaled_mesh_path.is_file():
            raise FileNotFoundError(
                "Scaled mesh가 없습니다: "
                f"{result.scaled_mesh_path}"
            )


def _evaluate_candidate_result(
    *,
    prepared_view: PreparedView,
    candidate_result: FoundationPoseCandidateResult,
    renderer: FoundationPoseMeshRenderer,
    output_directory: Path,
    weights: AlignmentScoreWeights,
    depth_trim_quantile: float,
    min_depth_overlap_pixels: int,
    free_space_absolute_tolerance_m: float,
    free_space_relative_tolerance: float,
) -> list[HypothesisAlignmentEvaluation]:
    """
    Scale 후보 하나의 FoundationPose top-K를 렌더링하고 평가한다.
    """

    hypotheses = tuple(
        sorted(
            candidate_result.hypotheses,
            key=lambda item: item.rank,
        )
    )

    poses = np.stack(
        [
            hypothesis.pose_cam_from_proxy
            for hypothesis in hypotheses
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    candidate_directory = (
        output_directory
        / f"candidate_{candidate_result.candidate_index:02d}"
    )

    render_directory = (
        candidate_directory / "render"
    )

    score_directory = (
        candidate_directory / "scores"
    )

    score_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    render_result = renderer.render(
        mesh_path=candidate_result.scaled_mesh_path,
        poses_camera_from_proxy=poses,
        camera_matrix=(
            prepared_view.view.camera_matrix
        ),
        image_height=(
            prepared_view.view.rgb.shape[0]
        ),
        image_width=(
            prepared_view.view.rgb.shape[1]
        ),
        output_directory=render_directory,
        filename_prefix="hypothesis",
    )

    if (
        render_result.rendered_masks.shape[0]
        != len(hypotheses)
    ):
        raise RuntimeError(
            "렌더링 결과 개수와 FoundationPose "
            "후보 개수가 다릅니다."
        )

    evaluations: list[
        HypothesisAlignmentEvaluation
    ] = []

    for render_index, hypothesis in enumerate(
        hypotheses
    ):
        alignment_score = score_alignment(
            observed_mask=(
                prepared_view
                .segmentation
                .mask_bool
            ),
            observed_depth_m=(
                prepared_view.view.depth_m
            ),
            rendered_mask=(
                render_result
                .rendered_masks[render_index]
            ),
            rendered_depth_m=(
                render_result
                .rendered_depth_m[render_index]
            ),
            object_scale_m=(
                candidate_result.scale_m
            ),
            weights=weights,
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

        score_path = (
            score_directory
            / (
                f"hypothesis_"
                f"{hypothesis.rank:02d}.json"
            )
        )

        save_alignment_score(
            result=alignment_score,
            output_path=score_path,
            view_name=(
                prepared_view.view.source.name
            ),
            candidate_index=(
                candidate_result.candidate_index
            ),
            hypothesis_rank=hypothesis.rank,
            scale_m=candidate_result.scale_m,
        )

        evaluations.append(
            HypothesisAlignmentEvaluation(
                candidate_result=candidate_result,
                hypothesis=hypothesis,
                alignment_score=alignment_score,
                render_directory=render_directory,
                score_path=score_path,
            )
        )

    return evaluations


def _evaluation_to_dict(
    evaluation: HypothesisAlignmentEvaluation,
) -> dict[str, object]:
    """평가 결과를 JSON 저장 형식으로 변환한다."""

    result = evaluation.candidate_result
    hypothesis = evaluation.hypothesis
    score = evaluation.alignment_score

    return {
        "candidate_index": result.candidate_index,
        "hypothesis_rank": hypothesis.rank,
        "scale_m": result.scale_m,
        "scaled_mesh_path": str(
            result.scaled_mesh_path
        ),
        "foundationpose_score": hypothesis.score,
        "alignment_score": asdict(score),
        "render_directory": str(
            evaluation.render_directory
        ),
        "score_path": str(
            evaluation.score_path
        ),
    }


def evaluate_foundationpose_alignments(
    *,
    prepared_view: PreparedView,
    candidate_results: Sequence[
        FoundationPoseCandidateResult
    ],
    renderer: FoundationPoseMeshRenderer,
    output_directory: Path,
    weights: AlignmentScoreWeights = (
        AlignmentScoreWeights()
    ),
    depth_trim_quantile: float = 0.90,
    min_depth_overlap_pixels: int = 50,
    free_space_absolute_tolerance_m: float = 0.005,
    free_space_relative_tolerance: float = 0.02,
) -> AlignmentEvaluationResult:
    """
    모든 scale·pose 후보를 동일한 mask/depth 기준으로 평가한다.

    Self-alignment에서 사용할 경우:
        여러 scale candidate × top-K pose 전체 평가

    Cross-alignment에서 사용할 경우:
        선택된 scale의 top-K pose 평가
    """

    _validate_candidate_results(
        prepared_view=prepared_view,
        candidate_results=candidate_results,
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

    all_evaluations: list[
        HypothesisAlignmentEvaluation
    ] = []

    for candidate_result in candidate_results:
        candidate_evaluations = (
            _evaluate_candidate_result(
                prepared_view=prepared_view,
                candidate_result=(
                    candidate_result
                ),
                renderer=renderer,
                output_directory=(
                    output_directory
                ),
                weights=weights,
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

        all_evaluations.extend(
            candidate_evaluations
        )

    sorted_evaluations = tuple(
        sorted(
            all_evaluations,
            key=lambda item: (
                item.alignment_score.total_loss,
                item.alignment_score.depth_loss,
                item.alignment_score.mask_loss,
                -item.hypothesis.score,
                item.candidate_result.candidate_index,
                item.hypothesis.rank,
            ),
        )
    )

    if not sorted_evaluations:
        raise RuntimeError(
            "Alignment 평가 결과가 생성되지 않았습니다."
        )

    summary_path = (
        output_directory
        / "alignment_evaluation.json"
    )

    best_evaluation = sorted_evaluations[0]

    metadata = {
        "view_name": (
            prepared_view.view.source.name
        ),
        "lower_is_better": True,
        "candidate_count": len(
            candidate_results
        ),
        "hypothesis_count": len(
            sorted_evaluations
        ),
        "weights": {
            "mask": weights.mask,
            "depth": weights.depth,
            "free_space": weights.free_space,
            "boundary": weights.boundary,
        },
        "parameters": {
            "depth_trim_quantile": (
                depth_trim_quantile
            ),
            "min_depth_overlap_pixels": (
                min_depth_overlap_pixels
            ),
            "free_space_absolute_tolerance_m": (
                free_space_absolute_tolerance_m
            ),
            "free_space_relative_tolerance": (
                free_space_relative_tolerance
            ),
        },
        "best": _evaluation_to_dict(
            best_evaluation
        ),
        "evaluations": [
            _evaluation_to_dict(item)
            for item in sorted_evaluations
        ],
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

    return AlignmentEvaluationResult(
        view_name=(
            prepared_view.view.source.name
        ),
        evaluations=sorted_evaluations,
        summary_path=summary_path,
    )


def select_best_self_alignment(
    evaluation_result: AlignmentEvaluationResult,
) -> SelfAlignmentSelection:
    """
    Self-alignment 평가에서 최저 loss의 scale·pose를 선택한다.

    반환 결과는 이후 cross-alignment에서
    동일한 scaled mesh를 재사용하는 데 사용한다.
    """

    best = evaluation_result.best

    return select_self_alignment(
        result=best.candidate_result,
        hypothesis_rank=(
            best.hypothesis.rank
        ),
        alignment_loss=(
            best.alignment_score.total_loss
        ),
    )