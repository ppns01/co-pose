from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from core.types import PreparedView, ViewName

if TYPE_CHECKING:
    from pose.foundationpose_runner import (
        FoundationPoseCandidateResult,
        FoundationPoseRunner,
    )
    from pose.relative_pose_builder import (
        SelfAlignmentSelection,
    )
    from scale.mesh_scaler import (
        ScaledMeshCandidate,
    )


CrossAlignmentPath = Literal[
    "reference_proxy_to_query",
    "query_proxy_to_reference",
]


@dataclass(frozen=True)
class CrossAlignmentResult:
    """
    한 proxy를 반대편 RGB-D에 정합한 결과.

    reference_proxy_to_query:
        Reference proxy Pr를 Query에 정합한다.
        출력 pose는 T_Cq_from_Pr이다.

    query_proxy_to_reference:
        Query proxy Pq를 Reference에 정합한다.
        출력 pose는 T_Cr_from_Pq이다.
    """

    path_name: CrossAlignmentPath

    source_proxy_view: ViewName
    target_view: ViewName

    self_alignment: SelfAlignmentSelection
    scaled_mesh_candidate: ScaledMeshCandidate

    foundationpose_result: FoundationPoseCandidateResult
    metadata_path: Path

    def __post_init__(self) -> None:
        if self.path_name not in (
            "reference_proxy_to_query",
            "query_proxy_to_reference",
        ):
            raise ValueError(
                "지원하지 않는 cross-alignment 경로입니다: "
                f"{self.path_name}"
            )

        if self.source_proxy_view == self.target_view:
            raise ValueError(
                "Cross-alignment의 source와 target view는 "
                "서로 달라야 합니다."
            )

        expected_path = _resolve_path_name(
            source_proxy_view=self.source_proxy_view,
            target_view=self.target_view,
        )

        if self.path_name != expected_path:
            raise ValueError(
                "Cross-alignment path 이름이 source/target과 "
                "일치하지 않습니다: "
                f"expected={expected_path}, "
                f"actual={self.path_name}"
            )

        if (
            self.self_alignment.proxy_view
            != self.source_proxy_view
        ):
            raise ValueError(
                "Self-alignment의 proxy view가 "
                "cross source와 다릅니다: "
                f"self={self.self_alignment.proxy_view}, "
                f"source={self.source_proxy_view}"
            )

        if (
            self.foundationpose_result.view_name
            != self.target_view
        ):
            raise ValueError(
                "FoundationPose 결과의 view가 "
                "cross target과 다릅니다: "
                f"result={self.foundationpose_result.view_name}, "
                f"target={self.target_view}"
            )

        if not self.foundationpose_result.hypotheses:
            raise ValueError(
                "Cross-alignment pose 후보가 없습니다."
            )

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                "Cross-alignment metadata가 없습니다: "
                f"{self.metadata_path}"
            )


@dataclass(frozen=True)
class BidirectionalCrossAlignmentResult:
    """두 방향의 cross-alignment 결과."""

    reference_proxy_to_query: CrossAlignmentResult
    query_proxy_to_reference: CrossAlignmentResult

    summary_path: Path

    def __post_init__(self) -> None:
        if (
            self.reference_proxy_to_query.path_name
            != "reference_proxy_to_query"
        ):
            raise ValueError(
                "Reference proxy cross 결과가 잘못되었습니다."
            )

        if (
            self.query_proxy_to_reference.path_name
            != "query_proxy_to_reference"
        ):
            raise ValueError(
                "Query proxy cross 결과가 잘못되었습니다."
            )

        if not self.summary_path.is_file():
            raise FileNotFoundError(
                "Bidirectional cross summary가 없습니다: "
                f"{self.summary_path}"
            )


def _resolve_path_name(
    source_proxy_view: ViewName,
    target_view: ViewName,
) -> CrossAlignmentPath:
    """Source proxy와 target view로 경로 이름을 결정한다."""

    if (
        source_proxy_view == "reference"
        and target_view == "query"
    ):
        return "reference_proxy_to_query"

    if (
        source_proxy_view == "query"
        and target_view == "reference"
    ):
        return "query_proxy_to_reference"

    raise ValueError(
        "지원하지 않는 cross-alignment 방향입니다: "
        f"{source_proxy_view} → {target_view}"
    )


def _validate_candidate_matches_self(
    *,
    self_alignment: SelfAlignmentSelection,
    scaled_mesh_candidate: ScaledMeshCandidate,
) -> None:
    """
    Self-alignment에서 선택한 mesh와 cross 단계의 mesh가
    완전히 동일한지 검사한다.

    이 조건이 깨지면 proxy local frame이 소거되지 않는다.
    """

    self_mesh_path = (
        self_alignment
        .scaled_mesh_path
        .expanduser()
        .resolve()
    )

    candidate_mesh_path = (
        scaled_mesh_candidate
        .scaled_mesh_path
        .expanduser()
        .resolve()
    )

    if self_mesh_path != candidate_mesh_path:
        raise ValueError(
            "Self와 cross에 서로 다른 scaled mesh가 "
            "사용되었습니다.\n"
            f"Self : {self_mesh_path}\n"
            f"Cross: {candidate_mesh_path}"
        )

    if (
        self_alignment.candidate_index
        != scaled_mesh_candidate.candidate_index
    ):
        raise ValueError(
            "Self와 cross의 scale candidate index가 "
            "다릅니다: "
            f"self={self_alignment.candidate_index}, "
            f"cross={scaled_mesh_candidate.candidate_index}"
        )

    if not np.isclose(
        self_alignment.scale_m,
        scaled_mesh_candidate.scale_m,
        atol=1e-8,
        rtol=1e-6,
    ):
        raise ValueError(
            "Self와 cross의 scale 값이 다릅니다: "
            f"self={self_alignment.scale_m}, "
            f"cross={scaled_mesh_candidate.scale_m}"
        )


def _validate_target_view(
    *,
    source_proxy_view: ViewName,
    target_view: PreparedView,
) -> ViewName:
    """Cross-alignment target view를 검증한다."""

    target_view_name = target_view.view.source.name

    if target_view_name == source_proxy_view:
        raise ValueError(
            "Proxy를 자기 view에 다시 정합하려고 합니다. "
            "Cross-alignment target은 반대편 view여야 합니다."
        )

    return target_view_name


def _validate_foundationpose_result(
    *,
    result: FoundationPoseCandidateResult,
    scaled_mesh_candidate: ScaledMeshCandidate,
    target_view_name: ViewName,
) -> None:
    """FoundationPose cross 결과를 검증한다."""

    if result.view_name != target_view_name:
        raise ValueError(
            "FoundationPose 결과 target view가 다릅니다: "
            f"expected={target_view_name}, "
            f"actual={result.view_name}"
        )

    if (
        result.candidate_index
        != scaled_mesh_candidate.candidate_index
    ):
        raise ValueError(
            "FoundationPose 결과의 candidate index가 "
            "입력 candidate와 다릅니다."
        )

    if not np.isclose(
        result.scale_m,
        scaled_mesh_candidate.scale_m,
        atol=1e-8,
        rtol=1e-6,
    ):
        raise ValueError(
            "FoundationPose 결과 scale이 입력 scale과 "
            "다릅니다: "
            f"input={scaled_mesh_candidate.scale_m}, "
            f"result={result.scale_m}"
        )

    result_mesh_path = (
        result.scaled_mesh_path
        .expanduser()
        .resolve()
    )

    input_mesh_path = (
        scaled_mesh_candidate
        .scaled_mesh_path
        .expanduser()
        .resolve()
    )

    if result_mesh_path != input_mesh_path:
        raise ValueError(
            "FoundationPose가 다른 scaled mesh를 "
            "사용했습니다:\n"
            f"input={input_mesh_path}\n"
            f"result={result_mesh_path}"
        )

    if not result.hypotheses:
        raise ValueError(
            "FoundationPose cross pose 후보가 없습니다."
        )


def _save_cross_metadata(
    *,
    path_name: CrossAlignmentPath,
    self_alignment: SelfAlignmentSelection,
    scaled_mesh_candidate: ScaledMeshCandidate,
    target_view_name: ViewName,
    foundationpose_result: FoundationPoseCandidateResult,
    output_directory: Path,
) -> Path:
    """한 방향의 cross-alignment 정보를 저장한다."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        output_directory
        / "cross_alignment.json"
    )

    metadata = {
        "path_name": path_name,
        "source_proxy_view": self_alignment.proxy_view,
        "target_view": target_view_name,
        "scale_fixed": True,
        "scale_m": self_alignment.scale_m,
        "candidate_index": (
            self_alignment.candidate_index
        ),
        "scaled_mesh_path": str(
            scaled_mesh_candidate.scaled_mesh_path
        ),
        "self_alignment": {
            "proxy_view": self_alignment.proxy_view,
            "candidate_index": (
                self_alignment.candidate_index
            ),
            "hypothesis_rank": (
                self_alignment.hypothesis_rank
            ),
            "foundationpose_score": (
                self_alignment.foundationpose_score
            ),
            "alignment_loss": (
                self_alignment.alignment_loss
            ),
            "pose_camera_from_proxy": (
                self_alignment
                .pose_camera_from_proxy
                .tolist()
            ),
        },
        "cross_alignment": {
            "target_view": (
                foundationpose_result.view_name
            ),
            "top_k": len(
                foundationpose_result.hypotheses
            ),
            "foundationpose_output_directory": str(
                foundationpose_result.output_directory
            ),
            "hypotheses": [
                {
                    "rank": hypothesis.rank,
                    "score": hypothesis.score,
                    "pose_camera_from_proxy": (
                        hypothesis
                        .pose_cam_from_proxy
                        .tolist()
                    ),
                }
                for hypothesis
                in foundationpose_result.hypotheses
            ],
        },
        "coordinate_rule": (
            "The same scaled proxy mesh and local frame "
            "are used in self and cross alignment."
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
            "Cross-alignment metadata가 저장되지 않았습니다: "
            f"{metadata_path}"
        )

    return metadata_path


def finalize_cross_alignment(
    *,
    foundationpose_result: FoundationPoseCandidateResult,
    self_alignment: SelfAlignmentSelection,
    scaled_mesh_candidate: ScaledMeshCandidate,
    target_view: PreparedView,
    output_directory: Path,
) -> CrossAlignmentResult:
    """
    이미 계산된 FoundationPose 결과를 검증하고 cross 결과로 저장한다.
    """

    _validate_candidate_matches_self(
        self_alignment=self_alignment,
        scaled_mesh_candidate=scaled_mesh_candidate,
    )

    target_view_name = _validate_target_view(
        source_proxy_view=self_alignment.proxy_view,
        target_view=target_view,
    )

    path_name = _resolve_path_name(
        source_proxy_view=self_alignment.proxy_view,
        target_view=target_view_name,
    )

    _validate_foundationpose_result(
        result=foundationpose_result,
        scaled_mesh_candidate=scaled_mesh_candidate,
        target_view_name=target_view_name,
    )

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    metadata_path = _save_cross_metadata(
        path_name=path_name,
        self_alignment=self_alignment,
        scaled_mesh_candidate=scaled_mesh_candidate,
        target_view_name=target_view_name,
        foundationpose_result=foundationpose_result,
        output_directory=output_directory,
    )

    return CrossAlignmentResult(
        path_name=path_name,
        source_proxy_view=self_alignment.proxy_view,
        target_view=target_view_name,
        self_alignment=self_alignment,
        scaled_mesh_candidate=scaled_mesh_candidate,
        foundationpose_result=foundationpose_result,
        metadata_path=metadata_path,
    )


def run_cross_alignment(
    *,
    runner: FoundationPoseRunner,
    self_alignment: SelfAlignmentSelection,
    scaled_mesh_candidate: ScaledMeshCandidate,
    target_view: PreparedView,
    output_directory: Path,
) -> CrossAlignmentResult:
    """
    선택된 self scale과 동일한 proxy mesh를 반대편 view에 정합한다.

    중요:
        runner의 output_root는 self 결과와 겹치지 않는
        cross 전용 폴더로 지정해야 한다.
    """

    _validate_candidate_matches_self(
        self_alignment=self_alignment,
        scaled_mesh_candidate=scaled_mesh_candidate,
    )
    _validate_target_view(
        source_proxy_view=self_alignment.proxy_view,
        target_view=target_view,
    )

    foundationpose_result = runner.run_candidate(
        candidate=scaled_mesh_candidate,
        prepared_view=target_view,
    )

    return finalize_cross_alignment(
        foundationpose_result=foundationpose_result,
        self_alignment=self_alignment,
        scaled_mesh_candidate=scaled_mesh_candidate,
        target_view=target_view,
        output_directory=output_directory,
    )


def combine_bidirectional_cross_alignment_results(
    *,
    reference_proxy_to_query: CrossAlignmentResult,
    query_proxy_to_reference: CrossAlignmentResult,
    output_directory: Path,
) -> BidirectionalCrossAlignmentResult:
    """두 방향의 검증된 cross 결과를 저장하고 하나로 묶는다."""

    if (
        reference_proxy_to_query.path_name
        != "reference_proxy_to_query"
    ):
        raise ValueError(
            "Reference proxy cross 결과가 잘못되었습니다."
        )

    if (
        query_proxy_to_reference.path_name
        != "query_proxy_to_reference"
    ):
        raise ValueError(
            "Query proxy cross 결과가 잘못되었습니다."
        )

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )
    summary_path = (
        output_directory
        / "bidirectional_cross_alignment.json"
    )

    summary = {
        "pose_convention": {
            "reference_proxy_to_query": (
                "T_query_camera_from_reference_proxy"
            ),
            "query_proxy_to_reference": (
                "T_reference_camera_from_query_proxy"
            ),
        },
        "scale_policy": (
            "Each proxy keeps the scale selected during "
            "self-alignment."
        ),
        "reference_proxy_to_query": {
            "scale_m": (
                reference_proxy_to_query
                .self_alignment
                .scale_m
            ),
            "mesh_path": str(
                reference_proxy_to_query
                .scaled_mesh_candidate
                .scaled_mesh_path
            ),
            "metadata_path": str(
                reference_proxy_to_query
                .metadata_path
            ),
            "top_k": len(
                reference_proxy_to_query
                .foundationpose_result
                .hypotheses
            ),
        },
        "query_proxy_to_reference": {
            "scale_m": (
                query_proxy_to_reference
                .self_alignment
                .scale_m
            ),
            "mesh_path": str(
                query_proxy_to_reference
                .scaled_mesh_candidate
                .scaled_mesh_path
            ),
            "metadata_path": str(
                query_proxy_to_reference
                .metadata_path
            ),
            "top_k": len(
                query_proxy_to_reference
                .foundationpose_result
                .hypotheses
            ),
        },
    }

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return BidirectionalCrossAlignmentResult(
        reference_proxy_to_query=(
            reference_proxy_to_query
        ),
        query_proxy_to_reference=(
            query_proxy_to_reference
        ),
        summary_path=summary_path,
    )


def run_bidirectional_cross_alignment(
    *,
    runner: FoundationPoseRunner,
    reference_self: SelfAlignmentSelection,
    query_self: SelfAlignmentSelection,
    reference_scaled_candidate: ScaledMeshCandidate,
    query_scaled_candidate: ScaledMeshCandidate,
    reference_view: PreparedView,
    query_view: PreparedView,
    output_directory: Path,
) -> BidirectionalCrossAlignmentResult:
    """
    Reference proxy→Query와 Query proxy→Reference를 연속 실행한다.

    동일 FoundationPoseRunner를 재사용하므로 scorer/refiner를
    다시 로드하지 않고 mesh만 reset_object()로 교체한다.
    """

    if reference_self.proxy_view != "reference":
        raise ValueError(
            "reference_self의 proxy_view는 "
            "'reference'여야 합니다."
        )

    if query_self.proxy_view != "query":
        raise ValueError(
            "query_self의 proxy_view는 "
            "'query'여야 합니다."
        )

    if reference_view.view.source.name != "reference":
        raise ValueError(
            "reference_view의 이름이 reference가 아닙니다."
        )

    if query_view.view.source.name != "query":
        raise ValueError(
            "query_view의 이름이 query가 아닙니다."
        )

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    reference_to_query_result = (
        run_cross_alignment(
            runner=runner,
            self_alignment=reference_self,
            scaled_mesh_candidate=(
                reference_scaled_candidate
            ),
            target_view=query_view,
            output_directory=(
                output_directory
                / "reference_proxy_to_query"
            ),
        )
    )

    query_to_reference_result = (
        run_cross_alignment(
            runner=runner,
            self_alignment=query_self,
            scaled_mesh_candidate=(
                query_scaled_candidate
            ),
            target_view=reference_view,
            output_directory=(
                output_directory
                / "query_proxy_to_reference"
            ),
        )
    )

    return combine_bidirectional_cross_alignment_results(
        reference_proxy_to_query=(
            reference_to_query_result
        ),
        query_proxy_to_reference=(
            query_to_reference_result
        ),
        output_directory=output_directory,
    )
