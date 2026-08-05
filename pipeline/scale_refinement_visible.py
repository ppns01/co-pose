from __future__ import annotations

from pathlib import Path
from typing import Sequence

from core.types import AlignedProxyState, PipelineConfig
from pipeline.scale_refinement_joint_shared import (
    _refine_aligned_states_joint_shared_scale,
)
from pipeline.scale_refinement_visible_independent import (
    _refine_aligned_states_visible_scale_independent,
)


def _normalized_visible_scale_policy(config: PipelineConfig) -> str:
    return config.visible_scale_policy.strip().lower()


def _pair_visible_scale_required(config: PipelineConfig) -> bool:
    return (
        bool(config.visible_scale_refinement_enabled)
        and _normalized_visible_scale_policy(config) == "joint_shared"
    )


def _refine_aligned_states_visible_scale(
    *,
    config: PipelineConfig,
    aligned_states: Sequence[AlignedProxyState],
    output_root: Path,
) -> tuple[AlignedProxyState, ...]:
    """
    Visible-scale 정책 dispatcher.

    scale.visible_refinement.policy:
        independent:
            기존 view별 독립 refinement.

        joint_shared:
            동일 absolute scale bank를 Reference와 Query에 적용하고,
            양쪽 self-alignment를 함께 설명하는 하나의 scale을 선택.
    """
    policy = _normalized_visible_scale_policy(config)

    if policy == "independent":
        return _refine_aligned_states_visible_scale_independent(
            config=config,
            aligned_states=aligned_states,
            output_root=output_root,
        )

    if policy == "joint_shared":
        normalized_states = tuple(aligned_states)

        if len(normalized_states) == 1:
            print(
                "[Joint shared scale deferred] "
                f"view={normalized_states[0].generated.view_name}; "
                "waiting for reference/query pair"
            )
            return normalized_states

        if not (
            config.visible_scale_refinement_reference_enabled
            and config.visible_scale_refinement_query_enabled
        ):
            raise ValueError(
                "joint_shared 정책은 reference_enabled=true와 "
                "query_enabled=true를 모두 요구합니다."
            )

        return _refine_aligned_states_joint_shared_scale(
            config=config,
            aligned_states=aligned_states,
            output_root=output_root,
        )

    raise ValueError(
        "지원하지 않는 scale.visible_refinement.policy: "
        f"{policy!r}; expected independent or joint_shared"
    )
