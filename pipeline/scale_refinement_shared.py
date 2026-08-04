from __future__ import annotations

import math
from typing import Any

from core.types import PipelineConfig


def _candidate_by_index(
    candidates: tuple[Any, ...],
    candidate_index: int,
) -> Any:
    matches = [
        candidate
        for candidate in candidates
        if candidate.candidate_index
        == candidate_index
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one scaled mesh candidate "
            f"for index {candidate_index}; "
            f"found {len(matches)}."
        )

    return matches[0]


def _visible_scale_enabled_for_view(
    *,
    config: PipelineConfig,
    view_name: str,
) -> bool:
    if view_name == "reference":
        return config.visible_scale_refinement_reference_enabled

    if view_name == "query":
        return config.visible_scale_refinement_query_enabled

    raise ValueError(
        f"Unsupported visible-scale view: {view_name}"
    )


def _visible_scale_loss_improved(
    *,
    coarse_loss: float | None,
    refined_loss: float | None,
    minimum_improvement_ratio: float,
) -> bool:
    if (
        coarse_loss is None
        or refined_loss is None
        or not math.isfinite(coarse_loss)
        or not math.isfinite(refined_loss)
        or coarse_loss < 0.0
        or refined_loss < 0.0
    ):
        return False

    required_maximum_loss = coarse_loss * (
        1.0 - minimum_improvement_ratio
    )
    return refined_loss <= required_maximum_loss
