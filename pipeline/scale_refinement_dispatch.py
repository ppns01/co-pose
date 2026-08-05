from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from core.types import AlignedProxyState, PipelineConfig
from pipeline.scale_refinement_independent_axis import (
    _refine_aligned_states_independent_axis_scale,
)
from pipeline.scale_refinement_shared_axis import (
    _refine_aligned_states_shared_axis_scale,
)
from pipeline.scale_refinement_visible import (
    _pair_visible_scale_required,
    _refine_aligned_states_visible_scale,
)


def _apply_pair_scale_refinement(
    *,
    config: PipelineConfig,
    aligned_states: Sequence[
        AlignedProxyState
    ],
    output_root: Path,
    reference_frame: Any = None,
    query_frame: Any = None,
    pair_visible_scale_pending: bool = False,
) -> tuple[AlignedProxyState, ...]:
    """
    Apply the pair-level (reference, query) scale-refinement stage.

    ``aligned_states`` is positionally (reference, query) and that order
    is preserved on return. ``reference_frame``/``query_frame`` are BOP
    frame selectors used only for the diagnostic GT report; they are
    never used in a transform.
    """
    states = tuple(aligned_states)
    if len(states) != 2:
        return states

    if pair_visible_scale_pending and _pair_visible_scale_required(config):
        states = _refine_aligned_states_visible_scale(
            config=config,
            aligned_states=states,
            output_root=output_root,
        )

    if config.pose_path == "self_mesh":
        if config.enable_shared_axis_scale_refinement:
            states = _refine_aligned_states_shared_axis_scale(
                config=config,
                aligned_states=states,
                output_root=output_root,
                reference_frame=(
                    config.reference if reference_frame is None else reference_frame
                ),
                query_frame=(
                    config.query if query_frame is None else query_frame
                ),
            )
        else:
            states = _refine_aligned_states_independent_axis_scale(
                config=config,
                aligned_states=states,
                output_root=output_root,
            )

    return states


def _apply_post_coarse_proxy_refinement(
    *,
    config: PipelineConfig,
    coarse_aligned_states: Sequence[
        AlignedProxyState
    ],
    output_root: Path,
    reference_frame: Any = None,
    query_frame: Any = None,
) -> tuple[AlignedProxyState, ...]:
    """Apply pair-shared S*, then per-view camera-axis shape correction."""
    refined_states = tuple(coarse_aligned_states)
    if config.visible_scale_refinement_enabled:
        refined_states = (
            _refine_aligned_states_visible_scale(
                config=config,
                aligned_states=refined_states,
                output_root=output_root,
            )
        )

    return _apply_pair_scale_refinement(
        config=config,
        aligned_states=refined_states,
        output_root=output_root,
        reference_frame=reference_frame,
        query_frame=query_frame,
        pair_visible_scale_pending=False,
    )
