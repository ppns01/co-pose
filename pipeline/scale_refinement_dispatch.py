from __future__ import annotations

from pathlib import Path
from typing import Sequence

from core.types import AlignedProxyState, PipelineConfig
from pipeline.scale_refinement_independent_axis import (
    _refine_aligned_states_independent_axis_scale,
)
from pipeline.scale_refinement_shared_axis import (
    _refine_aligned_states_shared_axis_scale,
)
from pipeline.scale_refinement_visible import (
    _refine_aligned_states_visible_scale,
)


def _apply_post_coarse_proxy_refinement(
    *,
    config: PipelineConfig,
    coarse_aligned_states: Sequence[
        AlignedProxyState
    ],
    output_root: Path,
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

    if config.pose_path == "self_mesh" and len(refined_states) == 2:
        if config.enable_shared_axis_scale_refinement:
            refined_states = _refine_aligned_states_shared_axis_scale(
                config=config,
                aligned_states=refined_states,
                output_root=output_root,
                reference_frame=config.reference,
                query_frame=config.query,
            )
        else:
            refined_states = _refine_aligned_states_independent_axis_scale(
                config=config,
                aligned_states=refined_states,
                output_root=output_root,
            )

    return refined_states
