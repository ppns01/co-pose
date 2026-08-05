from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.scale_refinement_dispatch import (
    _apply_pair_scale_refinement,
    _apply_post_coarse_proxy_refinement,
)


def _config(policy: str, shared_axis: bool) -> SimpleNamespace:
    return SimpleNamespace(
        visible_scale_refinement_enabled=True,
        visible_scale_policy=policy,
        visible_scale_refinement_reference_enabled=True,
        visible_scale_refinement_query_enabled=True,
        pose_path="self_mesh",
        enable_shared_axis_scale_refinement=shared_axis,
        reference=SimpleNamespace(scene_id=9, image_id=0, instance_index=0),
        query=SimpleNamespace(scene_id=9, image_id=0, instance_index=0),
    )


def _pair_states() -> tuple[SimpleNamespace, SimpleNamespace]:
    return (
        SimpleNamespace(generated=SimpleNamespace(view_name="reference")),
        SimpleNamespace(generated=SimpleNamespace(view_name="query")),
    )


class ScaleRefinementPairDispatchParityTests(unittest.TestCase):
    """
    Parity tests over the pair-level dispatch layer only.

    All leaf refiners (visible-scale independent/joint_shared and both
    axis-scale refiners) are patched with recording stand-ins, so these
    tests exercise no GPU, no dataset, and no FoundationPose/dGeDi
    process. The single-query path is emulated by calling
    ``_apply_post_coarse_proxy_refinement`` on a 2-tuple; the batch path
    is emulated by calling it twice on 1-tuples (reference then query)
    followed by ``_apply_pair_scale_refinement`` on the resulting pair.
    """

    def test_joint_shared_parity_between_single_query_and_batch(self) -> None:
        config = _config("joint_shared", shared_axis=True)
        stages: list[str] = []

        def fake_joint_shared(*, aligned_states, **_: object):
            stages.append("joint_shared")
            return tuple(aligned_states)

        def fake_shared_axis(*, aligned_states, **_: object):
            stages.append("shared_axis")
            return tuple(aligned_states)

        with (
            patch(
                "pipeline.scale_refinement_visible."
                "_refine_aligned_states_joint_shared_scale",
                side_effect=fake_joint_shared,
            ),
            patch(
                "pipeline.scale_refinement_dispatch."
                "_refine_aligned_states_shared_axis_scale",
                side_effect=fake_shared_axis,
            ),
        ):
            stages.clear()
            _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=_pair_states(),
                output_root=Path("/single"),
            )
            single_query_stages = list(stages)

            stages.clear()
            reference_generated, query_generated = _pair_states()
            (reference_state,) = _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=(reference_generated,),
                output_root=Path("/reference"),
            )
            (query_state,) = _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=(query_generated,),
                output_root=Path("/query"),
            )
            _apply_pair_scale_refinement(
                config=config,
                aligned_states=(reference_state, query_state),
                output_root=Path("/pair"),
                reference_frame=config.reference,
                query_frame=SimpleNamespace(
                    scene_id=9, image_id=1, instance_index=0
                ),
                pair_visible_scale_pending=True,
            )
            batch_stages = list(stages)

        self.assertEqual(single_query_stages, ["joint_shared", "shared_axis"])
        self.assertEqual(batch_stages, single_query_stages)

    def test_batch_call_forwards_loop_query_frame_not_config_query(self) -> None:
        config = _config("joint_shared", shared_axis=True)
        recorded_query_frames: list[object] = []

        def fake_joint_shared(*, aligned_states, **_: object):
            return tuple(aligned_states)

        def fake_shared_axis(*, aligned_states, query_frame, **_: object):
            recorded_query_frames.append(query_frame)
            return tuple(aligned_states)

        with (
            patch(
                "pipeline.scale_refinement_visible."
                "_refine_aligned_states_joint_shared_scale",
                side_effect=fake_joint_shared,
            ),
            patch(
                "pipeline.scale_refinement_dispatch."
                "_refine_aligned_states_shared_axis_scale",
                side_effect=fake_shared_axis,
            ),
        ):
            _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=_pair_states(),
                output_root=Path("/single"),
            )

            loop_query_frame = SimpleNamespace(
                scene_id=9, image_id=7, instance_index=0
            )
            _apply_pair_scale_refinement(
                config=config,
                aligned_states=_pair_states(),
                output_root=Path("/pair"),
                reference_frame=config.reference,
                query_frame=loop_query_frame,
                pair_visible_scale_pending=True,
            )

        self.assertEqual(len(recorded_query_frames), 2)
        self.assertIs(recorded_query_frames[0], config.query)
        self.assertIs(recorded_query_frames[1], loop_query_frame)
        self.assertIsNot(recorded_query_frames[1], config.query)

    def test_independent_policy_visible_refines_each_view_once_and_skips_pair_stage(
        self,
    ) -> None:
        config = _config("independent", shared_axis=False)
        visible_calls: list[str] = []

        def fake_independent_visible(*, aligned_states, **_: object):
            for state in aligned_states:
                visible_calls.append(state.generated.view_name)
            return tuple(aligned_states)

        def fake_independent_axis(*, aligned_states, **_: object):
            return tuple(aligned_states)

        with (
            patch(
                "pipeline.scale_refinement_visible."
                "_refine_aligned_states_visible_scale_independent",
                side_effect=fake_independent_visible,
            ),
            patch(
                "pipeline.scale_refinement_dispatch."
                "_refine_aligned_states_independent_axis_scale",
                side_effect=fake_independent_axis,
            ),
        ):
            visible_calls.clear()
            _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=_pair_states(),
                output_root=Path("/single"),
            )
            single_query_visible_calls = list(visible_calls)

            visible_calls.clear()
            reference_generated, query_generated = _pair_states()
            (reference_state,) = _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=(reference_generated,),
                output_root=Path("/reference"),
            )
            (query_state,) = _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=(query_generated,),
                output_root=Path("/query"),
            )
            before_pair_stage_calls = list(visible_calls)
            _apply_pair_scale_refinement(
                config=config,
                aligned_states=(reference_state, query_state),
                output_root=Path("/pair"),
                reference_frame=config.reference,
                query_frame=SimpleNamespace(
                    scene_id=9, image_id=3, instance_index=0
                ),
                pair_visible_scale_pending=True,
            )
            batch_visible_calls = list(visible_calls)

        self.assertEqual(sorted(single_query_visible_calls), ["query", "reference"])
        self.assertEqual(sorted(before_pair_stage_calls), ["query", "reference"])
        self.assertEqual(batch_visible_calls, before_pair_stage_calls)

    def test_independent_axis_branch_runs_once_only_on_the_pair(self) -> None:
        config = _config("independent", shared_axis=False)
        axis_calls: list[tuple[str, ...]] = []

        def fake_independent_visible(*, aligned_states, **_: object):
            return tuple(aligned_states)

        def fake_independent_axis(*, aligned_states, **_: object):
            axis_calls.append(
                tuple(state.generated.view_name for state in aligned_states)
            )
            return tuple(aligned_states)

        with (
            patch(
                "pipeline.scale_refinement_visible."
                "_refine_aligned_states_visible_scale_independent",
                side_effect=fake_independent_visible,
            ),
            patch(
                "pipeline.scale_refinement_dispatch."
                "_refine_aligned_states_independent_axis_scale",
                side_effect=fake_independent_axis,
            ),
        ):
            axis_calls.clear()
            _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=_pair_states(),
                output_root=Path("/single"),
            )
            self.assertEqual(axis_calls, [("reference", "query")])

            axis_calls.clear()
            reference_generated, query_generated = _pair_states()
            (reference_state,) = _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=(reference_generated,),
                output_root=Path("/reference"),
            )
            self.assertEqual(axis_calls, [])
            (query_state,) = _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=(query_generated,),
                output_root=Path("/query"),
            )
            self.assertEqual(axis_calls, [])
            _apply_pair_scale_refinement(
                config=config,
                aligned_states=(reference_state, query_state),
                output_root=Path("/pair"),
                reference_frame=config.reference,
                query_frame=SimpleNamespace(
                    scene_id=9, image_id=5, instance_index=0
                ),
                pair_visible_scale_pending=True,
            )
            self.assertEqual(axis_calls, [("reference", "query")])


if __name__ == "__main__":
    unittest.main()
