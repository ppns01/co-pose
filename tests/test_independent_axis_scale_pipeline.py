from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.types import AlignedProxyState
from pipeline.dgedi_final_stage import _run_aligned_pair
from pipeline.scale_refinement_dispatch import (
    _apply_post_coarse_proxy_refinement,
)
from pipeline.scale_refinement_independent_axis import (
    _refine_aligned_states_independent_axis_scale,
)
from pose.alignment_scorer import AlignmentScoreResult


def _score(loss: float) -> AlignmentScoreResult:
    return AlignmentScoreResult(
        total_loss=loss,
        mask_loss=loss,
        depth_loss=loss,
        free_space_loss=0.0,
        boundary_loss=0.0,
        mask_iou=1.0 - loss,
        depth_residual_m=0.001,
        depth_residual_normalized=0.01,
        overlap_pixel_count=100,
        valid_depth_overlap_count=100,
        rendered_pixel_count=100,
        free_space_violation_count=0,
    )


class IndependentAxisScalePipelineTests(unittest.TestCase):
    @staticmethod
    def _state(view_name: str, candidate: object) -> AlignedProxyState:
        return AlignedProxyState(
            generated=SimpleNamespace(
                view_name=view_name,
                prepared_view=SimpleNamespace(),
            ),
            self_results=(SimpleNamespace(),),
            self_evaluation=SimpleNamespace(),
            self_alignment=SimpleNamespace(
                pose_camera_from_proxy=np.eye(4),
                alignment_loss=0.2,
            ),
            selected_candidate=candidate,
        )

    @staticmethod
    def _config() -> SimpleNamespace:
        return SimpleNamespace(
            alignment_weight_mask=0.3,
            alignment_weight_depth=0.4,
            alignment_weight_free_space=0.2,
            alignment_weight_boundary=0.1,
            foundationpose_repository=Path("/foundationpose"),
            device="cuda:0",
            renderer_batch_size=8,
            renderer_maximum_texture_size=1024,
            alignment_depth_trim_quantile=0.9,
            alignment_minimum_depth_overlap_pixels=50,
            alignment_free_space_absolute_tolerance_m=0.005,
            alignment_free_space_relative_tolerance=0.02,
            normalization_quantile_low=0.01,
            normalization_quantile_high=0.99,
            normalization_sample_count=1000,
            normalization_random_seed=0,
            axis_scale_minimum_factor=0.85,
            axis_scale_maximum_factor=1.15,
            axis_scale_grid_step_count=5,
            axis_scale_penalty_weight=0.02,
            axis_scale_uncertainty_penalty_weight=0.10,
            axis_scale_minimum_loss_improvement_ratio=0.01,
            scale_quantile_low=0.05,
            scale_quantile_high=0.95,
            scale_minimum_valid_points=1,
            top_k=8,
            foundationpose_rotation_diversity_threshold_deg=30.0,
            refine_iterations=5,
            foundationpose_workers=2,
            foundationpose_debug=0,
        )

    def test_self_mesh_pair_runs_camera_axis_refinement_without_g0(self) -> None:
        states = (
            SimpleNamespace(generated=SimpleNamespace(view_name="reference")),
            SimpleNamespace(generated=SimpleNamespace(view_name="query")),
        )
        config = SimpleNamespace(
            visible_scale_refinement_enabled=True,
            pose_path="self_mesh",
            enable_shared_axis_scale_refinement=False,
        )
        with (
            patch(
                "pipeline.scale_refinement_dispatch."
                "_refine_aligned_states_visible_scale",
                return_value=states,
            ) as shared_scale,
            patch(
                "pipeline.scale_refinement_dispatch."
                "_refine_aligned_states_independent_axis_scale",
                return_value=states,
            ) as refine_axis,
        ):
            returned = _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=states,
                output_root=Path("/output"),
            )
        self.assertEqual(returned, states)
        shared_scale.assert_called_once()
        refine_axis.assert_called_once_with(
            config=config,
            aligned_states=states,
            output_root=Path("/output"),
        )

    def test_single_view_defers_axis_refinement(self) -> None:
        state = SimpleNamespace(
            generated=SimpleNamespace(view_name="reference")
        )
        config = SimpleNamespace(
            visible_scale_refinement_enabled=True,
            pose_path="self_mesh",
            enable_shared_axis_scale_refinement=False,
        )
        with (
            patch(
                "pipeline.scale_refinement_dispatch."
                "_refine_aligned_states_visible_scale",
                return_value=(state,),
            ),
            patch(
                "pipeline.scale_refinement_dispatch."
                "_refine_aligned_states_independent_axis_scale"
            ) as refine_axis,
        ):
            returned = _apply_post_coarse_proxy_refinement(
                config=config,
                coarse_aligned_states=(state,),
                output_root=Path("/output"),
            )
        self.assertEqual(returned, (state,))
        refine_axis.assert_not_called()

    def test_only_improved_view_is_rerun_and_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            for view_name in ("reference", "query"):
                (output_root / "axis_scale_refinement" / view_name).mkdir(
                    parents=True
                )
            reference_source = SimpleNamespace(
                name="reference_source",
                scale_m=0.15,
            )
            query_source = SimpleNamespace(
                name="query_source",
                scale_m=0.15,
            )
            reference_scaled = SimpleNamespace(name="reference_scaled")
            query_scaled = SimpleNamespace(name="query_scaled")
            states = (
                self._state("reference", reference_source),
                self._state("query", query_source),
            )
            reference_search = SimpleNamespace(
                applied=True,
                source_candidate=reference_source,
                selected_candidate=reference_scaled,
                selected_scale_factors=(1.1, 0.9, 1.0),
                baseline_score=_score(0.2),
                selected_score=_score(0.1),
                source_dimensions_m=(0.1, 0.2, 0.3),
                selected_dimensions_m=(0.11, 0.18, 0.3),
                depth_observability=0.25,
                axis_penalty_weights=(0.02, 0.02, 0.095),
                summary_path=output_root / "reference_search.json",
            )
            query_search = SimpleNamespace(
                applied=True,
                source_candidate=query_source,
                selected_candidate=query_scaled,
                selected_scale_factors=(0.9, 1.1, 1.0),
                baseline_score=_score(0.2),
                selected_score=_score(0.1),
                source_dimensions_m=(0.1, 0.2, 0.3),
                selected_dimensions_m=(0.09, 0.22, 0.3),
                depth_observability=0.50,
                axis_penalty_weights=(0.02, 0.02, 0.07),
                summary_path=output_root / "query_search.json",
            )
            renderer_context = MagicMock()
            renderer_context.__enter__.return_value = MagicMock()
            renderer_context.__exit__.return_value = False
            accepted_alignment = SimpleNamespace(alignment_loss=0.1)
            rejected_alignment = SimpleNamespace(alignment_loss=0.21)
            with (
                patch(
                    "pose.mesh_renderer.FoundationPoseMeshRenderer",
                    return_value=renderer_context,
                ),
                patch(
                    "scale.independent_axis_scale_refiner."
                    "fit_independent_axis_scale",
                    side_effect=(reference_search, query_search),
                ),
                patch(
                    "pose.foundationpose_process_pool."
                    "run_foundationpose_jobs",
                    return_value=(SimpleNamespace(), SimpleNamespace()),
                ) as run_jobs,
                patch(
                    "pose.alignment_evaluator."
                    "evaluate_foundationpose_alignments",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "pose.alignment_evaluator.select_best_self_alignment",
                    side_effect=(accepted_alignment, rejected_alignment),
                ),
            ):
                returned = _refine_aligned_states_independent_axis_scale(
                    config=self._config(),
                    aligned_states=states,
                    output_root=output_root,
                )
            self.assertIs(returned[0].selected_candidate, reference_scaled)
            self.assertIs(returned[1].selected_candidate, query_source)
            self.assertEqual(len(run_jobs.call_args.kwargs["jobs"]), 2)
            self.assertTrue(
                (
                    output_root
                    / "axis_scale_refinement"
                    / "reference"
                    / "selection.json"
                ).is_file()
            )
            query_selection = json.loads(
                (
                    output_root
                    / "axis_scale_refinement"
                    / "query"
                    / "selection.json"
                ).read_text(encoding="utf-8")
            )
            self.assertFalse(query_selection["accepted"])
            self.assertTrue(query_selection["pair_shared_scale_verified"])
            self.assertEqual(
                query_selection["selected_source"],
                "pre_axis_fallback",
            )
            self.assertEqual(
                query_selection["requested_scale_factors_xyz"],
                [0.9, 1.1, 1.0],
            )
            self.assertEqual(
                query_selection["applied_scale_factors_xyz"],
                [1.0, 1.0, 1.0],
            )
            self.assertEqual(
                query_selection["selected_dimensions_m"],
                [0.1, 0.2, 0.3],
            )

    def test_self_mesh_pair_uses_only_final_dgedi_helper(self) -> None:
        prepared_view = SimpleNamespace()

        def state(view_name: str) -> SimpleNamespace:
            return SimpleNamespace(
                generated=SimpleNamespace(prepared_view=prepared_view),
                self_alignment=SimpleNamespace(
                    proxy_view=view_name,
                    candidate_index=0,
                    scale_m=0.15,
                    scaled_mesh_path=Path("/missing.obj"),
                ),
            )

        expected = object()
        config = SimpleNamespace(pose_path="self_mesh")
        with patch(
            "pipeline.dgedi_final_stage._run_self_mesh_final_dgedi",
            return_value=expected,
        ) as final_dgedi:
            returned = _run_aligned_pair(
                config=config,
                reference_state=state("reference"),
                query_state=state("query"),
                reference_dino=None,
                reference_surface=None,
                query_dino=None,
                query_surface=None,
                output_root=Path("/output"),
                research_context=None,
            )
        self.assertIs(returned, expected)
        final_dgedi.assert_called_once()


if __name__ == "__main__":
    unittest.main()
