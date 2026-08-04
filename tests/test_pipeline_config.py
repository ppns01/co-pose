from __future__ import annotations

import unittest
from pathlib import Path

import main


TEST_DIRECTORY = Path(__file__).resolve().parent
FIXTURE_DIRECTORY = TEST_DIRECTORY / "fixtures"


class PipelineConfigFileTests(unittest.TestCase):
    def test_self_mesh_requires_pair_shared_scale_policy(self) -> None:
        config = main.build_config(
            main.parse_args(
                ["--pose-path", "self_mesh", "--disable-visible-scale-refinement"]
            )
        )
        with self.assertRaisesRegex(ValueError, "pair-shared S\\*"):
            main.validate_config(config)

    def test_camera_axis_and_dgedi_diagnostic_defaults_are_loaded(self) -> None:
        config = main.build_config(main.parse_args([]))

        self.assertAlmostEqual(
            config.axis_scale_uncertainty_penalty_weight,
            0.10,
        )
        self.assertAlmostEqual(
            config.foundationpose_rotation_diversity_threshold_deg,
            30.0,
        )
        self.assertAlmostEqual(
            config.dgedi_confidence_weight_correspondence,
            0.15,
        )
        self.assertAlmostEqual(config.dgedi_confidence_weight_depth, 0.25)
        self.assertAlmostEqual(
            config.dgedi_confidence_target_correspondence_fraction,
            0.10,
        )
        self.assertAlmostEqual(
            config.dgedi_selection_rotation_penalty_weight,
            0.05,
        )
        self.assertAlmostEqual(
            config.dgedi_selection_near_tie_margin,
            0.01,
        )

    def test_default_yaml_supplies_active_tunables(
        self,
    ) -> None:
        config = main.build_config(
            main.parse_args([])
        )

        self.assertEqual(
            config.source_config_path,
            main.DEFAULT_PIPELINE_CONFIG.resolve(),
        )
        self.assertEqual(config.random_seed, 42)
        self.assertAlmostEqual(
            config.sam3_confidence_threshold,
            0.30,
        )
        self.assertEqual(
            config.sam3_checkpoint,
            (
                main.PROJECT_ROOT
                / "sam3"
                / "weights"
                / "sam3.pt"
            ).resolve(),
        )
        self.assertEqual(
            config.instantmesh_diffusion_steps,
            150,
        )
        self.assertEqual(
            config.scale_multipliers,
            (0.8, 1.0, 1.25, 1.5, 1.8),
        )
        self.assertTrue(
            config.visible_scale_refinement_enabled
        )
        self.assertTrue(
            config
            .visible_scale_refinement_reference_enabled
        )
        self.assertTrue(
            config
            .visible_scale_refinement_query_enabled
        )
        self.assertAlmostEqual(
            config
            .visible_scale_minimum_loss_improvement_ratio,
            0.01,
        )
        self.assertTrue(config.storage_results_only)
        self.assertTrue(
            config.gt_mask_fallback_on_sam3_failure
        )
        self.assertEqual(
            config.linemod_all_object_ids,
            tuple(range(1, 16)),
        )
        self.assertEqual(config.visible_scale_policy, "joint_shared")
        self.assertEqual(config.axis_scale_grid_step_count, 7)
        self.assertAlmostEqual(
            config.axis_scale_minimum_factor,
            0.50,
        )
        self.assertAlmostEqual(
            config.axis_scale_maximum_factor,
            2.00,
        )
        self.assertAlmostEqual(config.axis_scale_penalty_weight, 0.02)
        self.assertAlmostEqual(
            config.axis_scale_minimum_loss_improvement_ratio,
            0.0,
        )
        self.assertEqual(config.dgedi_sample_count, 30000)
        self.assertAlmostEqual(
            config.dgedi_maximum_surface_depth_residual_m,
            0.010,
        )

    def test_custom_yaml_then_cli_precedence(
        self,
    ) -> None:
        config_path = (
            FIXTURE_DIRECTORY
            / "pipeline_override.yaml"
        )
        config = main.build_config(
            main.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--sam3-prompt",
                    "impact drill",
                    "--instantmesh-online",
                ]
            )
        )

        self.assertEqual(
            config.sam3_prompt,
            "impact drill",
        )
        self.assertEqual(
            config.instantmesh_diffusion_steps,
            30,
        )
        self.assertFalse(
            config.instantmesh_offline
        )

    def test_cli_single_query_clears_yaml_batch(
        self,
    ) -> None:
        config_path = (
            FIXTURE_DIRECTORY
            / "pipeline_batch_override.yaml"
        )
        config = main.build_config(
            main.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--query-image-id",
                    "7",
                ]
            )
        )

        self.assertIsNone(
            config.batch_query_image_ids
        )
        self.assertEqual(config.query.image_id, 7)

    def test_boolean_cli_can_reverse_yaml(
        self,
    ) -> None:
        config_path = (
            FIXTURE_DIRECTORY
            / "pipeline_boolean_override.yaml"
        )
        config = main.build_config(
            main.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--enable-dino",
                    "--instantmesh-online",
                    "--enable-visible-scale-refinement",
                ]
            )
        )

        self.assertTrue(config.dino_enabled)
        self.assertTrue(
            config.visible_scale_refinement_enabled
        )
        self.assertFalse(
            config.instantmesh_offline
        )

    def test_visible_scale_cli_can_disable_default(
        self,
    ) -> None:
        enabled = main.build_config(
            main.parse_args(
                [
                    "--enable-visible-scale-refinement",
                ]
            )
        )
        disabled = main.build_config(
            main.parse_args(
                [
                    "--disable-visible-scale-refinement",
                ]
            )
        )

        self.assertTrue(
            enabled.visible_scale_refinement_enabled
        )
        self.assertFalse(
            disabled.visible_scale_refinement_enabled
        )

    def test_visible_scale_loss_fallback_policy(
        self,
    ) -> None:
        self.assertTrue(
            main._visible_scale_loss_improved(
                coarse_loss=0.10,
                refined_loss=0.098,
                minimum_improvement_ratio=0.01,
            )
        )
        self.assertFalse(
            main._visible_scale_loss_improved(
                coarse_loss=0.10,
                refined_loss=0.0995,
                minimum_improvement_ratio=0.01,
            )
        )
        self.assertFalse(
            main._visible_scale_loss_improved(
                coarse_loss=0.10,
                refined_loss=0.101,
                minimum_improvement_ratio=0.01,
            )
        )

    def test_dependent_cli_values_are_rederived(
        self,
    ) -> None:
        config = main.build_config(
            main.parse_args(
                [
                    "--object-id",
                    "9",
                    "--dinov3-model",
                    "dinov3_vitb16",
                ]
            )
        )

        self.assertEqual(
            config.object_name,
            "duck",
        )
        self.assertEqual(
            config.sam3_prompt,
            "a small yellow rubber duck",
        )
        self.assertEqual(
            config.dinov3_checkpoint.name,
            (
                "dinov3-vitb16-"
                "pretrain-lvd1689m"
            ),
        )

    def test_dino_repository_rederives_checkpoint(
        self,
    ) -> None:
        config = main.build_config(
            main.parse_args(
                [
                    "--dinov3-repository",
                    "alternate_dinov3",
                ]
            )
        )

        self.assertEqual(
            config.dinov3_checkpoint.parent.parent,
            config.dinov3_repository,
        )

    def test_unknown_yaml_key_is_rejected(
        self,
    ) -> None:
        config_path = (
            FIXTURE_DIRECTORY
            / "pipeline_unknown_key.yaml"
        )

        with self.assertRaisesRegex(
            ValueError,
            "diffusion_step_typo",
        ):
            main.parse_args(
                [
                    "--config",
                    str(config_path),
                ]
            )


if __name__ == "__main__":
    unittest.main()
