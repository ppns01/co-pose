from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None

if cv2 is not None:
    from utils.pipeline_visualizer import (
        POSE_AXIS_LENGTH_RATIO,
        _draw_pose_center_and_axes,
        _project_proxy_point,
        save_pipeline_visualization_report,
    )


TEST_DIRECTORY = Path(__file__).resolve().parent


def _prepared_view() -> SimpleNamespace:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[16:48, 16:48] = np.array(
        [80, 120, 160],
        dtype=np.uint8,
    )
    mask = np.zeros((64, 64), dtype=np.bool_)
    mask[16:48, 16:48] = True
    depth = np.zeros((64, 64), dtype=np.float32)
    depth[mask] = np.float32(0.75)
    camera_matrix = np.array(
        [
            [80.0, 0.0, 32.0],
            [0.0, 80.0, 32.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return SimpleNamespace(
        view=SimpleNamespace(
            rgb=rgb,
            camera_matrix=camera_matrix,
        ),
        segmentation=SimpleNamespace(
            mask_bool=mask
        ),
        masked_depth_m=depth,
    )


def _alignment_evaluation(
    root: Path,
) -> SimpleNamespace:
    render_directory = root / "render"
    render_directory.mkdir(parents=True)

    rendered_rgb = np.zeros(
        (1, 64, 64, 3),
        dtype=np.uint8,
    )
    rendered_rgb[0, 16:48, 16:48] = np.array(
        [180, 80, 40],
        dtype=np.uint8,
    )
    rendered_masks = np.zeros(
        (1, 64, 64),
        dtype=np.bool_,
    )
    rendered_masks[0, 16:48, 16:48] = True

    np.save(
        render_directory / "rendered_rgb.npy",
        rendered_rgb,
        allow_pickle=False,
    )
    np.save(
        render_directory / "rendered_masks.npy",
        rendered_masks,
        allow_pickle=False,
    )

    pose_camera_from_proxy = np.eye(
        4,
        dtype=np.float32,
    )
    pose_camera_from_proxy[2, 3] = np.float32(1.0)
    hypothesis = SimpleNamespace(
        rank=0,
        pose_cam_from_proxy=pose_camera_from_proxy,
    )
    candidate_result = SimpleNamespace(
        candidate_index=0,
        scale_m=0.20,
        hypotheses=(hypothesis,),
    )
    evaluation = SimpleNamespace(
        candidate_result=candidate_result,
        hypothesis=hypothesis,
        render_directory=render_directory,
    )
    summary_path = root / "alignment_evaluation.json"
    summary_path.touch()

    return SimpleNamespace(
        best=evaluation,
        evaluations=(evaluation,),
        summary_path=summary_path,
    )


@unittest.skipIf(
    cv2 is None,
    "OpenCV is unavailable in this test environment.",
)
class PipelineVisualizerTests(unittest.TestCase):
    def test_draws_center_and_half_scale_xyz_axes(
        self,
    ) -> None:
        camera_matrix = np.array(
            [
                [80.0, 0.0, 32.0],
                [0.0, 80.0, 32.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = np.float32(1.0)
        scale_m = 0.20
        axis_length_m = (
            POSE_AXIS_LENGTH_RATIO * scale_m
        )

        self.assertAlmostEqual(
            axis_length_m,
            0.10,
        )
        self.assertEqual(
            _project_proxy_point(
                point_proxy=np.array(
                    [axis_length_m, 0.0, 0.0],
                    dtype=np.float64,
                ),
                pose_camera_from_proxy=pose,
                camera_matrix=camera_matrix,
            ),
            (40, 32),
        )

        rgb = np.zeros(
            (64, 64, 3),
            dtype=np.uint8,
        )
        identity_axes = _draw_pose_center_and_axes(
            rgb=rgb,
            pose_camera_from_proxy=pose,
            camera_matrix=camera_matrix,
            axis_length_m=axis_length_m,
        )
        x_patch = identity_axes[30:35, 38:43]
        y_patch = identity_axes[38:43, 30:35]
        self.assertTrue(
            np.any(
                (x_patch[:, :, 0] > 200)
                & (x_patch[:, :, 1] < 80)
                & (x_patch[:, :, 2] < 80)
            )
        )
        self.assertTrue(
            np.any(
                (y_patch[:, :, 0] < 80)
                & (y_patch[:, :, 1] > 200)
                & (y_patch[:, :, 2] < 80)
            )
        )
        self.assertTrue(
            np.array_equal(
                identity_axes[32, 32],
                np.array(
                    [255, 255, 0],
                    dtype=np.uint8,
                ),
            )
        )

        rotated_pose = pose.copy()
        rotated_pose[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        rotated_axes = _draw_pose_center_and_axes(
            rgb=rgb,
            pose_camera_from_proxy=rotated_pose,
            camera_matrix=camera_matrix,
            axis_length_m=axis_length_m,
        )
        z_patch = rotated_axes[22:27, 30:35]
        self.assertTrue(
            np.any(
                (z_patch[:, :, 0] < 80)
                & (z_patch[:, :, 1] < 80)
                & (z_patch[:, :, 2] > 200)
            )
        )

    def test_offscreen_center_skips_pose_overlay(
        self,
    ) -> None:
        rgb = np.zeros(
            (64, 64, 3),
            dtype=np.uint8,
        )
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = np.float32(100.0)
        pose[2, 3] = np.float32(1.0)
        camera_matrix = np.array(
            [
                [80.0, 0.0, 32.0],
                [0.0, 80.0, 32.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        result = _draw_pose_center_and_axes(
            rgb=rgb,
            pose_camera_from_proxy=pose,
            camera_matrix=camera_matrix,
            axis_length_m=0.10,
        )

        self.assertTrue(np.array_equal(result, rgb))

    def test_writes_stage_report_without_rerendering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            root = Path(temp_dir)
            reference_view = _prepared_view()
            query_view = _prepared_view()

            reference_evaluation = (
                _alignment_evaluation(
                    root / "reference_self"
                )
            )
            query_evaluation = (
                _alignment_evaluation(
                    root / "query_self"
                )
            )
            reference_cross = (
                _alignment_evaluation(
                    root / "reference_cross"
                )
            )
            query_cross = (
                _alignment_evaluation(
                    root / "query_cross"
                )
            )

            preview_path = root / "preview.png"
            preview_path.touch()
            reference_mesh = root / "reference.obj"
            query_mesh = root / "query.obj"
            reference_mesh.touch()
            query_mesh.touch()

            mesh_result_reference = SimpleNamespace(
                artifact_paths=(preview_path,),
                primary_output_path=reference_mesh,
            )
            mesh_result_query = SimpleNamespace(
                artifact_paths=(preview_path,),
                primary_output_path=query_mesh,
            )

            cross_summary = (
                root / "cross_evidence.json"
            )
            cross_summary.touch()
            cross_evidence = SimpleNamespace(
                reference_proxy=SimpleNamespace(
                    alignment_evaluation=(
                        reference_cross
                    )
                ),
                query_proxy=SimpleNamespace(
                    alignment_evaluation=query_cross
                ),
                summary_path=cross_summary,
            )
            pair = SimpleNamespace(
                reference_candidate_index=0,
                query_candidate_index=0,
                consistency_loss=0.125,
                passes_hard_gate=True,
            )
            consistency = SimpleNamespace(
                reference_candidate_count=1,
                query_candidate_count=1,
                pairs=(pair,),
            )
            final_result = SimpleNamespace(
                status="CONSISTENT",
                selected_path_name="reference_proxy",
                selected_candidate=SimpleNamespace(
                    path_name="reference_proxy",
                    cross_candidate_index=0,
                    cross_hypothesis_rank=0,
                ),
                selected_relative_pose_query_from_reference=(
                    np.eye(4, dtype=np.float32)
                ),
            )

            report_path = (
                save_pipeline_visualization_report(
                    output_root=root / "result",
                    reference_view=reference_view,
                    query_view=query_view,
                    reference_mesh_result=(
                        mesh_result_reference
                    ),
                    query_mesh_result=(
                        mesh_result_query
                    ),
                    reference_self_evaluation=(
                        reference_evaluation
                    ),
                    query_self_evaluation=(
                        query_evaluation
                    ),
                    cross_evidence=cross_evidence,
                    consistency_result=consistency,
                    final_result=final_result,
                )
            )

            self.assertTrue(report_path.is_file())
            self.assertTrue(
                (
                    report_path.parent
                    / "stage_visuals.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    report_path.parent
                    / "stage_08_final_selected_overlay.png"
                ).is_file()
            )
            final_overlay = cv2.imdecode(
                np.fromfile(
                    report_path.parent
                    / "stage_08_final_selected_overlay.png",
                    dtype=np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )
            self.assertIsNotNone(final_overlay)
            self.assertTrue(
                np.array_equal(
                    final_overlay[32, 32],
                    np.array(
                        [0, 255, 255],
                        dtype=np.uint8,
                    ),
                )
            )

            report_text = report_path.read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "1. RGB-D preparation and masks",
                report_text,
            )
            self.assertIn(
                "8. Final selection",
                report_text,
            )
            self.assertIn(
                "CONSISTENT",
                report_text,
            )
            self.assertIn(
                "Axis length is 50%",
                report_text,
            )


if __name__ == "__main__":
    unittest.main()
