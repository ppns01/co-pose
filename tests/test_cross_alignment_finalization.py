from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pose.cross_alignment_runner import (
    combine_bidirectional_cross_alignment_results,
    finalize_cross_alignment,
)


TEST_DIRECTORY = Path(__file__).resolve().parent


def _cross_inputs(
    *,
    directory: Path,
    source_view: str,
    target_view: str,
) -> tuple[
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
]:
    mesh_path = (
        directory
        / f"{source_view}_mesh.obj"
    )
    mesh_path.touch()
    pose = np.eye(4, dtype=np.float32)

    self_alignment = SimpleNamespace(
        proxy_view=source_view,
        candidate_index=2,
        scale_m=0.2,
        scaled_mesh_path=mesh_path,
        hypothesis_rank=0,
        foundationpose_score=1.0,
        alignment_loss=0.1,
        pose_camera_from_proxy=pose,
    )
    candidate = SimpleNamespace(
        candidate_index=2,
        scale_m=0.2,
        scaled_mesh_path=mesh_path,
    )
    prepared_view = SimpleNamespace(
        view=SimpleNamespace(
            source=SimpleNamespace(
                name=target_view,
            ),
        ),
    )
    foundationpose_result = SimpleNamespace(
        view_name=target_view,
        candidate_index=2,
        scale_m=0.2,
        scaled_mesh_path=mesh_path,
        output_directory=(
            directory
            / f"{target_view}_foundationpose"
        ),
        hypotheses=(
            SimpleNamespace(
                rank=0,
                score=2.0,
                pose_cam_from_proxy=pose,
            ),
        ),
    )

    return (
        self_alignment,
        candidate,
        prepared_view,
        foundationpose_result,
    )


class CrossAlignmentFinalizationTests(
    unittest.TestCase
):
    def test_precomputed_results_finalize_and_combine(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            root = Path(temp_dir)
            (
                reference_self,
                reference_candidate,
                query_view,
                reference_to_query_result,
            ) = _cross_inputs(
                directory=root,
                source_view="reference",
                target_view="query",
            )
            (
                query_self,
                query_candidate,
                reference_view,
                query_to_reference_result,
            ) = _cross_inputs(
                directory=root,
                source_view="query",
                target_view="reference",
            )

            reference_cross = finalize_cross_alignment(
                foundationpose_result=(
                    reference_to_query_result
                ),
                self_alignment=reference_self,
                scaled_mesh_candidate=(
                    reference_candidate
                ),
                target_view=query_view,
                output_directory=(
                    root
                    / "reference_proxy_to_query"
                ),
            )
            query_cross = finalize_cross_alignment(
                foundationpose_result=(
                    query_to_reference_result
                ),
                self_alignment=query_self,
                scaled_mesh_candidate=query_candidate,
                target_view=reference_view,
                output_directory=(
                    root
                    / "query_proxy_to_reference"
                ),
            )
            combined = (
                combine_bidirectional_cross_alignment_results(
                    reference_proxy_to_query=(
                        reference_cross
                    ),
                    query_proxy_to_reference=query_cross,
                    output_directory=(
                        root / "combined"
                    ),
                )
            )

            summary = json.loads(
                combined.summary_path.read_text(
                    encoding="utf-8",
                )
            )

        self.assertEqual(
            reference_cross.path_name,
            "reference_proxy_to_query",
        )
        self.assertEqual(
            query_cross.path_name,
            "query_proxy_to_reference",
        )
        self.assertEqual(
            summary[
                "reference_proxy_to_query"
            ]["top_k"],
            1,
        )
        self.assertEqual(
            summary[
                "query_proxy_to_reference"
            ]["top_k"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
