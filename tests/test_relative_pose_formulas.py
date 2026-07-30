from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np


TEST_DIRECTORY = Path(__file__).resolve().parent


def _translation_pose(
    x: float,
    y: float,
    z: float,
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = (x, y, z)
    return pose


class RelativePoseFormulaTests(unittest.TestCase):
    def test_both_proxy_paths_produce_query_from_reference(self) -> None:
        trimesh_context = (
            nullcontext()
            if importlib.util.find_spec("trimesh")
            is not None
            else patch.dict(
                sys.modules,
                {
                    "trimesh": types.ModuleType(
                        "trimesh"
                    )
                },
            )
        )

        with trimesh_context:
            from pose.foundationpose_runner import (
                FoundationPoseCandidateResult,
                FoundationPoseHypothesis,
            )
            from pose.relative_pose_builder import (
                SelfAlignmentSelection,
                build_bidirectional_relative_candidates,
            )

        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            temp_root = Path(temp_dir)
            reference_mesh = temp_root / "reference.obj"
            query_mesh = temp_root / "query.obj"
            reference_mesh.touch()
            query_mesh.touch()

            reference_output = (
                temp_root / "reference_output"
            )
            query_output = temp_root / "query_output"
            reference_output.mkdir()
            query_output.mkdir()

            reference_metadata = (
                reference_output / "meta.json"
            )
            query_metadata = (
                query_output / "meta.json"
            )
            reference_metadata.touch()
            query_metadata.touch()

            reference_self = SelfAlignmentSelection(
                proxy_view="reference",
                candidate_index=0,
                hypothesis_rank=0,
                scale_m=0.2,
                scaled_mesh_path=reference_mesh,
                pose_camera_from_proxy=(
                    _translation_pose(1.0, 0.0, 0.0)
                ),
                foundationpose_score=0.9,
            )

            query_self = SelfAlignmentSelection(
                proxy_view="query",
                candidate_index=0,
                hypothesis_rank=0,
                scale_m=0.3,
                scaled_mesh_path=query_mesh,
                pose_camera_from_proxy=(
                    _translation_pose(4.0, 2.0, 0.0)
                ),
                foundationpose_score=0.8,
            )

            reference_proxy_to_query = (
                FoundationPoseCandidateResult(
                    view_name="query",
                    candidate_index=0,
                    scale_m=0.2,
                    scaled_mesh_path=reference_mesh,
                    hypotheses=(
                        FoundationPoseHypothesis(
                            rank=0,
                            score=0.7,
                            pose_cam_from_proxy=(
                                _translation_pose(
                                    4.0,
                                    2.0,
                                    0.0,
                                )
                            ),
                        ),
                    ),
                    output_directory=reference_output,
                    metadata_path=reference_metadata,
                )
            )

            query_proxy_to_reference = (
                FoundationPoseCandidateResult(
                    view_name="reference",
                    candidate_index=0,
                    scale_m=0.3,
                    scaled_mesh_path=query_mesh,
                    hypotheses=(
                        FoundationPoseHypothesis(
                            rank=0,
                            score=0.6,
                            pose_cam_from_proxy=(
                                _translation_pose(
                                    1.0,
                                    0.0,
                                    0.0,
                                )
                            ),
                        ),
                    ),
                    output_directory=query_output,
                    metadata_path=query_metadata,
                )
            )

            candidates = (
                build_bidirectional_relative_candidates(
                    reference_self=reference_self,
                    query_self=query_self,
                    reference_proxy_to_query=(
                        reference_proxy_to_query
                    ),
                    query_proxy_to_reference=(
                        query_proxy_to_reference
                    ),
                )
            )

            expected = _translation_pose(
                3.0,
                2.0,
                0.0,
            )

            np.testing.assert_allclose(
                candidates
                .reference_proxy_candidates[0]
                .relative_pose_query_from_reference,
                expected,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                candidates
                .query_proxy_candidates[0]
                .relative_pose_query_from_reference,
                expected,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
