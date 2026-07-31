from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_io.linemod_loader import load_linemod_view
from evaluation.relative_pose_evaluator import (
    BOPFrameGTSpec,
    build_ground_truth_relative_pose,
    load_bop_linemod_absolute_gt_pose,
)
from mask_provider import generate_sam3_segmentation
from pose.dgedi_runner import _save_self_aligned_mesh
from pose.foundationpose_runner import FoundationPoseRunner
from preprocessing.mask_processing import prepare_masked_view
from scale.mesh_scaler import ScaledMeshCandidate
from scripts.visualize_dgedi_alignment import pose_error


def _prepare_true_model_candidate(
    *,
    model_path: Path,
    output_dir: Path,
) -> ScaledMeshCandidate:
    raw_model = o3d.io.read_triangle_mesh(str(model_path))
    vertices_m = np.asarray(raw_model.vertices, dtype=np.float64) * 0.001
    raw_model.vertices = o3d.utility.Vector3dVector(vertices_m)

    output_dir.mkdir(parents=True, exist_ok=True)
    scaled_mesh_path = output_dir / "gt_model_meters.obj"
    o3d.io.write_triangle_mesh(
        str(scaled_mesh_path),
        raw_model,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )

    diagonal_m = float(
        np.linalg.norm(vertices_m.max(axis=0) - vertices_m.min(axis=0))
    )

    metadata_path = output_dir / "gt_model_metadata.json"
    metadata_path.write_text(
        json.dumps({"source": str(model_path), "diagonal_m": diagonal_m}),
        encoding="utf-8",
    )

    return ScaledMeshCandidate(
        candidate_index=0,
        scale_m=diagonal_m,
        normalized_mesh_path=scaled_mesh_path,
        scaled_mesh_path=scaled_mesh_path,
        metadata_path=metadata_path,
        scale_transform=np.eye(4, dtype=np.float64),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "True BOP CAD model shape, but FoundationPose's own "
            "(non-cheating) self-pose ESTIMATION run against the real "
            "RGB-D -- then dGeDi. Isolates FoundationPose self-alignment "
            "accuracy from InstantMesh reconstruction quality."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, default=8)
    parser.add_argument("--reference-image-id", type=int, default=0)
    parser.add_argument("--query-image-id", type=int, default=4)
    parser.add_argument("--object-id", type=int, default=8)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--sam3-prompt", required=True)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument(
        "--foundationpose-repository",
        type=Path,
        default=PROJECT_ROOT / "FoundationPose",
    )
    parser.add_argument(
        "--dgedi-repository",
        type=Path,
        default=PROJECT_ROOT / "external_models" / "dGeDi",
    )
    parser.add_argument("--dgedi-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dgedi-config", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--refine-iterations", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=30000)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.expanduser().resolve()
    models_dir = (
        args.models_dir.expanduser().resolve()
        if args.models_dir is not None
        else dataset_root / "models"
    )
    model_path = models_dir / f"obj_{args.object_id:06d}.ply"

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    reference_gt = load_bop_linemod_absolute_gt_pose(
        BOPFrameGTSpec(
            dataset_root=dataset_root,
            split="test",
            scene_id=args.scene_id,
            image_id=args.reference_image_id,
            object_id=args.object_id,
            instance_index=0,
        )
    )
    query_gt = load_bop_linemod_absolute_gt_pose(
        BOPFrameGTSpec(
            dataset_root=dataset_root,
            split="test",
            scene_id=args.scene_id,
            image_id=args.query_image_id,
            object_id=args.object_id,
            instance_index=0,
        )
    )
    ground_truth_relative = build_ground_truth_relative_pose(
        reference_absolute_pose=reference_gt,
        query_absolute_pose=query_gt,
    ).astype(np.float64)

    candidate = _prepare_true_model_candidate(
        model_path=model_path,
        output_dir=output_root / "gt_model",
    )

    estimated_self_pose_by_view: dict[str, np.ndarray] = {}
    self_aligned_mesh_by_view: dict[str, Path] = {}

    # Run every SAM3 segmentation before FoundationPose ever loads its CUDA
    # context: a second in-process SAM3 call after FoundationPose has run
    # hits a pin_memory() bug in SAM3's geometry encoder (third-party,
    # order-dependent -- not something to patch here).
    prepared_view_by_name = {}
    for view_name, image_id in (
        ("reference", args.reference_image_id),
        ("query", args.query_image_id),
    ):
        loaded_view = load_linemod_view(
            dataset_root=dataset_root,
            view_name=view_name,
            scene_id=args.scene_id,
            image_id=image_id,
            object_name=args.object_name,
            object_id=args.object_id,
        )

        segmentation = generate_sam3_segmentation(
            view=loaded_view,
            output_directory=output_root / "views" / view_name / "segmentation",
            text_prompt=args.sam3_prompt,
        )

        prepared_view_by_name[view_name] = prepare_masked_view(
            view=loaded_view,
            segmentation=segmentation,
            output_directory=output_root / "views" / view_name / "prepared",
        )

    with FoundationPoseRunner(
        repository_path=args.foundationpose_repository,
        output_root=output_root / "foundationpose",
        top_k=args.top_k,
        refine_iterations=args.refine_iterations,
        device="cuda:0",
    ) as runner:
        for view_name in ("reference", "query"):
            result = runner.run_candidate(
                candidate=candidate,
                prepared_view=prepared_view_by_name[view_name],
            )

            best = result.hypotheses[0]

            print(
                f"[FoundationPose self pose] {view_name}: "
                f"rank={best.rank}, score={best.score:.4f}"
            )

            estimated_self_pose_by_view[view_name] = best.pose_cam_from_proxy.astype(
                np.float64
            )

            self_aligned_mesh_by_view[view_name] = _save_self_aligned_mesh(
                source_mesh_path=candidate.scaled_mesh_path,
                pose_camera_from_proxy=estimated_self_pose_by_view[view_name],
                output_mesh_path=(
                    output_root / f"{view_name}_gt_model_estimated_self_pose.obj"
                ),
            )

    true_absolute_by_view = {"reference": reference_gt, "query": query_gt}
    for view_name in ("reference", "query"):
        self_pose_error_deg, self_pose_error_cm = pose_error(
            estimated_self_pose_by_view[view_name],
            true_absolute_by_view[view_name],
        )
        print(
            f"[Self-pose accuracy] {view_name}: "
            f"rotation={self_pose_error_deg:.2f} deg, "
            f"translation={self_pose_error_cm:.2f} cm"
        )

    dgedi_repository = args.dgedi_repository.expanduser().resolve()
    dgedi_python = args.dgedi_python.expanduser().resolve()
    dgedi_config = (
        args.dgedi_config.expanduser().resolve()
        if args.dgedi_config is not None
        else dgedi_repository / "config_dgedi.yaml"
    )

    command = [
        str(dgedi_python),
        str(PROJECT_ROOT / "pose" / "dgedi_runner.py"),
        "--worker",
        "--repository",
        str(dgedi_repository),
        "--config",
        str(dgedi_config),
        "--reference-mesh",
        str(self_aligned_mesh_by_view["reference"]),
        "--query-mesh",
        str(self_aligned_mesh_by_view["query"]),
        "--output-directory",
        str(output_root),
        "--mode",
        "multi_scale",
        "--device",
        "cuda",
        "--sample-count",
        str(args.sample_count),
        "--ransac-threshold",
        "0.03",
        "--icp-threshold",
        "0.03",
    ]

    worker_environment = os.environ.copy()
    worker_environment.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    worker_environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    completed = subprocess.run(
        command,
        cwd=dgedi_repository,
        env=worker_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    print(completed.stdout[-3000:])
    if completed.returncode != 0:
        print(completed.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"dGeDi worker failed with code {completed.returncode}")

    estimate = np.load(
        output_root / "dgedi_proxy_pose_query_from_reference.npy",
        allow_pickle=False,
    ).astype(np.float64)

    rotation_error_deg, translation_error_cm = pose_error(
        estimate,
        ground_truth_relative,
    )

    result_summary = {
        "rotation_error_deg": rotation_error_deg,
        "translation_error_cm": translation_error_cm,
        "note": (
            "True CAD model shape + FoundationPose's own self-pose "
            "estimate (not GT) + dGeDi. Isolates FoundationPose "
            "self-alignment accuracy given perfect mesh shape."
        ),
    }

    (output_root / "result.json").write_text(
        json.dumps(result_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result_summary, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
