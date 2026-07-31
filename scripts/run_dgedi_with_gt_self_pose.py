from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.relative_pose_evaluator import (
    BOPFrameGTSpec,
    build_ground_truth_relative_pose,
    load_bop_linemod_absolute_gt_pose,
)
from pose.dgedi_runner import _save_self_aligned_mesh
from scripts.visualize_dgedi_alignment import pose_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bake the TRUE GT absolute pose (instead of FoundationPose's "
            "estimated self pose) into OUR reconstructed InstantMesh "
            "candidate meshes, then run dGeDi. Isolates residual mesh-"
            "shape error from self-pose-estimation error: if this comes "
            "out accurate, FoundationPose self-alignment was the "
            "dominant error source; if it's still bad, mesh shape is."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, default=8)
    parser.add_argument("--reference-image-id", type=int, default=0)
    parser.add_argument("--query-image-id", type=int, default=4)
    parser.add_argument("--object-id", type=int, default=8)
    parser.add_argument("--reference-mesh-path", type=Path, required=True)
    parser.add_argument("--query-mesh-path", type=Path, required=True)
    parser.add_argument(
        "--dgedi-repository",
        type=Path,
        default=PROJECT_ROOT / "external_models" / "dGeDi",
    )
    parser.add_argument("--dgedi-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dgedi-config", type=Path, default=None)
    parser.add_argument("--mode", choices=("single_scale", "multi_scale"), default="multi_scale")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-count", type=int, default=30000)
    parser.add_argument("--ransac-threshold", type=float, default=0.03)
    parser.add_argument("--icp-threshold", type=float, default=0.03)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.expanduser().resolve()

    dgedi_repository = args.dgedi_repository.expanduser().resolve()
    dgedi_python = args.dgedi_python.expanduser().resolve()
    dgedi_config = (
        args.dgedi_config.expanduser().resolve()
        if args.dgedi_config is not None
        else dgedi_repository / "config_dgedi.yaml"
    )

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

    # Bake TRUE GT pose onto OUR reconstructed (still possibly wrong-shaped)
    # meshes -- self-pose is now perfect by construction; mesh shape is not.
    reference_mesh = _save_self_aligned_mesh(
        source_mesh_path=args.reference_mesh_path,
        pose_camera_from_proxy=reference_gt,
        output_mesh_path=output_root / "reference_our_mesh_gt_pose.obj",
    )

    query_mesh = _save_self_aligned_mesh(
        source_mesh_path=args.query_mesh_path,
        pose_camera_from_proxy=query_gt,
        output_mesh_path=output_root / "query_our_mesh_gt_pose.obj",
    )

    print(f"[Reference: our mesh, GT pose] {reference_mesh}")
    print(f"[Query: our mesh, GT pose] {query_mesh}")

    command = [
        str(dgedi_python),
        str(PROJECT_ROOT / "pose" / "dgedi_runner.py"),
        "--worker",
        "--repository",
        str(dgedi_repository),
        "--config",
        str(dgedi_config),
        "--reference-mesh",
        str(reference_mesh),
        "--query-mesh",
        str(query_mesh),
        "--output-directory",
        str(output_root),
        "--mode",
        args.mode,
        "--device",
        args.device,
        "--sample-count",
        str(args.sample_count),
        "--ransac-threshold",
        str(args.ransac_threshold),
        "--icp-threshold",
        str(args.icp_threshold),
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

    print(completed.stdout[-4000:])
    if completed.returncode != 0:
        print(completed.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"dGeDi worker failed with code {completed.returncode}")

    proxy_pose_path = output_root / "dgedi_proxy_pose_query_from_reference.npy"
    estimate = np.load(proxy_pose_path, allow_pickle=False).astype(np.float64)

    rotation_error_deg, translation_error_cm = pose_error(
        estimate,
        ground_truth_relative,
    )

    result = {
        "rotation_error_deg": rotation_error_deg,
        "translation_error_cm": translation_error_cm,
        "reference_mesh_path": str(args.reference_mesh_path),
        "query_mesh_path": str(args.query_mesh_path),
        "note": (
            "Our reconstructed mesh shape, but TRUE GT self pose baked "
            "in (not FoundationPose's estimate). Residual error here "
            "is attributable to mesh SHAPE, not self-pose estimation."
        ),
    }

    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
