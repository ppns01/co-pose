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
            "Feed dGeDi the TRUE BOP CAD model (GT-posed into each "
            "camera frame) instead of our InstantMesh reconstruction, "
            "to see how accurate dGeDi's own registration is when "
            "reconstruction noise is removed entirely."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, default=8)
    parser.add_argument("--reference-image-id", type=int, default=0)
    parser.add_argument("--query-image-id", type=int, default=4)
    parser.add_argument("--object-id", type=int, default=8)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument(
        "--dgedi-repository",
        type=Path,
        default=PROJECT_ROOT / "external_models" / "dGeDi",
    )
    parser.add_argument(
        "--dgedi-python",
        type=Path,
        default=Path(sys.executable),
    )
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
    models_dir = (
        args.models_dir.expanduser().resolve()
        if args.models_dir is not None
        else dataset_root / "models"
    )
    model_path = models_dir / f"obj_{args.object_id:06d}.ply"

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    dgedi_repository = args.dgedi_repository.expanduser().resolve()
    dgedi_python = args.dgedi_python.expanduser().resolve()
    dgedi_config = (
        args.dgedi_config.expanduser().resolve()
        if args.dgedi_config is not None
        else dgedi_repository / "config_dgedi.yaml"
    )

    for path in (dgedi_repository, dgedi_python, dgedi_config):
        if not path.exists():
            raise FileNotFoundError(path)

    # ---- 1. BOP models are millimeters; rescale to meters up front ----
    raw_model = o3d.io.read_triangle_mesh(str(model_path))
    vertices_m = np.asarray(raw_model.vertices, dtype=np.float64) * 0.001
    raw_model.vertices = o3d.utility.Vector3dVector(vertices_m)

    scaled_model_path = output_root / "gt_model_meters.obj"
    o3d.io.write_triangle_mesh(
        str(scaled_model_path),
        raw_model,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )

    # ---- 2. absolute + relative GT poses (meters) ----
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

    # ---- 3. bake GT pose into camera-frame meshes, same convention the
    #          real pipeline uses for FoundationPose-estimated self poses ----
    reference_mesh = _save_self_aligned_mesh(
        source_mesh_path=scaled_model_path,
        pose_camera_from_proxy=reference_gt,
        output_mesh_path=output_root / "reference_gt_model_in_reference_camera.obj",
    )

    query_mesh = _save_self_aligned_mesh(
        source_mesh_path=scaled_model_path,
        pose_camera_from_proxy=query_gt,
        output_mesh_path=output_root / "query_gt_model_in_query_camera.obj",
    )

    print(f"[Reference GT-model mesh] {reference_mesh}")
    print(f"[Query GT-model mesh] {query_mesh}")

    # ---- 4. run the actual dGeDi worker on these meshes ----
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

    # ---- 5. compare dGeDi's estimate (on perfect geometry) to true GT ----
    proxy_pose_path = (
        output_root / "dgedi_proxy_pose_query_from_reference.npy"
    )

    estimate_from_gt_geometry = np.load(
        proxy_pose_path,
        allow_pickle=False,
    ).astype(np.float64)

    rotation_error_deg, translation_error_cm = pose_error(
        estimate_from_gt_geometry,
        ground_truth_relative,
    )

    result = {
        "rotation_error_deg": rotation_error_deg,
        "translation_error_cm": translation_error_cm,
        "note": (
            "dGeDi run directly on the true BOP CAD model, GT-posed "
            "into each camera frame -- zero reconstruction noise on "
            "either side. This isolates dGeDi/RANSAC/ICP registration "
            "accuracy from InstantMesh reconstruction quality."
        ),
    }

    result_path = output_root / "dgedi_on_gt_model_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
