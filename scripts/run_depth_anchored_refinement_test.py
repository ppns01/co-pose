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
from mesh_refinement.depth_anchored_visible_refiner import (
    refine_visible_surface_with_depth,
)
from mesh_refinement.mesh_scale_projector import reproject_to_target_scale
from pose.dgedi_runner import _diameter
from scripts.visualize_dgedi_alignment import pose_error


def _refine_one_view(
    *,
    mesh_path: Path,
    views_root: Path,
    view_name: str,
    camera_k: np.ndarray,
    output_dir: Path,
    max_displacement_m: float,
) -> tuple[Path, dict]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    points_camera = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    masked_depth_m = np.load(
        views_root / view_name / "prepared" / "masked_depth_m.npy"
    )
    mask_bool = np.load(views_root / view_name / "segmentation" / "mask_bool.npy")

    # S* was established upstream (joint_shared) using a different scale
    # convention than _diameter(). Re-projecting to that raw number here
    # would silently rescale by the conversion gap between the two
    # conventions, not by anything the local refinement actually did.
    # Instead, target *this mesh's own* pre-refinement _diameter() reading
    # -- same ruler, before vs after -- which preserves whatever S* already
    # established without introducing a cross-convention correction.
    scale_before_any_refinement = float(_diameter(points_camera))
    target_scale_m = scale_before_any_refinement

    refinement = refine_visible_surface_with_depth(
        points_camera=points_camera,
        triangles=triangles,
        vertex_normals_camera=vertex_normals,
        masked_depth_m=masked_depth_m,
        mask_bool=mask_bool,
        camera_k=camera_k,
        maximum_local_displacement_m=max_displacement_m,
    )

    reprojection = reproject_to_target_scale(
        points_camera=refinement.refined_points_camera,
        target_scale_m=target_scale_m,
        diameter_fn=_diameter,
    )

    mesh.vertices = o3d.utility.Vector3dVector(
        reprojection.reprojected_points_camera
    )
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    output_dir.mkdir(parents=True, exist_ok=True)
    refined_mesh_path = output_dir / f"{view_name}_depth_refined.obj"
    o3d.io.write_triangle_mesh(
        str(refined_mesh_path),
        mesh,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )

    diagnostics = dict(refinement.diagnostics)
    diagnostics.update(
        {
            "view_name": view_name,
            "scale_before_refinement_m": scale_before_any_refinement,
            "scale_after_raw_deformation_m": (
                reprojection.scale_before_reprojection_m
            ),
            "scale_after_reprojection_m": reprojection.scale_after_reprojection_m,
            "target_scale_m": target_scale_m,
            "compensation_beta": reprojection.compensation_beta,
        }
    )

    return refined_mesh_path, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--reference-image-id", type=int, required=True)
    parser.add_argument("--query-image-id", type=int, required=True)
    parser.add_argument("--reference-mesh-path", type=Path, required=True)
    parser.add_argument("--query-mesh-path", type=Path, required=True)
    parser.add_argument("--views-root", type=Path, required=True)
    parser.add_argument("--max-displacement-m", type=float, default=0.010)
    parser.add_argument(
        "--dgedi-repository",
        type=Path,
        default=PROJECT_ROOT / "external_models" / "dGeDi",
    )
    parser.add_argument("--dgedi-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dgedi-config", type=Path, default=None)
    parser.add_argument("--sample-count", type=int, default=30000)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.expanduser().resolve()

    with open(
        dataset_root
        / "test"
        / f"{args.scene_id:06d}"
        / "scene_camera.json"
    ) as camera_file:
        camera_records = json.load(camera_file)

    camera_k_reference = np.asarray(
        camera_records[str(args.reference_image_id)]["cam_K"]
    ).reshape(3, 3)
    camera_k_query = np.asarray(
        camera_records[str(args.query_image_id)]["cam_K"]
    ).reshape(3, 3)

    refined_reference_path, reference_diagnostics = _refine_one_view(
        mesh_path=args.reference_mesh_path,
        views_root=args.views_root,
        view_name="reference",
        camera_k=camera_k_reference,
        output_dir=output_root,
        max_displacement_m=args.max_displacement_m,
    )

    refined_query_path, query_diagnostics = _refine_one_view(
        mesh_path=args.query_mesh_path,
        views_root=args.views_root,
        view_name="query",
        camera_k=camera_k_query,
        output_dir=output_root,
        max_displacement_m=args.max_displacement_m,
    )

    print("[Reference diagnostics]", json.dumps(reference_diagnostics, indent=2))
    print("[Query diagnostics]", json.dumps(query_diagnostics, indent=2))

    (output_root / "refinement_diagnostics.json").write_text(
        json.dumps(
            {"reference": reference_diagnostics, "query": query_diagnostics},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---- run dGeDi on the refined meshes ----
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
        str(refined_reference_path),
        "--query-mesh",
        str(refined_query_path),
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

    rotation_error_deg, translation_error_cm = pose_error(
        estimate,
        ground_truth_relative,
    )

    result = {
        "rotation_error_deg": rotation_error_deg,
        "translation_error_cm": translation_error_cm,
    }

    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
