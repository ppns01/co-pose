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
from mask_provider import generate_sam3_segmentation, release_sam3_processor
from mesh_refinement.depth_anchored_visible_refiner import (
    refine_visible_surface_with_depth,
)
from mesh_refinement.mesh_scale_projector import reproject_to_target_scale
from pose.dgedi_runner import _diameter, _save_self_aligned_mesh
from preprocessing.mask_processing import prepare_masked_view
from scripts.visualize_dgedi_alignment import pose_error


DGEDI_REPOSITORY = PROJECT_ROOT / "external_models" / "dGeDi"
DGEDI_PYTHON = Path(sys.executable)
DGEDI_CONFIG = DGEDI_REPOSITORY / "config_dgedi.yaml"
INSTANTMESH_PYTHON = Path(
    "/home/park/miniforge3/envs/instantmesh_clean/bin/python3.10"
)
INSTANTMESH_REPO = PROJECT_ROOT / "InstantMesh"


def run_dgedi(reference_mesh: Path, query_mesh: Path, output_dir: Path) -> np.ndarray:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(DGEDI_PYTHON),
        str(PROJECT_ROOT / "pose" / "dgedi_runner.py"),
        "--worker",
        "--repository",
        str(DGEDI_REPOSITORY),
        "--config",
        str(DGEDI_CONFIG),
        "--reference-mesh",
        str(reference_mesh),
        "--query-mesh",
        str(query_mesh),
        "--output-directory",
        str(output_dir),
        "--mode",
        "multi_scale",
        "--device",
        "cuda",
        "--sample-count",
        "30000",
        "--ransac-threshold",
        "0.03",
        "--icp-threshold",
        "0.03",
    ]
    env = os.environ.copy()
    env.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    completed = subprocess.run(
        command, cwd=DGEDI_REPOSITORY, env=env, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"dGeDi failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    return np.load(
        output_dir / "dgedi_proxy_pose_query_from_reference.npy",
        allow_pickle=False,
    ).astype(np.float64)


def generate_instantmesh(segmented_rgb_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(INSTANTMESH_PYTHON),
        str(INSTANTMESH_REPO / "run.py"),
        str(PROJECT_ROOT / "configs" / "instantmesh_16gb" / "instant-mesh-large.yaml"),
        str(segmented_rgb_path),
        "--output_path",
        str(output_dir),
        "--diffusion_steps",
        "75",
        "--seed",
        "42",
        "--scale",
        "1.0",
        "--distance",
        "4.5",
        "--view",
        "6",
    ]
    completed = subprocess.run(
        command, cwd=INSTANTMESH_REPO, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"InstantMesh failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    mesh_path = (
        output_dir / "instant-mesh-large" / "meshes" / f"{segmented_rgb_path.stem}.obj"
    )
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    return mesh_path


def refine_mesh(
    *,
    mesh_path: Path,
    views_root: Path,
    view_name: str,
    camera_k: np.ndarray,
    output_dir: Path,
) -> tuple[Path, dict]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    points_camera = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    masked_depth_m = np.load(
        views_root / view_name / "prepared" / "masked_depth_m.npy"
    )
    mask_bool = np.load(views_root / view_name / "segmentation" / "mask_bool.npy")

    scale_before = float(_diameter(points_camera))

    refinement = refine_visible_surface_with_depth(
        points_camera=points_camera,
        triangles=triangles,
        vertex_normals_camera=vertex_normals,
        masked_depth_m=masked_depth_m,
        mask_bool=mask_bool,
        camera_k=camera_k,
        maximum_local_displacement_m=0.010,
        # condition C already bakes in TRUE GT self-pose, so there is no
        # genuine rigid error to remove here.
        remove_global_rigid_component=False,
    )

    reprojection = reproject_to_target_scale(
        points_camera=refinement.refined_points_camera,
        target_scale_m=scale_before,
        diameter_fn=_diameter,
    )

    mesh.vertices = o3d.utility.Vector3dVector(
        reprojection.reprojected_points_camera
    )
    mesh.compute_vertex_normals()

    output_dir.mkdir(parents=True, exist_ok=True)
    refined_path = output_dir / f"{view_name}_refined.obj"
    o3d.io.write_triangle_mesh(
        str(refined_path), mesh, write_ascii=False, compressed=False,
        print_progress=False,
    )

    diagnostics = dict(refinement.diagnostics)
    diagnostics["scale_before_m"] = scale_before
    diagnostics["scale_after_m"] = reprojection.scale_after_reprojection_m

    return refined_path, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--sam3-prompt", required=True)
    parser.add_argument("--reference-image-id", type=int, required=True)
    parser.add_argument(
        "--reference-condition-c-mesh", type=Path, required=True
    )
    parser.add_argument(
        "--reference-views-root", type=Path, required=True
    )
    parser.add_argument("--query-image-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.expanduser().resolve()

    with open(
        dataset_root / "test" / f"{args.scene_id:06d}" / "scene_camera.json"
    ) as camera_file:
        camera_records = json.load(camera_file)

    reference_gt = load_bop_linemod_absolute_gt_pose(
        BOPFrameGTSpec(
            dataset_root=dataset_root, split="test", scene_id=args.scene_id,
            image_id=args.reference_image_id, object_id=args.object_id,
            instance_index=0,
        )
    )

    camera_k_reference = np.asarray(
        camera_records[str(args.reference_image_id)]["cam_K"]
    ).reshape(3, 3)

    # Reference is fixed across every query -- refine it exactly once.
    reference_refined_mesh, reference_diagnostics = refine_mesh(
        mesh_path=args.reference_condition_c_mesh,
        views_root=args.reference_views_root,
        view_name="reference",
        camera_k=camera_k_reference,
        output_dir=output_root / "reference_refined",
    )
    print(
        "[Reference refinement] "
        f"unique_pixels={reference_diagnostics.get('unique_pixel_correspondence_count')}, "
        f"mean_vertices_per_pixel={reference_diagnostics.get('mean_vertices_per_pixel'):.2f}, "
        f"rigid_rot_deg={reference_diagnostics.get('rigid_residual_rotation_deg'):.2f}"
    )

    results = {}

    for query_image_id in args.query_image_ids:
        print(f"\n=== query {query_image_id} ===")
        query_root = output_root / f"query_{query_image_id:03d}"
        query_root.mkdir(parents=True, exist_ok=True)

        query_gt = load_bop_linemod_absolute_gt_pose(
            BOPFrameGTSpec(
                dataset_root=dataset_root, split="test", scene_id=args.scene_id,
                image_id=query_image_id, object_id=args.object_id,
                instance_index=0,
            )
        )
        ground_truth_relative = build_ground_truth_relative_pose(
            reference_absolute_pose=reference_gt,
            query_absolute_pose=query_gt,
        ).astype(np.float64)

        camera_k_query = np.asarray(
            camera_records[str(query_image_id)]["cam_K"]
        ).reshape(3, 3)

        # 1. prepare view
        loaded_view = load_linemod_view(
            dataset_root=dataset_root, view_name="query", scene_id=args.scene_id,
            image_id=query_image_id, object_name=args.object_name,
            object_id=args.object_id,
        )
        segmentation = generate_sam3_segmentation(
            view=loaded_view,
            output_directory=query_root / "views" / "query" / "segmentation",
            text_prompt=args.sam3_prompt,
        )
        prepared_view = prepare_masked_view(
            view=loaded_view, segmentation=segmentation,
            output_directory=query_root / "views" / "query" / "prepared",
        )

        # InstantMesh's mesh-extraction step peaks near this GPU's full
        # 15.47GB capacity; SAM3 left resident from segmentation above
        # is enough to push it into OOM, so release before spawning it.
        release_sam3_processor()

        # 2. generate + scale mesh, bake GT pose (condition C)
        raw_mesh_path = generate_instantmesh(
            segmented_rgb_path=prepared_view.segmented_rgb_path,
            output_dir=query_root / "generated",
        )

        raw_mesh = o3d.io.read_triangle_mesh(str(raw_mesh_path))
        raw_points = np.asarray(raw_mesh.vertices, dtype=np.float64)
        raw_diagonal = float(_diameter(raw_points))
        target_scale_m = float(
            _diameter(
                np.asarray(
                    o3d.io.read_triangle_mesh(
                        str(args.reference_condition_c_mesh)
                    ).vertices
                )
            )
        )
        centroid = raw_points.mean(axis=0)
        scaled_points = centroid + (target_scale_m / raw_diagonal) * (
            raw_points - centroid
        )
        raw_mesh.vertices = o3d.utility.Vector3dVector(scaled_points)
        scaled_mesh_path = query_root / "generated_scaled.obj"
        o3d.io.write_triangle_mesh(
            str(scaled_mesh_path), raw_mesh, write_ascii=False,
            compressed=False, print_progress=False,
        )

        condition_c_query_mesh = _save_self_aligned_mesh(
            source_mesh_path=scaled_mesh_path,
            pose_camera_from_proxy=query_gt,
            output_mesh_path=query_root / "query_condition_c.obj",
        )

        # 3. condition C: dGeDi(reference condition-C, query condition-C)
        estimate_c = run_dgedi(
            args.reference_condition_c_mesh,
            condition_c_query_mesh,
            query_root / "dgedi_c",
        )
        rot_c, trans_c = pose_error(estimate_c, ground_truth_relative)

        # 4. condition D: refine query, then dGeDi(reference refined, query refined)
        query_refined_mesh, query_diagnostics = refine_mesh(
            mesh_path=condition_c_query_mesh,
            views_root=query_root / "views",
            view_name="query",
            camera_k=camera_k_query,
            output_dir=query_root / "refined",
        )
        estimate_d = run_dgedi(
            reference_refined_mesh,
            query_refined_mesh,
            query_root / "dgedi_d",
        )
        rot_d, trans_d = pose_error(estimate_d, ground_truth_relative)

        rigid_rot = query_diagnostics.get("rigid_residual_rotation_deg")
        rigid_rot_text = f"{rigid_rot:.2f}deg" if rigid_rot is not None else "n/a"
        print(
            f"query {query_image_id}: "
            f"C={rot_c:.2f}deg/{trans_c:.2f}cm  "
            f"D={rot_d:.2f}deg/{trans_d:.2f}cm  "
            f"(status={query_diagnostics.get('status')}, "
            f"unique_px={query_diagnostics.get('unique_pixel_correspondence_count')}, "
            f"rigid_rot={rigid_rot_text})"
        )

        results[query_image_id] = {
            "condition_c_rotation_deg": rot_c,
            "condition_c_translation_cm": trans_c,
            "condition_d_rotation_deg": rot_d,
            "condition_d_translation_cm": trans_d,
            "query_refinement_diagnostics": query_diagnostics,
        }

    (output_root / "summary.json").write_text(
        json.dumps(
            {"reference_diagnostics": reference_diagnostics, "queries": results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    for query_image_id, r in results.items():
        print(
            f"query {query_image_id}: "
            f"C={r['condition_c_rotation_deg']:.2f}deg/{r['condition_c_translation_cm']:.2f}cm  "
            f"-> D={r['condition_d_rotation_deg']:.2f}deg/{r['condition_d_translation_cm']:.2f}cm"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
