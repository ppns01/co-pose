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
from mesh_refinement.silhouette_mesh_refiner import (
    refine_mesh_for_silhouette_and_depth,
)
from pose.dgedi_runner import _diameter
from scripts.visualize_dgedi_alignment import pose_error


def refine_one_view(
    *,
    mesh_path: Path,
    views_root: Path,
    view_name: str,
    camera_k: np.ndarray,
    output_dir: Path,
) -> tuple[Path, dict]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    points_camera = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)

    masked_depth_m = np.load(
        views_root / view_name / "prepared" / "masked_depth_m.npy"
    )
    mask_bool = np.load(views_root / view_name / "segmentation" / "mask_bool.npy")

    target_scale_m = float(_diameter(points_camera))

    result = refine_mesh_for_silhouette_and_depth(
        points_camera=points_camera,
        triangles=triangles,
        mask_bool=mask_bool,
        camera_k=camera_k,
        masked_depth_m=masked_depth_m,
        target_scale_m=target_scale_m,
        diameter_fn=_diameter,
    )

    mesh.vertices = o3d.utility.Vector3dVector(result.refined_points_camera)
    mesh.compute_vertex_normals()

    output_dir.mkdir(parents=True, exist_ok=True)
    refined_path = output_dir / f"{view_name}_silhouette_refined.obj"
    o3d.io.write_triangle_mesh(
        str(refined_path), mesh, write_ascii=False, compressed=False,
        print_progress=False,
    )

    return refined_path, result.diagnostics


def acceptance_gate(diag: dict) -> tuple[bool, list[str]]:
    reasons = []
    iou_improved = diag["iou_after"] >= diag["iou_before"]
    if not iou_improved:
        reasons.append(
            f"IoU did not improve ({diag['iou_before']:.3f} -> {diag['iou_after']:.3f})"
        )

    boundary_improved = (
        diag["boundary_distance_after_px"] is None
        or diag["boundary_distance_before_px"] is None
        or diag["boundary_distance_after_px"] <= diag["boundary_distance_before_px"] + 0.1
    )
    if not boundary_improved:
        reasons.append(
            f"boundary distance got worse ({diag['boundary_distance_before_px']:.2f} -> "
            f"{diag['boundary_distance_after_px']:.2f} px)"
        )

    scale_drift = None
    if diag["target_scale_m"]:
        scale_drift = abs(
            diag["scale_after_reprojection_m"] - diag["target_scale_m"]
        ) / diag["target_scale_m"]
        if scale_drift > 0.005:
            reasons.append(f"S* drift too large ({scale_drift*100:.2f}%)")

    if diag["centroid_drift_m"] > 0.003:
        reasons.append(f"centroid drift too large ({diag['centroid_drift_m']*1000:.2f}mm)")

    if diag["displacement_max_m"] > 0.009:
        reasons.append(f"max displacement near/at cap ({diag['displacement_max_m']*1000:.2f}mm)")

    accepted = len(reasons) == 0
    return accepted, reasons


def run_dgedi(reference_mesh: Path, query_mesh: Path, output_dir: Path) -> np.ndarray:
    output_dir.mkdir(parents=True, exist_ok=True)
    dgedi_repository = PROJECT_ROOT / "external_models" / "dGeDi"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "pose" / "dgedi_runner.py"),
        "--worker",
        "--repository", str(dgedi_repository),
        "--config", str(dgedi_repository / "config_dgedi.yaml"),
        "--reference-mesh", str(reference_mesh),
        "--query-mesh", str(query_mesh),
        "--output-directory", str(output_dir),
        "--mode", "multi_scale",
        "--device", "cuda",
        "--sample-count", "30000",
        "--ransac-threshold", "0.03",
        "--icp-threshold", "0.03",
    ]
    env = os.environ.copy()
    env.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    completed = subprocess.run(
        command, cwd=dgedi_repository, env=env, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout[-2000:] + completed.stderr[-2000:])
    return np.load(
        output_dir / "dgedi_proxy_pose_query_from_reference.npy", allow_pickle=False
    ).astype(np.float64)


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
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.expanduser().resolve()

    with open(
        dataset_root / "test" / f"{args.scene_id:06d}" / "scene_camera.json"
    ) as f:
        cams = json.load(f)
    k_ref = np.asarray(cams[str(args.reference_image_id)]["cam_K"]).reshape(3, 3)
    k_qry = np.asarray(cams[str(args.query_image_id)]["cam_K"]).reshape(3, 3)

    ref_refined, ref_diag = refine_one_view(
        mesh_path=args.reference_mesh_path, views_root=args.views_root,
        view_name="reference", camera_k=k_ref, output_dir=output_root,
    )
    qry_refined, qry_diag = refine_one_view(
        mesh_path=args.query_mesh_path, views_root=args.views_root,
        view_name="query", camera_k=k_qry, output_dir=output_root,
    )

    ref_ok, ref_reasons = acceptance_gate(ref_diag)
    qry_ok, qry_reasons = acceptance_gate(qry_diag)

    print("[reference]", json.dumps(ref_diag, indent=2))
    print("accepted:", ref_ok, ref_reasons)
    print("[query]", json.dumps(qry_diag, indent=2))
    print("accepted:", qry_ok, qry_reasons)

    reference_gt = load_bop_linemod_absolute_gt_pose(
        BOPFrameGTSpec(dataset_root=dataset_root, split="test", scene_id=args.scene_id,
                        image_id=args.reference_image_id, object_id=args.object_id, instance_index=0)
    )
    query_gt = load_bop_linemod_absolute_gt_pose(
        BOPFrameGTSpec(dataset_root=dataset_root, split="test", scene_id=args.scene_id,
                        image_id=args.query_image_id, object_id=args.object_id, instance_index=0)
    )
    ground_truth_relative = build_ground_truth_relative_pose(
        reference_absolute_pose=reference_gt, query_absolute_pose=query_gt
    ).astype(np.float64)

    estimate = run_dgedi(ref_refined, qry_refined, output_root / "dgedi")
    rot, trans = pose_error(estimate, ground_truth_relative)

    result = {
        "gate_accepted": bool(ref_ok and qry_ok),
        "reference_gate_reasons": ref_reasons,
        "query_gate_reasons": qry_reasons,
        "rotation_error_deg": rot,
        "translation_error_cm": trans,
        "reference_diagnostics": ref_diag,
        "query_diagnostics": qry_diag,
    }
    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k not in
                       ("reference_diagnostics", "query_diagnostics")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
