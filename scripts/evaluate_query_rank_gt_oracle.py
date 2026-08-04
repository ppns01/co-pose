from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"JSON root must be object: {path}"
        )

    return data


def image_record(
    data: dict[str, Any],
    image_id: int,
) -> Any:
    for key in (
        str(image_id),
        f"{image_id:06d}",
    ):
        if key in data:
            return data[key]

    raise KeyError(
        f"image_id={image_id} not found"
    )


def load_bop_gt_pose(
    *,
    scene_gt_path: Path,
    image_id: int,
    object_id: int,
    instance_index: int,
) -> np.ndarray:
    scene_gt = load_json(scene_gt_path)

    annotations = image_record(
        scene_gt,
        image_id,
    )

    matches = [
        item
        for item in annotations
        if int(item["obj_id"]) == object_id
    ]

    if instance_index >= len(matches):
        raise IndexError(
            "GT instance index out of range: "
            f"object_id={object_id}, "
            f"instance_index={instance_index}, "
            f"match_count={len(matches)}"
        )

    annotation = matches[instance_index]

    rotation = np.asarray(
        annotation["cam_R_m2c"],
        dtype=np.float64,
    ).reshape(3, 3)

    # BOP translation unit: millimeter
    translation_m = (
        np.asarray(
            annotation["cam_t_m2c"],
            dtype=np.float64,
        ).reshape(3)
        / 1000.0
    )

    pose = np.eye(
        4,
        dtype=np.float64,
    )

    pose[:3, :3] = rotation
    pose[:3, 3] = translation_m

    return pose


def rotation_error_deg(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    relative_rotation = (
        estimated[:3, :3]
        @ ground_truth[:3, :3].T
    )

    cosine = float(
        np.clip(
            (
                np.trace(relative_rotation)
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )
    )

    return math.degrees(
        math.acos(cosine)
    )


def translation_error_cm(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            estimated[:3, 3]
            - ground_truth[:3, 3]
        )
        * 100.0
    )


def translation_error_xyz_cm(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> list[float]:
    return (
        (
            estimated[:3, 3]
            - ground_truth[:3, 3]
        )
        * 100.0
    ).tolist()


def load_hypothesis_pose(
    *,
    foundationpose_json: Path,
    rank: int,
) -> tuple[np.ndarray, Path, float]:
    data = load_json(
        foundationpose_json
    )

    hypothesis = next(
        (
            item
            for item in data["hypotheses"]
            if int(item["rank"]) == rank
        ),
        None,
    )

    if hypothesis is None:
        raise KeyError(
            f"rank={rank} not found in "
            f"{foundationpose_json}"
        )

    pose = np.asarray(
        hypothesis["pose_cam_from_proxy"],
        dtype=np.float64,
    )

    if pose.shape != (4, 4):
        raise ValueError(
            f"Invalid pose shape: {pose.shape}"
        )

    mesh_path = Path(
        data["scaled_mesh_path"]
    ).expanduser().resolve()

    score = float(
        hypothesis["score"]
    )

    return pose, mesh_path, score


def bake_mesh(
    *,
    source_mesh_path: Path,
    pose_camera_from_proxy: np.ndarray,
    output_mesh_path: Path,
) -> Path:
    mesh = o3d.io.read_triangle_mesh(
        str(source_mesh_path),
        enable_post_processing=True,
    )

    if len(mesh.vertices) == 0:
        raise ValueError(
            f"Empty mesh: {source_mesh_path}"
        )

    mesh.transform(
        pose_camera_from_proxy
    )

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    output_mesh_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = o3d.io.write_triangle_mesh(
        str(output_mesh_path),
        mesh,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )

    if not success:
        raise IOError(
            f"Failed to write: {output_mesh_path}"
        )

    return output_mesh_path


def mesh_vertex_max_error_mm(
    first_path: Path,
    second_path: Path,
) -> float | None:
    if (
        not first_path.is_file()
        or not second_path.is_file()
    ):
        return None

    first = o3d.io.read_triangle_mesh(
        str(first_path)
    )

    second = o3d.io.read_triangle_mesh(
        str(second_path)
    )

    first_vertices = np.asarray(
        first.vertices,
        dtype=np.float64,
    )

    second_vertices = np.asarray(
        second.vertices,
        dtype=np.float64,
    )

    if first_vertices.shape != second_vertices.shape:
        return None

    return float(
        np.linalg.norm(
            first_vertices - second_vertices,
            axis=1,
        ).max()
        * 1000.0
    )


def evaluate_pose(
    pose: np.ndarray,
    gt_relative_pose: np.ndarray,
) -> dict[str, Any]:
    return {
        "rotation_error_deg": (
            rotation_error_deg(
                pose,
                gt_relative_pose,
            )
        ),
        "translation_error_cm": (
            translation_error_cm(
                pose,
                gt_relative_pose,
            )
        ),
        "translation_error_xyz_cm": (
            translation_error_xyz_cm(
                pose,
                gt_relative_pose,
            )
        ),
        "pose": pose.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--candidate-index",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--icp-threshold",
        type=float,
        default=0.03,
    )

    args = parser.parse_args()

    project_root = Path(
        __file__
    ).resolve().parents[1]

    run = args.run.expanduser().resolve()

    pipeline_config = load_json(
        run / "pipeline_config.json"
    )

    selection = load_json(
        run
        / "visible_scale_refinement"
        / "joint_shared"
        / "selection.json"
    )

    candidate_index = (
        int(
            selection[
                "selected_candidate_index"
            ]
        )
        if args.candidate_index is None
        else args.candidate_index
    )

    selected_query_rank = int(
        selection[
            "selected_record"
        ][
            "query_hypothesis_rank"
        ]
    )

    object_id = int(
        pipeline_config["object_id"]
    )

    split = str(
        pipeline_config["split"]
    )

    dataset_root = Path(
        pipeline_config["dataset_root"]
    ).expanduser().resolve()

    reference = pipeline_config[
        "reference"
    ]

    query = pipeline_config[
        "query"
    ]

    reference_scene_id = int(
        reference["scene_id"]
    )

    reference_image_id = int(
        reference["image_id"]
    )

    reference_instance_index = int(
        reference["instance_index"]
    )

    query_scene_id = int(
        query["scene_id"]
    )

    query_image_id = int(
        query["image_id"]
    )

    query_instance_index = int(
        query["instance_index"]
    )

    reference_gt = load_bop_gt_pose(
        scene_gt_path=(
            dataset_root
            / split
            / f"{reference_scene_id:06d}"
            / "scene_gt.json"
        ),
        image_id=reference_image_id,
        object_id=object_id,
        instance_index=(
            reference_instance_index
        ),
    )

    query_gt = load_bop_gt_pose(
        scene_gt_path=(
            dataset_root
            / split
            / f"{query_scene_id:06d}"
            / "scene_gt.json"
        ),
        image_id=query_image_id,
        object_id=object_id,
        instance_index=query_instance_index,
    )

    gt_relative_pose = (
        query_gt
        @ np.linalg.inv(reference_gt)
    )

    query_fp_json = (
        run
        / "foundationpose"
        / "self_joint_shared_scale"
        / "query"
        / f"candidate_{candidate_index:02d}"
        / "foundationpose_result.json"
    )

    original_dgedi_root = (
        run
        / "mesh_registration"
        / "dgedi"
    )

    reference_camera_mesh = (
        original_dgedi_root
        / "self_aligned_meshes"
        / (
            "reference_self_aligned_"
            "in_reference_camera.obj"
        )
    )

    original_query_camera_mesh = (
        original_dgedi_root
        / "self_aligned_meshes"
        / (
            "query_self_aligned_"
            "in_query_camera.obj"
        )
    )

    if not reference_camera_mesh.is_file():
        raise FileNotFoundError(
            "기존 Reference self-aligned mesh가 없습니다: "
            f"{reference_camera_mesh}"
        )

    dgedi_repository = Path(
        os.environ.get(
            "DGEDI_REPOSITORY",
            str(
                project_root
                / "external_models"
                / "dGeDi"
            ),
        )
    ).expanduser().resolve()

    dgedi_config = Path(
        os.environ.get(
            "DGEDI_CONFIG",
            str(
                dgedi_repository
                / "config_dgedi.yaml"
            ),
        )
    ).expanduser().resolve()

    dgedi_python = Path(
        os.environ.get(
            "DGEDI_PYTHON",
            sys.executable,
        )
    ).expanduser().resolve()

    output_root = (
        run
        / "diagnostics"
        / "query_rank_gt_oracle"
        / f"candidate_{candidate_index:02d}"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[dict[str, Any]] = []

    print("=== Query rank GT oracle ===")
    print("run                 :", run)
    print("candidate           :", candidate_index)
    print("pipeline selected   :", selected_query_rank)
    print("ranks               :", args.ranks)
    print("reference mesh      :", reference_camera_mesh)
    print("GT relative pose:")
    print(gt_relative_pose)
    print()

    for rank in args.ranks:
        rank_root = (
            output_root
            / f"rank_{rank:02d}"
        )

        rank_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            query_pose,
            query_proxy_mesh,
            foundationpose_score,
        ) = load_hypothesis_pose(
            foundationpose_json=query_fp_json,
            rank=rank,
        )

        query_camera_mesh = bake_mesh(
            source_mesh_path=query_proxy_mesh,
            pose_camera_from_proxy=query_pose,
            output_mesh_path=(
                rank_root
                / (
                    "query_rank_"
                    f"{rank:02d}_"
                    "self_aligned_camera.obj"
                )
            ),
        )

        baseline_mesh_difference_mm = None

        if rank == selected_query_rank:
            baseline_mesh_difference_mm = (
                mesh_vertex_max_error_mm(
                    query_camera_mesh,
                    original_query_camera_mesh,
                )
            )

        dgedi_output = (
            rank_root / "dgedi"
        )

        command = [
            str(dgedi_python),
            str(
                project_root
                / "pose"
                / "dgedi_runner.py"
            ),
            "--worker",
            "--repository",
            str(dgedi_repository),
            "--config",
            str(dgedi_config),
            "--reference-mesh",
            str(reference_camera_mesh),
            "--query-mesh",
            str(query_camera_mesh),
            "--output-directory",
            str(dgedi_output),
            "--mode",
            os.environ.get(
                "DGEDI_MODE",
                "multi_scale",
            ),
            "--device",
            os.environ.get(
                "DGEDI_DEVICE",
                "cuda",
            ),
            "--sample-count",
            str(args.sample_count),
            "--ransac-threshold",
            str(args.ransac_threshold),
            "--icp-threshold",
            str(args.icp_threshold),
        ]

        environment = os.environ.copy()

        environment.pop(
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
            None,
        )

        environment[
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        ] = "1"

        print(
            f"[rank {rank:02d}] dGeDi 실행"
        )

        completed = subprocess.run(
            command,
            cwd=dgedi_repository,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        (
            rank_root / "dgedi_stdout.txt"
        ).write_text(
            completed.stdout,
            encoding="utf-8",
        )

        (
            rank_root / "dgedi_stderr.txt"
        ).write_text(
            completed.stderr,
            encoding="utf-8",
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"dGeDi failed for rank={rank}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        metadata = load_json(
            dgedi_output
            / "dgedi_registration.json"
        )

        ransac_pose = np.asarray(
            metadata["ransac"]["pose"],
            dtype=np.float64,
        )

        icp_pose = np.asarray(
            metadata["icp"]["pose"],
            dtype=np.float64,
        )

        record = {
            "rank": rank,
            "pipeline_selected": (
                rank == selected_query_rank
            ),
            "foundationpose_score": (
                foundationpose_score
            ),
            "baseline_mesh_max_difference_mm": (
                baseline_mesh_difference_mm
            ),
            "query_pose_camera_from_proxy": (
                query_pose.tolist()
            ),
            "ransac": {
                "fitness": float(
                    metadata["ransac"]["fitness"]
                ),
                "inlier_rmse_m": float(
                    metadata["ransac"][
                        "inlier_rmse_m"
                    ]
                ),
                "correspondence_count": int(
                    metadata["ransac"][
                        "correspondence_count"
                    ]
                ),
                **evaluate_pose(
                    ransac_pose,
                    gt_relative_pose,
                ),
            },
            "icp": {
                "fitness": float(
                    metadata["icp"]["fitness"]
                ),
                "inlier_rmse_m": float(
                    metadata["icp"][
                        "inlier_rmse_m"
                    ]
                ),
                "correspondence_count": int(
                    metadata["icp"][
                        "correspondence_count"
                    ]
                ),
                **evaluate_pose(
                    icp_pose,
                    gt_relative_pose,
                ),
            },
        }

        results.append(record)

        print(
            f"rank={rank:02d} "
            f"RANSAC="
            f"{record['ransac']['rotation_error_deg']:.3f}deg/"
            f"{record['ransac']['translation_error_cm']:.3f}cm "
            f"ICP="
            f"{record['icp']['rotation_error_deg']:.3f}deg/"
            f"{record['icp']['translation_error_cm']:.3f}cm"
        )

    summary = {
        "run": str(run),
        "candidate_index": candidate_index,
        "pipeline_selected_rank": (
            selected_query_rank
        ),
        "pose_convention": (
            "T_query_camera_from_reference_camera"
        ),
        "ground_truth_relative_pose": (
            gt_relative_pose.tolist()
        ),
        "results": results,
    }

    summary_path = (
        output_root
        / "query_rank_gt_oracle.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=== ICP GT 기준 정렬 ===")

    for record in sorted(
        results,
        key=lambda item: (
            item["icp"]["rotation_error_deg"],
            item["icp"]["translation_error_cm"],
        ),
    ):
        marker = (
            " *PIPELINE_SELECTED"
            if record["pipeline_selected"]
            else ""
        )

        print(
            f"rank={record['rank']:02d} "
            f"rotation="
            f"{record['icp']['rotation_error_deg']:.3f}deg "
            f"translation="
            f"{record['icp']['translation_error_cm']:.3f}cm "
            f"fitness="
            f"{record['icp']['fitness']:.4f}"
            f"{marker}"
        )

    print()
    print("saved:", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
