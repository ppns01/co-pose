from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))

    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {path}")

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64,
    )

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64,
    )

    return vertices, triangles


def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    return points @ rotation.T + translation


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = (
        np.trace(rotation) - 1.0
    ) / 2.0

    cosine = float(
        np.clip(cosine, -1.0, 1.0)
    )

    return math.degrees(
        math.acos(cosine)
    )


def pose_difference(
    estimated: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    delta = estimated @ np.linalg.inv(reference)

    return {
        "rotation_difference_deg": (
            rotation_angle_deg(delta[:3, :3])
        ),
        "translation_difference_mm": float(
            np.linalg.norm(delta[:3, 3])
            * 1000.0
        ),
        "delta_matrix": delta.tolist(),
    }


def rigid_fit(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    if source.shape != target.shape:
        raise ValueError(
            f"Point shape mismatch: "
            f"{source.shape} != {target.shape}"
        )

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)

    source_zero = source - source_center
    target_zero = target - target_center

    covariance = source_zero.T @ target_zero

    u, _, vt = np.linalg.svd(
        covariance,
        full_matrices=False,
    )

    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T

    translation = (
        target_center
        - rotation @ source_center
    )

    transform = np.eye(
        4,
        dtype=np.float64,
    )

    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


def triangle_normals(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    faces = vertices[triangles]

    normals = np.cross(
        faces[:, 1] - faces[:, 0],
        faces[:, 2] - faces[:, 0],
    )

    lengths = np.linalg.norm(
        normals,
        axis=1,
    )

    valid = lengths > 1e-12

    normalized = np.zeros_like(
        normals,
        dtype=np.float64,
    )

    normalized[valid] = (
        normals[valid]
        / lengths[valid, None]
    )

    return normalized, valid


def load_foundationpose_selection(
    metadata_path: Path,
    hypothesis_rank: int,
) -> tuple[Path, np.ndarray, dict[str, Any]]:
    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8",
        )
    )

    hypotheses = metadata.get(
        "hypotheses",
        [],
    )

    selected = None

    for item in hypotheses:
        if int(item["rank"]) == hypothesis_rank:
            selected = item
            break

    if selected is None:
        raise KeyError(
            f"hypothesis rank {hypothesis_rank} "
            f"not found in {metadata_path}"
        )

    original_mesh_path = Path(
        metadata["scaled_mesh_path"]
    ).expanduser().resolve()

    if "pose_cam_from_proxy" in selected:
        pose = np.asarray(
            selected["pose_cam_from_proxy"],
            dtype=np.float64,
        )
    else:
        pose_path = Path(
            selected["pose_path"]
        ).expanduser().resolve()

        pose = np.load(
            pose_path,
            allow_pickle=False,
        ).astype(np.float64)

    if pose.shape != (4, 4):
        raise ValueError(
            f"T0 shape must be (4, 4): "
            f"{pose.shape}"
        )

    return (
        original_mesh_path,
        pose,
        metadata,
    )


def error_statistics(
    estimated: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    errors = np.linalg.norm(
        estimated - reference,
        axis=1,
    )

    return {
        "rmse_mm": float(
            np.sqrt(np.mean(errors**2))
            * 1000.0
        ),
        "median_mm": float(
            np.median(errors)
            * 1000.0
        ),
        "p95_mm": float(
            np.quantile(errors, 0.95)
            * 1000.0
        ),
        "maximum_mm": float(
            errors.max()
            * 1000.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--foundationpose-result",
        type=Path,
        required=True,
        help=(
            "Selected candidate의 "
            "foundationpose_result.json"
        ),
    )

    parser.add_argument(
        "--hypothesis-rank",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--m0-mesh",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--gt-pose-npy",
        type=Path,
        default=None,
        help="Optional T_camera_from_CAD",
    )

    parser.add_argument(
        "--cad-to-proxy-npy",
        type=Path,
        default=None,
        help=(
            "Optional T_proxy_from_CAD. "
            "GT pose 비교에 반드시 필요함."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/t0_m0_consistency.json"
        ),
    )

    args = parser.parse_args()

    metadata_path = (
        args.foundationpose_result
        .expanduser()
        .resolve()
    )

    m0_path = (
        args.m0_mesh
        .expanduser()
        .resolve()
    )

    (
        original_mesh_path,
        t0,
        metadata,
    ) = load_foundationpose_selection(
        metadata_path,
        args.hypothesis_rank,
    )

    original_vertices, original_triangles = (
        load_mesh(original_mesh_path)
    )

    m0_vertices, m0_triangles = load_mesh(
        m0_path
    )

    if (
        original_vertices.shape
        != m0_vertices.shape
    ):
        raise RuntimeError(
            "Vertex count mismatch: "
            f"original={original_vertices.shape}, "
            f"M0={m0_vertices.shape}"
        )

    triangles_equal = (
        original_triangles.shape
        == m0_triangles.shape
        and np.array_equal(
            original_triangles,
            m0_triangles,
        )
    )

    # 1. T0를 원본 proxy에 직접 적용한 예상 M0
    expected_m0 = transform_points(
        original_vertices,
        t0,
    )

    forward_error = error_statistics(
        m0_vertices,
        expected_m0,
    )

    # M0 vertex correspondence로 실제 bake 행렬 역산
    fitted_bake_pose = rigid_fit(
        original_vertices,
        m0_vertices,
    )

    bake_pose_difference = pose_difference(
        fitted_bake_pose,
        t0,
    )

    # 3. M0를 T0 inverse로 proxy frame 복원
    recovered_proxy = transform_points(
        m0_vertices,
        np.linalg.inv(t0),
    )

    inverse_error = error_statistics(
        recovered_proxy,
        original_vertices,
    )

    rotation = t0[:3, :3]

    rigid_validity = {
        "determinant": float(
            np.linalg.det(rotation)
        ),
        "orthogonality_error_fro": float(
            np.linalg.norm(
                rotation.T @ rotation
                - np.eye(3)
            )
        ),
    }

    # 4. triangle normal 변환 검증
    normal_report: dict[str, Any] = {
        "available": False,
    }

    if triangles_equal:
        original_normals, valid_original = (
            triangle_normals(
                original_vertices,
                original_triangles,
            )
        )

        m0_normals, valid_m0 = (
            triangle_normals(
                m0_vertices,
                m0_triangles,
            )
        )

        valid = valid_original & valid_m0

        expected_normals = (
            original_normals
            @ rotation.T
        )

        dots = np.sum(
            expected_normals[valid]
            * m0_normals[valid],
            axis=1,
        )

        dots = np.clip(
            dots,
            -1.0,
            1.0,
        )

        normal_report = {
            "available": True,
            "valid_face_count": int(
                np.count_nonzero(valid)
            ),
            "mean_dot": float(
                dots.mean()
            ),
            "median_dot": float(
                np.median(dots)
            ),
            "minimum_dot": float(
                dots.min()
            ),
            "flipped_fraction": float(
                np.mean(dots < 0.0)
            ),
            "over_0_999_fraction": float(
                np.mean(dots > 0.999)
            ),
        }

    gt_report: dict[str, Any]

    if (
        args.gt_pose_npy is not None
        and args.cad_to_proxy_npy
        is not None
    ):
        gt_pose = np.load(
            args.gt_pose_npy,
            allow_pickle=False,
        ).astype(np.float64)

        cad_to_proxy = np.load(
            args.cad_to_proxy_npy,
            allow_pickle=False,
        ).astype(np.float64)

        predicted_camera_from_cad = (
            t0 @ cad_to_proxy
        )

        gt_report = {
            "available": True,
            "note": (
                "predicted T_camera_from_CAD "
                "= T0 @ T_proxy_from_CAD"
            ),
            **pose_difference(
                predicted_camera_from_cad,
                gt_pose,
            ),
            "predicted_camera_from_cad": (
                predicted_camera_from_cad
                .tolist()
            ),
            "ground_truth_camera_from_cad": (
                gt_pose.tolist()
            ),
        }
    else:
        gt_report = {
            "available": False,
            "reason": (
                "T0 is camera<-proxy while GT is "
                "camera<-CAD. Direct rotation comparison "
                "is invalid without T_proxy_from_CAD."
            ),
        }

    report = {
        "foundationpose_metadata": str(
            metadata_path
        ),
        "candidate_index": metadata.get(
            "candidate_index"
        ),
        "hypothesis_rank": (
            args.hypothesis_rank
        ),
        "original_proxy_mesh": str(
            original_mesh_path
        ),
        "m0_camera_mesh": str(m0_path),
        "vertex_count": int(
            len(original_vertices)
        ),
        "triangle_count": int(
            len(original_triangles)
        ),
        "triangles_equal": bool(
            triangles_equal
        ),
        "t0_pose_camera_from_proxy": (
            t0.tolist()
        ),
        "fitted_bake_pose": (
            fitted_bake_pose.tolist()
        ),
        "t0_rigid_validity": (
            rigid_validity
        ),
        "forward_bake_vertex_error": (
            forward_error
        ),
        "fitted_bake_vs_t0": (
            bake_pose_difference
        ),
        "inverse_recovery_vertex_error": (
            inverse_error
        ),
        "triangle_normal_consistency": (
            normal_report
        ),
        "gt_pose_comparison": gt_report,
    }

    output_path = (
        args.output_json
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"\nsaved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
