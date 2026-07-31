from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from evaluation.relative_pose_evaluator import (
    BOPFrameGTSpec,
    build_ground_truth_relative_pose,
    load_bop_linemod_absolute_gt_pose,
)


def find_named_value(
    value: Any,
    target_name: str,
) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() == target_name.lower():
                return item

        for item in value.values():
            found = find_named_value(
                item,
                target_name,
            )

            if found is not None:
                return found

    elif isinstance(value, list):
        for item in value:
            found = find_named_value(
                item,
                target_name,
            )

            if found is not None:
                return found

    return None


def find_matrix(
    value: Any,
) -> np.ndarray | None:
    if isinstance(value, dict):
        preferred_keys = (
            "pose",
            "transformation",
            "transform",
            "matrix",
            "relative_pose",
        )

        for key in preferred_keys:
            if key in value:
                matrix = find_matrix(value[key])

                if matrix is not None:
                    return matrix

        for item in value.values():
            matrix = find_matrix(item)

            if matrix is not None:
                return matrix

    elif isinstance(value, list):
        try:
            matrix = np.asarray(
                value,
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            matrix = None

        if (
            matrix is not None
            and matrix.shape == (4, 4)
            and np.all(np.isfinite(matrix))
        ):
            return matrix

        for item in value:
            matrix = find_matrix(item)

            if matrix is not None:
                return matrix

    return None


def find_self_pose(
    data: Any,
    names: tuple[str, ...],
) -> np.ndarray | None:
    for name in names:
        value = find_named_value(
            data,
            name,
        )

        if value is None:
            continue

        matrix = find_matrix(value)

        if matrix is not None:
            return matrix

    return None


def pose_error(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, Any]:
    rotation_difference = (
        ground_truth[:3, :3].T
        @ estimate[:3, :3]
    )

    cosine = float(
        np.clip(
            (
                np.trace(rotation_difference)
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )
    )

    rotation_error_deg = float(
        np.degrees(
            np.arccos(cosine)
        )
    )

    translation_xyz_cm = (
        estimate[:3, 3]
        - ground_truth[:3, 3]
    ) * 100.0

    return {
        "rotation_error_deg": (
            rotation_error_deg
        ),
        "translation_error_cm": float(
            np.linalg.norm(
                translation_xyz_cm
            )
        ),
        "translation_xyz_cm": (
            translation_xyz_cm.tolist()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query-image-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--scene-id",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--reference-image-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--object-id",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    query_label = (
        f"q{args.scene_id:06d}_"
        f"{args.query_image_id:06d}_i00"
    )

    query_root = (
        output_root
        / "queries"
        / query_label
    )

    if not query_root.is_dir():
        query_root = output_root

    metadata_candidates = sorted(
        (
            query_root
            / "mesh_registration"
            / "dgedi"
        ).glob("*.json")
    )

    metadata_path = None
    metadata = None

    for candidate in metadata_candidates:
        try:
            candidate_data = json.loads(
                candidate.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if (
            find_named_value(
                candidate_data,
                "ransac",
            )
            is not None
            and find_named_value(
                candidate_data,
                "icp",
            )
            is not None
        ):
            metadata_path = candidate
            metadata = candidate_data
            break

    if metadata_path is None or metadata is None:
        raise FileNotFoundError(
            "RANSAC과 ICP가 포함된 dGeDi metadata를 "
            f"찾지 못했습니다: {metadata_candidates}"
        )

    ransac_section = find_named_value(
        metadata,
        "ransac",
    )

    icp_section = find_named_value(
        metadata,
        "icp",
    )

    ransac_raw = find_matrix(
        ransac_section
    )

    icp_raw = find_matrix(
        icp_section
    )

    if ransac_raw is None:
        raise KeyError(
            "RANSAC pose matrix를 찾지 못했습니다."
        )

    if icp_raw is None:
        raise KeyError(
            "ICP pose matrix를 찾지 못했습니다."
        )

    final_path = (
        query_root
        / "method_results"
        / "self_mesh"
        / "final_relative_pose.npy"
    )

    final_pose = np.load(
        final_path,
        allow_pickle=False,
    ).astype(np.float64)

    selection_path = (
        query_root
        / "method_results"
        / "self_mesh"
        / "final_selection.json"
    )

    selection = json.loads(
        selection_path.read_text(
            encoding="utf-8"
        )
    )

    reference_self = find_self_pose(
        selection,
        (
            "reference_self_pose",
            "reference_pose_camera_from_proxy",
        ),
    )

    query_self = find_self_pose(
        selection,
        (
            "query_self_pose",
            "query_pose_camera_from_proxy",
        ),
    )

    raw_final_gap = float(
        np.linalg.norm(
            icp_raw - final_pose
        )
    )

    composed_final_gap = float("inf")

    if (
        reference_self is not None
        and query_self is not None
    ):
        icp_composed = (
            query_self
            @ icp_raw
            @ np.linalg.inv(reference_self)
        )

        composed_final_gap = float(
            np.linalg.norm(
                icp_composed - final_pose
            )
        )

    if composed_final_gap < raw_final_gap:
        convention = (
            "proxy pose composed with self poses"
        )

        ransac_pose = (
            query_self
            @ ransac_raw
            @ np.linalg.inv(reference_self)
        )

        icp_pose = (
            query_self
            @ icp_raw
            @ np.linalg.inv(reference_self)
        )
    else:
        convention = (
            "direct camera-relative pose"
        )

        ransac_pose = ransac_raw
        icp_pose = icp_raw

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    reference_gt = (
        load_bop_linemod_absolute_gt_pose(
            BOPFrameGTSpec(
                dataset_root=dataset_root,
                split="test",
                scene_id=args.scene_id,
                image_id=args.reference_image_id,
                object_id=args.object_id,
                instance_index=0,
            )
        )
    )

    query_gt = (
        load_bop_linemod_absolute_gt_pose(
            BOPFrameGTSpec(
                dataset_root=dataset_root,
                split="test",
                scene_id=args.scene_id,
                image_id=args.query_image_id,
                object_id=args.object_id,
                instance_index=0,
            )
        )
    )

    ground_truth = (
        build_ground_truth_relative_pose(
            reference_absolute_pose=reference_gt,
            query_absolute_pose=query_gt,
        )
    ).astype(np.float64)

    ransac_error = pose_error(
        ransac_pose,
        ground_truth,
    )

    icp_error = pose_error(
        icp_pose,
        ground_truth,
    )

    delta_rotation = (
        icp_error["rotation_error_deg"]
        - ransac_error["rotation_error_deg"]
    )

    delta_translation = (
        icp_error["translation_error_cm"]
        - ransac_error["translation_error_cm"]
    )

    if (
        delta_rotation < 0.0
        and delta_translation < 0.0
    ):
        classification = "ICP improved both"
    elif (
        delta_rotation > 0.0
        and delta_translation > 0.0
    ):
        classification = "ICP worsened both"
    else:
        classification = "ICP produced mixed changes"

    result = {
        "query_image_id": args.query_image_id,
        "metadata_path": str(metadata_path),
        "inferred_convention": convention,
        "raw_icp_final_gap": raw_final_gap,
        "composed_icp_final_gap": (
            composed_final_gap
        ),
        "ransac": {
            "pose": ransac_pose.tolist(),
            **ransac_error,
        },
        "icp": {
            "pose": icp_pose.tolist(),
            **icp_error,
        },
        "icp_minus_ransac": {
            "rotation_error_deg": (
                delta_rotation
            ),
            "translation_error_cm": (
                delta_translation
            ),
        },
        "classification": classification,
    }

    output_path = (
        query_root
        / "mesh_registration"
        / "dgedi"
        / "stage_diagnosis.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nSaved:", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
