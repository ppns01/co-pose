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


REFERENCE_KEYS = {
    "reference_self_pose",
    "reference_pose_camera_from_proxy",
}

QUERY_KEYS = {
    "query_self_pose",
    "query_pose_camera_from_proxy",
}


def _as_matrix(value: Any) -> np.ndarray | None:
    try:
        matrix = np.asarray(
            value,
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return None

    if (
        matrix.shape == (4, 4)
        and np.all(np.isfinite(matrix))
    ):
        return matrix

    return None


def _find_value(
    value: Any,
    keys: set[str],
) -> np.ndarray | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys:
                matrix = _as_matrix(item)

                if matrix is not None:
                    return matrix

        for item in value.values():
            found = _find_value(item, keys)

            if found is not None:
                return found

    elif isinstance(value, list):
        for item in value:
            found = _find_value(item, keys)

            if found is not None:
                return found

    return None


def _find_matrix_in_jsons(
    root: Path,
    keys: set[str],
) -> tuple[np.ndarray, Path]:
    preferred = [
        root
        / "method_results"
        / "self_mesh"
        / "final_selection.json"
    ]

    candidates = preferred + sorted(
        root.rglob("*.json")
    )

    visited: set[Path] = set()

    for path in candidates:
        path = path.resolve()

        if path in visited or not path.is_file():
            continue

        visited.add(path)

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except Exception:
            continue

        matrix = _find_value(data, keys)

        if matrix is not None:
            return matrix, path

    raise KeyError(
        f"Could not find matrix keys: {sorted(keys)}"
    )


def _pose_error(
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
                np.trace(
                    rotation_difference
                )
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
        "--query-image-id",
        type=int,
        default=4,
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

    query_roots = [
        path
        for path in output_root.rglob(
            query_label
        )
        if path.is_dir()
    ]

    if len(query_roots) != 1:
        raise RuntimeError(
            f"Expected one query root for "
            f"{query_label}; found={len(query_roots)}"
        )

    query_root = query_roots[0]

    reference_self, reference_source = (
        _find_matrix_in_jsons(
            query_root,
            REFERENCE_KEYS,
        )
    )

    query_self, query_source = (
        _find_matrix_in_jsons(
            query_root,
            QUERY_KEYS,
        )
    )

    self_only = (
        query_self
        @ np.linalg.inv(
            reference_self
        )
    )

    final_pose_path = (
        query_root
        / "method_results"
        / "self_mesh"
        / "final_relative_pose.npy"
    )

    final_pose = np.load(
        final_pose_path,
        allow_pickle=False,
    ).astype(np.float64)

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
                image_id=(
                    args.reference_image_id
                ),
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
            reference_absolute_pose=(
                reference_gt
            ),
            query_absolute_pose=query_gt,
        )
    ).astype(np.float64)

    self_error = _pose_error(
        self_only,
        ground_truth,
    )

    final_error = _pose_error(
        final_pose,
        ground_truth,
    )

    result = {
        "query_root": str(query_root),
        "reference_self_source": str(
            reference_source
        ),
        "query_self_source": str(
            query_source
        ),
        "self_only": {
            "pose": self_only.tolist(),
            **self_error,
        },
        "final": {
            "pose": final_pose.tolist(),
            **final_error,
        },
        "ground_truth": (
            ground_truth.tolist()
        ),
    }

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    output_path = (
        query_root
        / "foundationpose_self_diagnosis.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved:", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
