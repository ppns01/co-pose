from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray

from evaluation.relative_pose_evaluator import (
    BOPFrameGTSpec,
    load_bop_linemod_absolute_gt_pose,
)


@dataclass(frozen=True)
class OfficialBOP19EvaluationResult:
    result_csv_path: Path
    targets_path: Path
    dataset_view_root: Path
    scores_path: Path
    scores: dict[str, float]
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_csv_path": str(self.result_csv_path),
            "targets_path": str(self.targets_path),
            "dataset_view_root": str(self.dataset_view_root),
            "scores_path": str(self.scores_path),
            "scores": self.scores,
            "command": list(self.command),
        }


def _validate_pose(
    pose: NDArray[np.floating],
    name: str,
) -> NDArray[np.float64]:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite (4, 4) pose.")
    if not np.allclose(
        value[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError(f"{name} is not a homogeneous transform.")
    return np.ascontiguousarray(value)


def relative_pose_to_query_absolute_mm(
    *,
    predicted_relative_pose: NDArray[np.floating],
    reference_frame: BOPFrameGTSpec,
) -> NDArray[np.float64]:
    """Anchor a relative estimate to reference GT for evaluation only."""
    relative_pose = _validate_pose(
        predicted_relative_pose,
        "predicted_relative_pose",
    )
    reference_pose_m = _validate_pose(
        load_bop_linemod_absolute_gt_pose(reference_frame),
        "reference_absolute_pose",
    )
    query_pose_m = _validate_pose(
        relative_pose @ reference_pose_m,
        "predicted_query_absolute_pose",
    )
    query_pose_mm = query_pose_m.copy()
    query_pose_mm[:3, 3] *= 1000.0
    return query_pose_mm


def _ensure_directory_symlink(
    *,
    link_path: Path,
    target_path: Path,
) -> None:
    resolved_target = target_path.expanduser().resolve()
    if not resolved_target.is_dir():
        raise FileNotFoundError(
            f"BOP dataset directory not found: {resolved_target}"
        )
    if os.path.lexists(link_path):
        if link_path.is_symlink() and link_path.resolve() == resolved_target:
            return
        raise FileExistsError(
            "Refusing to replace an existing BOP dataset-view path: "
            f"{link_path}"
        )
    link_path.symlink_to(resolved_target, target_is_directory=True)


def _camera_payload(
    *,
    dataset_root: Path,
    split: str,
    target: dict[str, int],
) -> dict[str, float | int]:
    scene_id = int(target["scene_id"])
    image_id = int(target["im_id"])
    scene_root = dataset_root / split / f"{scene_id:06d}"
    scene_camera_path = scene_root / "scene_camera.json"
    scene_camera = json.loads(
        scene_camera_path.read_text(encoding="utf-8")
    )
    camera = scene_camera.get(str(image_id))
    if not isinstance(camera, dict):
        raise KeyError(
            f"Image {image_id} is absent from {scene_camera_path}"
        )
    camera_matrix = np.asarray(
        camera["cam_K"],
        dtype=np.float64,
    ).reshape(3, 3)
    depth_path = scene_root / "depth" / f"{image_id:06d}.png"
    depth = iio.imread(depth_path)
    height, width = depth.shape[:2]
    return {
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "width": int(width),
        "height": int(height),
        "depth_scale": float(camera.get("depth_scale", 1.0)),
    }


def create_bop_linemod_dataset_view(
    *,
    dataset_root: Path,
    split: str,
    output_directory: Path,
    targets: Sequence[dict[str, int]],
) -> tuple[Path, Path]:
    """Create a small symlink-based dataset root accepted by BOP Toolkit."""
    if not targets:
        raise ValueError("At least one BOP evaluation target is required.")
    resolved_dataset_root = Path(dataset_root).expanduser().resolve()
    dataset_view_root = (
        Path(output_directory).expanduser().resolve() / "dataset_view"
    )
    linemod_root = dataset_view_root / "lm"
    linemod_root.mkdir(parents=True, exist_ok=True)

    _ensure_directory_symlink(
        link_path=linemod_root / split,
        target_path=resolved_dataset_root / split,
    )
    _ensure_directory_symlink(
        link_path=linemod_root / "models_eval",
        target_path=resolved_dataset_root / "models_eval",
    )

    camera_path = linemod_root / "camera.json"
    camera_path.write_text(
        json.dumps(
            _camera_payload(
                dataset_root=resolved_dataset_root,
                split=split,
                target=targets[0],
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    targets_path = linemod_root / "test_targets_relative_pose.json"
    targets_path.write_text(
        json.dumps(
            list(targets),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_view_root, targets_path


def run_official_bop19_evaluation(
    *,
    toolkit_root: Path,
    dataset_view_root: Path,
    result_csv_path: Path,
    targets_path: Path,
    evaluation_directory: Path,
    renderer_type: str = "python",
    num_workers: int = 1,
) -> OfficialBOP19EvaluationResult:
    """Run BOP Toolkit's unmodified eval_bop19_pose.py entrypoint."""
    if num_workers < 1:
        raise ValueError(f"num_workers must be positive: {num_workers}")
    resolved_toolkit_root = Path(toolkit_root).expanduser().resolve()
    evaluation_script = resolved_toolkit_root / "scripts" / "eval_bop19_pose.py"
    if not evaluation_script.is_file():
        raise FileNotFoundError(
            f"BOP evaluation script not found: {evaluation_script}"
        )
    resolved_result_csv = Path(result_csv_path).expanduser().resolve()
    resolved_targets = Path(targets_path).expanduser().resolve()
    resolved_evaluation_directory = (
        Path(evaluation_directory).expanduser().resolve()
    )
    resolved_evaluation_directory.mkdir(parents=True, exist_ok=True)

    command = (
        sys.executable,
        str(evaluation_script),
        f"--renderer_type={renderer_type}",
        f"--result_filenames={resolved_result_csv.name}",
        f"--results_path={resolved_result_csv.parent}",
        f"--eval_path={resolved_evaluation_directory}",
        f"--targets_filename={resolved_targets}",
        f"--num_workers={int(num_workers)}",
        "--cleanup_eval",
    )
    environment = os.environ.copy()
    environment["BOP_PATH"] = str(
        Path(dataset_view_root).expanduser().resolve()
    )
    environment["BOP_RESULTS_PATH"] = str(resolved_result_csv.parent)
    environment["BOP_EVAL_PATH"] = str(resolved_evaluation_directory)
    environment["BOP_NUM_WORKERS"] = str(int(num_workers))
    python_directory = str(Path(sys.executable).resolve().parent)
    environment["PATH"] = (
        python_directory
        + os.pathsep
        + environment.get("PATH", "")
    )

    completed = subprocess.run(
        command,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Official BOP19 evaluation failed with exit code "
            f"{completed.returncode}."
        )

    result_name = resolved_result_csv.stem
    scores_path = (
        resolved_evaluation_directory
        / result_name
        / "scores_bop19.json"
    )
    if not scores_path.is_file():
        raise FileNotFoundError(
            f"Official BOP score file was not produced: {scores_path}"
        )
    raw_scores = json.loads(
        scores_path.read_text(encoding="utf-8"),
        parse_constant=lambda _value: None,
    )
    scores = {
        str(name): float(value)
        for name, value in raw_scores.items()
        if isinstance(value, (int, float))
    }
    return OfficialBOP19EvaluationResult(
        result_csv_path=resolved_result_csv,
        targets_path=resolved_targets,
        dataset_view_root=Path(dataset_view_root).resolve(),
        scores_path=scores_path,
        scores=scores,
        command=command,
    )
