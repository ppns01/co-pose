from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track LINEMOD objects with SAM 3.1 and save "
            "compressed binary PNG masks."
        )
    )
    parser.add_argument(
        "--object-ids",
        type=int,
        nargs="+",
        default=[2],
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "sam3"
            / "weights"
            / "sam3.1_multiplex.pt"
        ),
    )
    parser.add_argument(
        "--sam3-repository",
        type=Path,
        default=PROJECT_ROOT / "sam3",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sam31_masks",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Override the prompt. This is only allowed with "
            "one object ID."
        ),
    )
    parser.add_argument(
        "--output-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Maximum frames per direction from the seed. "
            "Omit to track the whole sequence."
        ),
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--multiplex-count",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
    )
    return parser.parse_args(argv)


def _image_ids(rgb_directory: Path) -> list[int]:
    image_ids = sorted(
        int(path.stem)
        for path in rgb_directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            and path.stem.isdigit()
        )
    )
    if not image_ids:
        raise RuntimeError(
            f"No numbered RGB frames found: {rgb_directory}"
        )
    return image_ids


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _union_output_masks(
    outputs: dict[str, Any],
) -> tuple[np.ndarray | None, int]:
    binary_masks = outputs.get("out_binary_masks")
    if binary_masks is None:
        return None, 0

    masks = _to_numpy(binary_masks).astype(bool)
    if masks.size == 0:
        return None, 0
    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim != 3:
        raise ValueError(
            "Unexpected SAM 3.1 mask shape: "
            f"{tuple(masks.shape)}"
        )

    return np.logical_or.reduce(masks, axis=0), int(
        masks.shape[0]
    )


def _mask_iou(
    mask: np.ndarray,
    ground_truth_path: Path,
) -> float | None:
    if not ground_truth_path.is_file():
        return None
    ground_truth = (
        np.asarray(Image.open(ground_truth_path)) > 0
    )
    if ground_truth.shape != mask.shape:
        raise ValueError(
            "Mask shape mismatch: "
            f"prediction={mask.shape}, "
            f"ground_truth={ground_truth.shape}"
        )
    union = np.logical_or(mask, ground_truth).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(
        mask,
        ground_truth,
    ).sum()
    return float(intersection / union)


def _save_binary_png(
    mask: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L",
    ).save(
        output_path,
        format="PNG",
        optimize=True,
    )


def _write_summary(
    *,
    rows: list[dict[str, Any]],
    output_root: Path,
    metadata: dict[str, Any],
) -> None:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    csv_path = output_root / "summary.csv"
    json_path = output_root / "summary.json"

    fieldnames = [
        "object_id",
        "object_name",
        "prompt",
        "seed_image_id",
        "frame_index",
        "image_id",
        "status",
        "num_objects",
        "mask_pixels",
        "iou",
        "mask_path",
        "error",
    ]
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        **metadata,
        "passed": sum(
            row["status"] == "PASS"
            for row in rows
        ),
        "failed": sum(
            row["status"] != "PASS"
            for row in rows
        ),
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[SUMMARY CSV]  {csv_path}")
    print(f"[SUMMARY JSON] {json_path}")


def _frame_row(
    *,
    outputs: dict[str, Any],
    object_id: int,
    object_name: str,
    prompt: str,
    seed_image_id: int,
    frame_index: int,
    image_id: int,
    instance_index: int,
    object_directory: Path,
    mask_visib_directory: Path,
) -> dict[str, Any]:
    mask, num_objects = _union_output_masks(outputs)
    base = {
        "object_id": object_id,
        "object_name": object_name,
        "prompt": prompt,
        "seed_image_id": seed_image_id,
        "frame_index": frame_index,
        "image_id": image_id,
    }
    if mask is None or not mask.any():
        return {
            **base,
            "status": "FAIL",
            "num_objects": num_objects,
            "mask_pixels": 0,
            "iou": None,
            "mask_path": None,
            "error": "SAM 3.1 returned no mask.",
        }

    mask_path = (
        object_directory
        / f"{image_id:06d}.png"
    )
    _save_binary_png(mask, mask_path)
    ground_truth_path = (
        mask_visib_directory
        / (
            f"{image_id:06d}_"
            f"{instance_index:06d}.png"
        )
    )
    return {
        **base,
        "status": "PASS",
        "num_objects": num_objects,
        "mask_pixels": int(mask.sum()),
        "iou": _mask_iou(
            mask,
            ground_truth_path,
        ),
        "mask_path": str(mask_path),
        "error": None,
    }


def _track_object(
    *,
    predictor: Any,
    object_id: int,
    object_name: str,
    prompt: str,
    seed_image_id: int,
    image_ids: list[int],
    rgb_directory: Path,
    mask_visib_directory: Path,
    instance_index: int,
    output_root: Path,
    output_threshold: float,
    max_frames: int | None,
    rows_sink: list[dict[str, Any]],
) -> None:
    seed_frame_index = image_ids.index(
        seed_image_id
    )
    object_directory = (
        output_root
        / "masks"
        / f"object_{object_id:02d}_{object_name}"
    )
    session_id: str | None = None

    try:
        session = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(rgb_directory),
                "offload_video_to_cpu": True,
            }
        )
        session_id = session["session_id"]
        seed_response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": seed_frame_index,
                "text": prompt,
                "output_prob_thresh": (
                    output_threshold
                ),
            }
        )
        seed_outputs = seed_response.get(
            "outputs",
            {},
        )
        seed_mask, _ = _union_output_masks(
            seed_outputs
        )
        if seed_mask is None or not seed_mask.any():
            raise RuntimeError(
                "SAM 3.1 could not detect the object "
                f"on seed frame {seed_image_id:06d} "
                f"with prompt {prompt!r}."
            )

        request: dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both",
            "start_frame_index": seed_frame_index,
            "output_prob_thresh": output_threshold,
        }
        if max_frames is not None:
            request["max_frame_num_to_track"] = (
                max_frames
            )

        for response in (
            predictor.handle_stream_request(request)
        ):
            frame_index = int(
                response["frame_index"]
            )
            image_id = image_ids[frame_index]
            row = _frame_row(
                outputs=response.get("outputs", {}),
                object_id=object_id,
                object_name=object_name,
                prompt=prompt,
                seed_image_id=seed_image_id,
                frame_index=frame_index,
                image_id=image_id,
                instance_index=instance_index,
                object_directory=object_directory,
                mask_visib_directory=(
                    mask_visib_directory
                ),
            )
            rows_sink.append(row)
            iou = row["iou"]
            iou_text = (
                "n/a"
                if iou is None
                else f"{iou:.4f}"
            )
            print(
                f"[{row['status']}] "
                f"object={object_id:02d} "
                f"image={image_id:06d} "
                f"objects={row['num_objects']} "
                f"IoU={iou_text}"
            )
    finally:
        if session_id is not None:
            predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                }
            )

def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = _parse_args(argv)
    if (
        args.prompt is not None
        and len(args.object_ids) != 1
    ):
        raise ValueError(
            "--prompt requires exactly one object ID."
        )
    if (
        args.max_frames is not None
        and args.max_frames <= 0
    ):
        raise ValueError(
            "--max-frames must be positive."
        )

    from main import LINEMOD_OBJECT_METADATA
    from sam3_prompt_sweep import _selected_frames

    plans: list[dict[str, Any]] = []
    for object_id in args.object_ids:
        if object_id not in LINEMOD_OBJECT_METADATA:
            raise ValueError(
                f"Unknown LINEMOD object ID: {object_id}"
            )
        object_name, default_prompt = (
            LINEMOD_OBJECT_METADATA[object_id]
        )
        prompt = (
            args.prompt
            if args.prompt is not None
            else default_prompt
        )
        object_root = (
            args.dataset_root
            / args.split
            / f"{object_id:06d}"
        )
        rgb_directory = object_root / "rgb"
        image_ids = _image_ids(rgb_directory)
        selected_frames = dict(
            _selected_frames(
                dataset_root=args.dataset_root,
                split=args.split,
                object_id=object_id,
                instance_index=(
                    args.instance_index
                ),
            )
        )
        seed_image_id = int(
            selected_frames["best"]
        )
        plans.append(
            {
                "object_id": object_id,
                "object_name": object_name,
                "prompt": prompt,
                "object_root": object_root,
                "rgb_directory": rgb_directory,
                "mask_visib_directory": (
                    object_root / "mask_visib"
                ),
                "image_ids": image_ids,
                "seed_image_id": seed_image_id,
            }
        )
        print(
            f"[PLAN] object={object_id:02d} "
            f"{object_name} "
            f"frames={len(image_ids)} "
            f"seed={seed_image_id:06d} "
            f"prompt={prompt!r}"
        )

    print("[MODEL] SAM 3.1 Object Multiplex")
    print(f"[CHECKPOINT] {args.checkpoint}")
    print("[OUTPUT] compressed binary PNG masks only")
    if args.list_only:
        return 0

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            "SAM 3.1 checkpoint not found: "
            f"{args.checkpoint}"
        )
    if not args.sam3_repository.is_dir():
        raise FileNotFoundError(
            "SAM3 repository not found: "
            f"{args.sam3_repository}"
        )
    sys.path.insert(
        0,
        str(args.sam3_repository),
    )

    from sam3 import build_sam3_predictor

    started_at = time.perf_counter()
    predictor = build_sam3_predictor(
        checkpoint_path=str(args.checkpoint),
        bpe_path=str(
            args.sam3_repository
            / "sam3"
            / "assets"
            / "bpe_simple_vocab_16e6.txt.gz"
        ),
        version="sam3.1",
        compile=False,
        max_num_objects=args.max_objects,
        multiplex_count=args.multiplex_count,
        use_fa3=False,
        async_loading_frames=True,
        default_output_prob_thresh=(
            args.output_threshold
        ),
    )

    all_rows: list[dict[str, Any]] = []
    interrupted = False
    try:
        for plan in plans:
            _track_object(
                predictor=predictor,
                object_id=plan["object_id"],
                object_name=plan["object_name"],
                prompt=plan["prompt"],
                seed_image_id=(
                    plan["seed_image_id"]
                ),
                image_ids=plan["image_ids"],
                rgb_directory=(
                    plan["rgb_directory"]
                ),
                mask_visib_directory=(
                    plan[
                        "mask_visib_directory"
                    ]
                ),
                instance_index=(
                    args.instance_index
                ),
                output_root=args.output_root,
                output_threshold=(
                    args.output_threshold
                ),
                max_frames=args.max_frames,
                rows_sink=all_rows,
            )
    except KeyboardInterrupt:
        interrupted = True
        print(
            "\n[INTERRUPTED] Saving completed "
            "mask records."
        )
    finally:
        _write_summary(
            rows=all_rows,
            output_root=args.output_root,
            metadata={
                "model_version": "sam3.1",
                "checkpoint": str(args.checkpoint),
                "dataset_root": str(
                    args.dataset_root
                ),
                "split": args.split,
                "instance_index": (
                    args.instance_index
                ),
                "output_threshold": (
                    args.output_threshold
                ),
                "max_frames": args.max_frames,
                "elapsed_sec": (
                    time.perf_counter()
                    - started_at
                ),
                "interrupted": interrupted,
            },
        )

    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
