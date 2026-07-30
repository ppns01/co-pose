from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent

PROMPT_CANDIDATES: dict[int, tuple[str, ...]] = {
    1: (
        "brown figurine",
        "brown toy",
        "brown statue",
        "animal figurine",
        "toy statue",
        "small brown sculpture",
    ),
    2: (
        "a metallic bench vise tool with a screw handle",
        "metal bench vise",
        "blue metal vise",
        "metal clamp with a screw handle",
        "blue metal clamping tool",
        "table vise tool",
    ),
    10: (
        "container",
        "plastic container",
        "case",
        "plastic case",
        "tray",
        "plastic tray",
        "carton",
        "package",
        "lunchbox",
        "takeout box",
        "food box",
        "styrofoam container",
    ),
    15: (
        "phone handset",
        "charging base",
        "phone base",
        "telephone base",
        "charging dock",
        "phone dock",
    ),
}

OBJECT2_PROMPT_SWEEP_100: tuple[str, ...] = (
    "bench vise",
    "bench vice",
    "machine vise",
    "machine vice",
    "metal vise",
    "metal vice",
    "blue vise",
    "blue vice",
    "table vise",
    "workbench vise",
    "metalworking vise",
    "workshop vise",
    "mechanic vise",
    "engineer's vise",
    "rotary vise",
    "swivel vise",
    "clamp vise",
    "jaw vise",
    "screw vise",
    "tool vise",
    "blue metal tool",
    "blue mechanical tool",
    "blue machine tool",
    "blue workshop tool",
    "blue clamping tool",
    "blue metal clamp",
    "metal clamping device",
    "blue mechanical device",
    "blue metal device",
    "gray and blue metal device",
    "blue industrial tool",
    "metalwork tool",
    "mechanical fixture",
    "metal fixture",
    "blue metal fixture",
    "work holding tool",
    "workholding tool",
    "clamping machine",
    "small metal machine",
    "compact metal machine",
    "round blue metal tool",
    "circular blue metal tool",
    "blue circular machine",
    "blue metal apparatus",
    "mechanical apparatus",
    "blue base with a silver handle",
    "blue base with a vertical silver handle",
    "blue circular base with a metal handle",
    "round blue base with a silver bar",
    "blue machine with a vertical handle",
    "blue metal machine with a vertical handle",
    "blue metal device with a silver handle",
    "blue tool with a screw handle",
    "metal tool with a vertical silver bar",
    "mechanical tool with a vertical silver bar",
    "blue object with a vertical metal rod",
    "blue object with a silver rod",
    "blue round object with a metal rod",
    "circular metal base with a vertical rod",
    "blue metal base with a vertical rod",
    "round metal fixture with a handle",
    "blue clamp with a vertical handle",
    "metal clamp with a silver handle",
    "blue tool with a vertical screw",
    "blue machine with a screw handle",
    "metal device with a long handle",
    "blue device with a T-shaped handle",
    "blue tool with a T-shaped handle",
    "round blue tool with a vertical handle",
    "circular blue device with a metal handle",
    "blue metal object in the center",
    "blue mechanical object in the center",
    "blue tool in the center",
    "metal tool in the center",
    "blue machine in the center",
    "metal machine in the center",
    "round blue object in the center",
    "circular blue object in the center",
    "blue metal device in the center",
    "blue circular tool in the center",
    "object with vertical silver handle in the center",
    "metal object with vertical rod in the center",
    "blue base with silver handle in the center",
    "blue and gray object in the center",
    "central blue metal tool",
    "central metal device",
    "central blue mechanical device",
    "blue workshop tool at the center",
    "blue circular base at the center",
    "metal fixture at the center",
    "object between the markers",
    "blue object between the markers",
    "metal tool between the black and white markers",
    "blue machine on the marker board",
    "metal tool on the marker board",
    "blue object on the board",
    "round blue object on the board",
    (
        "blue metal machine with a vertical silver "
        "handle in the center"
    ),
    (
        "blue circular metal object with a silver "
        "rod in the center"
    ),
    "small blue metal machine in the middle",
)

PROMPT_SWEEP_CONCEPTS_20: dict[
    int,
    tuple[str, ...],
] = {
    1: (
        "brown toy",
        "brown figurine",
        "brown ape figurine",
        "ape toy",
        "monkey toy",
        "gorilla toy",
        "small brown sculpture",
        "brown animal figurine",
        "seated ape figurine",
        "sitting monkey toy",
        "toy primate",
        "brown statue",
        "small ape statue",
        "brown monkey figure",
        "brown animal toy",
        "primate figurine",
        "small brown toy animal",
        "seated brown toy",
        "toy gorilla",
        "brown plastic figurine",
    ),
    3: (
        "white bowl",
        "ceramic bowl",
        "white ceramic bowl",
        "round bowl",
        "small white bowl",
        "shallow bowl",
        "dish bowl",
        "round white dish",
        "white container bowl",
        "empty white bowl",
        "white kitchen bowl",
        "curved white bowl",
        "circular white bowl",
        "tabletop bowl",
        "white serving bowl",
        "small ceramic dish",
        "white concave object",
        "bowl-shaped object",
        "white round container",
        "smooth white bowl",
    ),
    4: (
        "camera",
        "black camera",
        "digital camera",
        "DSLR camera",
        "black DSLR camera",
        "photo camera",
        "camera body",
        "black camera body",
        "camera with lens",
        "black device with lens",
        "photographic camera",
        "compact camera",
        "box camera",
        "classic camera",
        "camera device",
        "dark camera",
        "black optical device",
        "handheld camera",
        "square camera",
        "lens camera",
    ),
    5: (
        "white watering can",
        "watering can",
        "plastic watering can",
        "white plastic watering can",
        "water jug",
        "white water jug",
        "watering pot",
        "white pouring container",
        "plastic water container",
        "white handled container",
        "container with spout",
        "white container with handle",
        "white jug with spout",
        "garden watering can",
        "small watering can",
        "white plastic jug",
        "handled water container",
        "white can",
        "plastic can",
        "pouring can",
    ),
    6: (
        "pink cat figurine",
        "cat figurine",
        "pink toy cat",
        "toy cat",
        "cat toy",
        "pink animal figurine",
        "small pink cat",
        "seated cat figurine",
        "pink cat statue",
        "cat sculpture",
        "kitten figurine",
        "pink kitten toy",
        "small cat statue",
        "plastic cat figurine",
        "animal toy",
        "pink animal toy",
        "sitting cat toy",
        "toy kitten",
        "cat-shaped figure",
        "pink plastic cat",
    ),
    7: (
        "cup",
        "blue cup",
        "drinking cup",
        "blue drinking cup",
        "plastic cup",
        "blue plastic cup",
        "mug",
        "blue mug",
        "drinking vessel",
        "small blue cup",
        "cylindrical cup",
        "open cup",
        "cup-shaped container",
        "blue container",
        "table cup",
        "beverage cup",
        "blue drinking vessel",
        "handled cup",
        "round blue cup",
        "plastic drinking vessel",
    ),
    8: (
        "power drill",
        "electric drill",
        "cordless drill",
        "handheld drill",
        "green power drill",
        "green electric drill",
        "drill tool",
        "drilling machine",
        "handheld power tool",
        "green handheld tool",
        "electric power tool",
        "cordless power tool",
        "drill with handle",
        "green drill",
        "black and green drill",
        "construction drill",
        "pistol-shaped drill",
        "drill machine",
        "power drilling tool",
        "mechanical drill",
    ),
    9: (
        "rubber duck",
        "yellow rubber duck",
        "toy duck",
        "yellow toy duck",
        "small yellow duck",
        "duck figurine",
        "bath duck",
        "bath toy duck",
        "plastic duck",
        "yellow duck toy",
        "duck-shaped toy",
        "small rubber duck",
        "yellow animal toy",
        "duck statue",
        "toy bird",
        "yellow bird figurine",
        "floating duck toy",
        "classic rubber duck",
        "small yellow bird toy",
        "rubber duck figurine",
    ),
    10: (
        "plastic case",
        "egg box",
        "eggbox",
        "egg carton",
        "white egg carton",
        "foam egg carton",
        "white plastic case",
        "white container",
        "plastic container",
        "white plastic box",
        "white ribbed box",
        "white segmented container",
        "white clamshell container",
        "closed white case",
        "white food container",
        "white tray",
        "plastic tray",
        "white package",
        "molded plastic case",
        "rectangular plastic case",
    ),
    11: (
        "glue bottle",
        "white glue bottle",
        "small glue bottle",
        "plastic glue bottle",
        "adhesive bottle",
        "white adhesive bottle",
        "bottle with nozzle",
        "glue container",
        "white bottle with nozzle",
        "small plastic bottle",
        "squeeze bottle",
        "white squeeze bottle",
        "craft glue bottle",
        "liquid glue bottle",
        "adhesive container",
        "nozzle bottle",
        "small white bottle",
        "glue dispenser",
        "white plastic container with nozzle",
        "bottle of glue",
    ),
    12: (
        "hole puncher",
        "hole punch",
        "paper hole punch",
        "blue hole punch",
        "blue paper hole punch",
        "desktop hole punch",
        "office hole punch",
        "metal hole punch",
        "punch machine",
        "paper punch machine",
        "blue office tool",
        "desktop puncher",
        "blue punch device",
        "metal and plastic hole punch",
        "hole punching tool",
        "paper punching device",
        "blue desktop tool",
        "office punching machine",
        "compact hole punch",
        "blue mechanical office tool",
    ),
    13: (
        "clothes iron",
        "clothing iron",
        "electric iron",
        "household iron",
        "blue clothes iron",
        "blue electric iron",
        "steam iron",
        "garment iron",
        "ironing tool",
        "household appliance iron",
        "blue household appliance",
        "clothes smoothing iron",
        "iron appliance",
        "electric clothes iron",
        "handheld iron",
        "blue ironing device",
        "laundry iron",
        "fabric iron",
        "triangular iron",
        "blue steam iron",
    ),
    14: (
        "desk lamp",
        "white desk lamp",
        "table lamp",
        "white table lamp",
        "reading lamp",
        "white reading lamp",
        "small desk lamp",
        "adjustable lamp",
        "task lamp",
        "white task lamp",
        "lamp with shade",
        "desktop light",
        "table light",
        "white lighting fixture",
        "articulated desk lamp",
        "small table light",
        "office desk lamp",
        "white lamp",
        "electric lamp",
        "desk reading light",
    ),
    15: (
        "phone handset",
        "cordless phone",
        "telephone",
        "gray telephone",
        "phone with base",
        "cordless phone with charging base",
        "charging base",
        "phone base",
        "telephone base",
        "phone dock",
        "charging dock",
        "handset and base",
        "gray phone handset",
        "wireless phone",
        "home telephone",
        "desk telephone",
        "landline phone",
        "phone device",
        "communication handset",
        "telephone set",
    ),
}

PROMPT_SWEEP_TEMPLATES_5: tuple[str, ...] = (
    "{concept}",
    "a {concept}",
    "{concept} in the center",
    "the {concept} on the marker board",
    "single {concept}",
)


def _expand_prompt_concepts(
    concepts: Sequence[str],
) -> tuple[str, ...]:
    prompts = tuple(
        template.format(concept=concept)
        for template in PROMPT_SWEEP_TEMPLATES_5
        for concept in concepts
    )
    if len(prompts) != 100 or len(set(prompts)) != 100:
        raise ValueError(
            "Each object prompt bank must contain exactly "
            "100 unique prompts."
        )
    return prompts


ALL_OBJECT_PROMPT_SWEEP_100: dict[
    int,
    tuple[str, ...],
] = {
    object_id: _expand_prompt_concepts(concepts)
    for object_id, concepts
    in PROMPT_SWEEP_CONCEPTS_20.items()
}
ALL_OBJECT_PROMPT_SWEEP_100[2] = (
    OBJECT2_PROMPT_SWEEP_100
)

OBJECT2_PRIMARY_PROMPT_COUNT = 2
OBJECT2_MIN_MASK_AREA_RATIO = 0.005
OBJECT2_MAX_MASK_AREA_RATIO = 0.12
OBJECT2_MAX_CENTER_DISTANCE_NORM = 0.50
OBJECT2_MIN_VALID_DEPTH_RATIO = 0.50


def _slug(text: str) -> str:
    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        text.lower(),
    ).strip("_")
    return slug or "prompt"


def _selected_frames(
    *,
    dataset_root: Path,
    split: str,
    object_id: int,
    instance_index: int,
) -> tuple[tuple[str, int], ...]:
    info_path = (
        dataset_root
        / split
        / f"{object_id:06d}"
        / "scene_gt_info.json"
    )
    if not info_path.is_file():
        raise FileNotFoundError(
            f"scene_gt_info.json not found: {info_path}"
        )

    info_by_image = json.loads(
        info_path.read_text(encoding="utf-8")
    )
    candidates: list[tuple[int, dict[str, Any]]] = []

    for image_key, instances in info_by_image.items():
        if instance_index >= len(instances):
            continue

        candidates.append(
            (
                int(image_key),
                instances[instance_index],
            )
        )

    if not candidates:
        raise ValueError(
            "No valid LINEMOD frames for "
            f"object={object_id}, instance={instance_index}."
        )

    image_ids = {
        image_id
        for image_id, _ in candidates
    }
    first_image_id = (
        0
        if 0 in image_ids
        else min(image_ids)
    )

    def best_key(
        item: tuple[int, dict[str, Any]],
    ) -> tuple[
        float,
        float,
        float,
        float,
        int,
    ]:
        image_id, info = item
        bbox = info.get("bbox_visib", [0, 0, 0, 0])
        bbox_area = float(bbox[2]) * float(bbox[3])
        visible_fraction = float(
            info.get("visib_fract", 0.0)
        )
        return (
            bbox_area * visible_fraction,
            bbox_area,
            float(info.get("px_count_visib", 0.0)),
            visible_fraction,
            -image_id,
        )

    best_image_id = max(
        candidates,
        key=best_key,
    )[0]
    frames = [
        ("first", first_image_id),
        ("best", best_image_id),
    ]

    return tuple(dict.fromkeys(frames))


def _all_frames(
    *,
    dataset_root: Path,
    split: str,
    object_id: int,
    instance_index: int,
) -> tuple[tuple[str, int], ...]:
    info_path = (
        dataset_root
        / split
        / f"{object_id:06d}"
        / "scene_gt_info.json"
    )
    if not info_path.is_file():
        raise FileNotFoundError(
            f"scene_gt_info.json not found: {info_path}"
        )

    info_by_image = json.loads(
        info_path.read_text(encoding="utf-8")
    )
    image_ids = sorted(
        int(image_key)
        for image_key, instances
        in info_by_image.items()
        if instance_index < len(instances)
    )
    if not image_ids:
        raise ValueError(
            "No valid LINEMOD frames for "
            f"object={object_id}, instance={instance_index}."
        )

    return tuple(
        ("all", image_id)
        for image_id in image_ids
    )


def _random_frames(
    *,
    dataset_root: Path,
    split: str,
    object_id: int,
    instance_index: int,
    count: int,
    seed: int,
) -> tuple[tuple[str, int], ...]:
    available_frames = _all_frames(
        dataset_root=dataset_root,
        split=split,
        object_id=object_id,
        instance_index=instance_index,
    )
    image_ids = [
        image_id
        for _, image_id in available_frames
    ]
    sample_count = min(count, len(image_ids))
    object_rng = random.Random(
        seed + object_id * 1_000_003
    )
    sampled_image_ids = sorted(
        object_rng.sample(
            image_ids,
            sample_count,
        )
    )
    return tuple(
        ("random", image_id)
        for image_id in sampled_image_ids
    )


def _load_gt_mask(
    *,
    dataset_root: Path,
    split: str,
    object_id: int,
    image_id: int,
    instance_index: int,
) -> np.ndarray:
    mask_path = (
        dataset_root
        / split
        / f"{object_id:06d}"
        / "mask_visib"
        / (
            f"{image_id:06d}_"
            f"{instance_index:06d}.png"
        )
    )
    if not mask_path.is_file():
        raise FileNotFoundError(
            f"GT visible mask not found: {mask_path}"
        )

    return np.ascontiguousarray(
        np.asarray(
            Image.open(mask_path).convert("L")
        )
        > 0,
        dtype=np.bool_,
    )


def _mask_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float | int]:
    prediction = np.asarray(
        prediction,
        dtype=np.bool_,
    )
    ground_truth = np.asarray(
        ground_truth,
        dtype=np.bool_,
    )
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            "Prediction and GT mask shapes differ: "
            f"{prediction.shape} vs {ground_truth.shape}"
        )

    intersection = int(
        np.count_nonzero(
            prediction & ground_truth
        )
    )
    union = int(
        np.count_nonzero(
            prediction | ground_truth
        )
    )
    prediction_pixels = int(
        np.count_nonzero(prediction)
    )
    ground_truth_pixels = int(
        np.count_nonzero(ground_truth)
    )

    return {
        "iou": (
            intersection / union
            if union > 0
            else 1.0
        ),
        "precision": (
            intersection / prediction_pixels
            if prediction_pixels > 0
            else 0.0
        ),
        "recall": (
            intersection / ground_truth_pixels
            if ground_truth_pixels > 0
            else 0.0
        ),
        "prediction_pixels": prediction_pixels,
        "ground_truth_pixels": ground_truth_pixels,
    }


def _mask_candidate_geometry(
    *,
    mask: np.ndarray,
    depth_m: np.ndarray,
) -> dict[str, float | int]:
    mask = np.asarray(mask, dtype=np.bool_)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if mask.ndim != 2 or mask.shape != depth_m.shape:
        raise ValueError(
            "Mask/depth shapes must match: "
            f"mask={mask.shape}, depth={depth_m.shape}"
        )

    pixel_count = int(np.count_nonzero(mask))
    if pixel_count == 0:
        return {
            "mask_pixels": 0,
            "area_ratio": 0.0,
            "center_distance_norm": float("inf"),
            "valid_depth_ratio": 0.0,
        }

    height, width = mask.shape
    y_coordinates, x_coordinates = np.nonzero(mask)
    image_center_x = (width - 1) * 0.5
    image_center_y = (height - 1) * 0.5
    half_diagonal = float(
        np.hypot(width * 0.5, height * 0.5)
    )
    center_distance_norm = float(
        np.hypot(
            float(np.mean(x_coordinates)) - image_center_x,
            float(np.mean(y_coordinates)) - image_center_y,
        )
        / half_diagonal
    )
    candidate_depth = depth_m[mask]
    valid_depth = (
        np.isfinite(candidate_depth)
        & (candidate_depth > 0.0)
    )

    return {
        "mask_pixels": pixel_count,
        "area_ratio": float(pixel_count / mask.size),
        "center_distance_norm": center_distance_norm,
        "valid_depth_ratio": float(
            np.count_nonzero(valid_depth)
            / pixel_count
        ),
    }


def _object2_candidate_rejection_reasons(
    geometry: dict[str, float | int],
) -> list[str]:
    reasons: list[str] = []
    area_ratio = float(geometry["area_ratio"])
    center_distance = float(
        geometry["center_distance_norm"]
    )
    valid_depth_ratio = float(
        geometry["valid_depth_ratio"]
    )

    if area_ratio < OBJECT2_MIN_MASK_AREA_RATIO:
        reasons.append("mask_too_small")
    if area_ratio > OBJECT2_MAX_MASK_AREA_RATIO:
        reasons.append("mask_too_large")
    if (
        center_distance
        > OBJECT2_MAX_CENTER_DISTANCE_NORM
    ):
        reasons.append("mask_too_far_from_center")
    if (
        valid_depth_ratio
        < OBJECT2_MIN_VALID_DEPTH_RATIO
    ):
        reasons.append("insufficient_valid_depth")
    return reasons


def _select_object2_candidate(
    *,
    candidates: Sequence[dict[str, Any]],
    depth_m: np.ndarray,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        geometry = _mask_candidate_geometry(
            mask=np.asarray(candidate["mask"]),
            depth_m=depth_m,
        )
        reasons = _object2_candidate_rejection_reasons(
            geometry
        )
        evaluated.append(
            {
                **candidate,
                **geometry,
                "rejection_reasons": reasons,
            }
        )

    valid = [
        candidate
        for candidate in evaluated
        if not candidate["rejection_reasons"]
    ]
    if not valid:
        return None, evaluated

    def selection_key(
        candidate: dict[str, Any],
    ) -> tuple[int, float, float, int]:
        prompt_index = int(candidate["prompt_index"])
        score = candidate.get("score")
        return (
            (
                1
                if prompt_index
                < OBJECT2_PRIMARY_PROMPT_COUNT
                else 0
            ),
            (
                float(score)
                if score is not None
                else float("-inf")
            ),
            -float(candidate["center_distance_norm"]),
            -prompt_index,
        )

    return max(valid, key=selection_key), evaluated


def _save_union_artifacts(
    *,
    image_rgb: np.ndarray,
    union_mask: np.ndarray,
    output_directory: Path,
    metadata: dict[str, Any],
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    union_mask = np.ascontiguousarray(
        union_mask,
        dtype=np.bool_,
    )
    mask_rgb = np.repeat(
        (
            union_mask.astype(np.uint8)
            * 255
        )[:, :, None],
        3,
        axis=2,
    )
    overlay = np.asarray(
        image_rgb,
        dtype=np.uint8,
    ).copy()
    source_pixels = overlay[
        union_mask
    ].astype(np.float32)
    overlay[union_mask] = np.clip(
        source_pixels * 0.55
        + np.array(
            [0, 255, 0],
            dtype=np.float32,
        )
        * 0.45,
        0.0,
        255.0,
    ).astype(np.uint8)

    np.save(
        output_directory / "mask_bool.npy",
        union_mask,
    )
    Image.fromarray(mask_rgb).save(
        output_directory / "mask_rgb.png"
    )
    Image.fromarray(overlay).save(
        output_directory / "overlay.png"
    )
    (
        output_directory / "meta.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _build_cached_masker(
    *,
    view: Any,
    repository_path: Path,
    checkpoint_path: Path,
    bpe_path: Path,
    device: str,
    use_amp: bool,
    confidence_threshold: float,
) -> tuple[Any, Any]:
    import torch

    from mask_provider import (
        _get_sam3_processor,
        _to_numpy,
    )
    from segmentation.sam3_masker import SAM3Masker

    resolved_device = torch.device(device)
    processor = _get_sam3_processor(
        repository_path=repository_path,
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=str(resolved_device),
        confidence_threshold=confidence_threshold,
    )
    image = Image.fromarray(
        np.asarray(view.rgb, dtype=np.uint8),
        mode="RGB",
    )

    with torch.inference_mode(), torch.autocast(
        device_type=resolved_device.type,
        dtype=torch.bfloat16,
        enabled=(
            use_amp
            and resolved_device.type == "cuda"
        ),
    ):
        inference_state = processor.set_image(image)

    def predict_cached(
        _image_rgb: np.ndarray,
        text_prompt: str,
    ) -> dict[str, np.ndarray]:
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.bfloat16,
                enabled=(
                    use_amp
                    and resolved_device.type == "cuda"
                ),
            ),
        ):
            output = processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt,
            )

        return {
            "masks": _to_numpy(output["masks"]),
            "scores": _to_numpy(
                output["scores"].to(
                    dtype=torch.float32
                )
            ),
        }

    return (
        SAM3Masker(predict_cached),
        predict_cached,
    )


def _build_sam31_predictor(
    *,
    repository_path: Path,
    checkpoint_path: Path,
    bpe_path: Path,
    confidence_threshold: float,
) -> Any:
    if not repository_path.is_dir():
        raise FileNotFoundError(
            f"SAM3 repository not found: {repository_path}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "SAM 3.1 checkpoint not found: "
            f"{checkpoint_path}"
        )
    if not bpe_path.is_file():
        raise FileNotFoundError(
            f"SAM3 BPE file not found: {bpe_path}"
        )

    repository_text = str(repository_path)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)

    from sam3 import build_sam3_predictor

    return build_sam3_predictor(
        checkpoint_path=str(checkpoint_path),
        bpe_path=str(bpe_path),
        version="sam3.1",
        compile=False,
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        async_loading_frames=False,
        default_output_prob_thresh=(
            confidence_threshold
        ),
    )


def _predict_sam31_single_frame(
    *,
    predictor: Any,
    image_rgb: np.ndarray,
    text_prompt: str,
    confidence_threshold: float,
    run_gc_collect: bool = True,
) -> tuple[np.ndarray, float | None]:
    session_id: str | None = None
    image = Image.fromarray(
        np.asarray(image_rgb, dtype=np.uint8),
        mode="RGB",
    )
    try:
        session = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": [image],
                "offload_video_to_cpu": False,
            }
        )
        session_id = session["session_id"]
        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": text_prompt,
                "output_prob_thresh": (
                    confidence_threshold
                ),
            }
        )
        outputs = response.get("outputs", {})
        masks = np.asarray(
            outputs.get("out_binary_masks", []),
            dtype=bool,
        )
        scores = np.asarray(
            outputs.get("out_probs", []),
            dtype=np.float32,
        ).reshape(-1)
        if masks.size == 0:
            raise ValueError(
                "SAM 3.1 returned no mask candidates."
            )
        if masks.ndim == 2:
            masks = masks[None, ...]
        elif (
            masks.ndim == 4
            and masks.shape[1] == 1
        ):
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError(
                "Unexpected SAM 3.1 mask shape: "
                f"{tuple(masks.shape)}"
            )
        if len(scores) != len(masks):
            raise ValueError(
                "SAM 3.1 mask/score count mismatch: "
                f"{len(masks)} masks, {len(scores)} scores"
            )
        selected_index = int(np.argmax(scores))
        return (
            masks[selected_index],
            float(scores[selected_index]),
        )
    finally:
        if session_id is not None:
            predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                    "run_gc_collect": run_gc_collect,
                }
            )


def _save_binary_mask_png(
    *,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Image.fromarray(
        np.asarray(mask, dtype=np.uint8) * 255,
        mode="L",
    ).save(
        output_path,
        format="PNG",
        optimize=True,
    )


def _predict_mask_in_memory(
    *,
    predict_function: Any,
    image_rgb: np.ndarray,
    text_prompt: str,
) -> tuple[np.ndarray, float | None]:
    from segmentation.sam3_masker import (
        _normalize_masks,
        _normalize_raw_output,
        _normalize_scores,
        _select_best_mask,
    )

    raw_output = predict_function(
        np.asarray(image_rgb, dtype=np.uint8),
        text_prompt,
    )
    raw_masks, raw_scores = (
        _normalize_raw_output(raw_output)
    )
    masks = _normalize_masks(
        raw_masks,
        expected_height=image_rgb.shape[0],
        expected_width=image_rgb.shape[1],
    )
    scores = _normalize_scores(
        raw_scores,
        mask_count=len(masks),
    )
    _, mask_bool, selected_score = (
        _select_best_mask(masks, scores)
    )
    return mask_bool, selected_score


def _write_summary(
    *,
    rows: list[dict[str, Any]],
    output_root: Path,
) -> tuple[Path, Path]:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    json_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "object_id",
        "object_name",
        "frame_type",
        "image_id",
        "prompt_index",
        "prompt",
        "status",
        "score",
        "selected_prompt",
        "selection_method",
        "candidate_count",
        "area_ratio",
        "center_distance_norm",
        "valid_depth_ratio",
        "iou",
        "precision",
        "recall",
        "prediction_pixels",
        "ground_truth_pixels",
        "elapsed_sec",
        "output_directory",
        "error_type",
        "error",
    ]
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            {
                field: row.get(field)
                for field in fieldnames
            }
            for row in rows
        )

    return json_path, csv_path


def _build_prompt_aggregate(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[int, str, int, str],
        list[dict[str, Any]],
    ] = {}
    for row in rows:
        prompt_index = int(
            row.get("prompt_index", -1)
        )
        if prompt_index < 0:
            continue
        key = (
            int(row["object_id"]),
            str(row["object_name"]),
            prompt_index,
            str(row["prompt"]),
        )
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (
        object_id,
        object_name,
        prompt_index,
        prompt,
    ), prompt_rows in grouped.items():
        passed_rows = [
            row
            for row in prompt_rows
            if row.get("status") == "PASS"
        ]
        all_ious = [
            (
                float(row.get("iou", 0.0))
                if row.get("status") == "PASS"
                else 0.0
            )
            for row in prompt_rows
        ]
        detected_ious = [
            float(row["iou"])
            for row in passed_rows
        ]
        detected_scores = [
            float(row["score"])
            for row in passed_rows
            if row.get("score") is not None
        ]
        image_count = len(prompt_rows)
        pass_count = len(passed_rows)
        aggregates.append(
            {
                "object_id": object_id,
                "object_name": object_name,
                "prompt_index": prompt_index,
                "prompt": prompt,
                "image_count": image_count,
                "pass_count": pass_count,
                "fail_count": (
                    image_count - pass_count
                ),
                "detection_rate": (
                    pass_count / image_count
                ),
                "mean_iou": float(
                    np.mean(all_ious)
                ),
                "median_iou": float(
                    np.median(all_ious)
                ),
                "mean_iou_detected": (
                    float(np.mean(detected_ious))
                    if detected_ious
                    else None
                ),
                "best_iou": (
                    max(detected_ious)
                    if detected_ious
                    else 0.0
                ),
                "mean_score_detected": (
                    float(
                        np.mean(detected_scores)
                    )
                    if detected_scores
                    else None
                ),
            }
        )

    ranked_aggregates: list[dict[str, Any]] = []
    object_ids = sorted(
        {
            int(row["object_id"])
            for row in aggregates
        }
    )
    for object_id in object_ids:
        object_rows = sorted(
            (
                row
                for row in aggregates
                if int(row["object_id"])
                == object_id
            ),
            key=lambda row: (
                float(row["mean_iou"]),
                float(row["detection_rate"]),
                float(row["median_iou"]),
                float(
                    row["mean_score_detected"]
                    if row["mean_score_detected"]
                    is not None
                    else -1.0
                ),
            ),
            reverse=True,
        )
        for rank, row in enumerate(
            object_rows,
            start=1,
        ):
            ranked_aggregates.append(
                {
                    "rank": rank,
                    **row,
                }
            )
    return ranked_aggregates


def _write_prompt_aggregate(
    *,
    rows: Sequence[dict[str, Any]],
    output_root: Path,
) -> tuple[
    list[dict[str, Any]],
    Path,
    Path,
]:
    aggregate_rows = _build_prompt_aggregate(rows)
    json_path = output_root / "prompt_aggregate.json"
    csv_path = output_root / "prompt_aggregate.csv"
    json_path.write_text(
        json.dumps(
            aggregate_rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fieldnames = [
        "rank",
        "object_id",
        "object_name",
        "prompt_index",
        "prompt",
        "image_count",
        "pass_count",
        "fail_count",
        "detection_rate",
        "mean_iou",
        "median_iou",
        "mean_iou_detected",
        "best_iou",
        "mean_score_detected",
    ]
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return aggregate_rows, json_path, csv_path


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    import main as pipeline_main

    pipeline_config = pipeline_main.build_config(
        pipeline_main.parse_args([])
    )
    parser = argparse.ArgumentParser(
        description=(
            "Sweep SAM text prompts for unresolved "
            "LINEMOD objects and report GT-mask IoU."
        )
    )
    parser.add_argument(
        "--sam-version",
        choices=("sam3", "sam3.1"),
        default="sam3.1",
        help=(
            "Segmentation backend. SAM 3.1 processes each "
            "image as an independent single-frame session."
        ),
    )
    parser.add_argument(
        "--sam31-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "sam3"
            / "weights"
            / "sam3.1_multiplex.pt"
        ),
    )
    parser.add_argument(
        "--object-ids",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--prompt-bank",
        choices=(
            "default",
            "object2-100",
            "all-100",
        ),
        default="default",
        help=(
            "Prompt collection to test. object2-100 runs "
            "100 unique prompts for LINEMOD object 02; "
            "all-100 does the same for every registered "
            "LINEMOD object."
        ),
    )
    parser.add_argument(
        "--image-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Run only these image IDs instead of the "
            "default first/best frame pair."
        ),
    )
    parser.add_argument(
        "--random-image-count",
        type=int,
        default=None,
        help=(
            "Uniformly sample this many valid images per "
            "object without replacement. Raw prompt-bank "
            "modes only."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help=(
            "Seed for reproducible per-object random image "
            "sampling."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help=(
            "Number of successful prompts to print after "
            "the complete sweep. All prompts still run."
        ),
    )
    parser.add_argument(
        "--final-prompts",
        action="store_true",
        help=(
            "Test final prompts for LINEMOD objects. "
            "Object 02 uses validated prompt fallback; "
            "object 15 uses a two-part union."
        ),
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help=(
            "With --final-prompts, test every valid image "
            "instead of only first/best."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=pipeline_config.dataset_root,
    )
    parser.add_argument(
        "--split",
        default=pipeline_config.split,
    )
    parser.add_argument(
        "--instance-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        default=pipeline_config.sam3_device,
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=(
            pipeline_config
            .sam3_confidence_threshold
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help=(
            "Print selected frames and prompts without "
            "loading the model."
        ),
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    import main as pipeline_main

    pipeline_config = pipeline_main.build_config(
        pipeline_main.parse_args([])
    )
    raw_prompt_sweep = args.prompt_bank in {
        "object2-100",
        "all-100",
    }
    if raw_prompt_sweep:
        if args.final_prompts or args.all_frames:
            raise ValueError(
                f"{args.prompt_bank} is a raw prompt "
                "sweep and "
                "cannot be combined with --final-prompts "
                "or --all-frames."
            )
        if args.prompt_bank == "all-100":
            prompt_candidates = (
                ALL_OBJECT_PROMPT_SWEEP_100
            )
            default_object_ids = tuple(
                sorted(prompt_candidates)
            )
        else:
            prompt_candidates = {
                2: OBJECT2_PROMPT_SWEEP_100,
            }
            default_object_ids = (2,)
        default_output_root = (
            PROJECT_ROOT
            / "outputs"
            / (
                (
                    "sam31_all_objects_prompt_100"
                    if args.sam_version == "sam3.1"
                    else "sam3_all_objects_prompt_100"
                )
                if args.prompt_bank == "all-100"
                else (
                    "sam31_object02_prompt_100"
                    if args.sam_version == "sam3.1"
                    else "sam3_object02_prompt_100"
                )
            )
        )
    elif args.final_prompts:
        prompt_candidates = {
            object_id: (metadata[1],)
            for object_id, metadata in (
                pipeline_main
                .LINEMOD_OBJECT_METADATA.items()
            )
        }
        object2_primary_prompts = (
            (
                "a metallic bench vise tool "
                "with a screw handle"
            ),
            "metal clamp with a screw handle",
        )
        prompt_candidates[2] = (
            object2_primary_prompts
            + (
                (
                    "blue metal object",
                    "blue mechanical tool",
                    "blue circular metal tool",
                    (
                        "metal object with a vertical "
                        "silver handle"
                    ),
                    "blue object in the center",
                )
                if args.sam_version == "sam3.1"
                else ()
            )
        )
        prompt_candidates[15] = (
            "phone handset",
            "phone base",
        )
        default_object_ids = tuple(
            sorted(prompt_candidates)
        )
        default_output_root = (
            PROJECT_ROOT
            / "outputs"
            / (
                (
                    "sam31_final_prompt_all_frames"
                    if args.sam_version == "sam3.1"
                    else "sam3_final_prompt_all_frames"
                )
                if args.all_frames
                else (
                    "sam31_final_prompt_smoke"
                    if args.sam_version == "sam3.1"
                    else "sam3_final_prompt_smoke"
                )
            )
        )
    else:
        prompt_candidates = PROMPT_CANDIDATES
        default_object_ids = (1, 10, 15)
        default_output_root = (
            PROJECT_ROOT
            / "outputs"
            / (
                "sam31_prompt_sweep"
                if args.sam_version == "sam3.1"
                else "sam3_prompt_sweep_round5"
            )
        )

    if (
        raw_prompt_sweep
        and args.random_image_count is not None
        and args.output_root is None
    ):
        default_output_root = (
            default_output_root.parent
            / (
                f"{default_output_root.name}_random"
                f"{args.random_image_count}_seed"
                f"{args.random_seed}"
            )
        )
    if args.all_frames and not args.final_prompts:
        raise ValueError(
            "--all-frames requires --final-prompts."
        )
    if (
        args.all_frames
        and (
            args.image_ids is not None
            or args.random_image_count is not None
        )
    ):
        raise ValueError(
            "--all-frames cannot be combined with "
            "--image-ids or --random-image-count."
        )
    if (
        args.image_ids is not None
        and args.random_image_count is not None
    ):
        raise ValueError(
            "--image-ids and --random-image-count cannot "
            "be used together."
        )
    if (
        args.random_image_count is not None
        and not raw_prompt_sweep
    ):
        raise ValueError(
            "--random-image-count requires object2-100 "
            "or all-100."
        )
    if (
        args.random_image_count is not None
        and args.random_image_count < 1
    ):
        raise ValueError(
            "--random-image-count must be at least 1."
        )
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if (
        args.image_ids is not None
        and any(image_id < 0 for image_id in args.image_ids)
    ):
        raise ValueError(
            "--image-ids values must be non-negative."
        )

    object_ids = (
        tuple(args.object_ids)
        if args.object_ids is not None
        else default_object_ids
    )
    unknown_object_ids = sorted(
        set(object_ids) - set(prompt_candidates)
    )
    if unknown_object_ids:
        raise ValueError(
            "No prompt candidates for object IDs: "
            f"{unknown_object_ids}"
        )

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )
    output_root = (
        (
            args.output_root
            if args.output_root is not None
            else default_output_root
        )
        .expanduser()
        .resolve()
    )

    plans: list[
        tuple[int, str, tuple[tuple[str, int], ...]]
    ] = []
    for object_id in object_ids:
        object_name = (
            pipeline_main
            .LINEMOD_OBJECT_METADATA[object_id][0]
        )
        if args.image_ids is not None:
            frames = tuple(
                ("selected", image_id)
                for image_id in dict.fromkeys(
                    args.image_ids
                )
            )
        elif args.random_image_count is not None:
            frames = _random_frames(
                dataset_root=dataset_root,
                split=args.split,
                object_id=object_id,
                instance_index=args.instance_index,
                count=args.random_image_count,
                seed=args.random_seed,
            )
        elif raw_prompt_sweep:
            frames = (("selected", 0),)
        elif args.all_frames:
            frames = _all_frames(
                dataset_root=dataset_root,
                split=args.split,
                object_id=object_id,
                instance_index=args.instance_index,
            )
        else:
            frames = _selected_frames(
                dataset_root=dataset_root,
                split=args.split,
                object_id=object_id,
                instance_index=args.instance_index,
            )
        plans.append(
            (object_id, object_name, frames)
        )
        if args.random_image_count is not None:
            print(
                f"[PLAN] object={object_id:02d} "
                f"{object_name}, random_frames="
                f"{len(frames)}, seed={args.random_seed}"
            )
        elif args.all_frames:
            print(
                f"[PLAN] object={object_id:02d} "
                f"{object_name}, frame_count={len(frames)}, "
                f"range={frames[0][1]:06d}-"
                f"{frames[-1][1]:06d}"
            )
        else:
            print(
                f"[PLAN] object={object_id:02d} "
                f"{object_name}, frames={list(frames)}"
            )
        if raw_prompt_sweep:
            print(
                "       prompt_count="
                f"{len(prompt_candidates[object_id])}"
            )
        else:
            for prompt in prompt_candidates[object_id]:
                print(f"       prompt={prompt!r}")

    if (
        raw_prompt_sweep
        and args.random_image_count is not None
    ):
        print(
            "[OUTPUT MODE] summary JSON/CSV and prompt "
            "aggregate only (no mask/union/overlay/NPY)"
        )
    elif raw_prompt_sweep:
        print(
            "[OUTPUT MODE] 100 raw prompts; compressed "
            "binary mask PNGs only (no union/overlay/NPY)"
        )
    elif args.final_prompts:
        print(
            "[OUTPUT MODE] compressed binary PNG masks "
            "only (no RGB, overlay, or NPY)"
        )
    print(
        f"[SAM VERSION] {args.sam_version} "
        "(independent image segmentation)"
    )

    if args.list_only:
        return 0

    if not (
        0.0
        <= args.confidence_threshold
        < 1.0
    ):
        raise ValueError(
            "confidence_threshold must be in [0, 1)."
        )

    from dataset_io.linemod_loader import (
        load_linemod_view,
    )
    from mask_provider import (
        release_sam3_processor,
    )

    sam31_predictor = None
    if args.sam_version == "sam3.1":
        sam31_predictor = _build_sam31_predictor(
            repository_path=(
                pipeline_config.sam3_repository
            ),
            checkpoint_path=(
                args.sam31_checkpoint
                .expanduser()
                .resolve()
            ),
            bpe_path=pipeline_config.sam3_bpe,
            confidence_threshold=(
                args.confidence_threshold
            ),
        )

    rows: list[dict[str, Any]] = []
    run_root = (
        output_root
        / (
            "threshold_"
            + str(args.confidence_threshold).replace(
                ".",
                "p",
            )
        )
    )

    interrupted = False
    sam31_call_count = 0
    try:
        for object_id, object_name, frames in plans:
            for frame_type, image_id in frames:
                view = load_linemod_view(
                    dataset_root=dataset_root,
                    view_name="reference",
                    scene_id=object_id,
                    image_id=image_id,
                    object_name=object_name,
                    object_id=object_id,
                    split=args.split,
                )
                ground_truth = _load_gt_mask(
                    dataset_root=dataset_root,
                    split=args.split,
                    object_id=object_id,
                    image_id=image_id,
                    instance_index=(
                        args.instance_index
                    ),
                )
                masker = None
                predict_cached = None
                if args.sam_version == "sam3":
                    (
                        masker,
                        predict_cached,
                    ) = _build_cached_masker(
                        view=view,
                        repository_path=(
                            pipeline_config
                            .sam3_repository
                        ),
                        checkpoint_path=(
                            pipeline_config
                            .sam3_checkpoint
                        ),
                        bpe_path=(
                            pipeline_config.sam3_bpe
                        ),
                        device=args.device,
                        use_amp=(
                            pipeline_config
                            .sam3_use_amp
                        ),
                        confidence_threshold=(
                            args.confidence_threshold
                        ),
                    )
                successful_masks: list[np.ndarray] = []
                successful_prompts: list[str] = []
                successful_candidates: list[
                    dict[str, Any]
                ] = []
                frame_row_start = len(rows)

                for prompt_index, prompt in enumerate(
                    prompt_candidates[object_id]
                ):
                    if (
                        args.final_prompts
                        and args.sam_version == "sam3.1"
                        and object_id == 2
                        and prompt_index
                        >= OBJECT2_PRIMARY_PROMPT_COUNT
                    ):
                        selected_primary, _ = (
                            _select_object2_candidate(
                                candidates=[
                                    candidate
                                    for candidate
                                    in successful_candidates
                                    if int(
                                        candidate[
                                            "prompt_index"
                                        ]
                                    )
                                    < OBJECT2_PRIMARY_PROMPT_COUNT
                                ],
                                depth_m=view.depth_m,
                            )
                        )
                        if selected_primary is not None:
                            print(
                                "[FALLBACK SKIP] "
                                f"object={object_id:02d} "
                                f"frame={frame_type} "
                                f"image={image_id:06d} "
                                "a validated primary "
                                "prompt succeeded"
                            )
                            break
                    prompt_directory = (
                        run_root
                        / f"object_{object_id:02d}_{object_name}"
                        / f"{frame_type}_{image_id:06d}"
                        / (
                            f"p{prompt_index:02d}_"
                            f"{_slug(prompt)}"
                        )
                    )
                    started_at = time.perf_counter()
                    row: dict[str, Any] = {
                        "object_id": object_id,
                        "object_name": object_name,
                        "frame_type": frame_type,
                        "image_id": image_id,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "output_directory": (
                            None
                            if (
                                args.final_prompts
                                or raw_prompt_sweep
                                or args.sam_version
                                == "sam3.1"
                            )
                            else str(prompt_directory)
                        ),
                    }

                    try:
                        if args.sam_version == "sam3.1":
                            if sam31_predictor is None:
                                raise RuntimeError(
                                    "SAM 3.1 predictor "
                                    "was not initialized."
                                )
                            sam31_call_count += 1
                            (
                                mask_bool,
                                selected_score,
                            ) = (
                                _predict_sam31_single_frame(
                                    predictor=(
                                        sam31_predictor
                                    ),
                                    image_rgb=view.rgb,
                                    text_prompt=prompt,
                                    confidence_threshold=(
                                        args
                                        .confidence_threshold
                                    ),
                                    run_gc_collect=(
                                        sam31_call_count
                                        % 32
                                        == 0
                                    ),
                                )
                            )
                            if (
                                not args.final_prompts
                                and args.random_image_count
                                is None
                            ):
                                prompt_mask_path = (
                                    prompt_directory
                                    / "mask.png"
                                )
                                _save_binary_mask_png(
                                    mask=mask_bool,
                                    output_path=(
                                        prompt_mask_path
                                    ),
                                )
                                row[
                                    "output_directory"
                                ] = str(
                                    prompt_directory
                                )
                        elif (
                            args.final_prompts
                            or raw_prompt_sweep
                        ):
                            if predict_cached is None:
                                raise RuntimeError(
                                    "SAM3 predictor was "
                                    "not initialized."
                                )
                            (
                                mask_bool,
                                selected_score,
                            ) = _predict_mask_in_memory(
                                predict_function=(
                                    predict_cached
                                ),
                                image_rgb=view.rgb,
                                text_prompt=prompt,
                            )
                            if (
                                raw_prompt_sweep
                                and args.random_image_count
                                is None
                            ):
                                prompt_mask_path = (
                                    prompt_directory
                                    / "mask.png"
                                )
                                _save_binary_mask_png(
                                    mask=mask_bool,
                                    output_path=(
                                        prompt_mask_path
                                    ),
                                )
                                row[
                                    "output_directory"
                                ] = str(
                                    prompt_directory
                                )
                        else:
                            if masker is None:
                                raise RuntimeError(
                                    "SAM3 masker was not "
                                    "initialized."
                                )
                            result = masker.segment(
                                view=view,
                                output_directory=(
                                    prompt_directory
                                ),
                                text_prompt=prompt,
                            )
                            mask_bool = result.mask_bool
                            selected_score = result.score

                        metrics = _mask_metrics(
                            mask_bool,
                            ground_truth,
                        )
                        row.update(
                            {
                                "status": "PASS",
                                "score": selected_score,
                                **metrics,
                            }
                        )
                        if not raw_prompt_sweep:
                            successful_masks.append(
                                mask_bool
                            )
                            successful_prompts.append(
                                prompt
                            )
                            successful_candidates.append(
                                {
                                    "mask": mask_bool,
                                    "prompt": prompt,
                                    "prompt_index": (
                                        prompt_index
                                    ),
                                    "score": selected_score,
                                }
                            )
                        if (
                            args.random_image_count
                            is None
                        ):
                            print(
                                f"[PASS] object={object_id:02d} "
                                f"frame={frame_type} "
                                f"image={image_id:06d} "
                                f"prompt={prompt!r} "
                                f"score={selected_score} "
                                f"IoU={metrics['iou']:.4f}"
                            )
                    except Exception as error:
                        row.update(
                            {
                                "status": "FAIL",
                                "error_type": (
                                    type(error).__name__
                                ),
                                "error": str(error),
                            }
                        )
                        if (
                            args.random_image_count
                            is None
                        ):
                            print(
                                f"[FAIL] object={object_id:02d} "
                                f"frame={frame_type} "
                                f"image={image_id:06d} "
                                f"prompt={prompt!r}: "
                                f"{type(error).__name__}: "
                                f"{error}"
                            )
                    finally:
                        row["elapsed_sec"] = (
                            time.perf_counter()
                            - started_at
                        )
                        rows.append(row)

                if raw_prompt_sweep:
                    if (
                        args.random_image_count
                        is not None
                    ):
                        frame_rows = rows[
                            frame_row_start:
                        ]
                        frame_passed = [
                            row
                            for row in frame_rows
                            if row.get("status")
                            == "PASS"
                        ]
                        best_iou = max(
                            (
                                float(row["iou"])
                                for row in frame_passed
                            ),
                            default=0.0,
                        )
                        print(
                            "[FRAME COMPLETE] "
                            f"object={object_id:02d} "
                            f"image={image_id:06d} "
                            f"passed={len(frame_passed)} "
                            f"failed="
                            f"{len(frame_rows) - len(frame_passed)} "
                            f"best_iou={best_iou:.4f}"
                        )
                    continue

                if (
                    args.final_prompts
                    and args.sam_version == "sam3.1"
                    and object_id == 2
                ):
                    selected_candidate, evaluated = (
                        _select_object2_candidate(
                            candidates=successful_candidates,
                            depth_m=view.depth_m,
                        )
                    )
                    for candidate in evaluated:
                        rejection_reasons = candidate[
                            "rejection_reasons"
                        ]
                        if rejection_reasons:
                            print(
                                "[CANDIDATE REJECT] "
                                f"object={object_id:02d} "
                                f"frame={frame_type} "
                                f"image={image_id:06d} "
                                f"prompt={candidate['prompt']!r} "
                                f"reasons={rejection_reasons}"
                            )

                    if selected_candidate is None:
                        error_message = (
                            "No prompt returned a mask."
                            if not successful_candidates
                            else (
                                "All returned masks failed "
                                "non-GT geometry/depth "
                                "validation."
                            )
                        )
                        rows.append(
                            {
                                "object_id": object_id,
                                "object_name": object_name,
                                "frame_type": frame_type,
                                "image_id": image_id,
                                "prompt_index": -1,
                                "prompt": "__selection__",
                                "status": "FAIL",
                                "score": None,
                                "candidate_count": len(
                                    successful_candidates
                                ),
                                "elapsed_sec": 0.0,
                                "output_directory": None,
                                "error_type": (
                                    "MaskSelectionError"
                                ),
                                "error": error_message,
                            }
                        )
                        print(
                            f"[MASK FAIL] object={object_id:02d} "
                            f"frame={frame_type} "
                            f"image={image_id:06d}: "
                            f"{error_message}"
                        )
                        continue

                    selected_mask = np.ascontiguousarray(
                        selected_candidate["mask"],
                        dtype=np.bool_,
                    )
                    metrics = _mask_metrics(
                        selected_mask,
                        ground_truth,
                    )
                    final_mask_path = (
                        run_root
                        / "masks"
                        / (
                            f"object_{object_id:02d}_"
                            f"{object_name}"
                        )
                        / f"{image_id:06d}.png"
                    )
                    _save_binary_mask_png(
                        mask=selected_mask,
                        output_path=final_mask_path,
                    )
                    selected_prompt_index = int(
                        selected_candidate["prompt_index"]
                    )
                    selection_method = (
                        "primary_prompt_confidence"
                        if selected_prompt_index
                        < OBJECT2_PRIMARY_PROMPT_COUNT
                        else "fallback_prompt_confidence"
                    )
                    rows.append(
                        {
                            "object_id": object_id,
                            "object_name": object_name,
                            "frame_type": frame_type,
                            "image_id": image_id,
                            "prompt_index": -1,
                            "prompt": "__selected__",
                            "status": "PASS",
                            "score": selected_candidate.get(
                                "score"
                            ),
                            "selected_prompt": (
                                selected_candidate["prompt"]
                            ),
                            "selection_method": (
                                selection_method
                            ),
                            "candidate_count": len(
                                successful_candidates
                            ),
                            "area_ratio": (
                                selected_candidate[
                                    "area_ratio"
                                ]
                            ),
                            "center_distance_norm": (
                                selected_candidate[
                                    "center_distance_norm"
                                ]
                            ),
                            "valid_depth_ratio": (
                                selected_candidate[
                                    "valid_depth_ratio"
                                ]
                            ),
                            **metrics,
                            "elapsed_sec": 0.0,
                            "output_directory": str(
                                final_mask_path
                            ),
                        }
                    )
                    print(
                        f"[MASK SELECTED] object={object_id:02d} "
                        f"frame={frame_type} "
                        f"image={image_id:06d} "
                        f"prompt={selected_candidate['prompt']!r} "
                        f"method={selection_method} "
                        f"IoU={metrics['iou']:.4f} "
                        f"path={final_mask_path}"
                    )
                    continue

                if successful_masks:
                    union_mask = np.logical_or.reduce(
                        successful_masks
                    )
                    metrics = _mask_metrics(
                        union_mask,
                        ground_truth,
                    )
                    if args.final_prompts:
                        final_mask_path = (
                            run_root
                            / "masks"
                            / (
                                f"object_{object_id:02d}_"
                                f"{object_name}"
                            )
                            / f"{image_id:06d}.png"
                        )
                        _save_binary_mask_png(
                            mask=union_mask,
                            output_path=final_mask_path,
                        )
                        final_prompt_label = (
                            "__union__"
                            if len(
                                prompt_candidates[
                                    object_id
                                ]
                            )
                            > 1
                            else "__final__"
                        )
                        rows.append(
                            {
                                "object_id": object_id,
                                "object_name": object_name,
                                "frame_type": frame_type,
                                "image_id": image_id,
                                "prompt_index": -1,
                                "prompt": (
                                    final_prompt_label
                                ),
                                "status": "PASS",
                                "score": None,
                                **metrics,
                                "elapsed_sec": 0.0,
                                "output_directory": str(
                                    final_mask_path
                                ),
                            }
                        )
                        print(
                            f"[MASK] object={object_id:02d} "
                            f"frame={frame_type} "
                            f"image={image_id:06d} "
                            f"IoU={metrics['iou']:.4f} "
                            f"path={final_mask_path}"
                        )
                        continue

                    union_directory = (
                        run_root
                        / f"object_{object_id:02d}_{object_name}"
                        / f"{frame_type}_{image_id:06d}"
                        / "union"
                    )
                    _save_union_artifacts(
                        image_rgb=view.rgb,
                        union_mask=union_mask,
                        output_directory=union_directory,
                        metadata={
                            "source": (
                                "sam3_prompt_union"
                            ),
                            "prompts": successful_prompts,
                            "confidence_threshold": (
                                args.confidence_threshold
                            ),
                            **metrics,
                        },
                    )
                    rows.append(
                        {
                            "object_id": object_id,
                            "object_name": object_name,
                            "frame_type": frame_type,
                            "image_id": image_id,
                            "prompt_index": -1,
                            "prompt": "__union__",
                            "status": "PASS",
                            "score": None,
                            **metrics,
                            "elapsed_sec": 0.0,
                            "output_directory": str(
                                union_directory
                            ),
                        }
                    )
                    print(
                        f"[UNION] object={object_id:02d} "
                        f"frame={frame_type} "
                        f"image={image_id:06d} "
                        f"IoU={metrics['iou']:.4f}"
                    )
    except KeyboardInterrupt:
        interrupted = True
        print(
            "\n[INTERRUPTED] Saving completed rows "
            "before exit."
        )
    finally:
        release_sam3_processor()
        json_path, csv_path = _write_summary(
            rows=rows,
            output_root=run_root,
        )
    aggregate_json_path = None
    aggregate_csv_path = None
    if (
        raw_prompt_sweep
        and args.random_image_count is not None
    ):
        (
            aggregate_rows,
            aggregate_json_path,
            aggregate_csv_path,
        ) = _write_prompt_aggregate(
            rows=rows,
            output_root=run_root,
        )
        for object_id in object_ids:
            object_ranked_rows = [
                row
                for row in aggregate_rows
                if int(row["object_id"])
                == object_id
            ]
            print(
                f"[TOP PROMPTS] object={object_id:02d} "
                f"showing="
                f"{min(args.top_k, len(object_ranked_rows))}"
            )
            for row in object_ranked_rows[
                : args.top_k
            ]:
                print(
                    f"  {int(row['rank']):02d}. "
                    f"mean_IoU="
                    f"{float(row['mean_iou']):.4f} "
                    f"detection="
                    f"{float(row['detection_rate']):.3f} "
                    f"prompt={row['prompt']!r}"
                )
    elif raw_prompt_sweep:
        for object_id, _, frames in plans:
            for _, image_id in frames:
                ranked_rows = sorted(
                    (
                        row
                        for row in rows
                        if (
                            int(row["object_id"])
                            == object_id
                            and int(row["image_id"])
                            == image_id
                            and row.get("status")
                            == "PASS"
                        )
                    ),
                    key=lambda row: (
                        float(row.get("iou", -1.0)),
                        float(
                            row.get("score")
                            if row.get("score")
                            is not None
                            else -1.0
                        ),
                    ),
                    reverse=True,
                )
                print(
                    f"[TOP PROMPTS] object={object_id:02d} "
                    f"image={image_id:06d} "
                    f"showing={min(args.top_k, len(ranked_rows))}"
                )
                for rank, row in enumerate(
                    ranked_rows[: args.top_k],
                    start=1,
                ):
                    print(
                        f"  {rank:02d}. "
                        f"IoU={float(row['iou']):.4f} "
                        f"score={row.get('score')} "
                        f"prompt={row['prompt']!r}"
                    )
    passed = sum(
        row.get("status") == "PASS"
        and not str(
            row.get("prompt", "")
        ).startswith("__")
        for row in rows
    )
    failed = sum(
        row.get("status") == "FAIL"
        for row in rows
    )
    print(
        f"[SWEEP COMPLETE] passed={passed}, "
        f"failed={failed}"
    )
    print(f"[SUMMARY JSON] {json_path}")
    print(f"[SUMMARY CSV]  {csv_path}")
    if aggregate_json_path is not None:
        print(
            "[PROMPT AGGREGATE JSON] "
            f"{aggregate_json_path}"
        )
    if aggregate_csv_path is not None:
        print(
            "[PROMPT AGGREGATE CSV]  "
            f"{aggregate_csv_path}"
        )
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
