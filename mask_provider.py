from __future__ import annotations

import gc
import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.types import LoadedView, SegmentationResult


_SAM3_PROCESSOR: Any | None = None
_SAM3_PROCESSOR_KEY: (
    tuple[str, str, str, str, float] | None
) = None


def prepare_object_mask(
    image_path: str | Path,
    output_path: str | Path,
    *,
    mask_path: str | Path | None = None,
    text_prompt: str | None = None,
    min_score: float = 0.0,
) -> Path:


    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"RGB 이미지가 없습니다: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    if mask_path is not None:
        mask = _load_existing_mask(
            mask_path=Path(mask_path),
            expected_size=image.size,
        )

    else:
        if text_prompt is None or not text_prompt.strip():
            raise ValueError(
                "기존 mask가 없으면 "
                "SAM3 text_prompt가 필요합니다."
            )

        mask = _generate_mask_with_sam3(
            image=image,
            text_prompt=text_prompt.strip(),
            min_score=min_score,
        )

    if not np.any(mask):
        raise ValueError(
            "생성된 mask에 foreground가 없습니다."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_image = Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L",
    )

    mask_image.save(output_path)

    return output_path


def load_existing_segmentation(
    *,
    view: LoadedView,
    mask_path: str | Path,
    output_directory: str | Path,
) -> SegmentationResult:
    """
    기존 BOP/LINEMOD mask를 공통 segmentation 계약으로 변환한다.

    원본 mask는 수정하지 않고, 파이프라인이 재사용할 Boolean NPY와
    3채널 0/255 RGB mask를 별도 출력 폴더에 저장한다.
    """

    resolved_mask_path = (
        Path(mask_path)
        .expanduser()
        .resolve()
    )

    image_height, image_width = (
        view.rgb.shape[:2]
    )

    mask_bool = np.ascontiguousarray(
        _load_existing_mask(
            mask_path=resolved_mask_path,
            expected_size=(
                image_width,
                image_height,
            ),
        ),
        dtype=np.bool_,
    )

    if not np.any(mask_bool):
        raise ValueError(
            "기존 mask에 foreground가 없습니다: "
            f"{resolved_mask_path}"
        )

    mask_rgb = np.repeat(
        (
            mask_bool.astype(np.uint8)
            * np.uint8(255)
        )[:, :, None],
        repeats=3,
        axis=2,
    )

    mask_rgb = np.ascontiguousarray(
        mask_rgb,
        dtype=np.uint8,
    )

    resolved_output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    resolved_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_bool_path = (
        resolved_output_directory
        / "mask_bool.npy"
    )

    mask_rgb_path = (
        resolved_output_directory
        / "mask_rgb.png"
    )

    metadata_path = (
        resolved_output_directory
        / "meta.json"
    )

    np.save(
        mask_bool_path,
        mask_bool,
        allow_pickle=False,
    )

    Image.fromarray(
        mask_rgb
    ).save(mask_rgb_path)

    metadata = {
        "source": "existing_mask",
        "view_name": view.source.name,
        "source_mask_path": str(
            resolved_mask_path
        ),
        "mask_bool_path": str(
            mask_bool_path
        ),
        "mask_rgb_path": str(
            mask_rgb_path
        ),
        "foreground_pixel_count": int(
            np.count_nonzero(mask_bool)
        ),
        "image_height": image_height,
        "image_width": image_width,
    }

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return SegmentationResult(
        mask_bool=mask_bool,
        mask_rgb=mask_rgb,
        mask_bool_path=mask_bool_path,
        mask_rgb_path=mask_rgb_path,
        score=None,
        overlay_path=None,
        metadata_path=metadata_path,
    )


def generate_sam3_segmentation(
    *,
    view: LoadedView,
    output_directory: str | Path,
    text_prompt: str | None = None,
    repository_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    bpe_path: str | Path | None = None,
    device: str | None = None,
    use_amp: bool = True,
    confidence_threshold: float = 0.30,
) -> SegmentationResult:
    """
    로컬 SAM3 모델로 segmentation을 생성하고 공통 산출물로 저장한다.

    후보 정규화와 선택, overlay 및 metadata 저장은 기존
    SAM3Masker 구현을 재사용한다.
    """

    from segmentation.sam3_masker import SAM3Masker

    masker = SAM3Masker(
        partial(
            _predict_with_sam3,
            repository_path=repository_path,
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            device=device,
            use_amp=use_amp,
            confidence_threshold=confidence_threshold,
        )
    )

    return masker.segment(
        view=view,
        output_directory=Path(output_directory),
        text_prompt=text_prompt,
    )


def _load_existing_mask(
    mask_path: Path,
    expected_size: tuple[int, int],
) -> np.ndarray:
    """
    기존 mask를 읽고 binary mask로 변환한다.
    """

    if not mask_path.is_file():
        raise FileNotFoundError(
            f"Mask 파일이 없습니다: {mask_path}"
        )

    mask_image = Image.open(mask_path).convert("L")

    if mask_image.size != expected_size:
        raise ValueError(
            "RGB와 mask 크기가 다릅니다: "
            f"rgb={expected_size}, "
            f"mask={mask_image.size}"
        )

    mask = np.asarray(mask_image) > 0

    return mask


def _generate_mask_with_sam3(
    image: Image.Image,
    text_prompt: str,
    min_score: float,
) -> np.ndarray:
    """
    SAM3 text prompt를 이용해 객체 mask를 생성한다.

    여러 instance가 검출되면 score가 가장 높은 mask 하나를 선택한다.
    """

    output = _predict_with_sam3(
        np.asarray(image, dtype=np.uint8),
        text_prompt,
    )

    masks = output["masks"]
    scores = output["scores"].reshape(-1)

    masks = np.squeeze(masks)

    if masks.ndim == 2:
        masks = masks[None, ...]

    if masks.ndim != 3:
        raise ValueError(
            f"예상하지 못한 SAM3 mask shape: {masks.shape}"
        )

    if masks.shape[0] != scores.shape[0]:
        raise ValueError(
            "SAM3 mask와 score 개수가 다릅니다: "
            f"masks={masks.shape[0]}, "
            f"scores={scores.shape[0]}"
        )

    valid_indices = np.flatnonzero(
        scores >= min_score
    )

    if valid_indices.size == 0:
        raise RuntimeError(
            "SAM3가 조건을 만족하는 객체를 찾지 못했습니다. "
            f"prompt={text_prompt!r}, "
            f"min_score={min_score}"
        )

    best_index = valid_indices[
        np.argmax(scores[valid_indices])
    ]

    best_mask = masks[best_index] > 0.5

    return best_mask


def _predict_with_sam3(
    image_rgb: np.ndarray,
    text_prompt: str,
    *,
    repository_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    bpe_path: str | Path | None = None,
    device: str | None = None,
    use_amp: bool = True,
    confidence_threshold: float = 0.30,
) -> dict[str, np.ndarray]:
    """SAM3 processor 출력을 CPU NumPy 배열로 변환한다."""

    import torch

    resolved_device = torch.device(
        device
        if device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )
    processor = _get_sam3_processor(
        repository_path=repository_path,
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=str(resolved_device),
        confidence_threshold=confidence_threshold,
    )
    image = Image.fromarray(
        np.asarray(image_rgb, dtype=np.uint8),
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
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt,
        )

    return {
        "masks": _to_numpy(output["masks"]),
        "scores": _to_numpy(
            output["scores"].to(dtype=torch.float32)
        ),
    }


def _get_sam3_processor(
    *,
    repository_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    bpe_path: str | Path | None = None,
    device: str | None = None,
    confidence_threshold: float = 0.30,
) -> Any:
    """
    로컬 checkpoint만 사용해 SAM3를 로드한다.

    인터넷이나 Hugging Face Hub에 접속하지 않는다.
    모델은 최초 한 번만 로드하고 이후 재사용한다.
    """

    import importlib
    import os
    import sys
    import torch

    global _SAM3_PROCESSOR
    global _SAM3_PROCESSOR_KEY

    # Hugging Face 및 Transformers의 외부 접속 차단
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    project_root = Path(__file__).resolve().parent
    sam3_repository = (
        Path(repository_path)
        if repository_path is not None
        else project_root / "sam3"
    ).expanduser().resolve()
    resolved_checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else (
            sam3_repository
            / "weights"
            / "sam3.pt"
        )
    ).expanduser().resolve()
    resolved_bpe_path = (
        Path(bpe_path)
        if bpe_path is not None
        else (
            sam3_repository
            / "sam3"
            / "assets"
            / "bpe_simple_vocab_16e6.txt.gz"
        )
    ).expanduser().resolve()
    resolved_device = torch.device(
        device
        if device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    if (
        not np.isfinite(confidence_threshold)
        or not 0.0 <= confidence_threshold < 1.0
    ):
        raise ValueError(
            "SAM3 confidence_threshold must be finite "
            "and in [0, 1)."
        )

    if (
        resolved_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "SAM3 CUDA device was requested but CUDA "
            f"is unavailable: {resolved_device}"
        )

    processor_key = (
        str(sam3_repository),
        str(resolved_checkpoint_path),
        str(resolved_bpe_path),
        str(resolved_device),
        float(confidence_threshold),
    )

    if _SAM3_PROCESSOR is not None:
        if _SAM3_PROCESSOR_KEY != processor_key:
            raise RuntimeError(
                "SAM3 is already loaded with different "
                "settings. Call release_sam3_processor() "
                "before changing its config."
            )

        return _SAM3_PROCESSOR

    if not resolved_checkpoint_path.is_file():
        raise FileNotFoundError(
            "SAM3 checkpoint가 없습니다:\n"
            f"{resolved_checkpoint_path}"
        )

    if not resolved_bpe_path.is_file():
        raise FileNotFoundError(
            "SAM3 BPE vocabulary가 없습니다:\n"
            f"{resolved_bpe_path}"
        )

    sam3_repository_string = str(sam3_repository)

    if sam3_repository_string not in sys.path:
        sys.path.insert(0, sam3_repository_string)

    importlib.invalidate_caches()

    try:
        from sam3.model_builder import (
            build_sam3_image_model,
        )

        from sam3.model.sam3_image_processor import (
            Sam3Processor,
        )

    except ImportError as error:
        raise ImportError(
            "SAM3 Python 패키지가 설치되어 있지 않습니다."
        ) from error

    model = build_sam3_image_model(
        checkpoint_path=str(
            resolved_checkpoint_path
        ),
        bpe_path=str(resolved_bpe_path),
        load_from_HF=False,
        device=str(resolved_device),
        eval_mode=True,
    )

    _SAM3_PROCESSOR = Sam3Processor(
        model,
        confidence_threshold=confidence_threshold,
    )
    _SAM3_PROCESSOR_KEY = processor_key

    print("[SAM3 로드 완료]")
    print(f"Device     : {resolved_device}")
    print(f"Checkpoint : {resolved_checkpoint_path}")
    print(f"BPE        : {resolved_bpe_path}")
    print(f"Confidence : {confidence_threshold}")

    return _SAM3_PROCESSOR


def release_sam3_processor() -> None:
    """SAM3 참조와 CUDA cache를 해제해 후속 모델의 VRAM을 확보한다."""

    global _SAM3_PROCESSOR
    global _SAM3_PROCESSOR_KEY

    processor = _SAM3_PROCESSOR
    _SAM3_PROCESSOR = None
    _SAM3_PROCESSOR_KEY = None

    if processor is None:
        return

    try:
        release_function = getattr(
            processor,
            "release",
            None,
        )

        if callable(release_function):
            release_function()
    finally:
        del processor
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def _to_numpy(value: Any) -> np.ndarray:
    """
    torch.Tensor 또는 일반 배열을 NumPy 배열로 변환한다.
    """

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        return value.numpy()

    return np.asarray(value)
