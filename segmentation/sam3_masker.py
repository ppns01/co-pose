from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from core.types import LoadedView, SegmentationResult


PredictFunction = Callable[
    [NDArray[np.uint8], str],
    tuple[Any, Any] | dict[str, Any],
]


def _write_rgb_png(
    output_path: Path,
    image_rgb: NDArray[np.uint8],
) -> None:
    """한글 경로를 지원하는 RGB PNG 저장 함수."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.dtype != np.uint8
    ):
        raise ValueError(
            "저장할 RGB 영상은 uint8 (H, W, 3)이어야 합니다: "
            f"shape={image_rgb.shape}, dtype={image_rgb.dtype}"
        )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    success, encoded = cv2.imencode(
        ".png",
        image_bgr,
    )

    if not success:
        raise RuntimeError(
            f"PNG 인코딩에 실패했습니다: {output_path}"
        )

    encoded.tofile(
        str(output_path)
    )


def _normalize_raw_output(
    raw_output: tuple[Any, Any] | dict[str, Any],
) -> tuple[Any, Any]:
    """Predict function 출력을 masks, scores로 분리한다."""

    if isinstance(raw_output, tuple):
        if len(raw_output) != 2:
            raise ValueError(
                "SAM3 predict function의 tuple 반환값은 "
                "(masks, scores) 형식이어야 합니다."
            )

        return raw_output[0], raw_output[1]

    if isinstance(raw_output, dict):
        if "masks" not in raw_output:
            raise KeyError(
                "SAM3 반환 dict에 'masks'가 없습니다."
            )

        return (
            raw_output["masks"],
            raw_output.get("scores"),
        )

    raise TypeError(
        "SAM3 predict function은 tuple 또는 dict를 "
        f"반환해야 합니다: {type(raw_output)}"
    )


def _normalize_masks(
    raw_masks: Any,
    *,
    expected_height: int,
    expected_width: int,
) -> list[NDArray[np.bool_]]:
    """SAM3 mask 출력을 Boolean mask 목록으로 변환한다."""

    if isinstance(raw_masks, np.ndarray):
        if raw_masks.ndim == 2:
            mask_items = [raw_masks]

        elif raw_masks.ndim == 3:
            mask_items = [
                raw_masks[index]
                for index in range(raw_masks.shape[0])
            ]

        elif (
            raw_masks.ndim == 4
            and raw_masks.shape[1] == 1
        ):
            mask_items = [
                raw_masks[index, 0]
                for index in range(raw_masks.shape[0])
            ]

        else:
            raise ValueError(
                "지원하지 않는 SAM3 mask 배열 shape입니다: "
                f"{raw_masks.shape}"
            )

    elif isinstance(raw_masks, (list, tuple)):
        mask_items = list(raw_masks)

    else:
        raise TypeError(
            "SAM3 masks는 numpy 배열 또는 list여야 합니다: "
            f"{type(raw_masks)}"
        )

    if not mask_items:
        raise ValueError(
            "SAM3가 mask 후보를 반환하지 않았습니다."
        )

    normalized_masks: list[NDArray[np.bool_]] = []

    for mask_index, mask_item in enumerate(mask_items):
        mask_array = np.asarray(mask_item)

        mask_array = np.squeeze(mask_array)

        if mask_array.shape != (
            expected_height,
            expected_width,
        ):
            raise ValueError(
                "SAM3 mask 해상도가 RGB와 다릅니다: "
                f"index={mask_index}, "
                f"expected={(expected_height, expected_width)}, "
                f"actual={mask_array.shape}"
            )

        if mask_array.dtype == np.bool_:
            mask_bool = mask_array
        else:
            if not np.isfinite(mask_array).all():
                raise ValueError(
                    "SAM3 mask에 NaN 또는 Inf가 있습니다: "
                    f"index={mask_index}"
                )

            mask_bool = mask_array > 0.5

        normalized_masks.append(
            np.ascontiguousarray(
                mask_bool,
                dtype=np.bool_,
            )
        )

    return normalized_masks


def _normalize_scores(
    raw_scores: Any,
    *,
    mask_count: int,
) -> list[float | None]:
    """SAM3 score 출력을 mask 개수와 맞춘다."""

    if raw_scores is None:
        if mask_count == 1:
            return [None]

        raise ValueError(
            "SAM3가 여러 mask를 반환했지만 scores가 없습니다."
        )

    score_array = np.asarray(
        raw_scores,
        dtype=np.float64,
    ).reshape(-1)

    if score_array.size != mask_count:
        raise ValueError(
            "SAM3 mask 개수와 score 개수가 다릅니다: "
            f"masks={mask_count}, scores={score_array.size}"
        )

    normalized_scores: list[float | None] = []

    for score_index, score_value in enumerate(score_array):
        score = float(score_value)

        if not np.isfinite(score):
            raise ValueError(
                "SAM3 score가 유한하지 않습니다: "
                f"index={score_index}, score={score}"
            )

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "SAM3 score는 0과 1 사이여야 합니다. "
                "현재 adapter가 logit을 반환한다면 확률로 "
                "변환해야 합니다: "
                f"index={score_index}, score={score}"
            )

        normalized_scores.append(score)

    return normalized_scores


def _select_best_mask(
    masks: list[NDArray[np.bool_]],
    scores: list[float | None],
) -> tuple[int, NDArray[np.bool_], float | None]:
    """빈 mask를 제외하고 최고 score 후보를 선택한다."""

    if len(masks) != len(scores):
        raise ValueError(
            "Mask와 score 개수가 다릅니다."
        )

    valid_candidates: list[
        tuple[int, NDArray[np.bool_], float | None]
    ] = []

    for candidate_index, (mask, score) in enumerate(
        zip(masks, scores)
    ):
        if not np.any(mask):
            continue

        valid_candidates.append(
            (
                candidate_index,
                mask,
                score,
            )
        )

    if not valid_candidates:
        raise ValueError(
            "SAM3가 반환한 모든 mask가 비어 있습니다."
        )

    if len(valid_candidates) == 1:
        return valid_candidates[0]

    if any(
        score is None
        for _, _, score in valid_candidates
    ):
        raise ValueError(
            "여러 유효 mask 중 일부에 score가 없습니다."
        )

    return max(
        valid_candidates,
        key=lambda item: float(item[2]),
    )


def _make_mask_rgb(
    mask_bool: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """Boolean mask를 0/255 RGB mask로 변환한다."""

    mask_gray = (
        mask_bool.astype(np.uint8)
        * np.uint8(255)
    )

    return np.ascontiguousarray(
        np.repeat(
            mask_gray[:, :, None],
            repeats=3,
            axis=2,
        ),
        dtype=np.uint8,
    )


def _make_overlay(
    image_rgb: NDArray[np.uint8],
    mask_bool: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """SAM3 결과 확인용 overlay를 생성한다."""

    overlay = image_rgb.copy()

    highlight = np.array(
        [0, 255, 0],
        dtype=np.float32,
    )

    source_pixels = overlay[
        mask_bool
    ].astype(np.float32)

    overlay[
        mask_bool
    ] = np.clip(
        source_pixels * 0.55
        + highlight * 0.45,
        0.0,
        255.0,
    ).astype(np.uint8)

    return np.ascontiguousarray(
        overlay,
        dtype=np.uint8,
    )


class SAM3Masker:
    """
    SAM3 inference adapter.

    실제 SAM3 모델 API는 predict_function으로 외부 주입한다.

    predict_function 입력:
        image_rgb: uint8 (H, W, 3)
        object_name: 객체 이름 text prompt

    predict_function 출력:
        (masks, scores)

    또는:
        {
            "masks": ...,
            "scores": ...
        }
    """

    def __init__(
        self,
        predict_function: PredictFunction,
    ) -> None:
        if not callable(predict_function):
            raise TypeError(
                "predict_function은 callable이어야 합니다."
            )

        self._predict_function = predict_function

    def segment(
        self,
        *,
        view: LoadedView,
        output_directory: Path,
        text_prompt: str | None = None,
    ) -> SegmentationResult:
        """객체 이름 prompt로 SAM3 mask를 생성하고 저장한다."""

        image_rgb = np.asarray(
            view.rgb,
            dtype=np.uint8,
        )

        if (
            image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
        ):
            raise ValueError(
                "입력 RGB shape은 (H, W, 3)이어야 합니다: "
                f"{image_rgb.shape}"
            )

        object_name = view.source.object_name.strip()

        if not object_name:
            raise ValueError(
                "SAM3 text prompt로 사용할 object_name이 "
                "비어 있습니다."
            )

        resolved_text_prompt = (
            object_name
            if text_prompt is None
            else text_prompt.strip()
        )

        if not resolved_text_prompt:
            raise ValueError(
                "SAM3 text prompt가 비어 있습니다."
            )

        raw_output = self._predict_function(
            image_rgb,
            resolved_text_prompt,
        )

        raw_masks, raw_scores = (
            _normalize_raw_output(
                raw_output
            )
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

        (
            selected_index,
            mask_bool,
            selected_score,
        ) = _select_best_mask(
            masks,
            scores,
        )

        mask_rgb = _make_mask_rgb(
            mask_bool
        )

        overlay_rgb = _make_overlay(
            image_rgb,
            mask_bool,
        )

        output_directory = (
            Path(output_directory)
            .expanduser()
            .resolve()
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask_bool_path = (
            output_directory
            / "mask_bool.npy"
        )

        mask_rgb_path = (
            output_directory
            / "mask_rgb.png"
        )

        overlay_path = (
            output_directory
            / "overlay.png"
        )

        metadata_path = (
            output_directory
            / "meta.json"
        )

        np.save(
            mask_bool_path,
            mask_bool,
            allow_pickle=False,
        )

        _write_rgb_png(
            mask_rgb_path,
            mask_rgb,
        )

        _write_rgb_png(
            overlay_path,
            overlay_rgb,
        )

        metadata = {
            "source": "sam3",
            "view_name": view.source.name,
            "object_name": object_name,
            "text_prompt": resolved_text_prompt,
            "prompt_type": "text",
            "selected_candidate_index": (
                selected_index
            ),
            "selected_score": selected_score,
            "candidate_count": len(masks),
            "mask_pixel_count": int(
                np.count_nonzero(mask_bool)
            ),
            "image_height": int(
                image_rgb.shape[0]
            ),
            "image_width": int(
                image_rgb.shape[1]
            ),
            "mask_bool_path": str(
                mask_bool_path
            ),
            "mask_rgb_path": str(
                mask_rgb_path
            ),
            "overlay_path": str(
                overlay_path
            ),
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
            score=selected_score,
            overlay_path=overlay_path,
            metadata_path=metadata_path,
        )

    def predict(
        self,
        *,
        view: LoadedView,
        output_directory: Path,
        text_prompt: str | None = None,
    ) -> SegmentationResult:
        """segment()의 호환용 별칭."""

        return self.segment(
            view=view,
            output_directory=output_directory,
            text_prompt=text_prompt,
        )

    def release(self) -> None:
        """외부 SAM3 adapter가 release를 지원하면 호출한다."""

        release_function = getattr(
            self._predict_function,
            "release",
            None,
        )

        if callable(release_function):
            release_function()

    def __enter__(self) -> SAM3Masker:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.release()
