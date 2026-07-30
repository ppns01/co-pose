from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F

from core.types import LoadedView


SUPPORTED_DINOV3_MODELS = {
    "dinov3_vits16",
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vith16plus",
}

DINOV3_HIDDEN_SIZES = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
}


@dataclass(frozen=True)
class DINOFeatureResult:
    """
    한 영상에서 추출한 DINOv3 dense patch feature.

    feature_map:
        L2 정규화된 DINOv3 patch feature.
        shape = (C, H_patch, W_patch)

    original_image_size:
        원본 RGB 크기.
        (height, width)

    resized_image_size:
        DINOv3 입력으로 변환된 영상 크기.
        (height, width)

    patch_size:
        DINOv3 patch 크기.
        일반적인 ViT 모델에서는 (16, 16).
    """

    view_name: str
    model_name: str

    feature_map: NDArray[np.float16]

    original_image_size: tuple[int, int]
    resized_image_size: tuple[int, int]
    patch_size: tuple[int, int]

    feature_path: Path
    metadata_path: Path

    def __post_init__(self) -> None:
        if self.view_name not in (
            "reference",
            "query",
        ):
            raise ValueError(
                f"지원하지 않는 view입니다: {self.view_name}"
            )

        if self.model_name not in SUPPORTED_DINOV3_MODELS:
            raise ValueError(
                "지원하지 않는 DINOv3 모델입니다: "
                f"{self.model_name}"
            )

        if self.feature_map.ndim != 3:
            raise ValueError(
                "DINO feature map shape은 "
                "(C, H, W)이어야 합니다: "
                f"{self.feature_map.shape}"
            )

        if self.feature_map.dtype != np.float16:
            raise TypeError(
                "저장된 DINO feature dtype은 "
                "float16이어야 합니다: "
                f"{self.feature_map.dtype}"
            )

        if not np.isfinite(self.feature_map).all():
            raise ValueError(
                "DINO feature에 NaN 또는 Inf가 있습니다."
            )

        if not self.feature_path.is_file():
            raise FileNotFoundError(
                "DINO feature 파일이 없습니다: "
                f"{self.feature_path}"
            )

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                "DINO metadata 파일이 없습니다: "
                f"{self.metadata_path}"
            )


class DINOv3Extractor:
    """
    로컬 DINOv3 가중치를 사용하는 dense patch
    feature extractor.

    지원 backend:
        - 원본 DINOv3 저장소 + PyTorch Hub .pth
        - 공식 Hugging Face 로컬 model.safetensors 디렉터리

    현재 pipeline 기본 모델:
        dinov3_vitl16

    메모리가 부족하면:
        dinov3_vitb16 또는 dinov3_vits16
    """

    def __init__(
        self,
        repository_path: Path,
        checkpoint_path: Path,
        *,
        model_name: str = "dinov3_vitb16",
        device: str = "cuda:0",
        target_long_side: int = 448,
        use_amp: bool = True,
        save_dtype: str = "float16",
    ) -> None:
        self._repository_path = (
            Path(repository_path)
            .expanduser()
            .resolve()
        )

        self._checkpoint_path = (
            Path(checkpoint_path)
            .expanduser()
            .resolve()
        )

        if model_name not in SUPPORTED_DINOV3_MODELS:
            raise ValueError(
                "지원하지 않는 DINOv3 모델입니다: "
                f"{model_name}\n"
                f"지원 모델: "
                f"{sorted(SUPPORTED_DINOV3_MODELS)}"
            )

        if not device.startswith("cuda"):
            raise ValueError(
                "현재 DINOv3 extractor는 CUDA 사용을 "
                "기준으로 합니다: "
                f"{device}"
            )

        if (
            isinstance(target_long_side, bool)
            or target_long_side < 64
        ):
            raise ValueError(
                "target_long_side는 64 이상인 "
                "정수여야 합니다: "
                f"{target_long_side}"
            )

        if save_dtype != "float16":
            raise ValueError(
                "현재 저장 dtype은 float16으로 고정합니다."
            )

        self._model_name = model_name
        self._device = torch.device(device)
        self._target_long_side = int(
            target_long_side
        )
        self._use_amp = bool(use_amp)

        self._model: torch.nn.Module | None = None
        self._patch_size: tuple[int, int] | None = None
        self._backend: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_torch_hub_model(
        self,
    ) -> torch.nn.Module:
        if not self._repository_path.is_dir():
            raise FileNotFoundError(
                "DINOv3 저장소가 없습니다: "
                f"{self._repository_path}"
            )

        hubconf_path = (
            self._repository_path / "hubconf.py"
        )

        if not hubconf_path.is_file():
            raise FileNotFoundError(
                "DINOv3 hubconf.py가 없습니다: "
                f"{hubconf_path}"
            )

        try:
            return torch.hub.load(
                repo_or_dir=str(
                    self._repository_path
                ),
                model=self._model_name,
                source="local",
                weights=str(
                    self._checkpoint_path
                ),
            )
        except Exception as error:
            raise RuntimeError(
                "PyTorch Hub DINOv3 모델을 "
                "로드하지 못했습니다.\n"
                f"Repository: {self._repository_path}\n"
                f"Checkpoint: {self._checkpoint_path}\n"
                f"Model: {self._model_name}"
            ) from error

    def _load_huggingface_model(
        self,
    ) -> torch.nn.Module:
        required_paths = (
            self._checkpoint_path / "config.json",
            self._checkpoint_path / "model.safetensors",
            (
                self._checkpoint_path
                / "preprocessor_config.json"
            ),
        )

        missing_paths = tuple(
            path
            for path in required_paths
            if not path.is_file()
        )

        if missing_paths:
            raise FileNotFoundError(
                "Hugging Face DINOv3 모델 디렉터리에 "
                "필수 파일이 없습니다:\n"
                + "\n".join(
                    str(path)
                    for path in missing_paths
                )
            )

        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError(
                "공식 Hugging Face DINOv3 모델에는 "
                "transformers>=4.56.0과 safetensors가 "
                "필요합니다."
            ) from error

        try:
            model = AutoModel.from_pretrained(
                str(self._checkpoint_path),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
        except Exception as error:
            raise RuntimeError(
                "로컬 Hugging Face DINOv3 모델을 "
                "로드하지 못했습니다. "
                "transformers>=4.56.0인지 확인하세요.\n"
                f"Model directory: "
                f"{self._checkpoint_path}\n"
                f"Model: {self._model_name}"
            ) from error

        config = getattr(
            model,
            "config",
            None,
        )

        if config is None:
            raise AttributeError(
                "Hugging Face DINOv3 모델에 "
                "config가 없습니다."
            )

        model_type = getattr(
            config,
            "model_type",
            None,
        )

        if model_type != "dinov3_vit":
            raise ValueError(
                "Hugging Face 모델이 DINOv3 ViT가 "
                "아닙니다: "
                f"{model_type}"
            )

        hidden_size = getattr(
            config,
            "hidden_size",
            None,
        )
        expected_hidden_size = (
            DINOV3_HIDDEN_SIZES[
                self._model_name
            ]
        )

        if hidden_size != expected_hidden_size:
            raise ValueError(
                "DINOv3 모델 이름과 Hugging Face "
                "가중치 크기가 다릅니다: "
                f"model={self._model_name}, "
                f"expected_hidden_size="
                f"{expected_hidden_size}, "
                f"actual_hidden_size={hidden_size}"
            )

        return model

    def load(self) -> None:
        """DINOv3 backbone을 GPU에 한 번 로드한다."""

        if self._model is not None:
            return

        if not (
            self._checkpoint_path.is_file()
            or self._checkpoint_path.is_dir()
        ):
            raise FileNotFoundError(
                "DINOv3 가중치 파일 또는 Hugging Face "
                "모델 디렉터리가 없습니다: "
                f"{self._checkpoint_path}"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch에서 CUDA를 사용할 수 없습니다."
            )

        torch.cuda.set_device(
            self._device
        )

        # 외부 다운로드를 방지한다.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        if self._checkpoint_path.is_dir():
            model = (
                self._load_huggingface_model()
            )
            backend = "huggingface_transformers"

        else:
            model = (
                self._load_torch_hub_model()
            )
            backend = "pytorch_hub"

        if not isinstance(
            model,
            torch.nn.Module,
        ):
            raise TypeError(
                "DINOv3 loader 결과가 "
                "torch.nn.Module이 아닙니다: "
                f"{type(model)}"
            )

        model = model.eval()
        model.requires_grad_(False)
        model = model.to(self._device)

        patch_size = (
            self._read_patch_size(model)
        )
        self._model = model
        self._backend = backend
        self._patch_size = patch_size

    @staticmethod
    def _read_patch_size(
        model: torch.nn.Module,
    ) -> tuple[int, int]:
        """DINOv3 patch embedding 크기를 읽는다."""

        patch_embed = getattr(
            model,
            "patch_embed",
            None,
        )

        if patch_embed is not None:
            patch_size_value = getattr(
                patch_embed,
                "patch_size",
                None,
            )

        else:
            config = getattr(
                model,
                "config",
                None,
            )
            patch_size_value = getattr(
                config,
                "patch_size",
                None,
            )

        if patch_size_value is None:
            raise AttributeError(
                "DINOv3 모델 설정에 "
                "patch_size가 없습니다."
            )

        if isinstance(
            patch_size_value,
            int,
        ):
            patch_height = patch_size_value
            patch_width = patch_size_value

        elif isinstance(
            patch_size_value,
            (tuple, list),
        ) and len(patch_size_value) == 2:
            patch_height = int(
                patch_size_value[0]
            )
            patch_width = int(
                patch_size_value[1]
            )

        else:
            raise TypeError(
                "지원하지 않는 patch_size 형식입니다: "
                f"{patch_size_value}"
            )

        if patch_height <= 0 or patch_width <= 0:
            raise ValueError(
                "patch_size는 양수여야 합니다: "
                f"{(patch_height, patch_width)}"
            )

        return (
            patch_height,
            patch_width,
        )

    def _compute_resized_size(
        self,
        original_height: int,
        original_width: int,
    ) -> tuple[int, int]:
        """
        원본 종횡비를 유지하면서 긴 변을 target_long_side로 맞춘다.

        최종 크기는 patch size의 배수가 되도록 조정한다.
        """

        if self._patch_size is None:
            raise RuntimeError(
                "DINOv3 모델이 로드되지 않았습니다."
            )

        patch_height, patch_width = (
            self._patch_size
        )

        scale = (
            self._target_long_side
            / max(
                original_height,
                original_width,
            )
        )

        resized_height = max(
            patch_height,
            int(round(original_height * scale)),
        )

        resized_width = max(
            patch_width,
            int(round(original_width * scale)),
        )

        resized_height = max(
            patch_height,
            int(
                round(
                    resized_height / patch_height
                )
                * patch_height
            ),
        )

        resized_width = max(
            patch_width,
            int(
                round(
                    resized_width / patch_width
                )
                * patch_width
            ),
        )

        return (
            resized_height,
            resized_width,
        )

    def _prepare_image(
        self,
        rgb: NDArray[np.uint8],
    ) -> tuple[
        torch.Tensor,
        tuple[int, int],
    ]:
        """
        uint8 RGB를 DINOv3 입력 tensor로 변환한다.

        ImageNet normalization:
            mean = (0.485, 0.456, 0.406)
            std  = (0.229, 0.224, 0.225)
        """

        if (
            rgb.ndim != 3
            or rgb.shape[2] != 3
        ):
            raise ValueError(
                "RGB shape은 (H, W, 3)이어야 합니다: "
                f"{rgb.shape}"
            )

        if rgb.dtype != np.uint8:
            raise TypeError(
                "RGB dtype은 uint8이어야 합니다: "
                f"{rgb.dtype}"
            )

        original_height, original_width = (
            rgb.shape[:2]
        )

        resized_size = (
            self._compute_resized_size(
                original_height=original_height,
                original_width=original_width,
            )
        )

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(rgb)
        )

        image_tensor = (
            image_tensor
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=self._device,
                dtype=torch.float32,
                non_blocking=True,
            )
            / 255.0
        )

        image_tensor = F.interpolate(
            image_tensor,
            size=resized_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=self._device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=self._device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        image_tensor = (
            image_tensor - mean
        ) / std

        return (
            image_tensor.contiguous(),
            resized_size,
        )

    def _forward_patch_features(
        self,
        image_tensor: torch.Tensor,
        resized_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        DINOv3 x_norm_patchtokens를 (1, C, Hp, Wp)로 변환한다.
        """

        if self._model is None:
            raise RuntimeError(
                "DINOv3 모델이 로드되지 않았습니다."
            )

        if self._patch_size is None:
            raise RuntimeError(
                "DINOv3 patch size가 설정되지 않았습니다."
            )

        resized_height, resized_width = (
            resized_size
        )

        patch_height, patch_width = (
            self._patch_size
        )

        feature_height = (
            resized_height // patch_height
        )

        feature_width = (
            resized_width // patch_width
        )

        amp_enabled = (
            self._use_amp
            and self._device.type == "cuda"
        )

        expected_token_count = (
            feature_height * feature_width
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                if self._backend == "pytorch_hub":
                    outputs = (
                        self._model.forward_features(
                            image_tensor
                        )
                    )

                elif (
                    self._backend
                    == "huggingface_transformers"
                ):
                    outputs = self._model(
                        pixel_values=image_tensor,
                        return_dict=True,
                    )

                else:
                    raise RuntimeError(
                        "DINOv3 backend가 설정되지 "
                        "않았습니다."
                    )

        if self._backend == "pytorch_hub":
            if not isinstance(outputs, dict):
                raise TypeError(
                    "DINOv3 forward_features() 결과가 "
                    "dict가 아닙니다: "
                    f"{type(outputs)}"
                )

            patch_tokens = outputs.get(
                "x_norm_patchtokens"
            )

            if patch_tokens is None:
                raise KeyError(
                    "DINOv3 출력에 "
                    "x_norm_patchtokens가 없습니다."
                )

        else:
            last_hidden_state = getattr(
                outputs,
                "last_hidden_state",
                None,
            )

            if last_hidden_state is None:
                raise AttributeError(
                    "Hugging Face DINOv3 출력에 "
                    "last_hidden_state가 없습니다."
                )

            config = getattr(
                self._model,
                "config",
                None,
            )
            register_token_count = getattr(
                config,
                "num_register_tokens",
                None,
            )

            if (
                isinstance(
                    register_token_count,
                    bool,
                )
                or not isinstance(
                    register_token_count,
                    int,
                )
                or register_token_count < 0
            ):
                raise ValueError(
                    "Hugging Face DINOv3의 "
                    "num_register_tokens가 올바르지 "
                    "않습니다: "
                    f"{register_token_count}"
                )

            prefix_token_count = (
                1 + register_token_count
            )
            expected_sequence_length = (
                prefix_token_count
                + expected_token_count
            )

            if (
                last_hidden_state.ndim != 3
                or last_hidden_state.shape[1]
                != expected_sequence_length
            ):
                raise ValueError(
                    "Hugging Face DINOv3 token 개수와 "
                    "예상 feature grid가 다릅니다: "
                    f"shape={last_hidden_state.shape}, "
                    f"expected_sequence_length="
                    f"{expected_sequence_length}"
                )

            patch_tokens = (
                last_hidden_state[
                    :,
                    prefix_token_count:,
                    :,
                ]
            )

        if not isinstance(
            patch_tokens,
            torch.Tensor,
        ):
            raise TypeError(
                "DINOv3 patch token이 Tensor가 "
                "아닙니다: "
                f"{type(patch_tokens)}"
            )

        if patch_tokens.ndim != 3:
            raise ValueError(
                "Patch token shape은 "
                "(B, N, C)이어야 합니다: "
                f"{patch_tokens.shape}"
            )

        batch_size, token_count, channel_count = (
            patch_tokens.shape
        )

        if batch_size != 1:
            raise ValueError(
                "현재 extractor는 batch size 1만 "
                "지원합니다: "
                f"{batch_size}"
            )

        if token_count != expected_token_count:
            raise ValueError(
                "Patch token 개수와 예상 feature grid가 "
                "다릅니다: "
                f"tokens={token_count}, "
                f"expected={expected_token_count}, "
                f"grid={(feature_height, feature_width)}"
            )

        feature_map = (
            patch_tokens
            .reshape(
                1,
                feature_height,
                feature_width,
                channel_count,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # cosine similarity 비교를 위해 채널 방향 L2 정규화
        feature_map = F.normalize(
            feature_map.float(),
            p=2,
            dim=1,
            eps=1e-6,
        )

        return feature_map

    @staticmethod
    def _save_result(
        *,
        view: LoadedView,
        model_name: str,
        backend: str,
        weights_path: Path,
        feature_map: torch.Tensor,
        resized_size: tuple[int, int],
        patch_size: tuple[int, int],
        output_directory: Path,
    ) -> DINOFeatureResult:
        """Dense feature map과 metadata를 저장한다."""

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        feature_path = (
            output_directory
            / "dinov3_feature_map.npy"
        )

        metadata_path = (
            output_directory
            / "dinov3_feature_metadata.json"
        )

        feature_numpy = (
            feature_map[0]
            .detach()
            .to(
                device="cpu",
                dtype=torch.float16,
            )
            .numpy()
        )

        feature_numpy = np.ascontiguousarray(
            feature_numpy,
            dtype=np.float16,
        )

        np.save(
            feature_path,
            feature_numpy,
            allow_pickle=False,
        )

        original_height, original_width = (
            view.rgb.shape[:2]
        )

        metadata: dict[str, Any] = {
            "view_name": view.source.name,
            "model_name": model_name,
            "backend": backend,
            "weights_path": str(weights_path),
            "feature_path": str(feature_path),
            "feature_shape": list(
                feature_numpy.shape
            ),
            "feature_dtype": "float16",
            "original_image_size": [
                original_height,
                original_width,
            ],
            "resized_image_size": list(
                resized_size
            ),
            "patch_size": list(
                patch_size
            ),
            "feature_normalization": "L2 channel-wise",
            "input_image": "original_rgb",
            "coordinate_convention": (
                "feature_map shape = "
                "(C, H_patch, W_patch)"
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

        return DINOFeatureResult(
            view_name=view.source.name,
            model_name=model_name,
            feature_map=feature_numpy,
            original_image_size=(
                original_height,
                original_width,
            ),
            resized_image_size=resized_size,
            patch_size=patch_size,
            feature_path=feature_path,
            metadata_path=metadata_path,
        )

    def extract(
        self,
        view: LoadedView,
        output_directory: Path,
    ) -> DINOFeatureResult:
        """
        원본 RGB에서 DINOv3 dense patch feature를 추출한다.

        생성 결과:
            dinov3_feature_map.npy
            dinov3_feature_metadata.json
        """

        self.load()

        if self._patch_size is None:
            raise RuntimeError(
                "DINOv3 patch size가 없습니다."
            )

        if self._backend is None:
            raise RuntimeError(
                "DINOv3 backend가 없습니다."
            )

        image_tensor, resized_size = (
            self._prepare_image(
                rgb=view.rgb
            )
        )

        feature_map = (
            self._forward_patch_features(
                image_tensor=image_tensor,
                resized_size=resized_size,
            )
        )

        result = self._save_result(
            view=view,
            model_name=self._model_name,
            backend=self._backend,
            weights_path=self._checkpoint_path,
            feature_map=feature_map,
            resized_size=resized_size,
            patch_size=self._patch_size,
            output_directory=(
                Path(output_directory)
                .expanduser()
                .resolve()
            ),
        )

        del image_tensor
        del feature_map

        torch.cuda.empty_cache()

        return result

    def release(self) -> None:
        """DINOv3 모델과 CUDA 메모리를 해제한다."""

        self._model = None
        self._patch_size = None
        self._backend = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def __enter__(
        self,
    ) -> DINOv3Extractor:
        self.load()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.release()
