from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from features.dinov3_extractor import (
    DINOv3Extractor,
)


class _FakeHuggingFaceModel(torch.nn.Module):
    def __init__(
        self,
        patch_tokens: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="dinov3_vit",
            hidden_size=1024,
            patch_size=16,
            num_register_tokens=4,
        )
        self.patch_tokens = patch_tokens

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        del pixel_values
        self.assert_return_dict = return_dict
        prefix = torch.full(
            (
                self.patch_tokens.shape[0],
                5,
                self.patch_tokens.shape[2],
            ),
            99.0,
            dtype=self.patch_tokens.dtype,
        )
        return SimpleNamespace(
            last_hidden_state=torch.cat(
                (prefix, self.patch_tokens),
                dim=1,
            )
        )


class _FakeTorchHubModel(torch.nn.Module):
    def forward_features(
        self,
        image_tensor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del image_tensor
        return {
            "x_norm_patchtokens": torch.ones(
                (1, 6, 3),
                dtype=torch.float32,
            )
        }


class DINOv3ExtractorTests(unittest.TestCase):
    @staticmethod
    def _extractor() -> DINOv3Extractor:
        return DINOv3Extractor(
            repository_path=Path("."),
            checkpoint_path=Path("."),
            model_name="dinov3_vitl16",
            device="cuda:0",
            use_amp=False,
        )

    def test_reads_patch_size_from_hf_config(
        self,
    ) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(
                patch_size=16,
            )
        )

        self.assertEqual(
            DINOv3Extractor._read_patch_size(
                model
            ),
            (16, 16),
        )

    def test_hf_output_removes_cls_and_register_tokens(
        self,
    ) -> None:
        patch_tokens = (
            torch.arange(
                1,
                13,
                dtype=torch.float32,
            )
            .reshape(1, 6, 2)
        )
        model = _FakeHuggingFaceModel(
            patch_tokens
        )
        extractor = self._extractor()
        extractor._model = model
        extractor._patch_size = (16, 16)
        extractor._backend = (
            "huggingface_transformers"
        )

        feature_map = (
            extractor._forward_patch_features(
                image_tensor=torch.zeros(
                    (1, 3, 32, 48),
                    dtype=torch.float32,
                ),
                resized_size=(32, 48),
            )
        )

        expected = (
            patch_tokens
            .reshape(1, 2, 3, 2)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        expected = F.normalize(
            expected,
            p=2,
            dim=1,
            eps=1e-6,
        )

        self.assertTrue(
            model.assert_return_dict
        )
        self.assertEqual(
            tuple(feature_map.shape),
            (1, 2, 2, 3),
        )
        torch.testing.assert_close(
            feature_map,
            expected,
        )

    def test_torch_hub_patch_output_still_works(
        self,
    ) -> None:
        extractor = self._extractor()
        extractor._model = _FakeTorchHubModel()
        extractor._patch_size = (16, 16)
        extractor._backend = "pytorch_hub"

        feature_map = (
            extractor._forward_patch_features(
                image_tensor=torch.zeros(
                    (1, 3, 32, 48),
                    dtype=torch.float32,
                ),
                resized_size=(32, 48),
            )
        )

        self.assertEqual(
            tuple(feature_map.shape),
            (1, 3, 2, 3),
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(
                feature_map,
                dim=1,
            ),
            torch.ones((1, 2, 3)),
        )

    def test_hf_output_rejects_wrong_token_count(
        self,
    ) -> None:
        model = _FakeHuggingFaceModel(
            torch.ones(
                (1, 5, 2),
                dtype=torch.float32,
            )
        )
        extractor = self._extractor()
        extractor._model = model
        extractor._patch_size = (16, 16)
        extractor._backend = (
            "huggingface_transformers"
        )

        with self.assertRaisesRegex(
            ValueError,
            "expected_sequence_length",
        ):
            extractor._forward_patch_features(
                image_tensor=torch.zeros(
                    (1, 3, 32, 48),
                    dtype=torch.float32,
                ),
                resized_size=(32, 48),
            )

    def test_hf_loader_is_local_and_safetensors_only(
        self,
    ) -> None:
        patch_tokens = torch.ones(
            (1, 6, 2),
            dtype=torch.float32,
        )
        model = _FakeHuggingFaceModel(
            patch_tokens
        )
        auto_model = SimpleNamespace(
            from_pretrained=MagicMock(
                return_value=model
            )
        )
        transformers_module = ModuleType(
            "transformers"
        )
        transformers_module.AutoModel = auto_model

        extractor = self._extractor()

        with (
            patch.dict(
                sys.modules,
                {
                    "transformers": (
                        transformers_module
                    )
                },
            ),
            patch.object(
                Path,
                "is_file",
                return_value=True,
            ),
        ):
            loaded_model = (
                extractor
                ._load_huggingface_model()
            )

        self.assertIs(
            loaded_model,
            model,
        )
        auto_model.from_pretrained.assert_called_once_with(
            str(extractor._checkpoint_path),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )

    def test_torch_hub_loader_keeps_legacy_contract(
        self,
    ) -> None:
        extractor = self._extractor()
        model = _FakeTorchHubModel()

        with (
            patch.object(
                Path,
                "is_dir",
                return_value=True,
            ),
            patch.object(
                Path,
                "is_file",
                return_value=True,
            ),
            patch(
                "features.dinov3_extractor"
                ".torch.hub.load",
                return_value=model,
            ) as hub_load,
        ):
            loaded_model = (
                extractor
                ._load_torch_hub_model()
            )

        self.assertIs(
            loaded_model,
            model,
        )
        hub_load.assert_called_once_with(
            repo_or_dir=str(
                extractor._repository_path
            ),
            model="dinov3_vitl16",
            source="local",
            weights=str(
                extractor._checkpoint_path
            ),
        )


if __name__ == "__main__":
    unittest.main()
