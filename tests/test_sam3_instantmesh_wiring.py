from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

import mask_provider
from core.types import LoadedView, ViewInput
from generators.base import MeshGenerationRequest
from generators.instantmesh_generator import (
    InstantMeshGenerator,
)


TEST_DIRECTORY = Path(__file__).resolve().parent


def _make_view(
    root: Path,
    *,
    height: int = 6,
    width: int = 8,
) -> LoadedView:
    return LoadedView(
        source=ViewInput(
            name="reference",
            rgb_path=root / "rgb.png",
            depth_path=root / "depth.png",
            intrinsics_path=root / "scene_camera.json",
            object_name="driller",
            scene_id=8,
            image_id=0,
            object_id=8,
        ),
        rgb=np.full(
            (height, width, 3),
            80,
            dtype=np.uint8,
        ),
        depth_m=np.ones(
            (height, width),
            dtype=np.float32,
        ),
        camera_matrix=np.array(
            [
                [500.0, 0.0, width / 2.0],
                [0.0, 500.0, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        depth_scale_to_m=0.001,
    )


class SAM3PipelineWiringTests(unittest.TestCase):
    def test_sam3_wrapper_forwards_runtime_config(
        self,
    ) -> None:
        fake_module = ModuleType(
            "segmentation.sam3_masker"
        )
        captured: dict[str, object] = {}
        expected_result = object()

        class FakeMasker:
            def __init__(self, predict_function) -> None:
                captured["predict"] = predict_function

            def segment(self, **_: object) -> object:
                return expected_result

        fake_module.SAM3Masker = FakeMasker
        predictor = MagicMock()
        repository = Path("/tmp/sam3")
        checkpoint = repository / "weights" / "sam3.pt"
        bpe = repository / "assets" / "bpe.gz"

        with (
            patch.dict(
                "sys.modules",
                {
                    "segmentation.sam3_masker": (
                        fake_module
                    )
                },
            ),
            patch.object(
                mask_provider,
                "_predict_with_sam3",
                predictor,
            ),
        ):
            result = (
                mask_provider
                .generate_sam3_segmentation(
                    view=object(),
                    output_directory=Path("/tmp/output"),
                    text_prompt="power drill",
                    repository_path=repository,
                    checkpoint_path=checkpoint,
                    bpe_path=bpe,
                    device="cuda:1",
                    use_amp=False,
                    confidence_threshold=0.30,
                )
            )
            image = np.zeros(
                (2, 3, 3),
                dtype=np.uint8,
            )
            captured["predict"](
                image,
                "power drill",
            )

        self.assertIs(result, expected_result)
        predictor.assert_called_once_with(
            image,
            "power drill",
            repository_path=repository,
            checkpoint_path=checkpoint,
            bpe_path=bpe,
            device="cuda:1",
            use_amp=False,
            confidence_threshold=0.30,
        )

    def test_sam3_wrapper_uses_prompt_and_saves_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            root = Path(temp_dir)
            view = _make_view(root)
            prompts: list[str] = []

            def fake_predict(
                image_rgb: np.ndarray,
                text_prompt: str,
                **_: object,
            ) -> dict[str, np.ndarray]:
                prompts.append(text_prompt)
                masks = np.zeros(
                    (2, *image_rgb.shape[:2]),
                    dtype=np.float32,
                )
                masks[0, 1:3, 1:3] = 1.0
                masks[1, 2:5, 2:7] = 1.0
                return {
                    "masks": masks,
                    "scores": np.array(
                        [0.2, 0.9],
                        dtype=np.float32,
                    ),
                }

            with patch.object(
                mask_provider,
                "_predict_with_sam3",
                side_effect=fake_predict,
            ):
                result = (
                    mask_provider
                    .generate_sam3_segmentation(
                        view=view,
                        output_directory=(
                            root / "segmentation"
                        ),
                        text_prompt="power drill",
                    )
                )

            self.assertEqual(prompts, ["power drill"])
            self.assertEqual(
                int(np.count_nonzero(result.mask_bool)),
                15,
            )
            self.assertTrue(result.mask_bool_path.is_file())
            self.assertTrue(result.mask_rgb_path.is_file())
            self.assertTrue(result.overlay_path.is_file())

            metadata = json.loads(
                result.metadata_path.read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(metadata["source"], "sam3")
            self.assertEqual(
                metadata["text_prompt"],
                "power drill",
            )
            self.assertEqual(
                metadata["selected_candidate_index"],
                1,
            )

    def test_instantmesh_receives_sam3_mask_as_alpha(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            root = Path(temp_dir)
            repository = root / "InstantMesh"
            repository.mkdir()
            (repository / "run.py").touch()
            config_path = root / "config.yaml"
            config_path.touch()
            python_path = root / "python"
            python_path.touch()

            image_rgb = np.full(
                (5, 7, 3),
                120,
                dtype=np.uint8,
            )
            segmented_rgb_path = (
                root / "segmented_rgb.png"
            )
            Image.fromarray(image_rgb).save(
                segmented_rgb_path
            )

            mask_bool = np.zeros(
                (5, 7),
                dtype=np.bool_,
            )
            mask_bool[1:4, 2:6] = True
            mask_bool_path = root / "mask_bool.npy"
            np.save(
                mask_bool_path,
                mask_bool,
                allow_pickle=False,
            )
            mask_rgb_path = root / "mask_rgb.png"
            Image.fromarray(
                mask_bool.astype(np.uint8) * 255
            ).save(mask_rgb_path)

            request = MeshGenerationRequest(
                view_name="reference",
                segmented_rgb_path=segmented_rgb_path,
                output_directory=root / "generated",
                mask_bool_path=mask_bool_path,
                mask_rgb_path=mask_rgb_path,
            )
            generator = InstantMeshGenerator(
                repository_path=repository,
                python_executable=python_path,
                config_path=config_path,
                use_rembg=True,
            )

            trusted_input_path = (
                generator
                ._prepare_trusted_alpha_input(
                    request
                )
            )

            with Image.open(
                trusted_input_path
            ) as trusted_image:
                self.assertEqual(
                    trusted_image.mode,
                    "RGBA",
                )
                trusted_rgba = np.asarray(
                    trusted_image
                )

            np.testing.assert_array_equal(
                trusted_rgba[:, :, :3],
                image_rgb,
            )
            np.testing.assert_array_equal(
                trusted_rgba[:, :, 3],
                mask_bool.astype(np.uint8) * 255,
            )

            command = generator._build_command(
                request,
                input_image_path=trusted_input_path,
            )
            self.assertIn(
                str(trusted_input_path.resolve()),
                command,
            )
            self.assertNotIn("--no_rembg", command)
            self.assertEqual(
                generator
                ._get_expected_mesh_path(request)
                .name,
                "segmented_rgb.obj",
            )


if __name__ == "__main__":
    unittest.main()
