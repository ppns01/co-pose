from __future__ import annotations

import sys
import types
import unittest


for optional_module_name in (
    "cv2",
    "trimesh",
):
    try:
        __import__(optional_module_name)
    except ModuleNotFoundError:
        sys.modules[optional_module_name] = (
            types.ModuleType(optional_module_name)
        )


from pose.torch_state import (
    preserve_torch_default_tensor_type,
)


class _FakeTensor:
    def __init__(self, tensor_type: str) -> None:
        self._tensor_type = tensor_type

    def type(self) -> str:
        return self._tensor_type


class _FakeTorch:
    def __init__(self) -> None:
        self.default_tensor_type = (
            "torch.FloatTensor"
        )

    def empty(self, _size: int) -> _FakeTensor:
        return _FakeTensor(
            self.default_tensor_type
        )

    def set_default_tensor_type(
        self,
        tensor_type: str,
    ) -> None:
        self.default_tensor_type = tensor_type


class PreserveTorchDefaultTensorTypeTest(
    unittest.TestCase
):
    def test_restores_tensor_type_after_success(
        self,
    ) -> None:
        fake_torch = _FakeTorch()

        with preserve_torch_default_tensor_type(
            fake_torch
        ):
            fake_torch.set_default_tensor_type(
                "torch.cuda.FloatTensor"
            )

        self.assertEqual(
            fake_torch.default_tensor_type,
            "torch.FloatTensor",
        )

    def test_restores_tensor_type_after_failure(
        self,
    ) -> None:
        fake_torch = _FakeTorch()

        with self.assertRaisesRegex(
            RuntimeError,
            "render failed",
        ):
            with preserve_torch_default_tensor_type(
                fake_torch
            ):
                fake_torch.set_default_tensor_type(
                    "torch.cuda.FloatTensor"
                )
                raise RuntimeError(
                    "render failed"
                )

        self.assertEqual(
            fake_torch.default_tensor_type,
            "torch.FloatTensor",
        )


if __name__ == "__main__":
    unittest.main()
