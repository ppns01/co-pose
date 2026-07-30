from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType


@contextmanager
def preserve_torch_default_tensor_type(
    torch_module: ModuleType,
) -> Iterator[None]:
    """
    외부 코드가 변경한 PyTorch 기본 tensor type을 호출 뒤 복구한다.

    FoundationPose Utils.py는 실행 중 기본 tensor type을 CUDA로
    변경하므로, 이후 같은 프로세스에서 실행되는 SAM3의 CPU
    pin-memory 경로가 실패하지 않도록 전역 상태를 격리한다.
    """

    previous_tensor_type = (
        torch_module.empty(0).type()
    )

    try:
        yield
    finally:
        current_tensor_type = (
            torch_module.empty(0).type()
        )

        if current_tensor_type != previous_tensor_type:
            torch_module.set_default_tensor_type(
                previous_tensor_type
            )
