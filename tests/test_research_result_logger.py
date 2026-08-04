from __future__ import annotations

import unittest

from evaluation.research_result_logger import (
    _cuda_version_from_torch_version,
)


class ResearchResultLoggerTests(unittest.TestCase):
    def test_extracts_cuda_version_from_torch_build(
        self,
    ) -> None:
        self.assertEqual(
            _cuda_version_from_torch_version(
                "2.11.0+cu128"
            ),
            "12.8",
        )


if __name__ == "__main__":
    unittest.main()
