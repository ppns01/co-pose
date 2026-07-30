from __future__ import annotations

import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, sentinel

from pose import foundationpose_process_pool
from pose.foundationpose_process_pool import (
    FoundationPoseProcessJob,
    run_foundationpose_jobs,
)


TEST_DIRECTORY = Path(__file__).resolve().parent


def _job(
    *,
    job_name: str,
    view_name: str,
    candidate_index: int,
) -> FoundationPoseProcessJob:
    return FoundationPoseProcessJob(
        job_name=job_name,
        candidate=SimpleNamespace(
            candidate_index=candidate_index,
            scaled_mesh_path=Path(
                f"{view_name}_{candidate_index}.obj"
            ),
        ),
        prepared_view=SimpleNamespace(
            view=SimpleNamespace(
                source=SimpleNamespace(
                    name=view_name,
                ),
            ),
        ),
    )


class _FakeExecutor:
    last_instance: _FakeExecutor | None = None

    def __init__(
        self,
        *,
        max_workers: int,
        mp_context: object,
        initializer: object,
        initargs: tuple[object, ...],
    ) -> None:
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.initializer = initializer
        self.initargs = initargs
        self.chunksize: int | None = None
        _FakeExecutor.last_instance = self

    def __enter__(self) -> _FakeExecutor:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        return None

    def map(
        self,
        function: object,
        jobs: tuple[
            FoundationPoseProcessJob,
            ...,
        ],
        *,
        chunksize: int,
    ) -> list[SimpleNamespace]:
        del function
        self.chunksize = chunksize

        return [
            SimpleNamespace(
                view_name=(
                    job.prepared_view.view.source.name
                ),
                candidate_index=(
                    job.candidate.candidate_index
                ),
                scaled_mesh_path=(
                    job.candidate.scaled_mesh_path
                ),
            )
            for job in jobs
        ]


class _FakeTensor:
    def __init__(self, tensor_type: str) -> None:
        self._tensor_type = tensor_type

    def type(self) -> str:
        return self._tensor_type


class _FakeTorch(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
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


class FoundationPoseProcessPoolTests(
    unittest.TestCase
):
    def test_job_is_pickleable(self) -> None:
        job = _job(
            job_name="query:2",
            view_name="query",
            candidate_index=2,
        )
        restored_job = pickle.loads(
            pickle.dumps(job)
        )

        self.assertEqual(
            (
                restored_job.job_name,
                restored_job
                .prepared_view
                .view
                .source
                .name,
                restored_job
                .candidate
                .candidate_index,
            ),
            ("query:2", "query", 2),
        )

    def test_parallel_jobs_preserve_order_and_cap_workers(
        self,
    ) -> None:
        jobs = (
            _job(
                job_name="reference:0",
                view_name="reference",
                candidate_index=0,
            ),
            _job(
                job_name="query:0",
                view_name="query",
                candidate_index=0,
            ),
            _job(
                job_name="reference:1",
                view_name="reference",
                candidate_index=1,
            ),
        )

        with tempfile.TemporaryDirectory(
            dir=TEST_DIRECTORY,
        ) as temp_dir:
            with (
                patch.object(
                    foundationpose_process_pool,
                    "ProcessPoolExecutor",
                    _FakeExecutor,
                ),
                patch.object(
                    foundationpose_process_pool
                    .multiprocessing,
                    "get_context",
                    return_value=sentinel.spawn_context,
                ) as get_context,
            ):
                results = run_foundationpose_jobs(
                    jobs=jobs,
                    repository_path=Path(temp_dir),
                    output_root=(
                        Path(temp_dir) / "outputs"
                    ),
                    top_k=3,
                    refine_iterations=5,
                    device="cuda:0",
                    worker_count=8,
                )

        executor = _FakeExecutor.last_instance
        self.assertIsNotNone(executor)
        assert executor is not None

        self.assertEqual(executor.max_workers, 3)
        self.assertIs(
            executor.mp_context,
            sentinel.spawn_context,
        )
        self.assertEqual(executor.chunksize, 1)
        get_context.assert_called_once_with("spawn")
        self.assertEqual(
            [
                (
                    result.view_name,
                    result.candidate_index,
                )
                for result in results
            ],
            [
                ("reference", 0),
                ("query", 0),
                ("reference", 1),
            ],
        )

    def test_sequential_runner_restores_default_tensor_type(
        self,
    ) -> None:
        fake_torch = _FakeTorch()

        class FakeRunner:
            def __init__(
                self,
                **_kwargs: object,
            ) -> None:
                pass

            def __enter__(self) -> FakeRunner:
                return self

            def __exit__(
                self,
                exception_type: object,
                exception: object,
                traceback: object,
            ) -> None:
                del exception_type
                del exception
                del traceback

            def run_candidate(
                self,
                *,
                candidate: object,
                prepared_view: object,
            ) -> SimpleNamespace:
                fake_torch.set_default_tensor_type(
                    "torch.cuda.FloatTensor"
                )
                return SimpleNamespace(
                    view_name=(
                        prepared_view
                        .view
                        .source
                        .name
                    ),
                    candidate_index=(
                        candidate.candidate_index
                    ),
                    scaled_mesh_path=(
                        candidate.scaled_mesh_path
                    ),
                )

        job = _job(
            job_name="reference:0",
            view_name="reference",
            candidate_index=0,
        )

        with (
            patch.dict(
                sys.modules,
                {"torch": fake_torch},
            ),
            patch(
                "pose.foundationpose_runner.FoundationPoseRunner",
                FakeRunner,
            ),
        ):
            run_foundationpose_jobs(
                jobs=(job,),
                repository_path=TEST_DIRECTORY,
                output_root=TEST_DIRECTORY,
                top_k=3,
                refine_iterations=5,
                device="cuda:0",
                worker_count=1,
            )

        self.assertEqual(
            fake_torch.default_tensor_type,
            "torch.FloatTensor",
        )

    def test_rejects_duplicate_output_key(self) -> None:
        duplicate_jobs = (
            _job(
                job_name="first",
                view_name="reference",
                candidate_index=0,
            ),
            _job(
                job_name="second",
                view_name="reference",
                candidate_index=0,
            ),
        )

        with self.assertRaises(ValueError):
            run_foundationpose_jobs(
                jobs=duplicate_jobs,
                repository_path=Path("."),
                output_root=Path("."),
                top_k=3,
                refine_iterations=5,
                device="cuda:0",
                worker_count=2,
            )

    def test_rejects_zero_workers(self) -> None:
        with self.assertRaises(ValueError):
            run_foundationpose_jobs(
                jobs=(
                    _job(
                        job_name="reference:0",
                        view_name="reference",
                        candidate_index=0,
                    ),
                ),
                repository_path=Path("."),
                output_root=Path("."),
                top_k=3,
                refine_iterations=5,
                device="cuda:0",
                worker_count=0,
            )


if __name__ == "__main__":
    unittest.main()
