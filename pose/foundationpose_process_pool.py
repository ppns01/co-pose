from __future__ import annotations

import atexit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from core.types import PreparedView

if TYPE_CHECKING:
    from pose.foundationpose_runner import (
        FoundationPoseCandidateResult,
    )
    from scale.mesh_scaler import (
        ScaledMeshCandidate,
    )


_WORKER_RUNNER: Any | None = None


@dataclass(frozen=True)
class FoundationPoseProcessJob:
    """독립 프로세스에서 실행할 FoundationPose 후보 하나."""

    job_name: str
    candidate: ScaledMeshCandidate
    prepared_view: PreparedView

    def __post_init__(self) -> None:
        normalized_name = self.job_name.strip()

        if not normalized_name:
            raise ValueError(
                "FoundationPose job_name은 비어 있을 수 없습니다."
            )

        object.__setattr__(
            self,
            "job_name",
            normalized_name,
        )


def _release_worker_runner() -> None:
    global _WORKER_RUNNER

    if _WORKER_RUNNER is not None:
        _WORKER_RUNNER.release()
        _WORKER_RUNNER = None


def _initialize_worker_runner(
    repository_path: Path,
    output_root: Path,
    top_k: int,
    rotation_diversity_threshold_deg: float,
    refine_iterations: int,
    debug: int,
    device: str,
) -> None:
    """Spawn된 자식 프로세스에서 모델을 한 번만 로드한다."""

    global _WORKER_RUNNER

    from pose.foundationpose_runner import (
        FoundationPoseRunner,
    )

    _WORKER_RUNNER = FoundationPoseRunner(
        repository_path=repository_path,
        output_root=output_root,
        top_k=top_k,
        rotation_diversity_threshold_deg=(
            rotation_diversity_threshold_deg
        ),
        refine_iterations=refine_iterations,
        debug=debug,
        device=device,
    )
    _WORKER_RUNNER.load()
    atexit.register(_release_worker_runner)


def _run_worker_job(
    job: FoundationPoseProcessJob,
) -> FoundationPoseCandidateResult:
    runner = _WORKER_RUNNER

    if runner is None:
        raise RuntimeError(
            "FoundationPose worker가 초기화되지 않았습니다."
        )

    print(
        "[FoundationPose job starting] "
        f"job={job.job_name} "
        f"view={job.prepared_view.view.source.name} "
        f"candidate={job.candidate.candidate_index}"
    )

    try:
        return runner.run_candidate(
            candidate=job.candidate,
            prepared_view=job.prepared_view,
        )

    except Exception as error:
        raise RuntimeError(
            "FoundationPose 독립 프로세스 작업에 실패했습니다: "
            f"job={job.job_name}, "
            f"view={job.prepared_view.view.source.name}, "
            f"candidate={job.candidate.candidate_index}, "
            f"mesh={job.candidate.scaled_mesh_path}"
        ) from error


def _validate_jobs(
    jobs: Sequence[FoundationPoseProcessJob],
    worker_count: int,
) -> tuple[FoundationPoseProcessJob, ...]:
    if (
        isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or worker_count < 1
    ):
        raise ValueError(
            "foundationpose_workers는 1 이상의 정수여야 합니다: "
            f"{worker_count}"
        )

    normalized_jobs = tuple(jobs)

    if not normalized_jobs:
        raise ValueError(
            "FoundationPose process job이 없습니다."
        )

    job_names = tuple(
        job.job_name
        for job in normalized_jobs
    )

    if len(set(job_names)) != len(job_names):
        raise ValueError(
            "FoundationPose job_name이 중복되었습니다."
        )

    output_keys = tuple(
        (
            job.prepared_view.view.source.name,
            job.candidate.candidate_index,
        )
        for job in normalized_jobs
    )

    if len(set(output_keys)) != len(output_keys):
        raise ValueError(
            "동일한 view와 candidate index의 FoundationPose "
            "작업이 중복되었습니다."
        )

    return normalized_jobs


def _validate_results(
    jobs: Sequence[FoundationPoseProcessJob],
    results: Sequence[FoundationPoseCandidateResult],
) -> tuple[FoundationPoseCandidateResult, ...]:
    normalized_results = tuple(results)

    if len(normalized_results) != len(jobs):
        raise RuntimeError(
            "FoundationPose job과 결과 개수가 다릅니다: "
            f"jobs={len(jobs)}, "
            f"results={len(normalized_results)}"
        )

    for job, result in zip(
        jobs,
        normalized_results,
        strict=True,
    ):
        expected_view = (
            job.prepared_view.view.source.name
        )

        if result.view_name != expected_view:
            raise RuntimeError(
                "FoundationPose 결과 view가 작업과 다릅니다: "
                f"job={job.job_name}, "
                f"expected={expected_view}, "
                f"actual={result.view_name}"
            )

        if (
            result.candidate_index
            != job.candidate.candidate_index
        ):
            raise RuntimeError(
                "FoundationPose 결과 candidate가 작업과 다릅니다: "
                f"job={job.job_name}, "
                f"expected={job.candidate.candidate_index}, "
                f"actual={result.candidate_index}"
            )

        expected_mesh = (
            Path(job.candidate.scaled_mesh_path)
            .expanduser()
            .resolve()
        )
        actual_mesh = (
            Path(result.scaled_mesh_path)
            .expanduser()
            .resolve()
        )

        if actual_mesh != expected_mesh:
            raise RuntimeError(
                "FoundationPose 결과 mesh가 작업과 다릅니다: "
                f"job={job.job_name}, "
                f"expected={expected_mesh}, "
                f"actual={actual_mesh}"
            )

    return normalized_results


def run_foundationpose_jobs(
    *,
    jobs: Sequence[FoundationPoseProcessJob],
    repository_path: Path,
    output_root: Path,
    top_k: int,
    rotation_diversity_threshold_deg: float = 0.0,
    refine_iterations: int,
    device: str,
    worker_count: int,
    debug: int = 0,
) -> tuple[FoundationPoseCandidateResult, ...]:
    """
    FoundationPose 후보들을 지정한 수의 독립 프로세스로 실행한다.

    worker_count=1이면 현재 프로세스에서 기존 순차 방식으로 실행한다.
    2 이상이면 CUDA fork를 피하기 위해 spawn context를 사용한다.
    반환 순서는 입력 jobs 순서와 동일하다.
    """

    normalized_jobs = _validate_jobs(
        jobs=jobs,
        worker_count=worker_count,
    )

    repository_path = (
        Path(repository_path)
        .expanduser()
        .resolve()
    )
    output_root = (
        Path(output_root)
        .expanduser()
        .resolve()
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if worker_count == 1:
        import torch

        from pose.foundationpose_runner import (
            FoundationPoseRunner,
        )
        from pose.torch_state import (
            preserve_torch_default_tensor_type,
        )

        with (
            preserve_torch_default_tensor_type(
                torch
            ),
            FoundationPoseRunner(
                repository_path=repository_path,
                output_root=output_root,
                top_k=top_k,
                rotation_diversity_threshold_deg=(
                    rotation_diversity_threshold_deg
                ),
                refine_iterations=refine_iterations,
                debug=debug,
                device=device,
            ) as runner,
        ):
            results = tuple(
                runner.run_candidate(
                    candidate=job.candidate,
                    prepared_view=job.prepared_view,
                )
                for job in normalized_jobs
            )

        return _validate_results(
            jobs=normalized_jobs,
            results=results,
        )

    active_workers = min(
        worker_count,
        len(normalized_jobs),
    )

    print(
        "[FoundationPose process pool] "
        f"workers={active_workers}, "
        f"requested={worker_count}, "
        f"jobs={len(normalized_jobs)}"
    )

    process_context = (
        multiprocessing.get_context("spawn")
    )

    try:
        with ProcessPoolExecutor(
            max_workers=active_workers,
            mp_context=process_context,
            initializer=_initialize_worker_runner,
            initargs=(
                repository_path,
                output_root,
                top_k,
                rotation_diversity_threshold_deg,
                refine_iterations,
                debug,
                device,
            ),
        ) as executor:
            results = tuple(
                executor.map(
                    _run_worker_job,
                    normalized_jobs,
                    chunksize=1,
                )
            )

    except Exception as error:
        raise RuntimeError(
            "FoundationPose 독립 프로세스 실행에 실패했습니다: "
            f"workers={active_workers}, "
            f"jobs={len(normalized_jobs)}, "
            f"output_root={output_root}"
        ) from error

    return _validate_results(
        jobs=normalized_jobs,
        results=results,
    )
