from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


POSE_PATHS = (
    "self_mesh",
    "self_cross",
    "cross_mesh",
)


def run_pose_path_entrypoint(
    *,
    pose_path: str,
    entrypoint_path: Path,
    argv: Sequence[str] | None = None,
) -> int:
    if pose_path not in POSE_PATHS:
        raise ValueError(
            f"Unsupported pose path: {pose_path}"
        )

    materialized_argv = (
        list(argv)
        if argv is not None
        else sys.argv[1:]
    )

    if any(
        argument == "--pose-path"
        or argument.startswith("--pose-path=")
        for argument in materialized_argv
    ):
        raise ValueError(
            "A method-specific main file cannot be "
            "combined with --pose-path."
        )

    all_linemod_count = sum(
        argument == "--all-linemod"
        for argument in materialized_argv
    )

    if all_linemod_count > 1:
        raise ValueError(
            "--all-linemod may be specified only once."
        )

    if all_linemod_count == 1:
        from linemod_all_runner import (
            run_all_linemod_sequence,
        )

        cleaned_argv = [
            argument
            for argument in materialized_argv
            if argument != "--all-linemod"
        ]
        return run_all_linemod_sequence(
            pose_path=pose_path,
            entrypoint_path=entrypoint_path,
            argv=cleaned_argv,
        )

    from main import main

    return main(
        [
            *materialized_argv,
            "--pose-path",
            pose_path,
        ]
    )
