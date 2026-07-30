from __future__ import annotations

from pathlib import Path

from pose_path_entrypoint import (
    run_pose_path_entrypoint,
)


if __name__ == "__main__":
    raise SystemExit(
        run_pose_path_entrypoint(
            pose_path="self_cross",
            entrypoint_path=Path(__file__),
        )
    )
