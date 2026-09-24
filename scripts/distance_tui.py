"""Launch the verified local distance demo paused; n steps, space resumes, q quits."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from habfly.knowledge import load_knowledge_pack
from habfly.runtime import RunOptions
from habfly.training.distance_session import load_distance_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/distance_tui.json"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--check", action="store_true", help="Validate artifacts without launching the TUI")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    payload = json.loads(args.profile.read_text())
    if args.seed is not None:
        payload["seed"] = args.seed
    settings = RunOptions.model_validate(payload)
    if (
        settings.task != "distance"
        or settings.policy != "checkpoint"
        or settings.backend != "local"
        or settings.environment != "simulator"
        or settings.spreadsheet_config
        or not settings.graph
        or not settings.checkpoint
        or not settings.dataset
    ):
        parser.error("This launcher supports only the local distance checkpoint profile")
    load_distance_session(
        settings.dataset, settings.checkpoint, load_knowledge_pack(settings.knowledge_pack), settings.seed
    )
    if args.check:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "scope": "distance_only",
                    "paused": settings.paused,
                    "seed": settings.seed,
                    "checkpoint": str(settings.checkpoint),
                }
            )
        )
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("Run this command in an interactive terminal, or use --check")
    command = [
        "cargo",
        "run",
        "--manifest-path",
        "tui/Cargo.toml",
        "--locked",
        "--offline",
        "--",
        "--python",
        sys.executable,
        "--start-payload",
        json.dumps(payload),
        "--autostart",
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
