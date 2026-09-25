"""Inspect a local color checkpoint, including failed experiments. Never opens a browser."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from habfly.training.color import GRAPH, load_color_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--seed", type=int, default=10000000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _, _, report = load_color_experiment(args.experiment)
    if not 10000000 <= args.seed <= 10000099:
        parser.error("Manual seeds are 10000000 through 10000099")
    print(
        json.dumps(
            {
                "scope": report["scope"],
                "experimental_local_checkpoint": True,
                "ready_for_final_test": report["ready_for_final_test"],
                "browser": "disabled",
            }
        ),
        flush=True,
    )
    if args.check:
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("Use an interactive terminal or --check")
    payload = {
        "task": "color",
        "policy": "checkpoint",
        "environment": "simulator",
        "calculation_backend": "local",
        "graph": str(GRAPH),
        "dataset": str(args.experiment),
        "checkpoint": str(args.experiment / "training/checkpoint.pt"),
        "seed": args.seed,
        "paused": True,
        "artifact_dir": "experiments/color-runs",
    }
    raise SystemExit(
        subprocess.call(
            [
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
        )
    )


if __name__ == "__main__":
    main()
