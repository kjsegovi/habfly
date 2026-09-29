"""Offline paused planet-calculation demo; reuses development, not sealed tests."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from habfly.runtime import RunOptions
from habfly.training.planet_session import load_planet_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/planet_tui.json"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--check", action="store_true", help="Validate locally without launching the TUI")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    payload = json.loads(args.profile.read_text())
    if args.seed is not None:
        payload["seed"] = args.seed
    options = RunOptions.model_validate(payload)
    _, env, provenance = load_planet_session(options)
    env.close()
    if args.check:
        print(json.dumps({"status": "ready", "paused": options.paused, "seed": options.seed, **provenance}))
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("Use an interactive terminal, or --check")
    print("Local supplied-measurement planet calculations only; no browser or training.")
    print("Recorded development case, not a new unseen test. n steps; space resumes; a aborts; q quits.")
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
