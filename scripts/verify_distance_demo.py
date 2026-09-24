"""Exercise the real checkpoint through the TUI's Python runtime, offline and without training."""

import argparse
import hashlib
import io
import json
import socket
import subprocess
import sys
from pathlib import Path

from habfly.runtime import Runtime, read_trace
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.chained_workflow import workflow_spec
from habfly.training.stellar import write_json


def deny(*args, **kwargs):
    raise RuntimeError("Demo verification attempted network or spreadsheet access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", type=Path, default=Path("configs/distance_tui.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh verification directory")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    SpreadsheetAdapter.__init__ = deny
    payload = json.loads(args.profile.read_text())
    manifest = json.loads((Path(payload["dataset"]) / "manifest.json").read_text())
    checkpoint = Path(payload["checkpoint"])
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    summaries = []
    for seed in manifest["content"]["workflow_splits"]["manual"]["seeds"]:
        runtime = Runtime(io.StringIO())
        try:
            runtime.command(
                {
                    "command": "start",
                    "payload": {
                        **payload,
                        "seed": seed,
                        "paused": True,
                        "artifact_dir": str(args.output / "traces"),
                    },
                }
            )
            expected_steps = (
                10
                if payload["task"] == "distance"
                else workflow_spec(payload["task"]).expected_steps(runtime.env.case)
            )
            assert runtime.status == "paused" and runtime.env.steps == 0
            runtime.command({"command": "step"})
            assert runtime.status == "paused" and runtime.env.steps == 1
            runtime.command({"command": "resume"})
            runtime.tick()
            runtime.command({"command": "pause"})
            assert runtime.env.steps == 2
            runtime.command({"command": "step"})
            assert runtime.status == "paused" and runtime.env.steps == 3
            runtime.command({"command": "resume"})
            while runtime.status == "running":
                runtime.tick()
            assert runtime.status == "completed" and runtime.env.steps == expected_steps
            saved = args.output / f"{seed}.events.jsonl"
            runtime.command({"command": "save_trace", "payload": {"path": str(saved)}})
            events = read_trace(saved)
            summary = next(e.payload for e in events if e.event == "episode_summary")
            assert summary["completed"] and summary["policy"] == "checkpoint"
            actions = [e.payload for e in events if e.event == "action_proposed"]
            neural = [e.payload for e in events if e.event == "neural_activity"]
            assert len(actions) == len(neural) == expected_steps
            assert all(a["action_source"] == "checkpoint" for a in actions)
            assert all(
                a["calibrated"] == (runtime.policy.calibration.get("status") == "calibrated") for a in actions
            )
            assert all(n["top_neurons"] and n["activity_source"] == "checkpoint" for n in neural)
            assert not any(e.event == "error" for e in events)
            replay = subprocess.run(
                [sys.executable, "scripts/habfly_offline.py", "replay", str(saved)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            assert [json.loads(line) for line in replay.stdout.splitlines()] == [
                json.loads(line) for line in saved.read_text().splitlines()
            ]
            summaries.append({"seed": seed, **summary, "events": len(events), "exact_offline_replay": True})
            write_json(args.output / "progress.json", summaries)
        finally:
            runtime.close()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    write_json(
        args.output / "report.json",
        {
            "scope": "manual_demo_runtime_verification_not_new_test",
            "completed": len(summaries),
            "requested": 12,
            "episodes": summaries,
            "checkpoint_sha256": before,
            "checkpoint_unchanged": True,
            "optimizer_updates": 0,
            "pause_step_resume_save_replay_verified": True,
            "network_and_sheets_blocked": True,
        },
    )
    print(
        json.dumps(
            {
                "completed": len(summaries),
                "runtime_verified": True,
                "report": str(args.output / "report.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
