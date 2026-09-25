"""Offline peak-wavelength color checkpoint. Never opens or writes to HabWorlds."""

import argparse
import json
import socket
from pathlib import Path


def deny(*args, **kwargs):
    raise RuntimeError("Offline color experiment attempted network or Sheets access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    train = sub.add_parser("train")
    train.add_argument("output", type=Path)
    train.add_argument("--profile", choices=("smoke", "pilot"), default="smoke")
    refine = sub.add_parser(
        "refine", help="One frozen-workflow, color-head-only comparison on existing cases"
    )
    refine.add_argument("source", type=Path)
    refine.add_argument("output", type=Path)
    final = sub.add_parser("test")
    final.add_argument("experiment", type=Path)
    args = parser.parse_args()
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    from habfly.spreadsheet import SpreadsheetAdapter

    SpreadsheetAdapter.__init__ = deny
    from habfly.color_reference import load_color_reference, validate_color_reference
    from habfly.training.color import refine_color, test_color, train_color

    if args.command == "validate":
        result = validate_color_reference(load_color_reference())
    elif args.command in {"train", "refine"}:
        report = (
            refine_color(args.source, args.output)
            if args.command == "refine"
            else train_color(args.output, profile=args.profile)
        )
        result = {
            key: report[key]
            for key in (
                "optimizer_updates",
                "checkpoint_reload_verified",
                "ready_for_final_test",
                "browser_eligible",
            )
        }
        result.update(
            train_completed=report["train"]["completed"],
            development_completed=report["development"]["completed"],
            report=str(args.output / "report.json"),
        )
        if args.command == "refine":
            result["cumulative_color_optimizer_updates"] = report["cumulative_color_optimizer_updates"]
            result["frozen_workflow_verified"] = report["frozen_workflow_verified"]
    else:
        report = test_color(args.experiment)
        result = {key: report[key] for key in ("episodes", "completed", "browser_eligible")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
