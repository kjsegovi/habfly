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
    stabilize = sub.add_parser(
        "stabilize", help="Lower-rate continuation with development checkpoint selection"
    )
    stabilize.add_argument("source", type=Path)
    stabilize.add_argument("output", type=Path)
    boundaries = sub.add_parser(
        "boundaries", help="Half-retained, half-new boundary curriculum with original-case regression"
    )
    boundaries.add_argument("source", type=Path)
    boundaries.add_argument("output", type=Path)
    isolated = sub.add_parser(
        "isolate", help="Selected-measurement graph path; one capped existing-case comparison"
    )
    isolated.add_argument("source", type=Path)
    isolated.add_argument("output", type=Path)
    ordered = sub.add_parser(
        "order", help="Train-only ordered readout initialization; capped boundary training"
    )
    ordered.add_argument("source", type=Path)
    ordered.add_argument("output", type=Path)
    final = sub.add_parser("test")
    final.add_argument("experiment", type=Path)
    args = parser.parse_args()
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    from habfly.spreadsheet import SpreadsheetAdapter

    SpreadsheetAdapter.__init__ = deny
    from habfly.color_reference import load_color_reference, validate_color_reference
    from habfly.training.color import refine_color, test_color, train_color
    from habfly.training.color_boundary import train_color_boundaries
    from habfly.training.color_isolation import isolate_color
    from habfly.training.color_ordering import order_color
    from habfly.training.color_stabilization import stabilize_color

    if args.command == "validate":
        result = validate_color_reference(load_color_reference())
    elif args.command in {"train", "refine", "stabilize", "boundaries", "isolate", "order"}:
        if args.command == "train":
            report = train_color(args.output, profile=args.profile)
        elif args.command == "refine":
            report = refine_color(args.source, args.output)
        elif args.command == "stabilize":
            report = stabilize_color(args.source, args.output)
        elif args.command == "isolate":
            report = isolate_color(args.source, args.output)
        elif args.command == "order":
            report = order_color(args.source, args.output)
        else:
            report = train_color_boundaries(args.source, args.output)
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
        if args.command in {"refine", "stabilize", "boundaries", "isolate", "order"}:
            result["cumulative_color_optimizer_updates"] = report["cumulative_color_optimizer_updates"]
            result["frozen_workflow_verified"] = report["frozen_workflow_verified"]
        if args.command in {"stabilize", "boundaries", "isolate", "order"}:
            result["selected_epoch"] = report["selected_epoch"]
            result["selected_checkpoint_optimizer_updates"] = report["selected_checkpoint_optimizer_updates"]
        if args.command in {"boundaries", "isolate", "order"}:
            result["original_cases_completed"] = report["regression"]["completed"]
            result["regression_gate_passed"] = report["regression_gate_passed"]
        if args.command == "order":
            result["closed_form_fits"] = report["closed_form_fits"]
            result["ordering_after"] = report["ordering_after"]
    else:
        report = test_color(args.experiment)
        result = {key: report[key] for key in ("episodes", "completed", "browser_eligible")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
