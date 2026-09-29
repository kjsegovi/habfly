"""One explicit bounded local planet experiment. No browser or spreadsheet access."""

import argparse
import socket
from pathlib import Path
from unittest.mock import patch

from habfly.data import load_graph
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.planet_calculations import print_progress, run_smoke


def deny(*_args, **_kwargs):
    raise RuntimeError("Offline planet smoke attempted network or spreadsheet access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--graph", type=Path, default=Path("data/processed/graphs-v2/graph-2000"))
    parser.add_argument(
        "--sequence-from", type=Path, help="Explicit 80-update follow-up from a completed smoke directory"
    )
    parser.add_argument(
        "--stellar-parent", type=Path, help="Explicit stellar v6 transfer for the same 80-update experiment"
    )
    parser.add_argument(
        "--period-from",
        type=Path,
        help="Explicit 80-update period-conversion prerequisite from a planet experiment",
    )
    parser.add_argument(
        "--curriculum-from",
        type=Path,
        help="Explicit parent directory for the next 80-update curriculum stage",
    )
    parser.add_argument("--stage", choices=("orbit", "mass", "radius", "full"))
    parser.add_argument(
        "--pilot-from", type=Path, help="Full-scope diagnostic parent; 64 cases, five epochs, 320 updates"
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Explicit additional 80 updates from a completed same-stage parent",
    )
    parser.add_argument(
        "--resume-optimizer", action="store_true", help="Preserve same-stage AdamW moments; requires --refine"
    )
    args = parser.parse_args()
    if args.pilot_from and any(
        (
            args.sequence_from,
            args.stellar_parent,
            args.period_from,
            args.curriculum_from,
            args.stage,
            args.refine,
            args.resume_optimizer,
        )
    ):
        parser.error("--pilot-from is a separate fixed-budget experiment mode")
    if bool(args.curriculum_from) != bool(args.stage):
        parser.error("--curriculum-from and --stage must be supplied together")
    if args.refine and not args.curriculum_from:
        parser.error("--refine requires --curriculum-from and --stage")
    if args.resume_optimizer and not args.refine:
        parser.error("--resume-optimizer requires --refine")
    if args.curriculum_from and (args.period_from or args.sequence_from or args.stellar_parent):
        parser.error("Choose exactly one explicit experiment mode")
    if args.period_from and (args.sequence_from or args.stellar_parent):
        parser.error("Choose the period diagnostic or the full-sequence diagnostic, not both")
    if args.stellar_parent and not args.sequence_from:
        parser.error("--stellar-parent requires --sequence-from for validated planet demonstrations")
    with (
        patch.object(socket, "create_connection", deny),
        patch.object(socket.socket, "connect", deny),
        patch.object(socket.socket, "connect_ex", deny),
        patch.object(SpreadsheetAdapter, "__init__", deny),
    ):
        if args.pilot_from:
            from habfly.training.planet_period import run_planet_pilot

            report = run_planet_pilot(
                args.pilot_from, load_graph(args.graph), args.output, progress=print_progress
            )
        elif args.curriculum_from:
            from habfly.training.planet_period import run_curriculum_diagnostic

            report = run_curriculum_diagnostic(
                args.curriculum_from,
                load_graph(args.graph),
                args.output,
                stage=args.stage,
                progress=print_progress,
                refine=args.refine,
                resume_optimizer=args.resume_optimizer,
            )
        elif args.period_from:
            from habfly.training.planet_period import run_period_diagnostic

            report = run_period_diagnostic(
                args.period_from, load_graph(args.graph), args.output, progress=print_progress
            )
        elif args.sequence_from:
            from habfly.training.planet_sequence import run_sequence_followup

            report = run_sequence_followup(
                args.sequence_from,
                load_graph(args.graph),
                args.output,
                progress=print_progress,
                stellar_parent=args.stellar_parent,
            )
        else:
            report = run_smoke(load_graph(args.graph), args.output, progress=print_progress)
    print_progress({"report": str(args.output / "report.json"), "closed_loop": report["closed_loop"]})


if __name__ == "__main__":
    main()
