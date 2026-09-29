"""Fixed 32-update temperature smoke or explicit 80-update whole-episode diagnostic."""

import argparse
import json
import socket
from pathlib import Path
from unittest.mock import patch

from habfly.data import load_graph
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.habitability_calculations import run_smoke


def deny(*args, **kwargs):
    raise RuntimeError("Offline habitability smoke attempted network or spreadsheet access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--graph", type=Path, default=Path("data/processed/graphs-v2/graph-2000"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--sequence-from", type=Path, help="Reuse a verified smoke dataset/checkpoint; no new cases"
    )
    modes.add_argument("--diagnose-from", type=Path, help="Frozen decision checks on a sequence follow-up")
    parser.add_argument(
        "--planet-parent", type=Path, help="Explicit semantic transfer; requires --sequence-from"
    )
    parser.add_argument(
        "--resume-from", type=Path, help="Explicit same-task 80-update refinement; requires --sequence-from"
    )
    parser.add_argument(
        "--pilot", action="store_true", help="Five epochs on 64 cases after a successful resumed diagnostic"
    )
    args = parser.parse_args()
    if args.planet_parent and not args.sequence_from:
        parser.error("--planet-parent requires --sequence-from")
    if args.resume_from and (not args.sequence_from or args.planet_parent):
        parser.error("--resume-from requires --sequence-from and cannot be mixed with transfer")
    if args.pilot and not args.resume_from:
        parser.error("--pilot requires --resume-from and --sequence-from")
    with (
        patch.object(socket, "create_connection", deny),
        patch.object(socket.socket, "connect", deny),
        patch.object(socket.socket, "connect_ex", deny),
        patch.object(SpreadsheetAdapter, "__init__", deny),
    ):
        progress = lambda row: print(json.dumps(row), flush=True)
        graph = load_graph(args.graph)
        if args.diagnose_from is not None:
            from habfly.training.habitability_sequence import diagnose_followup

            diagnose_followup(args.diagnose_from, graph, args.output, progress=progress)
            print(json.dumps({"report": str(args.output / "report.json"), "optimizer_updates": 0}))
            return
        elif args.sequence_from is None:
            report = run_smoke(graph, args.output, progress=progress)
        else:
            from habfly.training.habitability_sequence import run_sequence_followup

            report = run_sequence_followup(
                args.sequence_from,
                graph,
                args.output,
                progress=progress,
                planet_parent=args.planet_parent,
                resume_from=args.resume_from,
                pilot=args.pilot,
            )
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                "closed_loop": report["closed_loop"],
                "code_smoke_passed": report["code_smoke_passed"],
                "learning_gate_passed": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
