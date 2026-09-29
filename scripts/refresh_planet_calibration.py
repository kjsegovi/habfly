"""Refresh frozen planet confidence on recorded cases only; no browser or training."""

import argparse
import json
import socket
from pathlib import Path
from unittest.mock import patch

from habfly.data import load_graph
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.planet_recalibration import refresh_planet_calibration


def deny(*args, **kwargs):
    raise RuntimeError("Offline recalibration attempted network or spreadsheet access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--graph", type=Path, default=Path("data/processed/graphs-v2/graph-2000"))
    args = parser.parse_args()
    with (
        patch.object(socket, "create_connection", deny),
        patch.object(socket.socket, "connect", deny),
        patch.object(socket.socket, "connect_ex", deny),
        patch.object(SpreadsheetAdapter, "__init__", deny),
    ):
        report = refresh_planet_calibration(
            args.source,
            load_graph(args.graph),
            args.output,
            progress=lambda row: print(json.dumps(row), flush=True),
        )
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                "development": report["development"],
                "optimizer_updates": 0,
                "final_test_episodes": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
