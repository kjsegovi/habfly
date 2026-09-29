"""One gated, frozen 100-case temperature test; no browser, network or training."""

import argparse
import json
import socket
from pathlib import Path
from unittest.mock import patch

from habfly.data import load_graph
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.habitability_evaluation import run_final_evaluation


def deny(*args, **kwargs):
    raise RuntimeError("Offline temperature evaluation attempted network or spreadsheet access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--graph", type=Path, default=Path("data/processed/graphs-v2/graph-2000"))
    args = parser.parse_args()
    with (
        patch.object(socket, "create_connection", deny),
        patch.object(socket.socket, "connect", deny),
        patch.object(socket.socket, "connect_ex", deny),
        patch.object(SpreadsheetAdapter, "__init__", deny),
    ):
        run_final_evaluation(
            args.pilot,
            load_graph(args.graph),
            args.output,
            registry=Path("experiments"),
            progress=lambda row: print(json.dumps(row), flush=True),
        )


if __name__ == "__main__":
    main()
