"""Frozen offline diagnostics on a planet experiment's existing train/dev data."""

import argparse
import socket
from pathlib import Path
from unittest.mock import patch

import torch

from habfly.data import load_graph
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint
from habfly.training.planet_calculations import print_progress
from habfly.training.planet_diagnostics import diagnose_trajectories
from habfly.training.planet_sequence import file_hash, read
from habfly.training.stellar import write_json
from habfly.training.train import prepare_output_directory


def deny(*_args, **_kwargs):
    raise RuntimeError("Offline diagnostic attempted network or spreadsheet access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--graph", type=Path, default=Path("data/processed/graphs-v2/graph-2000"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    with (
        patch.object(socket, "create_connection", deny),
        patch.object(socket.socket, "connect", deny),
        patch.object(socket.socket, "connect_ex", deny),
        patch.object(SpreadsheetAdapter, "__init__", deny),
    ):
        checkpoint = args.experiment / "training/checkpoint.pt"
        content = read(checkpoint.with_suffix(".pt.json"))["provenance"]["planet_calculations"]
        policy, _ = load_checkpoint(checkpoint, load_graph(args.graph), content_pack=content)
        if policy.observation_encoding != "structured_planet_tool_v1":
            raise ValueError("Not a compatible planet checkpoint")
        checkpoint_hash = file_hash(checkpoint)
        directory = prepare_output_directory(args.output)
        report = {"checkpoint_sha256": checkpoint_hash, "content": content, "splits": {}}
        for split in ("train", "development"):
            path = args.experiment / f"expert-{split}/episodes.json"
            report["splits"][split] = {
                "trajectory_sha256": file_hash(path),
                **diagnose_trajectories(policy, read(path)),
            }
            print_progress({"split": split, **report["splits"][split]})
        if file_hash(checkpoint) != checkpoint_hash:
            raise RuntimeError("Diagnostic modified source checkpoint")
        write_json(directory / "report.json", report)


if __name__ == "__main__":
    main()
