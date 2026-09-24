"""Frozen manual-case counterfactuals: same numbers/instruction, all three classes.

This is diagnostic rehearsal, not another fresh test or classification training.
"""

import argparse
import copy
import hashlib
import json
import socket
from pathlib import Path

import torch

from habfly.data import load_graph
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.chained_workflow import workflow_spec
from habfly.training.checkpoints import load_checkpoint
from habfly.training.luminosity import clean, rollout
from habfly.training.luminosity_session import load_chain_session
from habfly.training.stellar import write_json


def deny(*args, **kwargs):
    raise RuntimeError("Conditional workflow diagnostic attempted network or Sheets access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", type=Path, default=Path("configs/mass_tui.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    settings = json.loads(args.profile.read_text())
    if settings["task"] not in {"mass", "radius"} or settings["calculation_backend"] != "local":
        parser.error("Only local mass/radius checkpoints are supported")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    SpreadsheetAdapter.__init__ = deny
    torch.set_num_threads(1)
    calculator = LocalCalculator(load_knowledge_pack())
    content, _ = load_chain_session(
        settings["dataset"], settings["checkpoint"], calculator.pack, settings["seed"], task=settings["task"]
    )
    checkpoint = Path(settings["checkpoint"])
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    policy, _ = load_checkpoint(checkpoint, load_graph(settings["graph"]), content_pack=content)
    policy.eval().requires_grad_(False)
    tensors = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    original = json.loads((Path(settings["dataset"]) / "manual.json").read_text())[:4]
    workflow = workflow_spec(settings["task"])
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for cls in ("main_sequence", "white_dwarf", "giant"):
        cases = copy.deepcopy(original)
        for case in cases:
            case["star_class"] = cls
            case["required"] = workflow.required_for(cls)
            answers = calculator.reference_answers(case["inputs"], cls)
            case["expected"] = {key: answers[key] for key in case["required"]}
        write_json(args.output / f"{cls}-cases.json", cases)
        reports[cls] = rollout(policy, calculator, cases, args.output / cls, environment=workflow.environment)
    unchanged = hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before and all(
        torch.equal(v, policy.state_dict()[k]) for k, v in tensors.items()
    )
    passed = unchanged and all(clean(r, 4, steps=workflow.expected_steps) for r in reports.values())
    report = {
        "scope": "manual_counterfactual_supplied_class_not_fresh_acceptance",
        "only_changes": ["supplied_star_class", "visible_required_fields", "private_grading_references"],
        "same_numbers_measurement_ids_instruction_and_control_shuffle_seed": True,
        "requested": 12,
        "completed": sum(r["completed"] for r in reports.values()),
        "passed": passed,
        "by_class": reports,
        "checkpoint_sha256": before,
        "model_state_unchanged": unchanged,
        "optimizer_updates": 0,
        "network_and_sheets_blocked": True,
    }
    write_json(args.output / "report.json", report)
    print(json.dumps({"passed": passed, "completed": report["completed"], "requested": 12}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
