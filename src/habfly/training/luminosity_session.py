"""Read-only promotion gate for the two-output manual TUI, never a live browser."""

import hashlib
import json
from pathlib import Path

from .chained_workflow import workflow_spec
from .checkpoints import source_hash
from .luminosity import ERRORS


def load_luminosity_session(directory, checkpoint, pack, seed):
    return load_chain_session(directory, checkpoint, pack, seed, task="luminosity")


def load_temperature_session(directory, checkpoint, pack, seed):
    return load_chain_session(directory, checkpoint, pack, seed, task="temperature")


def load_chain_session(directory, checkpoint, pack, seed, *, task):
    workflow = workflow_spec(task)
    directory, checkpoint = Path(directory), Path(checkpoint)
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    final = json.loads((directory / "final/report.json").read_text())
    content, rollout = manifest["content"], final["closed_loop"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (
        content != report["content"]
        or not report["development_gate_passed"]
        or not final["chain_gate_passed"]
        or final["scope"] != workflow.scope
        or not final["model_state_unchanged"]
        or final["optimizer_updates"] != 0
        or rollout["requested"] != 100
        or len(rollout["episodes"]) != 100
        or rollout["completed"] < 90
        or rollout["chained"] < 90
        or any(rollout[key] for key in ERRORS)
        or digest != report["checkpoint_sha256"]
        or digest != final["checkpoint_sha256"]
        or content["scope"] != workflow.scope
        or content["calculation_backend"] != "local"
        or content["required_fields"] != list(workflow.required)
        or content["max_steps"] != workflow.max_steps
        or content["knowledge_pack_hash"] != pack.checksum
    ):
        raise ValueError(f"{task.title()} session requires compatible artifacts and passed chain gates")
    cases = json.loads((directory / "manual.json").read_text())
    identity = content["workflow_splits"]["manual"]
    if (
        source_hash(cases) != identity["sha256"]
        or len(cases) != identity["count"]
        or [c["seed"] for c in cases] != identity["seeds"]
        or any(c["required"] != content["required_fields"] for c in cases)
    ):
        raise ValueError(f"Manual {task} dataset mismatch")
    selected = next((c for c in cases if c["seed"] == seed), None)
    if selected is None:
        raise ValueError(f"Choose a manual seed from {identity['seeds']}")
    return content, selected
