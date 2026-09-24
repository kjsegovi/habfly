"""Read-only, hash-checked manual distance sessions from a promoted local experiment."""

import hashlib
import json
from pathlib import Path

from .checkpoints import source_hash


def load_distance_session(directory, checkpoint, pack, seed):
    directory, checkpoint = Path(directory), Path(checkpoint)
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    final = json.loads((directory / "final/report.json").read_text())
    content = manifest["content"]
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rollout = final["closed_loop"]
    if (
        content != report["content"]
        or not report["development_gate_passed"]
        or not final["distance_gate_passed"]
        or final["scope"] != "local_tool_assisted_distance_only"
        or rollout["requested_episodes"] != 100
        or len(rollout["episodes"]) != 100
        or rollout["completed"] < 90
        or any(
            rollout[k] for k in ("invalid_actions", "tool_errors", "api_failures", "infrastructure_failures")
        )
        or checkpoint_hash != report["checkpoint_sha256"]
        or checkpoint_hash != final["checkpoint_sha256"]
        or content["calculation_backend"] != "local"
        or content["required_fields"] != ["distance"]
        or content["max_steps"] != 32
        or content["knowledge_pack_hash"] != pack.checksum
    ):
        raise ValueError("Distance session requires a compatible checkpoint and passed distance gates")
    cases = json.loads((directory / "manual.json").read_text())
    identity = content["workflow_splits"]["manual"]
    if (
        source_hash(cases) != identity["sha256"]
        or len(cases) != identity["count"]
        or [c["seed"] for c in cases] != identity["seeds"]
        or any(c["required"] != ["distance"] for c in cases)
    ):
        raise ValueError("Manual distance dataset mismatch")
    selected = next((c for c in cases if c["seed"] == seed), None)
    if selected is None:
        raise ValueError(f"Choose a manual seed from {identity['seeds']}")
    return content, selected
