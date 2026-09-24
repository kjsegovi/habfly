import hashlib
import json
from functools import partial

import pytest

from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.training.chained_workflow import workflow_spec
from habfly.training.checkpoints import source_hash
from habfly.training.luminosity_session import load_chain_session
from habfly.training.stellar import write_json


@pytest.fixture(params=("luminosity", "temperature"))
def session(tmp_path, request):
    pack = load_knowledge_pack()
    workflow = workflow_spec(request.param)
    cases = workflow.cases("manual", 2, LocalCalculator(pack), offset=50000)
    content = {
        **pack.content_identity(),
        "scope": workflow.scope,
        "required_fields": list(workflow.required),
        "max_steps": workflow.max_steps,
        "workflow_splits": {
            "manual": {"sha256": source_hash(cases), "count": 2, "seeds": [c["seed"] for c in cases]}
        },
    }
    # Loader checks identity only; actual checkpoint loading is separately tested.
    checkpoint = tmp_path / "fixture-checkpoint"
    checkpoint.write_bytes(b"synthetic test artifact, not a learned model")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(tmp_path / "manifest.json", {"content": content})
    write_json(
        tmp_path / "report.json",
        {"content": content, "development_gate_passed": True, "checkpoint_sha256": digest},
    )
    write_json(tmp_path / "manual.json", cases)
    (tmp_path / "final").mkdir()
    write_json(
        tmp_path / "final/report.json",
        {
            "scope": workflow.scope,
            "chain_gate_passed": True,
            "model_state_unchanged": True,
            "optimizer_updates": 0,
            "checkpoint_sha256": digest,
            "closed_loop": {
                "requested": 100,
                "completed": 100,
                "chained": 100,
                "episodes": [{}] * 100,
                "invalid_actions": 0,
                "tool_errors": 0,
                "api_failures": 0,
                "infrastructure_failures": 0,
            },
        },
    )
    return tmp_path, checkpoint, pack, cases, partial(load_chain_session, task=request.param)


def test_manual_session_scope_hash_and_seed(session):
    directory, checkpoint, pack, cases, loader = session
    content, selected = loader(directory, checkpoint, pack, cases[0]["seed"])
    assert selected == cases[0] and content["required_fields"] == cases[0]["required"]
    with pytest.raises(ValueError, match="manual seed"):
        loader(directory, checkpoint, pack, 0)
    cases[0]["expected"]["distance"] = 0
    write_json(directory / "manual.json", cases)
    with pytest.raises(ValueError, match="dataset mismatch"):
        loader(directory, checkpoint, pack, cases[0]["seed"])


@pytest.mark.parametrize(
    "mutation",
    ["gate", "scope", "checkpoint", "count", "completion", "reuse", "errors", "updated_weights", "optimizer"],
)
def test_reject_unready_chain(session, mutation):
    directory, checkpoint, pack, cases, loader = session
    path = directory / "final/report.json"
    final = json.loads(path.read_text())
    if mutation == "gate":
        final["chain_gate_passed"] = False
    elif mutation == "scope":
        final["scope"] = "local_tool_assisted_distance_only"
    elif mutation == "checkpoint":
        final["checkpoint_sha256"] = "different"
    elif mutation == "count":
        final["closed_loop"]["episodes"].pop()
    elif mutation == "completion":
        final["closed_loop"]["completed"] = 89
    elif mutation == "reuse":
        final["closed_loop"]["chained"] = 0
    elif mutation == "errors":
        final["closed_loop"]["tool_errors"] = 1
    elif mutation == "updated_weights":
        final["model_state_unchanged"] = False
    else:
        final["optimizer_updates"] = 1
    write_json(path, final)
    with pytest.raises(ValueError, match="passed chain gates"):
        loader(directory, checkpoint, pack, cases[0]["seed"])
