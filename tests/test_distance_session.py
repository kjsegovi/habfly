import hashlib
import io
import json

import pytest

from habfly.data import make_demo_graph
from habfly.knowledge import LocalCalculator, load_knowledge_pack
from habfly.model import ConnectomePolicy
from habfly.runtime import Runtime, read_trace
from habfly.training.checkpoints import save_checkpoint, source_hash
from habfly.training.distance_session import load_distance_session
from habfly.training.source_aliases import workflow_cases
from habfly.training.stellar import write_json


@pytest.fixture
def session(tmp_path):
    pack, graph = load_knowledge_pack(), make_demo_graph(16)
    cases = workflow_cases("manual", 2, LocalCalculator(pack))
    content = {
        **pack.content_identity(),
        "required_fields": ["distance"],
        "max_steps": 32,
        "workflow_splits": {
            "manual": {"sha256": source_hash(cases), "count": 2, "seeds": [c["seed"] for c in cases]}
        },
    }
    policy = ConnectomePolicy(
        graph,
        hidden_size=16,
        observation_encoding="structured_tool_v3",
        selection_mode="measurement_source_v2",
    )
    checkpoint = tmp_path / "training/checkpoint.pt"
    save_checkpoint(checkpoint, policy, stage="synthetic_test_only", seed=0, content_pack=content)
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(tmp_path / "manifest.json", {"content": content})
    write_json(tmp_path / "manual.json", cases)
    write_json(
        tmp_path / "report.json",
        {"content": content, "development_gate_passed": True, "checkpoint_sha256": checksum},
    )
    (tmp_path / "final").mkdir()
    write_json(
        tmp_path / "final/report.json",
        {
            "scope": "local_tool_assisted_distance_only",
            "distance_gate_passed": True,
            "checkpoint_sha256": checksum,
            "closed_loop": {
                "requested_episodes": 100,
                "episodes": [{}] * 100,
                "completed": 100,
                "invalid_actions": 0,
                "tool_errors": 0,
                "api_failures": 0,
                "infrastructure_failures": 0,
            },
        },
    )
    return tmp_path, checkpoint, pack, graph, cases


def test_session_hash_scope_and_seed_gates(session):
    directory, checkpoint, pack, _, cases = session
    content, case = load_distance_session(directory, checkpoint, pack, cases[0]["seed"])
    assert content["required_fields"] == ["distance"] and case == cases[0]
    with pytest.raises(ValueError, match="manual seed"):
        load_distance_session(directory, checkpoint, pack, 0)
    cases[0]["instruction"] = "altered"
    write_json(directory / "manual.json", cases)
    with pytest.raises(ValueError, match="dataset mismatch"):
        load_distance_session(directory, checkpoint, pack, cases[0]["seed"])


@pytest.mark.parametrize("mutation", ["failed", "wrong_checkpoint", "missing_episodes", "errors"])
def test_session_rejects_unready_or_wrong_model(session, mutation):
    directory, checkpoint, pack, _, cases = session
    path = directory / "final/report.json"
    final = json.loads(path.read_text())
    if mutation == "failed":
        final["distance_gate_passed"] = False
    elif mutation == "wrong_checkpoint":
        final["checkpoint_sha256"] = "wrong"
    elif mutation == "missing_episodes":
        final["closed_loop"]["episodes"].pop()
    else:
        final["closed_loop"]["tool_errors"] = 1
    write_json(path, final)
    with pytest.raises(ValueError, match="passed distance gates"):
        load_distance_session(directory, checkpoint, pack, cases[0]["seed"])


def test_distance_runtime_pause_step_abort_and_offline_replay(session, monkeypatch):
    import habfly.data
    from habfly.spreadsheet import SpreadsheetAdapter

    directory, checkpoint, _, graph, cases = session
    monkeypatch.setattr(habfly.data, "load_graph", lambda _: graph)

    def forbidden(*args, **kwargs):
        raise AssertionError("No spreadsheet access in distance runtime")

    monkeypatch.setattr(SpreadsheetAdapter, "__init__", forbidden)
    runtime = Runtime(io.StringIO())
    runtime.command(
        {
            "command": "start",
            "payload": {
                "task": "distance",
                "policy": "checkpoint",
                "calculation_backend": "local",
                "dataset": str(directory),
                "checkpoint": str(checkpoint),
                "graph": "synthetic-fixture",
                "seed": cases[0]["seed"],
                "paused": True,
                "artifact_dir": str(directory / "runs"),
            },
        }
    )
    assert runtime.status == "paused" and runtime.env.steps == 0
    assert runtime.observation.instruction == cases[0]["instruction"]
    assert "expected" not in runtime.observation.model_dump_json()
    runtime.command({"command": "step"})
    assert runtime.status == "paused" and runtime.env.steps == 1
    path = runtime.trace_path
    runtime.command({"command": "abort"})
    events = read_trace(path)
    assert next(e for e in events if e.event == "state").payload["stage"] == "distance"
    assert next(e for e in events if e.event == "action_proposed").payload["action_source"] == "checkpoint"
    runtime.command({"command": "replay", "payload": {"path": str(path)}})
    while runtime.status == "running":
        runtime.tick()
    runtime.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"environment": "browser"},
        {"calculation_backend": "google_sheets"},
        {"policy": "expert"},
        {"graph": None},
    ],
)
def test_distance_runtime_rejects_other_backends_before_access(tmp_path, settings):
    runtime = Runtime(io.StringIO())
    with pytest.raises(ValueError):
        runtime.command(
            {
                "command": "start",
                "payload": {
                    "task": "distance",
                    "policy": "checkpoint",
                    "dataset": "missing",
                    "graph": "missing",
                    "checkpoint": "missing",
                    "artifact_dir": str(tmp_path / "never_created"),
                    **settings,
                },
            }
        )
    assert not (tmp_path / "never_created").exists()
