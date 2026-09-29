"""Display/replay only; two sampled declines never certify recurrence or a task."""

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_project_compact_wire import envelope, parent, wrap

from habfly.contracts import RuntimeEvent
from habfly.project_wire import compact_project_event
from habfly.runtime import Runtime

FIXTURES = Path(__file__).resolve().parents[1] / "tui/tests/fixtures"
MODE = "approximate_reference_two_visible_tooltips_v1"
SCOPE = {
    "confirmed_feature_count": 2,
    "observed_interval_count": 1,
    "consistency_redundancy": 0,
    "consecutive_events_assumed": True,
    "recurrence_confirmed": False,
    "observed_recurrence_compatible": False,
    "single_spacing_compatible": True,
}


def reference():
    event = json.loads((FIXTURES / "two-tooltip-reference-measurements.jsonl").read_text().splitlines()[1])
    value = event["payload"]["reference_measurements"]
    value["star"] = "Example"
    return value


@pytest.mark.parametrize("kind", ["hello", "state", "episode_summary"])
def test_compact_reference_and_nested_sensor_scope_survive_without_promoting_child_completion(kind):
    view = reference()
    child = {
        "mode": "cooperative_shallow_transit_measurements",
        "measurement_mode": MODE,
        "phase": "measurements_ready",
        "task_completed": True,  # Deliberately hostile child claim must stay scoped.
        "project_completed": True,
        **SCOPE,
    }
    event, relayed = wrap(
        envelope(kind, child),
        components=("project.campaign", "campaign.owner", "star.workflow", "planet.shallow"),
    )
    source = parent(
        relayed,
        submitted=False,
        reference_measurements=view,
        project_owner={"star": "Example", "star_component": {"component": child}},
    )
    original = deepcopy(source)
    result = compact_project_event(event, source)
    nested = result["project_owner"]["star_component"]["component"]
    assert {key: nested[key] for key in SCOPE} == SCOPE
    assert {key: result["project_wire"]["leaf_summary"][key] for key in SCOPE} == SCOPE
    assert result["reference_measurements"] == view
    assert result["task_completed"] is result["project_completed"] is result["submitted"] is False
    assert result["target_workflows_verified"] is False
    assert not SCOPE.keys() & result.keys()
    assert result["project_wire"]["evidence_receipt"] is False
    assert source == original


@pytest.mark.parametrize("value", [False, 0.0, "0", None])
def test_wire_preserves_invalid_metadata_without_coercing_it_to_a_valid_integer(value):
    view = reference()
    view["consistency_redundancy"] = value
    result = compact_project_event("state", parent({}, reference_measurements=view))
    actual = result["reference_measurements"]["consistency_redundancy"]
    assert type(actual) is type(value) and actual == value


@pytest.mark.parametrize(
    "name,expected",
    [
        ("reference-measurements", "ddc1ec3769fc7f1fe8c4e7944a6c3069a9b34c4a3c928151fb09e8c5e814af39"),
        (
            "tooltip-reference-measurements",
            "fe438eb8396f053c4f9275fa4c0997827a2d76c378b5073d3a263000093b3cfd",
        ),
    ],
)
def test_legacy_reference_projection_bytes_are_unchanged(name, expected):
    event = json.loads((FIXTURES / f"{name}.jsonl").read_text().splitlines()[1])
    value = compact_project_event(event["event"], event["payload"])
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(raw).hexdigest() == expected


@pytest.mark.parametrize("compact", [False, True])
def test_v1_replay_preserves_single_spacing_scope_and_false_completion(tmp_path, monkeypatch, compact):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Display replay cannot start network, environment or model")

    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("habfly.browser.TorusBrowser.__init__", forbidden)
    monkeypatch.setattr("torch.load", forbidden)
    monkeypatch.setattr(Runtime, "start", forbidden)
    view = reference()
    payload = parent({}, submitted=False, task="browser_project", reference_measurements=view)
    if compact:
        payload = compact_project_event("state", payload)
    trace = tmp_path / "two-event.jsonl"
    trace.write_text(
        RuntimeEvent(event="state", sequence=0, run_id="fixture", payload=payload).model_dump_json() + "\n"
    )
    original = trace.read_bytes()
    runtime = Runtime(io.StringIO())
    runtime.command({"command": "replay", "payload": {"path": str(trace)}})
    runtime.tick()
    runtime.tick()
    events = [json.loads(line) for line in runtime.output.getvalue().splitlines()]
    assert events[-1]["payload"]["replay_finished"] is True
    assert events[-1]["payload"]["reference_measurements"] == view
    for key in ("task_completed", "project_completed", "submitted", "target_workflows_verified"):
        assert events[-1]["payload"][key] is False
    assert runtime.env is None and trace.read_bytes() == original
