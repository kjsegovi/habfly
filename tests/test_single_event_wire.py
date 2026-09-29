"""Single-event display evidence stays scoped and never grants completion."""
# ruff: noqa: F401

import io
import json
from copy import deepcopy

import pytest
from test_project_window_wait_wire import nested_window, offline, parent, relay_window, window

from habfly.contracts import RuntimeEvent
from habfly.project_wire import compact_project_event
from habfly.runtime import Runtime

VERSION = "user_approved_5000_day_single_event_no_planet_v1"
POLICY = {"version": VERSION, "sha256": "a" * 64, "omitted_recipe": {"exact": "raw journal"}}
ANALYSIS = {
    "status": "assume_no_planet",
    "reason": "user_approved_single_event_shortcut",
    "approximation": "user_approved_single_event_no_planet_shortcut",
    "visible_candidate_events": 1,
    "possible_planet_ignored": True,
}


@pytest.mark.parametrize("kind", ["hello", "state", "episode_summary"])
def test_recorded_policy_and_actual_shortcut_remain_nested(kind):
    child = window(policy=POLICY, analysis={**ANALYSIS, "single_event_evidence": {"large": []}})
    payload = parent(child)
    before = deepcopy(payload)
    compact = compact_project_event(kind, payload)
    result = nested_window(compact)
    assert result["policy"] == {key: POLICY[key] for key in ("version", "sha256")}
    assert result["analysis"] == ANALYSIS
    assert not ((ANALYSIS.keys() - {"status"}) & compact.keys())
    assert compact["status"] == "running"
    assert compact["task_completed"] is compact["project_completed"] is compact["submitted"] is False
    assert payload == before
    event, relay = relay_window(kind, child)
    compact = compact_project_event(event, parent(window(), relay))
    assert compact["project_wire"]["leaf_summary"]["analysis"] == ANALYSIS
    assert "analysis" not in compact


def test_enabled_policy_without_shortcut_does_not_invent_ignored_feature():
    compact = compact_project_event(
        "state",
        parent(
            window(
                policy=POLICY, analysis={"status": "assume_no_planet", "reason": "no_below_baseline_pixels"}
            )
        ),
    )
    assert nested_window(compact)["analysis"] == {
        "status": "assume_no_planet",
        "reason": "no_below_baseline_pixels",
    }
    assert "possible_planet_ignored" not in nested_window(compact)["analysis"]


@pytest.mark.parametrize("policy", [None, {}, {"version": "older_policy"}])
def test_legacy_projection_retains_same_summary_and_no_empty_analysis(policy):
    base = compact_project_event("state", parent(window()))
    legacy = compact_project_event("state", parent(window(policy=policy, analysis=ANALYSIS)))
    # Source-event integrity hashes naturally retain the distinct original bytes;
    # the display summary itself remains exactly unchanged for old policies.
    assert nested_window(legacy) == nested_window(base)
    assert "analysis" not in legacy["project_wire"]["leaf_summary"]


def test_project_v1_replay_preserves_exact_new_nested_view_and_source_bytes(tmp_path):
    payload = compact_project_event("state", parent(window(policy=POLICY, analysis=ANALYSIS)))
    path = tmp_path / "trace.jsonl"
    raw = RuntimeEvent(event="state", sequence=0, run_id="fixture", payload=payload).model_dump_json() + "\n"
    path.write_text(raw)
    runtime = Runtime(io.StringIO())
    runtime.command({"command": "replay", "payload": {"path": str(path)}})
    for _ in range(2):
        runtime.tick()
    emitted = [json.loads(line) for line in runtime.output.getvalue().splitlines()]
    assert any(
        item["event"] == "state" and item["payload"].get("project_owner") == payload["project_owner"]
        for item in emitted
    )
    assert path.read_text() == raw
    assert all(item["payload"].get("task_completed", False) is False for item in emitted)
