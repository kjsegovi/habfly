"""Optional window-wait display contract; no browser, model or live run."""

import hashlib
import io
import json
from copy import deepcopy

import pytest

from habfly.contracts import RuntimeEvent
from habfly.project_events import ProjectEventRelay
from habfly.project_wire import compact_project_event
from habfly.runtime import Runtime, read_trace

RULE = "complete_rendered_window_before_positive_handoff_v1"
WAIT = {
    "scheduling_rule": RULE,
    "prior_dip_observed": True,
    "waiting_for_positive_endpoint": True,
}
LIFECYCLE = ("hello", "state", "episode_summary")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Window-wire tests must not start an environment, model or network")

    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("habfly.browser.TorusBrowser.__init__", forbidden)
    monkeypatch.setattr("torch.load", forbidden)
    monkeypatch.setattr(Runtime, "start", forbidden)


def window(*, waiting=True, **changes):
    return {
        "mode": "bounded_planet_window",
        "phase": "observing",
        "observation_limit_days": 5000,
        "polls": 3,
        "max_polls": 20,
        "window_status": "dip_observed",
        "task_completed": False,
        **(WAIT if waiting else {}),
        **changes,
    }


def parent(chart, relay=None):
    return {
        **(relay or {}),
        "task": "browser_project",
        "runtime_task": "browser_project",
        "status": "running",
        "phase": "active",
        "browser_phase": "active",
        "finished": False,
        "task_completed": False,
        "project_completed": False,
        "submitted": False,
        "target_workflows_verified": False,
        "current_star": {"star": "Example", "ordinal": 1},
        "project_progress": {"collected": 1, "verified": 0},
        "project_owner": {
            "star": "Example",
            "phase": "active",
            "task_completed": False,
            "star_component": {"phase": "planet_window", "component": chart},
        },
    }


def nested_window(payload):
    return payload["project_owner"]["star_component"]["component"]


def relay_window(kind, payload):
    current = RuntimeEvent(event=kind, run_id="window", sequence=2, payload=payload).model_dump(mode="json")
    for component in reversed(("project.campaign", "campaign.owner", "star.workflow", "planet.window")):
        emitted = []
        relay = ProjectEventRelay(lambda *args, output=emitted: output.append(args))
        forward = relay.bind(component, star="Example")
        if set(current) == {"event", "payload"}:
            forward(current["event"], current["payload"])
        else:
            forward(current)
        next_kind, next_payload = emitted.pop()
        current = {"event": next_kind, "payload": next_payload}
    return current["event"], current["payload"]


@pytest.mark.parametrize("kind", LIFECYCLE)
@pytest.mark.parametrize("prior,waiting", [(False, False), (True, False), (True, True)])
def test_optional_wait_fields_survive_only_at_their_nested_window_location(kind, prior, waiting):
    chart = window(prior_dip_observed=prior, waiting_for_positive_endpoint=waiting)
    source = parent(chart)
    before = deepcopy(source)
    result = compact_project_event(kind, source)
    assert nested_window(result) == chart
    assert all(key not in result for key in WAIT)
    assert result["task_completed"] is result["project_completed"] is result["submitted"] is False
    assert result["project_progress"] == {"collected": 1, "verified": 0}
    nested_window(result)["prior_dip_observed"] = not prior
    assert source == before


@pytest.mark.parametrize("kind", LIFECYCLE)
def test_child_window_claims_never_promote_parent_completion(kind):
    child = window(task_completed=True, project_completed=True, submitted=True, finished=True)
    event, relay = relay_window(kind, child)
    result = compact_project_event(event, parent(window(), relay))
    leaf = result["project_wire"]["leaf_summary"]
    assert {key: leaf[key] for key in WAIT} == WAIT
    assert leaf["task_completed"] is True
    for key in ("finished", "task_completed", "project_completed", "submitted", "target_workflows_verified"):
        assert result[key] is False
    assert result["status"] == "running" and result["phase"] == "active"
    assert result["project_wire"]["evidence_receipt"] is False


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("hello", "9ef38558a17b5bbbee1377f9db86a7a7a6048a369d87c5ab5b65df82dbb52067"),
        ("state", "811048d5d4cab01b5a8ad5a3f77e35784dd203d9a02b95314cfec931ecc69e31"),
        ("episode_summary", "f94f172b1b983a9bfe036a493bcd9d4930854a8023a5f6df46a4f5bcaec6836f"),
    ],
)
def test_legacy_window_projection_bytes_stay_exact_without_optional_fields(kind, expected):
    # Captured before adding the three optional window summary keys.
    result = compact_project_event(kind, parent(window(waiting=False)))
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(raw).hexdigest() == expected
    assert all(key not in nested_window(result) for key in WAIT)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("waiting", [False, True])
def test_version_one_replay_retains_window_wait_without_success_or_source_mutation(
    tmp_path, compact, waiting
):
    original = parent(window(waiting=waiting))
    payload = compact_project_event("state", original) if compact else original
    events = [
        RuntimeEvent(event="hello", run_id="saved", sequence=0, payload={"task": "browser_project"}),
        RuntimeEvent(event="state", run_id="saved", sequence=1, payload=payload),
    ]
    path = tmp_path / "window.jsonl"
    path.write_text("".join(event.model_dump_json() + "\n" for event in events))
    before = path.read_bytes()
    assert all(event.version == 1 for event in read_trace(path))
    runtime = Runtime(io.StringIO())
    runtime.command({"command": "replay", "payload": {"path": str(path)}})
    for _ in range(len(events) + 1):
        runtime.tick()
    assert runtime.replay_events is None and runtime.env is None
    played = [json.loads(line) for line in runtime.output.getvalue().splitlines()]
    recorded_state = played[-2]["payload"]
    final = played[-1]["payload"]
    assert nested_window(recorded_state) == original["project_owner"]["star_component"]["component"]
    assert nested_window(final) == nested_window(recorded_state)
    assert final["replay_finished"] is True and final["recorded_status"] == "running"
    assert final["task_completed"] is final["project_completed"] is final["submitted"] is False
    assert final["project_progress"] == original["project_progress"]
    assert path.read_bytes() == before
