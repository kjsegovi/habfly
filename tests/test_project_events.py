"""Component evidence survives replay without taking over project lifecycle."""

import copy
import io
import json

import pytest

from habfly.project_events import ProjectEventRelay
from habfly.runtime import Runtime, read_trace


def envelope(event, payload, sequence=0, run_id="child"):
    return {"version": 1, "event": event, "payload": payload, "sequence": sequence, "run_id": run_id}


@pytest.mark.parametrize(
    "kind,key",
    [("hello", "component_hello"), ("state", "component_state"), ("episode_summary", "component_summary")],
)
def test_child_lifecycle_is_nested(kind, key):
    out = []
    relay = ProjectEventRelay(lambda *args: out.append(args))
    source = envelope(kind, {"status": "completed", "completed": True, "task_completed": True})
    expected = copy.deepcopy(source)
    relay.bind("planet.derived", star="Dulat")(source)
    event, payload = out[0]
    assert event == "state"
    assert payload[key] == source["payload"]
    assert payload["component_event"] == expected
    assert not {"status", "completed", "task_completed"} & payload.keys()
    source["payload"]["status"] = "changed"
    assert payload["component_event"] == expected


@pytest.mark.parametrize(
    "kind", ["observation", "action_proposed", "action_result", "neural_activity", "error"]
)
def test_data_events_keep_original_envelope(kind):
    out = []
    relay = ProjectEventRelay(lambda *args: out.append(args))
    source = envelope(kind, {"value": "0.0035000000000001", "component": "untrusted-child-name"})
    relay.bind("stellar.numeric", star="Example")(source)
    event, payload = out[0]
    assert event == kind
    assert payload["value"] == source["payload"]["value"]
    assert payload["component"] == "stellar.numeric"
    assert payload["component_star"] == "Example"
    assert payload["component_event"] == source


def test_retired_previous_star_cannot_replace_observation():
    out = []
    relay = ProjectEventRelay(lambda *args: out.append(args))
    old = relay.bind("stellar.numeric", star="First")
    new = relay.bind("stellar.numeric", star="Second")
    with pytest.raises(ValueError, match="Stale"):
        old(envelope("observation", {}))
    new(envelope("observation", {}))
    relay.retire()
    with pytest.raises(ValueError, match="Stale"):
        new(envelope("observation", {}, sequence=1))
    assert len(out) == 1


@pytest.mark.parametrize("second", [0, 1])
def test_nonmonotonic_child_fail_closed(second):
    out = []
    relay = ProjectEventRelay(lambda *args: out.append(args))
    send = relay.bind("raw")
    send(envelope("state", {}, sequence=1))
    with pytest.raises(ValueError, match="forwarding failed"):
        send(envelope("action_proposed", {}, sequence=second))
    assert relay.failed and len(out) == 1
    with pytest.raises(ValueError):
        relay.bind("next")


def test_outer_style_does_not_fabricate_child_sequence():
    out = []
    send = ProjectEventRelay(lambda *args: out.append(args)).bind("chart")
    send("state", {"phase": "observing"})
    assert out[0][1]["component_event"] == {"event": "state", "payload": {"phase": "observing"}}


def test_callback_failure_is_sanitized_and_sticky():
    def receiver(*args):
        raise RuntimeError("private-url?credential=secret")

    relay = ProjectEventRelay(receiver)
    send = relay.bind("raw")
    with pytest.raises(ValueError, match="^Project component event forwarding failed$"):
        send(envelope("state", {}))
    assert relay.failed
    with pytest.raises(ValueError, match="Stale or failed"):
        send(envelope("action_proposed", {}, sequence=1))


@pytest.mark.parametrize("item", [{}, {"event": "new-unversioned-kind"}, "not-an-event", 1])
def test_malformed_events_never_reach_receiver(item):
    out = []
    relay = ProjectEventRelay(lambda *args: out.append(args))
    with pytest.raises(ValueError):
        relay.bind("raw")(item)
    assert relay.failed and out == []


def test_runtime_replay_preserves_one_outer_sequence_and_child_identity(tmp_path):
    stream = io.StringIO()
    runtime = Runtime(stream)
    runtime.run_id = "project"
    relay = ProjectEventRelay(runtime.emit)
    for name in ("stellar.numeric", "stellar.color"):
        send = relay.bind(name, star="Dulat")
        send(envelope("hello", {}, 0, name))
        send(envelope("action_result", {"number": "6.961534486328223"}, 1, name))
        send(envelope("episode_summary", {"completed": True}, 2, name))
    path = tmp_path / "project.jsonl"
    path.write_text(stream.getvalue())
    replay = read_trace(path)
    assert [e.sequence for e in replay] == list(range(6))
    assert {e.run_id for e in replay} == {"project"}
    assert all(e.event != "episode_summary" for e in replay)
    assert all("completed" not in e.payload for e in replay)
    assert [json.loads(line) for line in stream.getvalue().splitlines()] == [e.model_dump() for e in replay]
