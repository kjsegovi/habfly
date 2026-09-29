"""Pure projection/injected runtime gates; never starts a browser or edits a run."""

import hashlib
import io
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
from test_runtime_browser_project import (  # noqa: F401 - injected fixture dependencies
    bridge_rig,
    command,
    integrated,
    start,
)

from habfly.contracts import RuntimeEvent
from habfly.project_events import ProjectEventRelay
from habfly.project_wire import compact_project_event, reconstruct_native_event
from habfly.runtime import RunOptions, Runtime, parse_run_options, read_trace

DATA = ("observation", "action_proposed", "action_result", "neural_activity", "error")
LIFECYCLE = ("hello", "state", "episode_summary")


def test_submission_refusal_projection_does_not_promote_completion():
    feedback = {"status": "course_refusal", "refusal_visible": True, "canonical_receipt": False}
    result = compact_project_event(
        "state",
        {"submission_feedback": feedback, "project_completed": False, "submitted": False},
        source_trace="events.jsonl",
    )
    assert result["submission_feedback"] == feedback
    assert result["project_completed"] is result["submitted"] is False


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def envelope(kind, payload):
    return {"version": 1, "run_id": "native-child", "sequence": 17, "event": kind, "payload": payload}


def wrap(
    native,
    components=("project.campaign", "campaign.owner", "star.workflow", "stellar.numeric"),
    *,
    star="Example",
):
    current = deepcopy(native)
    for component in reversed(components):
        result = []
        relay = ProjectEventRelay(lambda *args, received=result: received.append(args))
        forward = relay.bind(component, star=star)
        if set(current) == {"event", "payload"}:
            forward(current["event"], current["payload"])
        else:
            forward(current)
        kind, payload = result.pop()
        current = {"event": kind, "payload": payload}
    return current["event"], current["payload"]


def parent(payload, **changes):
    return {
        **payload,
        "status": "running",
        "phase": "active",
        "browser_phase": "active",
        "finished": False,
        "task_completed": False,
        "project_completed": False,
        "target_workflows_verified": False,
        "current_star": {"star": "Example", "ordinal": 1},
        "project_progress": {"collected": 1, "verified": 0},
        "reference_measurements": None,
        "project_owner": {"phase": "active", "star": "Example", "task_completed": False},
        **changes,
    }


def independent_reconstruction(payload):
    wire = payload["project_wire"]
    native = {**wire["leaf_header"], "payload": {k: payload[k] for k in wire["leaf_payload_keys"]}}
    assert digest(native["payload"]) == wire["leaf_payload_sha256"]
    assert digest(native) == wire["leaf_event_sha256"]
    return native


@pytest.mark.parametrize("kind", DATA)
def test_every_native_kind_exact_once_with_independent_reconstruction(kind):
    source = envelope(
        kind,
        {
            "instruction": "Unique native payload Ω-37",
            "value": "0.0035000000000001",
            "observation": {"controls": [{"id": "stable-7", "value": 0.0}]},
            "top_neurons": [{"body_id": "123", "activity": 0.02}],
            "task_completed": True,
            "terminated": True,
        },
    )
    event, payload = wrap(source)
    payload = parent(payload)
    before = deepcopy(payload)
    result = compact_project_event(event, payload)
    assert reconstruct_native_event(result) == independent_reconstruction(result) == source
    assert {k: result[k] for k in source["payload"]} == source["payload"]
    assert json.dumps(result, ensure_ascii=False).count("Unique native payload Ω-37") == 1
    assert result["project_wire"]["parent"]["task_completed"] is False
    assert result["project_wire"]["parent"]["project_completed"] is False
    assert result["project_wire"]["source_event_sha256"] == digest({"event": event, "payload": payload})
    assert result["project_wire"]["evidence_receipt"] is False
    assert payload == before
    result["observation"]["controls"][0]["id"] = "mutated-output"
    assert payload == before


@pytest.mark.parametrize("kind", LIFECYCLE)
def test_child_success_never_promotes_parent_lifecycle(kind):
    event, payload = wrap(
        envelope(
            kind,
            {
                "status": "completed",
                "phase": "verified",
                "task_completed": True,
                "project_completed": True,
                "completed": True,
                "finished": True,
            },
        )
    )
    result = compact_project_event(event, parent(payload))
    assert result["status"] == "running" and result["phase"] == "active"
    for key in ("task_completed", "project_completed", "finished", "target_workflows_verified"):
        assert result[key] is False
    assert result["project_wire"]["leaf_summary"]["task_completed"] is True
    assert result["project_wire"]["leaf_payload_inline"] is False
    with pytest.raises(ValueError, match="invalid_native_envelope"):
        reconstruct_native_event(result)


@pytest.mark.parametrize("status", ("running", "paused", "stopped", "aborted", "handoff", "completed"))
def test_parent_fields_and_explicit_nulls_are_preserved(status):
    payload = parent(
        {},
        status=status,
        finished=status == "completed",
        task_completed=status == "completed",
        project_completed=False,
        reference_measurements=None,
        class_setup=None,
    )
    output = compact_project_event("state", payload)
    for key in payload:
        assert output[key] == payload[key]


@pytest.mark.parametrize("kind", LIFECYCLE)
@pytest.mark.parametrize("enabled", (True, False))
@pytest.mark.parametrize("flag", ("supplied_stellar_inputs_enabled", "baseline_edge_reference_enabled"))
def test_explicit_reference_option_survives_only_as_parent_lifecycle_context(kind, enabled, flag):
    source = parent({}, **{flag: enabled})
    result = compact_project_event(kind, source)
    assert result[flag] is enabled
    assert result["task_completed"] is result["project_completed"] is False
    assert flag not in result["project_wire"]["leaf_summary"]
    event, child = wrap(envelope(kind, {flag: True}))
    result = compact_project_event(event, parent(child))
    assert flag not in result
    assert result["task_completed"] is result["project_completed"] is False


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("hello", "c8b0f5767d704f88d2eb08da29d0a0d761dd30baf618543774d95689d0a6f2a0"),
        ("state", "3e615e839b748c701d2e22a958503284266efb478f510749e9845549cd29cd76"),
        ("episode_summary", "04b48ccbe2f8b2b3e34c1b4ce2e4400cf809b3eefed9f22ae4d858dcbc10f918"),
    ],
)
def test_legacy_lifecycle_bytes_unchanged_without_supplied_option(kind, expected):
    # Captured before adding the optional display scalar; no default is inserted.
    source = {
        "task": "browser_project",
        "status": "paused",
        "task_completed": False,
        "project_completed": False,
    }
    result = compact_project_event(kind, source)
    assert "supplied_stellar_inputs_enabled" not in result
    assert "baseline_edge_reference_enabled" not in result
    assert digest(result) == expected


def test_scoped_tui_context_and_source_maps_not_recursive_snapshots():
    chart = {
        "mode": "bounded_planet_window",
        "phase": "observing",
        "polls": 3,
        "max_polls": 20,
        "observation_limit_days": 5000,
        "window_status": "partial",
        "waiting_for_clean_partial_chart": True,
    }
    payload = parent(
        {},
        class_setup={
            "phase": "setting_lifetime_prefix",
            "selected_class": "main_sequence",
            "lifetime_prefix": "Ga",
            "setup_verified": False,
        },
    )
    payload["project_owner"]["star_component"] = {"phase": "planet_window", "component": chart}
    payload["campaign"] = {
        "verified_stars": 0,
        "target_stars": 2,
        "max_seconds": 3600,
        "project_owner": deepcopy(payload["project_owner"]),
        "next_star": {"phase": "waiting"},
    }
    payload["model_source_sha256"] = {f"model-{i}": "e" * 64 for i in range(1000)}
    result = compact_project_event("state", payload)
    assert result["project_owner"]["star_component"]["component"] == chart
    assert result["class_setup"] == payload["class_setup"]
    assert result["campaign"]["next_star"] == {"phase": "waiting"}
    assert "project_owner" not in result["campaign"] and "model_source_sha256" not in result
    assert result["project_wire"]["source_event_sha256"] == digest({"event": "state", "payload": payload})


def test_outer_style_native_does_not_invent_sequence():
    source = {"event": "action_result", "payload": {"progress": 4}}
    kind, payload = wrap(source, ("project.owner", "star.workflow", "planet.window"))
    native = reconstruct_native_event(compact_project_event(kind, payload))
    assert native == source and set(native) == {"event", "payload"}


@pytest.mark.parametrize("component", ("raw", "derived"))
@pytest.mark.parametrize("kind", DATA + LIFECYCLE)
def test_positive_reference_wrapper_is_supported_without_inventing_claims(component, kind):
    source = envelope(kind, {"value": "12.000", "task_completed": True})
    payload = {"component_identity": component, "component_event": deepcopy(source)}
    if kind in LIFECYCLE:
        payload["component_summary"] = deepcopy(source["payload"])
        forwarded = "state"
    else:
        payload.update(source["payload"])
        forwarded = kind
    outer, incoming = wrap(
        {"event": forwarded, "payload": payload}, ("project.owner", "star.workflow", "planet.positive")
    )
    result = compact_project_event(outer, parent(incoming))
    assert result["project_wire"]["ancestry"][-1]["component"] == component
    if kind in DATA:
        assert reconstruct_native_event(result) == source
    else:
        assert result["task_completed"] is False


@pytest.mark.parametrize("kind", DATA + LIFECYCLE)
@pytest.mark.parametrize("root", [("project.owner",), ("project.campaign", "campaign.owner")])
def test_shallow_probe_leaf_reconstruction_and_parent_claims(kind, root):
    source = envelope(
        kind,
        {
            "chart": {"source": "visible_tooltips", "day": 901, "brightness_percent": "99.9991000"},
            "source_sha256": {"before.png": "b" * 64},
            "value": "unique-shallow-payload",
            "task_completed": True,
            "project_completed": True,
            "status": "completed",
        },
    )
    components = (*root, "star.workflow", "planet.shallow", "shallow.probe")
    event, payload = wrap(source, components)
    payload = parent(payload)
    before = deepcopy(payload)
    result = compact_project_event(event, payload)
    wire = result["project_wire"]
    assert [row["component"] for row in wire["ancestry"]] == list(components)
    assert wire["source_event_sha256"] == digest({"event": event, "payload": before})
    if kind in DATA:
        assert reconstruct_native_event(result) == independent_reconstruction(result) == source
        assert json.dumps(result).count("unique-shallow-payload") == 1
        assert wire["parent"]["task_completed"] is False
        assert wire["parent"]["project_completed"] is False
    else:
        assert result["status"] == "running"
        assert result["task_completed"] is False and result["project_completed"] is False
        assert wire["leaf_summary"]["task_completed"] is True
    assert payload == before


def test_shallow_owner_two_argument_native_event_does_not_invent_sequence():
    source = {
        "event": "observation",
        "payload": {"source": "explicit_tooltip_reference", "maximum_sampled_decline": "0.00009"},
    }
    event, payload = wrap(source, ("project.owner", "star.workflow", "planet.shallow"))
    assert reconstruct_native_event(compact_project_event(event, payload)) == source


@pytest.mark.parametrize("identity", ["invented.probe", "stellar.numeric", "shallow.probe.child"])
def test_shallow_accepts_only_explicit_probe_ancestry(identity):
    event, payload = wrap(
        envelope("action_proposed", {"kind": "HOVER"}),
        ("project.owner", "star.workflow", "planet.shallow", identity),
    )
    with pytest.raises(ValueError, match="^project_compact_wire_invalid_event$"):
        compact_project_event(event, payload)


def test_shallow_probe_cannot_appear_as_unscoped_root_or_skip_owner():
    for components in [("planet.shallow", "shallow.probe"), ("project.owner", "shallow.probe")]:
        event, payload = wrap(envelope("action_result", {"kind": "HOVER"}), components)
        with pytest.raises(ValueError, match="^project_compact_wire_invalid_event$"):
            compact_project_event(event, payload)


def test_shallow_progress_is_scoped_without_duplicate_cached_probe_data():
    shallow = {
        "mode": "cooperative_shallow_transit_measurements",
        "phase": "probe_active",
        "star": "Example",
        "native_action_attempts": 12,
        "native_actions_confirmed": 11,
        "max_native_actions": 64,
        "advances": 18,
        "max_advances": 80,
        "max_seconds": 900,
        "probe_index": 1,
        "completed_probes": [
            {"directory": "star/shallow/probe-01", "report_sha256": "a" * 64, "browser_actions": 9}
        ],
        "window_evidence": {
            "progress_path": "star/shallow/restore-01/progress/report.json",
            "progress_sha256": "b" * 64,
            "chart_sha256": "c" * 64,
            "chart_path": "star/shallow/restore-01/progress/chart.png",
        },
        "native_action_outcome_uncertain": True,
        "measurement_mode": "approximate_reference_visible_tooltips_v1",
        "learned_perception": False,
        "scientific_verified": False,
        "training_label": False,
        "task_completed": False,
        "project_completed": False,
    }
    payload = parent({})
    payload["project_owner"]["star_component"] = {
        "phase": "shallow_reference",
        "shallow_reference_enabled": True,
        "component": {**shallow, "component_state": {"huge_cache": "x" * 10000}},
    }
    result = compact_project_event("state", payload)
    star = result["project_owner"]["star_component"]
    assert star["component"] == shallow and star["shallow_reference_enabled"] is True
    assert "huge_cache" not in json.dumps(result)
    assert result["task_completed"] is False and result["project_completed"] is False


def test_completed_shallow_receipts_stay_scoped_to_sensor_not_arbitrary_lifecycle():
    result = compact_project_event(
        "state", {"completed_probes": [{"arbitrary": "cache"}], "window_evidence": {"unscoped": True}}
    )
    assert "completed_probes" not in result and "window_evidence" not in result


@pytest.mark.parametrize(
    "directory,expected_sha256",
    [
        ("shallow-probe-408", "007b21f09c7cbe6285154e6d06d2e21379144787ad2200c5ee524bce1ade93a1"),
        ("shallow-probe-410", "3d2ea48179f83ab3cca9e02f8b3a375b6c5f866b713fcb9a447d14b5116a42b4"),
        ("shallow-probe-412", "32d04530b2fbda4fbaa0bef7f2c376e5c9ec7f8b1b3d9ed937b01f3bd169ac35"),
    ],
)
def test_completed_visible_probe_events_streamed_exactly_if_available(directory, expected_sha256):
    """Read-only known development diagnostics, not a new native acceptance run."""
    path = Path("experiments/full-stellar-probe/20260926-005") / directory / "events.jsonl"
    if not path.is_file():
        pytest.skip("Local completed development diagnostic is not distributed")
    sha, counts = hashlib.sha256(), Counter()
    with path.open("rb") as stream:
        for line in stream:
            assert line.endswith(b"\n")
            sha.update(line)
            original = json.loads(line)
            kind, payload = wrap(
                original,
                ("project.campaign", "campaign.owner", "star.workflow", "planet.shallow", "shallow.probe"),
                star="AYISTASH",
            )
            result = compact_project_event(kind, parent(payload, current_star={"star": "AYISTASH"}))
            if kind in DATA:
                assert reconstruct_native_event(result) == independent_reconstruction(result) == original
                assert result["project_wire"]["parent"]["project_completed"] is False
                counts[kind] += 1
            else:
                assert result["project_completed"] is False and result["task_completed"] is False
    assert sha.hexdigest() == expected_sha256
    assert counts["observation"] == 1
    assert counts["action_proposed"] == counts["action_result"] >= 6


@pytest.mark.parametrize(
    "mutation",
    (
        lambda p: p.update(project_wire={}),
        lambda p: p["component_event"]["payload"].update(project_wire={}),
        lambda p: p["component_event"].update(unrecognized="bad"),
        lambda p: p["component_event"].update(sequence=-1),
        lambda p: p["component_event"].update(sequence=True),
        lambda p: p["component_event"].update(version=2),
        lambda p: p["component_event"].update(event="hello"),
        lambda p: p.update(component="invented.agent"),
        lambda p: p.update(component="stellar.numeric"),
        lambda p: p.update(component_star="private\nstar"),
    ),
)
def test_collisions_and_fake_envelopes_fail_closed(mutation):
    kind, payload = wrap(envelope("action_proposed", {"kind": "CLICK"}), ("browser.reference_class",))
    mutation(payload)
    with pytest.raises(ValueError, match="^project_compact_wire_invalid_event$"):
        compact_project_event(kind, payload)


def test_lifecycle_duplicate_mismatch_rejected():
    kind, payload = wrap(envelope("state", {"phase": "ready"}), ("browser.setup",))
    payload["component_state"] = {"phase": "fake"}
    with pytest.raises(ValueError, match="invalid_event"):
        compact_project_event(kind, payload)


@pytest.mark.parametrize(
    "trace",
    (
        "/tmp/events.jsonl",
        "../events.jsonl",
        "a/../events.jsonl",
        "https://private/events.jsonl",
        "events.jsonl?token=secret",
        "a\n.jsonl",
    ),
)
def test_source_trace_is_relative_local_redacted_binding(trace):
    with pytest.raises(ValueError, match="^project_compact_wire_invalid_event$") as error:
        compact_project_event("state", {}, source_trace=trace)
    assert trace not in str(error.value)


@pytest.mark.parametrize(
    "payload",
    (
        {"password": "sensitive-never-echo"},
        {"nested": {"access_token": "sensitive-never-echo"}},
        {"instruction": "http://localhost/preview?preview_sequence_id=sensitive-never-echo"},
        {"instruction": "https://user:secret@example.invalid"},
        {"value": float("nan")},
    ),
)
def test_native_secret_or_nonfinite_data_rejected_not_silently_edited(payload):
    kind, incoming = wrap(envelope("observation", payload))
    with pytest.raises(ValueError, match="^project_compact_wire_invalid_event$") as error:
        compact_project_event(kind, incoming)
    assert "sensitive" not in str(error.value)


def test_unknown_parent_private_fields_dropped_without_copying_or_echoing():
    payload = parent({}, password="private-value", session_url="http://localhost?secret=private-value")
    result = compact_project_event("state", payload)
    assert "private-value" not in json.dumps(result)
    assert result["project_wire"]["source_event_sha256"] == digest({"event": "state", "payload": payload})
    with pytest.raises(ValueError, match="invalid_event"):
        compact_project_event("state", parent({}, failure_reason="http://localhost?token=private-value"))


@pytest.mark.parametrize(
    "mutation",
    (
        lambda p: p.update(value="changed"),
        lambda p: p.update(invented="changed"),
        lambda p: p["project_wire"]["leaf_header"].update(sequence=18),
        lambda p: p["project_wire"]["leaf_header"].update(event="observation"),
        lambda p: p["project_wire"]["leaf_payload_keys"].append("value"),
    ),
)
def test_independent_reconstruction_rejects_tampering(mutation):
    kind, incoming = wrap(envelope("action_result", {"value": "12.0"}))
    result = compact_project_event(kind, incoming)
    mutation(result)
    with pytest.raises(ValueError, match="invalid_native_envelope"):
        reconstruct_native_event(result)


@pytest.mark.parametrize(
    "components",
    [
        ("project.campaign", "campaign.owner", "star.workflow", "stellar.numeric"),
        ("project.campaign", "campaign.owner", "star.workflow", "planet.shallow", "shallow.probe"),
    ],
)
def test_bounded_payload_size_and_linear_native_growth(components):
    sizes = []
    for count in (100, 1000, 10000):
        source = envelope("observation", {"instruction": "x" * count})
        kind, incoming = wrap(source, components)
        # Cached relay snapshots can contain the same large observation many
        # times; none is an extra native payload on the compact wire.
        incoming = parent(incoming, campaign={"phase": "active", "component_state": deepcopy(incoming)})
        incoming["model_source_sha256"] = {f"source-{i}": "b" * 64 for i in range(count)}
        result = compact_project_event(kind, incoming)
        sizes.append(len(canonical(result)))
        assert reconstruct_native_event(result) == source
        assert len(canonical(result)) < count + 2500
    assert sizes[2] - sizes[1] == 9000
    assert sizes[1] - sizes[0] == 900


def test_runtime_default_legacy_and_optin_native_and_replay(tmp_path):
    source = envelope("neural_activity", {"top_neurons": [{"body_id": "456", "activity": 0.9}]})
    kind, incoming = wrap(source)
    legacy = Runtime(io.StringIO())
    legacy.project_event(kind, incoming)
    assert json.loads(legacy.output.getvalue())["payload"] == incoming
    compact = Runtime(io.StringIO())
    compact.options = RunOptions(task="browser_project", project_compact_wire=True)
    compact.run_id, compact.trace_path = "outer", tmp_path / "outer.jsonl"
    compact.project_event("state", parent({}))
    compact.project_event(kind, incoming)
    compact.trace_path.write_text(compact.output.getvalue())
    recorded = list(read_trace(compact.trace_path))
    assert recorded[0].payload["status"] == "running"
    assert recorded[0].payload["task_completed"] is False
    assert reconstruct_native_event(recorded[1].payload) == source
    replay = Runtime(io.StringIO())
    replay.command({"command": "replay", "payload": {"path": str(compact.trace_path)}})
    while replay.replay_events is not None:
        replay.tick()
    played = [json.loads(line) for line in replay.output.getvalue().splitlines()]
    neural = next(item for item in played if item["event"] == "neural_activity")
    native_payload = {key: value for key, value in neural["payload"].items() if key != "replay"}
    assert reconstruct_native_event(native_payload) == source
    assert played[-1]["payload"].get("task_completed") is False


def test_runtime_rejects_before_outer_emit_or_status_change():
    runtime = Runtime(io.StringIO())
    runtime.options = RunOptions(task="browser_project", project_compact_wire=True)
    runtime.status = "running"
    kind, payload = wrap(envelope("episode_summary", {"task_completed": True}))
    payload["component"] = "fake.owner"
    with pytest.raises(ValueError, match="^project_compact_wire_invalid_event$"):
        runtime.project_event(kind, parent(payload, status="completed"))
    assert runtime.sequence == 0 and runtime.output.getvalue() == ""
    assert runtime.status == "running"


def test_compact_bridge_launch_class_handoff_and_abort_are_injected_only(integrated):  # noqa: F811
    rig = integrated
    bridge = start(rig, project_compact_wire=True)
    assert not rig.calls
    for _ in range(12):
        command(rig, "step")
        if bridge.phase == "awaiting_class_source":
            break
    assert bridge.phase == "awaiting_class_source"
    command(rig, "abort")
    assert rig.runtime.status == "aborted"
    incoming = [json.loads(line) for line in (bridge.output / "events.jsonl").read_text().splitlines()]
    outgoing = [json.loads(line) for line in rig.runtime.output.getvalue().splitlines()]
    bindings = {digest({"event": item["event"], "payload": item["payload"]}) for item in incoming}
    assert incoming and outgoing
    for item in outgoing:
        wire = item["payload"].get("project_wire")
        if wire:
            assert wire["source_event_sha256"] in bindings
    terminal = next(item for item in reversed(outgoing) if item["event"] in {"state", "episode_summary"})
    assert terminal["payload"]["task_completed"] is False
    assert terminal["payload"]["project_completed"] is False
    assert terminal["payload"]["status"] == "aborted"


def test_source_binding_changes_when_dropped_evidence_changes():
    payload = parent({}, source_sha256={"owned/report.json": "b" * 64})
    first = compact_project_event("state", payload)
    payload["source_sha256"]["owned/report.json"] = "c" * 64
    second = compact_project_event("state", payload)
    assert first["project_wire"]["source_event_sha256"] != second["project_wire"]["source_event_sha256"]
    assert {k: v for k, v in first.items() if k != "project_wire"} == {
        k: v for k, v in second.items() if k != "project_wire"
    }


def test_raw_native_arbitrary_component_label_is_not_used_as_ancestry():
    source = envelope("action_result", {"component": "untrusted-native-label", "value": "1.0"})
    kind, payload = wrap(source)
    result = compact_project_event(kind, payload)
    assert reconstruct_native_event(result) == source
    assert result["project_wire"]["ancestry"][-1]["component"] == "stellar.numeric"


def test_native_cannot_smuggle_an_extra_envelope_in_leaf():
    source = envelope(
        "observation",
        {
            "component": "browser.setup",
            "component_event": {"event": "action_proposed", "payload": {"kind": "CLICK"}},
        },
    )
    kind, payload = wrap(source)
    with pytest.raises(ValueError, match="invalid_event"):
        compact_project_event(kind, payload)


def test_leaf_growth_is_constant_across_supported_relay_depths():
    source = envelope("neural_activity", {"top_neurons": [{"body_id": "456", "activity": 0.9}]})
    lengths = []
    for path in (
        ("browser.reference_class",),
        ("project.owner", "star.workflow", "stellar.numeric"),
        ("project.campaign", "campaign.owner", "star.workflow", "stellar.numeric"),
    ):
        kind, payload = wrap(source, path)
        result = compact_project_event(kind, payload)
        assert reconstruct_native_event(result) == source
        assert json.dumps(result).count('"body_id"') == 1
        lengths.append(len(canonical(result)))
    assert max(lengths) - min(lengths) < 250


@pytest.mark.parametrize("value", ("true", "false", 0, 1, None, [], {}))
def test_option_is_strict_and_default_legacy(value):
    assert RunOptions().project_compact_wire is False
    with pytest.raises(ValueError):
        RunOptions(project_compact_wire=value)


def test_option_project_only_and_future_profile_optin():
    with pytest.raises(ValueError, match="requires_project_task"):
        parse_run_options({"task": "mini_habworlds", "project_compact_wire": True})
    assert parse_run_options({}).project_compact_wire is False
    profile = json.loads(Path("configs/browser_project_two_star.json").read_text())
    assert profile["project_compact_wire"] is True


def test_recorded_first_star_fixed_prefix_streamed_if_available():
    """Optional local regression: stream only1700 complete lines, never whole trace.

    This reads an immutable first-star prefix of an append-only development run.
    No snapshot/rewriting of active artifacts; no sealed evaluation cases used.
    """
    root = Path("experiments/browser-project-two-star")
    source = root / "ebc9eb3a887741fb88cc7c7380fa5138/events.jsonl"
    outer = root / "ebc9eb3a887741fb88cc7c7380fa5138.jsonl"
    if not source.is_file() or not outer.is_file():
        pytest.skip("Local development trace prefix is not distributed")
    counts, old_size, new_size, max_size = Counter(), 0, 0, 0
    with source.open() as raw, outer.open() as original:
        for _ in range(1700):
            raw_line, outer_line = raw.readline(), original.readline()
            assert raw_line.endswith("\n") and outer_line.endswith("\n")
            item, previous = json.loads(raw_line), json.loads(outer_line)
            assert item["event"] == previous["event"]
            assert (
                item["payload"].get("current_star", {}).get("ordinal", 1) <= 1
                if item["payload"].get("current_star")
                else True
            )
            result = compact_project_event(item["event"], item["payload"])
            assert result["project_wire"]["source_event_sha256"] == digest(
                {"event": item["event"], "payload": item["payload"]}
            )
            if item["event"] in DATA:
                leaf = item
                while "component_event" in leaf["payload"]:
                    leaf = leaf["payload"]["component_event"]
                assert independent_reconstruction(result) == reconstruct_native_event(result) == leaf
                counts[item["event"]] += 1
            else:
                for key in (
                    "status",
                    "phase",
                    "browser_phase",
                    "finished",
                    "task_completed",
                    "project_completed",
                    "current_star",
                    "project_progress",
                    "reference_measurements",
                    "target_workflows_verified",
                ):
                    if key in item["payload"]:
                        assert result[key] == item["payload"][key]
                # Apply the same runtime-owned fields as the original stdout.
                for key in (
                    "task",
                    "runtime_task",
                    "environment",
                    "seed",
                    "stars",
                    "project_campaign",
                    "policy",
                    "graph",
                    "checkpoint",
                    "project_trace_path",
                    "artifact_root",
                    "trace_path",
                ):
                    result[key] = previous["payload"][key]
                result["project_compact_wire"] = True
            projected = RuntimeEvent.model_validate({**previous, "payload": result})
            length = len(projected.model_dump_json().encode()) + 1
            old_size += len(outer_line.encode())
            new_size += length
            max_size = max(max_size, length)
    assert counts == {"action_proposed": 73, "action_result": 74, "observation": 91, "neural_activity": 63}
    assert old_size == 127497438
    assert new_size < old_size * 0.09 and max_size < 16000
