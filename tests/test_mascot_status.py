"""Presentation-only event reduction; no browser, model, trace writes or secrets."""

from copy import deepcopy

import pytest

from habfly.mascot_status import ALLOWED_CAPTIONS, MascotStatusReducer

SECRET = "admin@example.edu https://private.example/token password=private 123456.789"
OPERATIONS = {
    "distance": "distance",
    "luminosity": "luminosity",
    "temperature": "temperature",
    "mass": "stellar mass",
    "radius": "stellar radius",
    "lifetime": "stellar lifetime",
    "period_years": "orbital period",
    "radial_velocity": "radial velocity",
    "orbital_radius": "orbital radius",
    "planet_mass": "planet mass",
    "planet_radius": "planet radius",
    "planet_density": "planet density",
    "equilibrium_temp": "equilibrium temperature",
    "surface_temp": "surface temperature",
}


def event(kind, payload, **header):
    return {"version": 1, "event": kind, "payload": payload, **header}


def observation(operation="distance", destination="distance", revision=0):
    return {
        "revision": revision,
        "instruction": SECRET,
        "feedback": SECRET,
        "values": {"answers": {"distance": SECRET}},
        "calculation": {"operation": operation, "destination": destination, "tool_error": None},
    }


def nested(item, *, star="EXAMPLE", component="stellar.numeric", minimal=False):
    if minimal:
        item = {"event": item["event"], "payload": item["payload"]}
    kind = "state" if item["event"] in ("hello", "state", "episode_summary") else item["event"]
    return event(kind, {"component": component, "component_star": star, "component_event": item})


def assert_private_free(status):
    assert set(status) == {"caption", "activity", "source_label"}
    assert status["caption"] in ALLOWED_CAPTIONS
    assert all(type(value) is str and len(value) <= 80 for value in status.values())
    assert not any(
        text in str(status) for text in ("admin@", "https://", "password", "123456.789", "private")
    )


@pytest.mark.parametrize("operation,label", OPERATIONS.items())
@pytest.mark.parametrize("wrapped", [False, True])
def test_real_flat_planet_and_wrapped_stellar_action_shapes(operation, label, wrapped):
    reducer = MascotStatusReducer()
    obs = observation(operation, operation)
    reducer.consume(event("observation", obs))

    def act(kind, key, value=None):
        payload = {
            "kind": kind,
            "target": "0:" + key,
            "value": value,
            "observation_revision": 0,
            "action_confidence": 0.999,
            "calibrated": True,
            "reasoning": SECRET,
        }
        return reducer.consume(
            event("action_proposed", {"action": payload, "action_source": SECRET} if wrapped else payload)
        )

    assert act("SELECT", "operation", operation)["caption"] == f"Selecting {label} calculation"
    proposed = act("CLICK", "execute")
    assert proposed["caption"] == f"Preparing {label} calculation"
    assert proposed["source_label"] == "Proposed action"
    obs["calculation"]["last_operation"] = {
        "kind": "calculate",
        "operation": operation,
        "inputs": {"secret": SECRET},
        "result": {"ok": True, "value": SECRET},
    }
    result = reducer.consume(event("action_result", {"observation": obs, "failure_reason": None}))
    assert result["caption"] == f"Calculated {label}"
    assert result["source_label"] == "Recorded tool result"
    assert_private_free(result)


@pytest.mark.parametrize("field,label", OPERATIONS.items())
def test_copy_and_unit_captions_never_include_number_or_unit_payload(field, label):
    reducer = MascotStatusReducer()
    obs = observation(destination=field)
    reducer.consume(event("observation", obs))
    proposed = reducer.consume(
        event("action_proposed", {"kind": "CLICK", "target": "0:copy", "value": SECRET})
    )
    assert proposed["caption"] == f"Preparing copy to {label}"
    units = reducer.consume(
        event("action_proposed", {"kind": "SELECT", "target": "0:unit_" + field, "value": SECRET})
    )
    assert units["caption"] == f"Selecting {label} units"
    obs["calculation"]["last_operation"] = {
        "kind": "copy",
        "destination": field,
        "value": SECRET,
        "unit": SECRET,
    }
    copied = reducer.consume(event("action_result", {"observation": obs}))
    assert copied["caption"] == f"Copied to {label}"
    assert_private_free(copied)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "TYPE", "destination": "planet_mass", "target": "opaque", "value": SECRET},
        {"action": {"kind": "TYPE", "target": "live:1:planet_mass", "value": SECRET}},
    ],
)
def test_native_numeric_copy_proposals_do_not_claim_verified_readback(payload):
    reducer = MascotStatusReducer()
    result = reducer.consume(event("action_proposed", payload))
    assert result["caption"] == "Preparing copy to planet mass"
    assert result["source_label"] == "Proposed action"
    result = reducer.consume(
        event(
            "action_result",
            {"native_copy": {"destination": "planet_mass", "value": SECRET, "readback_verified": True}},
        )
    )
    assert result["caption"] == "Checked planet mass entry"
    assert_private_free(result)


@pytest.mark.parametrize("minimal", [False, True])
def test_nested_project_component_events_use_current_star_context(minimal):
    reducer = MascotStatusReducer()
    reducer.consume(nested(event("observation", observation("luminosity")), minimal=minimal))
    action = event("action_proposed", {"kind": "CLICK", "target": "0:execute"})
    assert reducer.consume(nested(action, minimal=minimal))["caption"] == "Preparing luminosity calculation"
    assert (
        reducer.consume(nested(action, star="ANOTHER", minimal=minimal))["caption"] == "Preparing calculation"
    )
    reducer.consume(nested(event("observation", observation("mass")), minimal=minimal))
    changed = reducer.consume(nested(action, component="terrestrial.temperature", minimal=minimal))
    assert changed["caption"] == "Preparing calculation"


@pytest.mark.parametrize(
    "status,extra,caption",
    [
        ("paused", {}, "Paused"),
        ("paused", {"replay": True}, "Replay paused"),
        ("stopped", {}, "Stopped"),
        ("aborted", {}, "Stopped"),
        ("completed", {}, "Run ended"),
        ("completed", {"completed": True}, "Task completed"),
        ("completed", {"project_completed": True}, "Project completed"),
        (
            "handoff",
            {"phase": "awaiting_assessment", "target_workflows_verified": True},
            "Ready for assessment",
        ),
        ("unknown_pending", {"submitted": True}, "Awaiting confirmation"),
        ("replay_completed", {}, "Replay ended"),
    ],
)
def test_lifecycle_truth_requires_explicit_success_not_termination(status, extra, caption):
    result = MascotStatusReducer().consume(event("state", {"status": status, "private": SECRET, **extra}))
    assert result["caption"] == caption
    assert_private_free(result)


@pytest.mark.parametrize(
    "payload", [{"terminated": True}, {"truncated": True}, {"completed": False}, {"submitted": True}]
)
def test_episode_termination_and_submission_alone_are_not_success(payload):
    assert MascotStatusReducer().consume(event("episode_summary", payload))["caption"] == "Run ended"


def test_component_success_and_top_level_pause_do_not_promote_project_success():
    reducer = MascotStatusReducer()
    child = nested(event("episode_summary", {"completed": True, "project_completed": True}))
    assert reducer.consume(child)["caption"] == "Step finished"
    child["payload"]["status"] = "paused"
    assert reducer.consume(child)["caption"] == "Paused"
    child["payload"]["status"] = "stopped"
    assert reducer.consume(child)["caption"] == "Stopped"


@pytest.mark.parametrize("problem", ["failure", "tool", "false_ok"])
def test_errors_do_not_echo_diagnostics_or_claim_calculation_success(problem):
    obs = observation()
    obs["calculation"]["last_operation"] = {
        "kind": "calculate",
        "operation": "distance",
        "result": {"ok": problem != "false_ok", "error": SECRET},
    }
    if problem == "tool":
        obs["calculation"]["tool_error"] = SECRET
    status = MascotStatusReducer().consume(
        event(
            "action_result", {"observation": obs, "failure_reason": SECRET if problem == "failure" else None}
        )
    )
    assert "attention" in status["caption"]
    assert_private_free(status)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        1,
        SECRET,
        {},
        {"event": []},
        event("unknown", {"text": SECRET}),
        event("state", [], version=1),
        event("state", {}, version=True),
        event("state", {}, version=2),
        event("state", {}, sequence=True),
    ],
)
def test_malformed_and_unknown_are_neutral_without_exceptions(bad):
    assert MascotStatusReducer().consume(bad)["caption"] == "Status unavailable"


def test_deep_or_cyclic_component_payload_is_bounded():
    value = event("state", {})
    value["payload"].update(component="loop", component_event=value)
    assert MascotStatusReducer().consume(value)["caption"] == "Status unavailable"
    value = event("observation", observation())
    for _ in range(10):
        value = nested(value)
    assert MascotStatusReducer().consume(value)["caption"] == "Status unavailable"


def test_replay_is_deterministic_input_is_unchanged_and_reset_clears_context():
    stream = [
        event("observation", observation("radius"), sequence=0, run_id="recorded"),
        event("action_proposed", {"kind": "CLICK", "target": "0:execute"}, sequence=1, run_id="recorded"),
        event(
            "episode_summary", {"completed": False, "failure_reason": SECRET}, sequence=2, run_id="recorded"
        ),
    ]
    before = deepcopy(stream)
    reducer = MascotStatusReducer()
    first = [reducer.consume(item) for item in stream]
    reducer.reset()
    assert first == [reducer.consume(item) for item in stream]
    assert stream == before
    assert reducer.consume(stream[0]) == first[-1]  # Duplicate/out-of-order event is not replayed.
    changed = reducer.consume(
        event("action_proposed", {"kind": "CLICK", "target": "0:execute"}, sequence=0, run_id="new")
    )
    assert changed["caption"] == "Preparing calculation"


@pytest.mark.parametrize("field", ["operation", "destination", "instruction", "feedback"])
def test_arbitrary_observation_values_cannot_become_captions(field):
    reducer = MascotStatusReducer()
    obs = observation()
    obs[field] = SECRET
    obs["calculation"][field] = SECRET
    reducer.consume(event("observation", obs))
    for key in ("operation", "execute", "copy", "destination"):
        result = reducer.consume(
            event(
                "action_proposed",
                {
                    "kind": "SELECT" if key in ("operation", "destination") else "CLICK",
                    "target": "0:" + key,
                    "value": SECRET,
                },
            )
        )
        assert_private_free(result)


def test_stale_revision_unknown_operation_and_neural_activity_never_invent_reasoning():
    reducer = MascotStatusReducer()
    reducer.consume(event("observation", observation("distance", revision=4)))
    assert (
        reducer.consume(event("action_proposed", {"kind": "CLICK", "target": "3:execute"}))["caption"]
        == "Status unavailable"
    )
    paused = reducer.consume(event("state", {"status": "paused"}))
    assert reducer.consume(event("neural_activity", {"thoughts": SECRET, "confidence": 1.0})) == paused
    unknown = reducer.consume(
        event("action_proposed", {"kind": "SELECT", "target": "4:operation", "value": [SECRET]})
    )
    assert unknown["caption"] == "Status unavailable"


@pytest.mark.parametrize(
    "kind,surface,caption",
    [
        ("HOVER", "spectrum", "Reading spectrum"),
        ("HOVER", "chart", "Reading light curve"),
        ("SCROLL", "chart", "Adjusting chart view"),
    ],
)
def test_visible_sensing_actions_are_not_reported_as_reasoning(kind, surface, caption):
    result = MascotStatusReducer().consume(
        event("action_proposed", {"kind": kind, "surface": surface, "value": SECRET})
    )
    assert result["caption"] == caption
    assert result["source_label"] == "Proposed action"


def test_snapshot_is_detached_and_old_envelope_defaults_are_supported():
    reducer = MascotStatusReducer()
    snapshot = reducer.snapshot()
    snapshot["caption"] = SECRET
    assert reducer.snapshot()["caption"] == "Ready"
    assert reducer.consume({"event": "state", "payload": {"status": "paused"}})["caption"] == "Paused"


def calculated(reducer, *, wrapped=False):
    obs = observation()
    obs["calculation"]["last_operation"] = {
        "kind": "calculate",
        "operation": "distance",
        "result": {"ok": True},
    }
    item = event("action_result", {"observation": obs})
    return reducer.consume(nested(item) if wrapped else item)


@pytest.mark.parametrize("wrapped", [False, True])
def test_generic_heartbeats_preserve_recent_result_without_claiming_ongoing_calculation(wrapped):
    reducer = MascotStatusReducer()
    assert calculated(reducer, wrapped=wrapped)["caption"] == "Calculated distance"
    for item in [
        event("state", {"status": "running"}),
        event("state", {}),
        event("neural_activity", {"thoughts": SECRET}),
    ]:
        result = reducer.consume(item)
        assert result == {
            "caption": "Calculated distance",
            "activity": "recent",
            "source_label": "Recent result",
        }
    item = event("observation", observation())
    assert reducer.consume(nested(item) if wrapped else item)["caption"] == "Calculated distance"


def test_recent_proposal_has_a_distinct_label_and_pause_stop_error_override():
    reducer = MascotStatusReducer()
    action = event("action_proposed", {"kind": "SELECT", "target": "0:operation", "value": "distance"})
    reducer.consume(action)
    assert reducer.consume(event("state", {"status": "running"})) == {
        "caption": "Selecting distance calculation",
        "activity": "recent",
        "source_label": "Recent action",
    }
    for item, caption in [
        (event("state", {"status": "paused"}), "Paused"),
        (event("state", {"status": "stopped"}), "Stopped"),
        (event("error", {"message": SECRET}), "Stopped"),
    ]:
        reducer.consume(action)
        assert reducer.consume(item)["caption"] == caption
        assert reducer.consume(event("neural_activity", {}))["caption"] == caption


def test_changed_star_or_component_clears_recent_caption_instead_of_showing_old_work():
    reducer = MascotStatusReducer()
    calculated(reducer, wrapped=True)
    other = nested(event("state", {"status": "running"}), star="ANOTHER")
    assert reducer.consume(other)["caption"] == "Running"
    calculated(reducer, wrapped=True)
    changed = event("state", {"status": "running", "current_star": {"star": "ANOTHER", "ordinal": 2}})
    assert reducer.consume(changed)["caption"] == "Running"
    calculated(reducer, wrapped=True)
    changed = nested(event("state", {"status": "running"}), component="stellar.color")
    assert reducer.consume(changed)["caption"] == "Running"


@pytest.mark.parametrize("bad", [{}, [], {"credential": SECRET}, [SECRET], True, 1.0])
def test_unhashable_and_malformed_scalars_are_safe_everywhere(bad):
    reducer = MascotStatusReducer()
    items = [
        {"event": bad, "payload": {}},
        event("state", {"status": bad}),
        event("state", {"phase": bad}),
        event("state", {"current_star": bad}),
        event("action_proposed", {"kind": bad, "target": "0:operation", "value": "distance"}),
        event("action_proposed", {"kind": "SELECT", "target": "0:operation", "value": bad}),
        event("action_proposed", {"kind": "TYPE", "target": bad, "destination": bad, "value": SECRET}),
        event("action_result", {"observation": {"calculation": {"operation": bad, "destination": bad}}}),
        nested(event("observation", observation()), component=bad),
    ]
    for item in items:
        assert_private_free(reducer.consume(item))
