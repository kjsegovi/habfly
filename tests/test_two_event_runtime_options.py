"""Two-event reference permission gates; all browser/model components injected."""
# ruff: noqa: F401,F811

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_browser_project_runtime import rig as bridge_rig
from test_browser_project_steps import rig as owner_rig
from test_browser_star_session import advance_to, create, create_positive, positive_rig, rig
from test_browser_star_shallow import shallow_rig
from test_runtime_browser_project import command, integrated, options, start

from habfly.browser import BrowserSafetyStop
from habfly.browser_project_steps import BrowserProjectSteps
from habfly.runtime import RunOptions, parse_run_options


def enabled_options(root, **changes):
    return {
        "project_two_event_reference": True,
        "project_reference_shallow_transits": True,
        "project_reference_planet_continuation": True,
        "browser_planet_pilot": str(root / "planet"),
        "browser_planet_final_evaluation": str(root / "planet-final"),
        **changes,
    }


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_runtime_flag_rejects_non_boolean_without_artifacts(tmp_path, value):
    with pytest.raises(ValueError):
        parse_run_options(options(tmp_path, **enabled_options(tmp_path, project_two_event_reference=value)))
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("task", [None, "stellar", "planet_calculations"])
def test_non_project_optin_is_rejected_and_default_is_false(task):
    assert RunOptions().project_two_event_reference is False
    payload = {"project_two_event_reference": True}
    if task is not None:
        payload["task"] = task
    with pytest.raises(ValueError, match="requires_project_task"):
        parse_run_options(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"project_reference_shallow_transits": False},
        {"project_reference_planet_continuation": False},
        {"browser_planet_pilot": None},
        {"browser_planet_final_evaluation": None},
        {"browser_planet_pilot": None, "browser_planet_final_evaluation": None},
    ],
)
def test_explicit_shallow_and_planet_dependencies_cannot_be_omitted(tmp_path, change):
    with pytest.raises(ValueError):
        parse_run_options(options(tmp_path, **enabled_options(tmp_path, **change)))
    assert not (tmp_path / "runs").exists()


def test_valid_optin_only_parses_existing_sources_without_launch(tmp_path):
    parsed = parse_run_options(options(tmp_path, **enabled_options(tmp_path)))
    assert parsed.project_two_event_reference is True
    assert parsed.project_reference_shallow_transits is True
    assert parsed.project_reference_planet_continuation is True
    assert parsed.stars == 1 and parsed.paused is True
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_forwards_explicit_permission_and_preserves_legacy_omission(integrated, enabled):
    changes = enabled_options(integrated.root, project_two_event_reference=enabled)
    bridge = start(integrated, **changes)
    assert not integrated.calls  # Even the injected browser driver has not started.
    assert ("two_event_reference" in bridge.model_options) is enabled
    assert bridge.model_options.get("two_event_reference", False) is enabled
    assert ("two_event_reference_enabled" in bridge.scope) is enabled
    assert bridge.scope.get("two_event_reference_enabled", False) is enabled
    seen, factory = [], bridge._steps_factory

    def capture(*args, **kwargs):
        seen.append(deepcopy(kwargs["model_options"]))
        return factory(*args, **kwargs)

    bridge._steps_factory = capture
    for _ in range(10):
        command(integrated, "step")
        if bridge.phase == "awaiting_class_source":
            break
    assert bridge.phase == "awaiting_class_source" and len(seen) == 1
    assert seen[0] == bridge.model_options
    assert not bridge.state()["task_completed"] and not bridge.state()["project_completed"]


@pytest.mark.parametrize("value", [False, 1, None])
def test_runtime_permission_mutation_prevents_even_injected_launch(integrated, value):
    start(integrated, **enabled_options(integrated.root))
    integrated.runtime.options.project_two_event_reference = value
    with pytest.raises(ValueError, match="authorization_changed"):
        integrated.runtime.advance_if_due()
    assert not integrated.calls


@pytest.mark.parametrize("target", ["model", "scope", "shallow_model", "shallow_options"])
def test_bridge_permission_mutation_stops_before_injected_driver(integrated, target):
    bridge = start(integrated, **enabled_options(integrated.root))
    if target == "model":
        bridge.model_options["two_event_reference"] = 1
    elif target == "scope":
        bridge.scope["two_event_reference_enabled"] = 1
    elif target == "shallow_model":
        bridge.model_options["shallow_reference"] = False
    else:
        integrated.runtime.options.project_reference_shallow_transits = False
    bridge.step()
    assert bridge.finished and bridge.phase == "stopped"
    assert "two_event_permission_changed" in bridge.failure
    assert not integrated.calls
    assert not bridge.state()["task_completed"]


def owner(rig, *, enabled=True, **changes):
    model = {key: key for key in ("dataset", "checkpoint", "color_experiment", "graph_path")}
    if enabled:
        model.update(
            shallow_reference=True,
            two_event_reference=True,
            planet_pilot="planet",
            planet_final_evaluation="planet-final",
        )
    model.update(changes)
    return BrowserProjectSteps(
        rig.page,
        rig.config,
        rig.root / "owner",
        run_history=rig.root,
        journal=rig.journal,
        star="ALPHA",
        model_options=model,
        reference_planet_continuation=enabled,
        component_factory=rig.factory,
        emit=lambda *event: rig.events.append(event),
        _clock=lambda: rig.clock.now,
    )


@pytest.mark.parametrize("changes", [{"two_event_reference": 1}, {"shallow_reference": False}])
def test_project_owner_rejects_invalid_permission_before_artifacts(owner_rig, changes):
    with pytest.raises(BrowserSafetyStop, match="invalid_two_event_reference"):
        owner(owner_rig, **changes)
    assert not (owner_rig.root / "owner").exists() and not owner_rig.calls


@pytest.mark.parametrize("enabled", [False, True])
def test_project_owner_forwards_only_explicit_optin_to_star(owner_rig, enabled):
    instance = owner(owner_rig, enabled=enabled)
    instance.start()
    instance.provide_class(
        class_dir=owner_rig.root / "class", selected_class="main_sequence", lifetime_prefix="Ga"
    )
    instance.step()
    assert len(owner_rig.children) == 1
    forwarded = owner_rig.children[0].options
    assert ("two_event_reference" in forwarded) is enabled
    assert forwarded.get("two_event_reference", False) is enabled
    assert instance.scope.get("two_event_reference_enabled", False) is enabled
    assert not instance.state()["task_completed"]


@pytest.mark.parametrize("target", ["model", "scope"])
def test_owner_permission_drift_prevents_child_construction(owner_rig, target):
    instance = owner(owner_rig)
    instance.start()
    instance.provide_class(
        class_dir=owner_rig.root / "class", selected_class="main_sequence", lifetime_prefix="Ga"
    )
    mapping, key = (
        (instance.model_options, "two_event_reference")
        if target == "model"
        else (instance.scope, "two_event_reference_enabled")
    )
    mapping[key] = 1
    instance.step()
    assert instance.finished and instance.phase == "stopped"
    assert "two_event_permission_changed" in instance.failure
    assert not owner_rig.children


@pytest.mark.parametrize("value", [1, None, "true"])
def test_star_permission_is_strict_before_construction(rig, value):
    with pytest.raises(ValueError):
        create_positive(rig, shallow_reference=True, two_event_reference=value)
    assert not rig.calls and not (rig.root / "run").exists()


def test_star_two_event_permission_cannot_enable_shallow_or_planet_implicitly(rig):
    with pytest.raises(ValueError):
        create(rig, two_event_reference=True)
    assert not rig.calls and not (rig.root / "run").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_star_passes_explicit_permission_to_sensor_constructor_only(shallow_rig, enabled):
    session = create_positive(shallow_rig, shallow_reference=True, two_event_reference=enabled)
    advance_to(session, "shallow_reference")
    session.advance()  # Constructor only: no probe or browser action.
    assert len(shallow_rig.sensor_options) == 1
    received = shallow_rig.sensor_options[0]
    assert ("allow_two_events" in received) is enabled
    assert received.get("allow_two_events", False) is enabled
    assert not any(call == ("advance", "shallow") for call in shallow_rig.calls)
    assert not session.state()["task_completed"]


@pytest.mark.parametrize("target", ["permission", "scope"])
def test_star_permission_tamper_stops_before_sensor_constructor(shallow_rig, target):
    session = create_positive(shallow_rig, shallow_reference=True, two_event_reference=True)
    advance_to(session, "shallow_reference")
    if target == "permission":
        session.two_event_reference = 1
    else:
        session.scope["two_event_reference_enabled"] = 1
    session.advance()
    assert session.finished and session.phase == "stopped"
    assert session.failure == "star_session_two_event_permission_changed"
    assert not shallow_rig.sensors


def test_new_three_star_profile_preserves_caps_and_held_thirty_stays_paused():
    base = json.loads(Path("configs/browser_project_supplied_three_star.json").read_bytes())
    selected = json.loads(Path("configs/browser_project_two_event_three_star.json").read_bytes())
    assert {key for key in base.keys() | selected.keys() if base.get(key) != selected.get(key)} == {
        "artifact_dir",
        "project_two_event_reference",
    }
    assert selected["project_two_event_reference"] is True
    assert selected["project_reference_shallow_transits"] is True
    assert selected["stars"] == 3 and selected["paused"] is True
    assert selected["project_max_advances"] == 512
    assert selected["project_max_seconds"] == 1800
    assert selected["project_campaign_max_seconds"] == base["project_campaign_max_seconds"]
    assert selected.get("project_allow_scoring", False) is False
    assert selected.get("project_allow_submission", False) is False
    held = json.loads(Path("configs/browser_project_thirty_star.json").read_bytes())
    assert held["stars"] == 30 and held["paused"] is True
    assert held["project_two_event_reference"] is True
    assert held["project_campaign_max_seconds"] == "uncapped"
