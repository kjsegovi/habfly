"""Injected parent scheduling guards; no browser, checkpoints or model runs."""

# ruff: noqa: F811 - imported pytest fixtures

from pathlib import Path

import pytest
from test_browser_project_steps import rig as project_rig  # noqa: F401
from test_browser_star_session import advance_to, create
from test_browser_star_session import rig as star_rig  # noqa: F401

import habfly.browser_project_steps as project
import habfly.browser_star_session as star
from habfly.browser import BrowserSafetyStop

FLAGS = {"allow_baseline_edge_reference": True, "allow_baseline_band_reference": True}


def owner(rig, *, model_options=None, emit=None):
    options = {key: key for key in ("dataset", "checkpoint", "color_experiment", "graph_path")}
    options.update(model_options or {})
    return project.BrowserProjectSteps(
        rig.page,
        rig.config,
        rig.root / "runtime",
        run_history=rig.root,
        journal=rig.journal,
        star="ALPHA",
        model_options=options,
        component_factory=rig.factory,
        emit=emit or (lambda *event: rig.events.append(event)),
        _clock=lambda: rig.clock.now,
    )


def class_ready(rig, component):
    component.start()
    component.provide_class(
        class_dir=rig.root / "class", selected_class="main_sequence", lifetime_prefix="Ga"
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_star_requires_strict_band_boolean_before_components_or_output(star_rig, value):
    with pytest.raises(ValueError):
        create(star_rig, allow_baseline_edge_reference=True, allow_baseline_band_reference=value)
    assert not star_rig.calls and not (star_rig.root / "run").exists()


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_project_requires_strict_band_boolean_before_components_or_output(project_rig, value):
    with pytest.raises(BrowserSafetyStop, match="invalid_baseline_band_reference"):
        owner(project_rig, model_options={**FLAGS, "allow_baseline_band_reference": value})
    assert not project_rig.calls and not (project_rig.root / "runtime").exists()


def test_both_parent_layers_require_existing_edge_permission(star_rig, project_rig):
    with pytest.raises(ValueError):
        create(star_rig, allow_baseline_band_reference=True)
    with pytest.raises(BrowserSafetyStop, match="invalid_baseline_band_reference"):
        owner(project_rig, model_options={"allow_baseline_band_reference": True})
    assert not star_rig.calls and not project_rig.calls


@pytest.mark.parametrize("enabled", [False, True])
def test_star_forwards_exact_window_optin_only_when_explicit(star_rig, monkeypatch, enabled):
    original, seen = star.PlanetWindowSession, []

    def window(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(star, "PlanetWindowSession", window)
    session = create(star_rig, **(FLAGS if enabled else {}))
    advance_to(session, "planet_window")
    session.advance()
    assert len(seen) == 1
    assert set(seen[0]) == {"run_history", "select_no", "emit"} | (FLAGS.keys() if enabled else set())
    assert seen[0]["select_no"] is True and seen[0]["run_history"] == star_rig.root
    for key, value in FLAGS.items():
        assert (seen[0].get(key) is value) if enabled else key not in seen[0]
    assert (
        (session.scope.get("baseline_band_reference_enabled") is True)
        if enabled
        else ("baseline_band_reference_enabled" not in session.scope)
    )
    session.abort()


@pytest.mark.parametrize("enabled", [False, True])
def test_project_forwards_unchanged_model_flags_only_when_explicit(project_rig, enabled):
    component = owner(project_rig, model_options=FLAGS if enabled else {})
    try:
        class_ready(project_rig, component)
        component.step()
        options = project_rig.children[0].options
        for key, value in FLAGS.items():
            assert (options.get(key) is value) if enabled else key not in options
        assert (
            (component.scope.get("baseline_band_reference_enabled") is True)
            if enabled
            else ("baseline_band_reference_enabled" not in component.scope)
        )
        assert not any(call[0] == "dispatch" for call in project_rig.calls)
    finally:
        component.close()


def mutate(component, kind, *, project_owner=False):
    if kind in {"option", "alias"}:
        value = False if kind == "option" else 1
        if project_owner:
            component.model_options["allow_baseline_band_reference"] = value
        else:
            component.allow_baseline_band_reference = value
    elif kind == "scope":
        component.scope["baseline_band_reference_enabled"] = False
    elif kind == "edge_scope":
        component.scope["baseline_edge_reference_enabled"] = False
    elif kind == "disk":
        path = component.output / "scope.json"
        path.write_bytes(path.read_bytes() + b" ")


def patched_source(monkeypatch, component, kind):
    changed = [False]
    path = component._baseline_edge_source if kind == "edge_source" else component._baseline_band_source
    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: original(p) + b" " if p == path and changed[0] else original(p)
    )
    return changed


@pytest.mark.parametrize("kind", ["option", "alias", "scope", "disk", "source"])
def test_star_midrun_permission_drift_stops_before_child_decision(star_rig, monkeypatch, kind):
    session = create(star_rig, **FLAGS)
    session.advance()
    if kind == "source":
        patched_source(monkeypatch, session, kind)[0] = True
    else:
        mutate(session, kind)
    before = list(star_rig.calls)
    session.advance()
    assert session.phase == "stopped" and not session.state()["task_completed"]
    assert not any(call[0] == "advance" for call in star_rig.calls[len(before) :])
    assert session.failure in {
        "star_session_baseline_band_permission_changed",
        "star_session_source_hash_changed",
    }


@pytest.mark.parametrize("kind", ["option", "alias", "scope", "disk", "source"])
def test_project_midrun_permission_drift_prevents_child_dispatch(project_rig, monkeypatch, kind):
    component = owner(project_rig, model_options=FLAGS)
    try:
        class_ready(project_rig, component)
        component.step()
        if kind == "source":
            patched_source(monkeypatch, component, kind)[0] = True
        else:
            mutate(component, kind, project_owner=True)
        component.step()
        assert component.status == "stopped" and not component.state()["task_completed"]
        assert not any(call[0] == "dispatch" for call in project_rig.calls)
    finally:
        component.close()


@pytest.mark.parametrize("kind", ["option", "scope", "disk", "source", "edge_scope", "edge_source"])
def test_star_action_callback_revocation_never_returns_to_native_dispatch(star_rig, monkeypatch, kind):
    session = create(star_rig, **FLAGS)
    session.advance()
    child = star_rig.components[0]
    changed = patched_source(monkeypatch, session, kind)
    native = []

    def proposed():
        child.emit(
            {"event": "action_proposed", "run_id": "numeric", "sequence": 0, "payload": {"kind": "TYPE"}}
        )
        native.append("would_dispatch")

    def revoke(event, payload):
        if event == "action_proposed":
            if kind in {"source", "edge_source"}:
                changed[0] = True
            else:
                mutate(session, kind)

    child.advance, session._callback = proposed, revoke
    session.advance()
    assert session.phase == "stopped" and not native
    assert not session.state()["task_completed"]


@pytest.mark.parametrize("kind", ["option", "scope", "disk", "source"])
def test_project_action_callback_revocation_prevents_following_dispatch(project_rig, monkeypatch, kind):
    component = owner(project_rig, model_options=FLAGS)
    try:
        class_ready(project_rig, component)
        component.step()
        changed = patched_source(monkeypatch, component, kind)

        def revoke(event, payload):
            if event == "action_proposed":
                if kind == "source":
                    changed[0] = True
                else:
                    mutate(component, kind, project_owner=True)

        component._callback = revoke
        component.step()
        assert component.status == "stopped" and not component.state()["task_completed"]
        assert not any(call[0] == "dispatch" for call in project_rig.calls)
    finally:
        component.close()
