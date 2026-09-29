"""Explicit demo shortcut scheduling only; no browser, models or real writes."""
# ruff: noqa: F811

from pathlib import Path

import pytest
from test_baseline_band_owner import class_ready, owner
from test_browser_project_steps import rig as project_rig  # noqa: F401

from habfly.browser import BrowserSafetyStop

FLAGS = {
    "allow_baseline_edge_reference": True,
    "allow_baseline_band_reference": True,
    "allow_single_event_reference": True,
}


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_strict_optin_before_output_or_components(project_rig, value):
    with pytest.raises(BrowserSafetyStop, match="invalid_single_event_reference"):
        owner(project_rig, model_options={**FLAGS, "allow_single_event_reference": value})
    assert not project_rig.calls and not (project_rig.root / "runtime").exists()


def test_shortcut_requires_explicit_baseline_band_permission(project_rig):
    with pytest.raises(BrowserSafetyStop, match="invalid_single_event_reference"):
        owner(project_rig, model_options={**FLAGS, "allow_baseline_band_reference": False})
    assert not project_rig.calls and not (project_rig.root / "runtime").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_project_only_forwards_explicit_shortcut_flag(project_rig, enabled):
    component = owner(project_rig, model_options=FLAGS if enabled else {})
    try:
        class_ready(project_rig, component)
        component.step()
        options = project_rig.children[0].options
        assert options.get("allow_single_event_reference", False) is enabled
        assert ("allow_single_event_reference" in options) is enabled
        assert ("single_event_reference_enabled" in component.scope) is enabled
        assert ("single_event_source_sha256" in component.scope) is enabled
        assert not any(call[0] == "dispatch" for call in project_rig.calls)
    finally:
        component.close()


@pytest.mark.parametrize("kind", ["option", "alias", "band", "scope", "hash", "disk", "source"])
@pytest.mark.parametrize("callback", [False, True])
def test_shortcut_revocation_blocks_following_native_dispatch(project_rig, monkeypatch, kind, callback):
    component = owner(project_rig, model_options=FLAGS)
    changed = [False]
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            original(path) + b" " if path == component._single_event_source and changed[0] else original(path)
        ),
    )

    def revoke():
        if kind in {"option", "alias"}:
            component.model_options["allow_single_event_reference"] = False if kind == "option" else 1
        elif kind == "band":
            component.model_options["allow_baseline_band_reference"] = False
        elif kind == "scope":
            component.scope["single_event_reference_enabled"] = False
        elif kind == "hash":
            component.scope["single_event_source_sha256"] = "0" * 64
        elif kind == "disk":
            path = component.output / "scope.json"
            path.write_bytes(path.read_bytes() + b" ")
        else:
            changed[0] = True

    try:
        class_ready(project_rig, component)
        component.step()
        if callback:

            def on_event(event, payload):
                if event == "action_proposed":
                    revoke()

            component._callback = on_event
        else:
            revoke()
        component.step()
        assert component.status == "stopped"
        assert not component.state()["task_completed"]
        assert not any(call[0] == "dispatch" for call in project_rig.calls)
    finally:
        component.close()
