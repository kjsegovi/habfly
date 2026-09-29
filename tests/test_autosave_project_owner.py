"""Autosave is explicit readback authority, never a persistence acknowledgement."""
# ruff: noqa: F811

from pathlib import Path

import pytest
from test_baseline_band_owner import class_ready, owner
from test_browser_project_steps import rig as project_rig  # noqa: F401

from habfly.browser import BrowserSafetyStop


@pytest.mark.parametrize("value", [None, False, True, 0, 1, [], {}, "auto", "AUTOSAVE", ""])
def test_invalid_strategy_fails_before_any_output_or_child(project_rig, value):
    with pytest.raises(BrowserSafetyStop, match="invalid_save_strategy"):
        owner(project_rig, model_options={"save_strategy": value})
    assert not project_rig.calls and not (project_rig.root / "runtime").exists()


@pytest.mark.parametrize("selected", [None, "explicit", "autosave"])
def test_only_selected_strategy_forwards_and_autosave_has_no_persistence_claim(project_rig, selected):
    settings = {} if selected is None else {"save_strategy": selected}
    component = owner(project_rig, model_options=settings)
    try:
        class_ready(project_rig, component)
        component.step()
        child = project_rig.children[0]
        assert ("save_strategy" in child.options) is (selected is not None)
        assert child.options.get("save_strategy", "explicit") == (selected or "explicit")
        assert ("save_strategy" in component.scope) is (selected == "autosave")
        if selected == "autosave":
            assert component.scope["persistence_verified"] is False
            assert component.scope["autosave_source_sha256"] == component._autosave_sha
        assert not any(call[0] == "dispatch" for call in project_rig.calls)
    finally:
        component.close()


@pytest.mark.parametrize("kind", ["strategy", "alias", "scope", "proof", "hash", "disk", "source"])
@pytest.mark.parametrize("callback", [False, True])
def test_autosave_permission_drift_blocks_next_child_action(project_rig, monkeypatch, kind, callback):
    component = owner(project_rig, model_options={"save_strategy": "autosave"})
    original = Path.read_bytes
    changed = [False]
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            original(path) + b" " if changed[0] and path == component._autosave_source else original(path)
        ),
    )

    def revoke():
        if kind in {"strategy", "alias"}:
            component.model_options["save_strategy"] = "explicit" if kind == "strategy" else True
        elif kind == "scope":
            component.scope["save_strategy"] = "explicit"
        elif kind == "proof":
            component.scope["persistence_verified"] = True
        elif kind == "hash":
            component.scope["autosave_source_sha256"] = "0" * 64
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
        assert component.status == "stopped" and not component.state()["task_completed"]
        assert not any(call[0] == "dispatch" for call in project_rig.calls)
    finally:
        component.close()
