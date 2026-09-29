"""Explicit capture strategy only: no browser launch or changed activity scope."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_probe import config

import habfly.browser_probe as module


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_capture_configuration_rejects_boolean_aliases(value):
    data = config().model_dump()
    data["pinned_control_capture"] = value
    with pytest.raises(ValueError):
        module.BrowserProbeConfig.model_validate(data)


@pytest.mark.parametrize("configured,explicit", [(False, False), (False, True), (True, False), (True, True)])
def test_configured_or_explicit_capture_path_is_chosen_before_reading(configured, explicit, monkeypatch):
    settings = config()
    settings.pinned_control_capture = configured
    frame = object()
    page = SimpleNamespace(url=settings.url, main_frame=frame)
    monkeypatch.setattr(module, "_check_auth_and_modals", lambda _: None)
    calls = []

    class Selected(Exception):
        pass

    def reader(name):
        def capture(f, prefix, cfg):
            assert (f, prefix, cfg) == (frame, "outer", settings)
            calls.append(name)
            raise Selected

        return capture

    monkeypatch.setattr(module, "_read_controls", reader("default"))
    monkeypatch.setattr(module, "_read_controls_pinned", reader("pinned"))
    with pytest.raises(Selected):
        module.inspect_page(page, settings, pin_controls=explicit)
    assert calls == ["pinned" if configured or explicit else "default"]


@pytest.mark.parametrize("value", [None, 1, 0, "true"])
def test_mutated_config_is_rejected_without_reading_any_page(value):
    settings = config()
    settings.pinned_control_capture = value
    with pytest.raises(ValueError, match="pinned_control_capture must be a boolean"):
        module.inspect_page(object(), settings)


def test_new_probe_changes_capture_only_and_preserves_every_boundary():
    original = json.loads(Path("configs/browser_probe.example.json").read_bytes())
    selected = json.loads(Path("configs/browser_probe_pinned.example.json").read_bytes())
    assert selected == {**original, "pinned_control_capture": True}
    assert module.BrowserProbeConfig.model_validate(original).pinned_control_capture is False
    assert module.BrowserProbeConfig.model_validate(selected).pinned_control_capture is True
    profiles = [
        json.loads(Path(f"configs/browser_project_{name}.json").read_bytes())
        for name in ("baseline_band_three_star", "thirty_star")
    ]
    for profile in profiles:
        assert profile["browser_config"] == "configs/browser_probe_pinned.example.json"
        assert profile["paused"] is True
        assert profile["project_max_advances"] == 512 and profile["project_max_seconds"] == 1800
    assert profiles[0]["stars"] == 3 and profiles[1]["stars"] == 30
    assert profiles[1]["project_campaign_max_seconds"] == "uncapped"
