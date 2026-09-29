"""Offline viewport configuration/driver injection only; never launch a browser."""
# ruff: noqa: F811

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_browser_project_runtime import (
    EMAIL,
    PASSWORD,
    create,
    ready,
    rig,  # noqa: F401
    text_artifacts,
)
from test_runtime_browser_project import options
from typer.testing import CliRunner

import habfly.runtime as runtime_module
from habfly.browser import BrowserSafetyStop
from habfly.cli import app
from habfly.runtime import ProjectViewport, RunOptions, parse_run_options

VIEWPORT = {"width": 1600, "height": 1100}


def configured(rig):
    rig.options = rig.options.model_copy(
        update={"task": "browser_project", "project_viewport": ProjectViewport(**VIEWPORT)}
    )
    return rig


@pytest.mark.parametrize(
    "value",
    [
        {"width": True, "height": 1100},
        {"width": 1600, "height": False},
        {"width": 1600.0, "height": 1100},
        {"width": "1600", "height": 1100},
        {"width": 949, "height": 1100},
        {"width": 2401, "height": 1100},
        {"width": 1600, "height": 599},
        {"width": 1600, "height": 1801},
        {"width": 1600},
        {"width": 1600, "height": 1100, "session": "private-fixture-value"},
        [1600, 1100],
        "private-fixture-value",
    ],
)
def test_invalid_viewport_is_rejected_before_artifacts_or_driver(tmp_path, value):
    payload = options(tmp_path, project_viewport=value)
    with pytest.raises(ValueError, match="browser_project_invalid_start_options") as error:
        parse_run_options(payload)
    assert "private-fixture-value" not in str(error.value)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "viewport", [None, VIEWPORT, {"width": 950, "height": 600}, {"width": 2400, "height": 1800}]
)
def test_viewport_roundtrip_and_bounds_are_explicit(tmp_path, viewport):
    parsed = parse_run_options(options(tmp_path, project_viewport=viewport))
    assert parsed.model_dump(mode="json")["project_viewport"] == viewport
    assert RunOptions.model_validate_json(parsed.model_dump_json()) == parsed


def test_viewport_option_is_not_silently_ignored_by_other_tasks():
    assert parse_run_options({"task": "mini_habworlds"}).project_viewport is None
    with pytest.raises(ValueError, match="project_viewport_requires_project_task"):
        parse_run_options({"task": "mini_habworlds", "project_viewport": VIEWPORT})


def test_current_two_star_profile_opts_in_without_widening_campaign_scope():
    profile = RunOptions.model_validate_json(Path("configs/browser_project_two_star.json").read_text())
    assert profile.project_viewport.model_dump() == VIEWPORT
    assert profile.stars == 2 and profile.project_campaign and profile.paused
    assert profile.project_campaign_max_seconds == 3600
    assert profile.project_max_advances == 512 and profile.project_max_seconds == 1800


def test_cli_forwards_safe_viewport_using_existing_v1_start_options(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime_module, "serve", lambda **kwargs: calls.append(kwargs))
    path = tmp_path / "options.json"
    path.write_text(json.dumps(options(tmp_path, project_viewport=VIEWPORT)))
    result = CliRunner().invoke(app, ["runtime", "--jsonl", "--start-options", str(path)])
    assert result.exit_code == 0, result.output
    assert calls[0]["start_options"]["project_viewport"] == VIEWPORT


def test_omitted_viewport_preserves_original_context_call_and_scope(rig):
    bridge = create(rig)
    assert "project_viewport" not in bridge.scope and "viewport_policy" not in bridge.scope
    bridge.start()
    bridge.step()
    assert ("new_context",) in rig.calls
    assert not any(c[0] == "new_context" and len(c) > 1 for c in rig.calls)
    assert bridge.phase == "opening_preview"
    bridge.close()


def test_fixed_viewport_is_passed_before_page_and_recorded_without_credentials(rig):
    configured(rig)
    bridge = create(rig)
    assert not rig.calls and not bridge.output.exists()
    assert bridge.scope["project_viewport"] == VIEWPORT
    assert bridge.scope["viewport_policy"] == "fixed_context_css_pixels"
    # The caller's options are copied; later input mutation cannot resize the run.
    rig.options.project_viewport.width = 1700
    bridge.start()
    bridge.step()
    assert ("new_context", {"viewport": VIEWPORT}) in rig.calls
    assert rig.calls.index(("new_context", {"viewport": VIEWPORT})) < rig.calls.index(("new_page",))
    assert bridge.page.viewport_size == VIEWPORT and bridge.phase == "opening_preview"
    assert json.loads((bridge.output / "scope.json").read_bytes())["project_viewport"] == VIEWPORT
    assert any(payload.get("project_viewport") == VIEWPORT for _, payload in rig.events)
    text = text_artifacts(bridge.output)
    assert EMAIL not in text and PASSWORD not in text and "preview_sequence_id" not in text
    bridge.close()


@pytest.mark.parametrize(
    "value", [{"width": True, "height": 1100}, {"width": 1600}, {"width": 2401, "height": 1100}]
)
def test_direct_bridge_validation_cannot_bypass_strict_option_contract(rig, value):
    rig.options = rig.options.model_copy(update={"task": "browser_project", "project_viewport": value})
    with pytest.raises(BrowserSafetyStop, match="invalid_project_viewport"):
        create(rig)
    assert not rig.calls and not (rig.root / "run").exists()


@pytest.mark.parametrize("actual", [None, {"width": 1280, "height": 720}, {"width": 1600.0, "height": 1100}])
def test_wrong_actual_context_viewport_stops_before_setup_or_navigation(rig, actual):
    configured(rig)
    rig.behaviour["context_viewport_override"] = actual
    bridge = create(rig)
    bridge.start()
    bridge.step()
    assert bridge.status == "stopped" and bridge.failure == "project_runtime_viewport_changed"
    assert not any(c[0] in {"goto", "setup_construct", "capture_initial"} for c in rig.calls)
    assert bridge.closed and ("context_close",) in rig.calls


@pytest.mark.parametrize("where", ["request", "scope", "policy"])
def test_changed_viewport_declaration_stops_before_launch(rig, where):
    configured(rig)
    bridge = create(rig)
    bridge.start()
    if where == "request":
        bridge.project_viewport["height"] = 1200
    elif where == "scope":
        bridge.scope["project_viewport"]["width"] = 1700
    else:
        bridge.scope["viewport_policy"] = "automatic_resize"
    bridge.step()
    assert bridge.status == "stopped" and bridge.failure == "project_runtime_viewport_settings_changed"
    assert not rig.calls


def test_viewport_drift_before_next_boundary_never_navigates_or_auto_resizes(rig):
    configured(rig)
    bridge = create(rig)
    bridge.start()
    bridge.step()
    bridge.page.viewport_size = {"width": 1280, "height": 720}
    bridge.step()
    assert bridge.failure == "project_runtime_viewport_changed"
    assert not any(c[0] == "goto" for c in rig.calls)
    assert bridge.page.viewport_size == {"width": 1280, "height": 720}
    bridge.close()


def test_setup_event_callback_viewport_drift_stops_before_next_operation(rig):
    configured(rig)
    holder = {}

    def callback(kind, payload):
        rig.events.append((kind, payload))
        if payload.get("component") == "browser.setup":
            holder["bridge"].page.viewport_size = {"width": 1280, "height": 720}

    bridge = create(rig, emit=callback)
    holder["bridge"] = bridge
    bridge.start()
    bridge.step()
    bridge.step()
    bridge.step()
    # The component relay deliberately sanitizes downstream guard exceptions.
    assert bridge.status == "stopped" and bridge.failure == "project_runtime_operation_failed"
    assert bridge._relay.failed
    assert not any(c[0] in {"setup_advance", "capture_initial", "steps_construct"} for c in rig.calls)
    bridge.close()


def test_terminal_callback_viewport_drift_revokes_clean_handoff(rig):
    configured(rig)
    holder = {}

    def callback(kind, payload):
        rig.events.append((kind, payload))
        if kind == "episode_summary" and payload.get("mode") == "fresh_browser_single_star_project_runtime":
            holder["bridge"].page.viewport_size = {"width": 1280, "height": 720}

    bridge = ready(rig, emit=callback)
    holder["bridge"] = bridge
    bridge.step()
    assert bridge.status == "stopped"
    assert bridge.failure == "project_runtime_terminal_campaign_validation_failed"
    assert not bridge.report["task_completed"] and not bridge.report["project_completed"]
    bridge.close()


def test_scope_viewport_does_not_alias_runtime_or_original_request(rig):
    configured(rig)
    bridge = create(rig)
    before = deepcopy(bridge.project_viewport)
    bridge.scope["project_viewport"]["width"] = 1900
    assert bridge.project_viewport == before and rig.options.project_viewport.width == 1600
