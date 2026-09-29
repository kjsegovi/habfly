"""Opt-in shortcut plumbing only: no browser, model, training or course run."""
# ruff: noqa: F401,F811

import hashlib
import json
from pathlib import Path

import pytest
from test_autonomous_validation import no_network_or_models, source_fixture
from test_autonomous_validation import options as source_options
from test_baseline_band_runtime import permissions as band_permissions
from test_browser_project_runtime import rig as bridge_rig
from test_browser_project_tui_launcher import launcher
from test_runtime_browser_project import command, integrated, options, start

from habfly.autonomous_validation import AutonomousValidationError, validate_autonomous_decision_sources
from habfly.runtime import RunOptions, parse_run_options

PROFILE = Path("configs/browser_project_single_event_three_star.json")


def permissions(root, **changes):
    return {**band_permissions(root), "project_single_event_reference": True, **changes}


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_single_event_boolean_aliases_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        parse_run_options(options(tmp_path, **permissions(tmp_path, project_single_event_reference=value)))
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("task", [None, "stellar", "planet_calculations"])
def test_single_event_defaults_false_and_requires_project(task):
    assert RunOptions().project_single_event_reference is False
    payload = {"project_single_event_reference": True, **({"task": task} if task else {})}
    with pytest.raises(ValueError, match="requires_project_task"):
        parse_run_options(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"project_baseline_band_reference": False},
        {"project_baseline_edge_reference": False},
        {"project_reference_planet_continuation": False},
        {"browser_planet_pilot": None},
        {"browser_planet_final_evaluation": None},
    ],
)
def test_single_event_requires_band_and_its_explicit_dependencies(tmp_path, change):
    with pytest.raises(ValueError):
        parse_run_options(options(tmp_path, **permissions(tmp_path, **change)))
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_forwarding_optional_manifest_and_false_completion(integrated, enabled):
    bridge = start(integrated, **permissions(integrated.root, project_single_event_reference=enabled))
    assert not integrated.calls
    assert ("allow_single_event_reference" in bridge.model_options) is enabled
    assert bridge.model_options.get("allow_single_event_reference", False) is enabled
    for key in (
        "single_event_reference_enabled",
        "single_event_source_sha256",
        "single_event_policy",
        "single_event_policy_sha256",
    ):
        assert (key in bridge.scope) is enabled
    if enabled:
        from habfly.planet_window_single_event import policy_manifest

        assert bridge.scope["single_event_policy"] == policy_manifest()
        assert (
            bridge.scope["single_event_source_sha256"]
            == hashlib.sha256(bridge._single_event_source.read_bytes()).hexdigest()
        )
        assert (
            bridge.scope["single_event_policy_sha256"]
            == hashlib.sha256(bridge._single_event_policy).hexdigest()
        )
    seen, factory = [], bridge._steps_factory

    def child(*args, **kwargs):
        seen.append(kwargs["model_options"].copy())
        return factory(*args, **kwargs)

    bridge._steps_factory = child
    for _ in range(10):
        command(integrated, "step")
        if bridge.phase == "awaiting_class_source":
            break
    assert bridge.phase == "awaiting_class_source" and seen == [bridge.model_options]
    assert not bridge.state()["task_completed"] and not bridge.state()["project_completed"]


@pytest.mark.parametrize("value", [False, 1, None])
def test_runtime_authorization_revokes_new_option_before_any_driver(integrated, value):
    start(integrated, **permissions(integrated.root))
    integrated.runtime.options.project_single_event_reference = value
    with pytest.raises(ValueError, match="authorization_changed"):
        integrated.runtime.advance_if_due()
    assert not integrated.calls


@pytest.mark.parametrize(
    "change",
    [
        "model",
        "scope",
        "hash",
        "manifest",
        "manifest_hash",
        "source",
        "live_manifest",
        "band_model",
        "band_options",
    ],
)
def test_bridge_pins_option_manifest_and_source_before_dispatch(integrated, monkeypatch, change):
    bridge = start(integrated, **permissions(integrated.root))
    if change == "model":
        bridge.model_options["allow_single_event_reference"] = 1
    elif change == "scope":
        bridge.scope["single_event_reference_enabled"] = 1
    elif change == "hash":
        bridge.scope["single_event_source_sha256"] = "0" * 64
    elif change == "manifest":
        bridge.scope["single_event_policy"]["unexpected"] = True
    elif change == "manifest_hash":
        bridge.scope["single_event_policy_sha256"] = "0" * 64
    elif change == "source":
        original = Path.read_bytes
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda p: original(p) + b" " if p == bridge._single_event_source else original(p),
        )
    elif change == "live_manifest":
        monkeypatch.setattr(bridge, "_single_event_policy_bytes", lambda: b"{}")
    elif change == "band_model":
        bridge.model_options["allow_baseline_band_reference"] = False
    else:
        integrated.runtime.options.project_baseline_band_reference = False
    bridge.step()
    assert bridge.finished and bridge.phase == "stopped"
    assert "single_event_permission_changed" in bridge.failure
    assert not integrated.calls and not bridge.state()["task_completed"]


def test_single_event_callback_revocation_prevents_next_action(integrated):
    bridge = start(integrated, **permissions(integrated.root))
    bridge._callback = lambda *_: bridge.scope.update(single_event_reference_enabled=False)
    bridge.step()
    bridge.step()
    assert bridge.finished and bridge.phase == "stopped"
    assert not any(call[0] in {"goto", "setup_advance", "model_step"} for call in integrated.calls)


def test_only_new_small_profile_and_held_thirty_enable_single_event_without_budget_change():
    base_path = Path("configs/browser_project_baseline_band_three_star.json")
    before = base_path.read_bytes()
    base, selected = json.loads(before), json.loads(PROFILE.read_bytes())
    assert selected == {
        **base,
        "project_single_event_reference": True,
        "project_save_strategy": "autosave",
        "artifact_dir": "experiments/browser-project-single-event-three-star",
    }
    assert selected["stars"] == 3 and selected["paused"] is True
    assert selected["project_campaign_max_seconds"] == 5400
    assert selected["project_max_seconds"] == 1800 and selected["project_max_advances"] == 512
    assert selected["project_allow_scoring"] is selected["project_allow_submission"] is False
    for path in Path("configs").glob("browser_project*.json"):
        assert json.loads(path.read_bytes()).get("project_single_event_reference", False) is (
            path in {PROFILE, Path("configs/browser_project_thirty_star.json")}
        )
    held = json.loads(Path("configs/browser_project_thirty_star.json").read_bytes())
    assert held["stars"] == 30 and held["paused"] is True
    assert held["project_campaign_max_seconds"] == "uncapped"
    assert held["project_max_seconds"] == 1800 and held["project_max_advances"] == 512
    assert base_path.read_bytes() == before


@pytest.mark.parametrize("enabled", [False, True])
def test_preflight_pins_single_event_policy_without_model_or_launch(source_options, enabled):
    source_options.update(
        project_baseline_edge_reference=True,
        project_baseline_band_reference=True,
        project_single_event_reference=enabled,
    )
    report = validate_autonomous_decision_sources(source_options)
    references = report["references"]
    assert ("single_event_policy" in references) is enabled
    assert ("reference.single_event_implementation" in report["source_files"]) is enabled
    if enabled:
        from habfly.planet_window_single_event import policy_manifest

        row = report["source_files"]["reference.single_event_implementation"]
        assert references["single_event_policy"] == policy_manifest()
        assert references["single_event_implementation_sha256"] == row["sha256"]
    for key in (
        "model_loaded",
        "inference_executed",
        "training_executed",
        "browser_readiness_verified",
        "scientific_verified",
        "task_completed",
        "project_completed",
        "launch_authorized",
        "thirty_star_launch_authorized",
    ):
        assert report[key] is False
    assert not Path(source_options["artifact_dir"]).exists()


def test_preflight_rechecks_single_event_source_after_initial_hash(source_options, monkeypatch):
    import habfly.autonomous_validation as validation

    source_options.update(
        project_baseline_edge_reference=True,
        project_baseline_band_reference=True,
        project_single_event_reference=True,
    )
    original, seen = validation._Sources.file, []

    def changed(book, role, source, **kwargs):
        if role == "reference.single_event_implementation":
            seen.append(role)
            if len(seen) == 2:
                raise AutonomousValidationError("autonomous_validation_source_changed")
        return original(book, role, source, **kwargs)

    monkeypatch.setattr(validation._Sources, "file", changed)
    with pytest.raises(AutonomousValidationError, match="source_changed"):
        validate_autonomous_decision_sources(source_options)
    assert len(seen) == 2 and not Path(source_options["artifact_dir"]).exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_launcher_labels_enabled_assumption_and_keeps_launch_hold(launcher, capsys, enabled):
    launcher.test_payload.update(
        permissions(launcher.test_profile.parent, project_single_event_reference=enabled)
    )
    launcher.test_profile.write_text(json.dumps(launcher.test_payload))
    assert launcher.main(["--check"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert ("single_event_reference_enabled" in value) is enabled
    assert ("single_event_interpretation" in value) is enabled
    if enabled:
        assert "assumed No despite one possible dip" in value["single_event_interpretation"]
    assert value["credentials_requested"] is value["browser_launched"] is False
    assert value["launch_authorized"] is value["thirty_star_launch_authorized"] is False
