"""Optional baseline-band permission/source checks; never a browser or policy run."""
# ruff: noqa: F401,F811

import hashlib
import json
from pathlib import Path

import pytest
from test_autonomous_validation import no_network_or_models, source_fixture
from test_autonomous_validation import options as source_options
from test_browser_project_runtime import rig as bridge_rig
from test_browser_project_tui_launcher import launcher
from test_runtime_browser_project import command, integrated, options, start

from habfly.autonomous_validation import AutonomousValidationError, validate_autonomous_decision_sources
from habfly.runtime import RunOptions, parse_run_options

PROFILE = Path("configs/browser_project_baseline_band_three_star.json")


def permissions(root, **changes):
    return {
        "project_baseline_band_reference": True,
        "project_baseline_edge_reference": True,
        "project_reference_planet_continuation": True,
        "browser_planet_pilot": str(root / "planet"),
        "browser_planet_final_evaluation": str(root / "planet-final"),
        **changes,
    }


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_boolean_aliases_are_rejected_before_artifacts(tmp_path, value):
    with pytest.raises(ValueError):
        parse_run_options(options(tmp_path, **permissions(tmp_path, project_baseline_band_reference=value)))
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("task", [None, "stellar", "planet_calculations"])
def test_default_false_and_non_project_optin_rejected(task):
    assert RunOptions().project_baseline_band_reference is False
    payload = {"project_baseline_band_reference": True}
    if task:
        payload["task"] = task
    with pytest.raises(ValueError, match="requires_project_task"):
        parse_run_options(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"project_baseline_edge_reference": False},
        {"project_reference_planet_continuation": False},
        {"browser_planet_pilot": None},
        {"browser_planet_final_evaluation": None},
        {"browser_planet_pilot": None, "browser_planet_final_evaluation": None},
    ],
)
def test_new_flag_requires_edge_and_explicit_planet_settings(tmp_path, change):
    with pytest.raises(ValueError):
        parse_run_options(options(tmp_path, **permissions(tmp_path, **change)))
    assert not (tmp_path / "runs").exists()


def test_valid_baseline_band_does_not_implicitly_enable_shallow_or_two_events(tmp_path):
    value = parse_run_options(options(tmp_path, **permissions(tmp_path)))
    assert value.project_baseline_band_reference is True
    assert value.project_reference_shallow_transits is False
    assert value.project_two_event_reference is False
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_deferred_runtime_forwarding_and_default_omission(integrated, enabled):
    bridge = start(integrated, **permissions(integrated.root, project_baseline_band_reference=enabled))
    assert not integrated.calls
    assert ("allow_baseline_band_reference" in bridge.model_options) is enabled
    assert bridge.model_options.get("allow_baseline_band_reference", False) is enabled
    assert ("baseline_band_reference_enabled" in bridge.scope) is enabled
    assert ("baseline_band_source_sha256" in bridge.scope) is enabled
    if enabled:
        expected = hashlib.sha256(bridge._baseline_band_source.read_bytes()).hexdigest()
        assert bridge.scope["baseline_band_source_sha256"] == expected
    seen, factory = [], bridge._steps_factory

    def child(*args, **kwargs):
        seen.append(kwargs["model_options"].copy())
        return factory(*args, **kwargs)

    bridge._steps_factory = child
    for _ in range(10):
        command(integrated, "step")
        if bridge.phase == "awaiting_class_source":
            break
    assert bridge.phase == "awaiting_class_source" and len(seen) == 1
    assert seen[0] == bridge.model_options
    assert not bridge.state()["task_completed"] and not bridge.state()["project_completed"]


@pytest.mark.parametrize("value", [False, 1, None])
def test_runtime_permission_drift_stops_before_injected_driver(integrated, value):
    start(integrated, **permissions(integrated.root))
    integrated.runtime.options.project_baseline_band_reference = value
    with pytest.raises(ValueError, match="authorization_changed"):
        integrated.runtime.advance_if_due()
    assert not integrated.calls


@pytest.mark.parametrize(
    "change",
    ["model", "scope", "hash", "source", "edge_model", "edge_options", "continuation", "pilot", "final"],
)
def test_bridge_rechecks_flag_dependencies_and_source_before_dispatch(integrated, monkeypatch, change):
    bridge = start(integrated, **permissions(integrated.root))
    if change == "model":
        bridge.model_options["allow_baseline_band_reference"] = 1
    elif change == "scope":
        bridge.scope["baseline_band_reference_enabled"] = 1
    elif change == "hash":
        bridge.scope["baseline_band_source_sha256"] = "0" * 64
    elif change == "source":
        original = Path.read_bytes
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda p: original(p) + b" " if p == bridge._baseline_band_source else original(p),
        )
    elif change == "edge_model":
        bridge.model_options["allow_baseline_edge_reference"] = False
    elif change == "edge_options":
        integrated.runtime.options.project_baseline_edge_reference = False
    elif change == "continuation":
        bridge.reference_planet_continuation = False
    elif change == "pilot":
        bridge.model_options["planet_pilot"] = Path("different")
    else:
        bridge.model_options["planet_final_evaluation"] = Path("different")
    bridge.step()
    assert bridge.finished and bridge.phase == "stopped"
    assert "baseline_band_permission_changed" in bridge.failure
    assert not integrated.calls and not bridge.state()["task_completed"]


def test_callback_revocation_blocks_next_scheduled_stage(integrated):
    bridge = start(integrated, **permissions(integrated.root))

    def revoke(*_):
        bridge.scope["baseline_band_reference_enabled"] = False

    bridge._callback = revoke
    bridge.step()
    # This callback follows driver initialization, not a pre-launch boundary.
    # Its revocation must prevent navigation or any subsequent setup action.
    bridge.step()
    assert bridge.finished and bridge.phase == "stopped"
    assert not any(call[0] in {"goto", "setup_advance", "model_step"} for call in integrated.calls)


def test_new_profile_is_exact_bounded_copy_and_only_held_full_profile_is_aligned():
    base = json.loads(Path("configs/browser_project_two_event_three_star.json").read_bytes())
    selected = json.loads(PROFILE.read_bytes())
    assert selected == {
        **base,
        "project_baseline_band_reference": True,
        "browser_config": "configs/browser_probe_pinned.example.json",
        "artifact_dir": "experiments/browser-project-baseline-band-three-star",
    }
    assert selected["stars"] == 3 and selected["paused"] is True
    assert selected["project_campaign_max_seconds"] == 5400
    assert selected["project_max_seconds"] == 1800 and selected["project_max_advances"] == 512
    assert selected["project_allow_scoring"] is selected["project_allow_submission"] is False
    for path in Path("configs").glob("browser_project*.json"):
        value = json.loads(path.read_bytes())
        assert value.get("project_baseline_band_reference", False) is (
            path
            in {
                PROFILE,
                Path("configs/browser_project_single_event_three_star.json"),
                Path("configs/browser_project_thirty_star.json"),
            }
        )
    held = json.loads(Path("configs/browser_project_thirty_star.json").read_bytes())
    assert held["stars"] == 30 and held["paused"] is True
    assert held["project_baseline_band_reference"] is True
    assert held["project_campaign_max_seconds"] == "uncapped"


@pytest.mark.parametrize("enabled", [False, True])
def test_preflight_pins_only_explicit_new_policy_without_model_or_launch(source_options, enabled):
    source_options.update(project_baseline_edge_reference=True, project_baseline_band_reference=enabled)
    report = validate_autonomous_decision_sources(source_options)
    references = report["references"]
    assert ("baseline_band_policy" in references) is enabled
    assert ("reference.baseline_band_implementation" in report["source_files"]) is enabled
    if enabled:
        from habfly.planet_window_baseline_band import policy_manifest

        row = report["source_files"]["reference.baseline_band_implementation"]
        assert references["baseline_band_policy"] == policy_manifest()
        assert references["baseline_band_implementation_sha256"] == row["sha256"]
        assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
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


def test_preflight_rechecks_new_source_after_initial_hash(source_options, monkeypatch):
    import habfly.autonomous_validation as validation

    source_options.update(project_baseline_edge_reference=True, project_baseline_band_reference=True)
    original = validation._Sources.file
    seen = []

    def changed(book, role, source, **kwargs):
        if role == "reference.baseline_band_implementation":
            seen.append(role)
            if len(seen) == 2:
                raise AutonomousValidationError("autonomous_validation_source_changed")
        return original(book, role, source, **kwargs)

    monkeypatch.setattr(validation._Sources, "file", changed)
    with pytest.raises(AutonomousValidationError, match="source_changed"):
        validate_autonomous_decision_sources(source_options)
    assert len(seen) == 2 and not Path(source_options["artifact_dir"]).exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_launcher_check_labels_assumption_without_granting_authority(launcher, capsys, enabled):
    launcher.test_payload.update(
        permissions(launcher.test_profile.parent, project_baseline_band_reference=enabled)
    )
    launcher.test_profile.write_text(json.dumps(launcher.test_payload))
    original = launcher.test_profile.read_bytes()
    assert launcher.main(["--check"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert ("baseline_band_reference_enabled" in value) is enabled
    assert ("baseline_band_interpretation" in value) is enabled
    if enabled:
        assert value["baseline_band_reference_enabled"] is True
        assert "assumed No, not proven absence or learned perception" in value["baseline_band_interpretation"]
    assert value["credentials_requested"] is value["browser_launched"] is False
    assert value["launch_authorized"] is value["thirty_star_launch_authorized"] is False
    assert launcher.test_profile.read_bytes() == original
    assert not Path(launcher.test_payload["artifact_dir"]).exists()
