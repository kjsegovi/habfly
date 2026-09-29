"""Optional autosave plumbing; no browser, credentials, model or live actions."""
# ruff: noqa: F401,F811

import hashlib
import io
import json
from pathlib import Path

import pytest
from test_autonomous_validation import no_network_or_models, source_fixture
from test_autonomous_validation import options as source_options
from test_browser_project_runtime import rig as bridge_rig
from test_browser_project_tui_launcher import launcher
from test_project_window_wait_wire import parent, relay_window, window
from test_runtime_browser_project import command, integrated, options, start

from habfly.autonomous_validation import AutonomousValidationError, validate_autonomous_decision_sources
from habfly.contracts import RuntimeEvent
from habfly.project_wire import compact_project_event
from habfly.runtime import RunOptions, Runtime, parse_run_options


@pytest.mark.parametrize(
    "value", [None, 0, 1, True, False, b"autosave", [], {}, "AUTOsave", "auto", "", " autosave", "autosave\n"]
)
def test_save_strategy_strict_string_enum(tmp_path, value):
    with pytest.raises(ValueError):
        parse_run_options(options(tmp_path, project_save_strategy=value))
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("task", [None, "stellar", "browser_numeric", "planet_calculations"])
def test_autosave_is_project_only_and_default_remains_explicit(task):
    assert RunOptions().project_save_strategy == "explicit"
    assert parse_run_options({"project_save_strategy": "explicit"}).project_save_strategy == "explicit"
    with pytest.raises(ValueError, match="project_autosave_requires_project_task"):
        parse_run_options({"project_save_strategy": "autosave", **({"task": task} if task else {})})


@pytest.mark.parametrize("strategy", ["explicit", "autosave"])
def test_strategy_forwarding_and_manifest_pins_are_optin_only(integrated, strategy):
    bridge = start(integrated, project_save_strategy=strategy)
    enabled = strategy == "autosave"
    assert not integrated.calls
    assert ("save_strategy" in bridge.model_options) is enabled
    assert bridge.model_options.get("save_strategy", "explicit") == strategy
    for key in (
        "save_strategy",
        "autosave_source_sha256",
        "autosave_manifest",
        "autosave_manifest_sha256",
        "persistence_verified",
    ):
        assert (key in bridge.scope) is enabled
    if enabled:
        from habfly.browser_autosave import autosave_manifest

        assert bridge.scope["autosave_manifest"] == autosave_manifest()
        assert (
            bridge.scope["autosave_source_sha256"]
            == hashlib.sha256(bridge._autosave_source.read_bytes()).hexdigest()
        )
        assert (
            bridge.scope["autosave_manifest_sha256"] == hashlib.sha256(bridge._autosave_manifest).hexdigest()
        )
        assert bridge.scope["persistence_verified"] is False
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


@pytest.mark.parametrize("value", ["explicit", "unknown", None, 1, True])
def test_runtime_authorization_drift_stops_before_driver(integrated, value):
    start(integrated, project_save_strategy="autosave")
    integrated.runtime.options.project_save_strategy = value
    with pytest.raises(ValueError, match="authorization_changed"):
        integrated.runtime.advance_if_due()
    assert not integrated.calls


@pytest.mark.parametrize(
    "change",
    [
        "model",
        "scope",
        "options",
        "source",
        "hash",
        "manifest",
        "manifest_boolean_alias",
        "manifest_hash",
        "live_manifest",
        "persistence",
    ],
)
def test_bridge_revalidates_strategy_and_source_before_dispatch(integrated, monkeypatch, change):
    bridge = start(integrated, project_save_strategy="autosave")
    if change == "model":
        bridge.model_options["save_strategy"] = "explicit"
    elif change == "scope":
        bridge.scope["save_strategy"] = "explicit"
    elif change == "options":
        integrated.runtime.options.project_save_strategy = "explicit"
    elif change == "source":
        original = Path.read_bytes
        monkeypatch.setattr(
            Path, "read_bytes", lambda p: original(p) + b" " if p == bridge._autosave_source else original(p)
        )
    elif change == "hash":
        bridge.scope["autosave_source_sha256"] = "0" * 64
    elif change == "manifest":
        bridge.scope["autosave_manifest"]["unexpected"] = True
    elif change == "manifest_boolean_alias":
        bridge.scope["autosave_manifest"]["user_approved"] = 1
    elif change == "manifest_hash":
        bridge.scope["autosave_manifest_sha256"] = "0" * 64
    elif change == "live_manifest":
        monkeypatch.setattr(bridge, "_autosave_manifest_bytes", lambda: b"{}")
    else:
        bridge.scope["persistence_verified"] = 0
    bridge.step()
    assert bridge.finished and bridge.phase == "stopped"
    assert "save_strategy_changed" in bridge.failure
    assert not integrated.calls and not bridge.state()["task_completed"]


def test_strategy_revoked_in_callback_cannot_proceed_to_navigation(integrated):
    bridge = start(integrated, project_save_strategy="autosave")
    bridge._callback = lambda *_: bridge.scope.update(save_strategy="explicit")
    bridge.step()
    bridge.step()
    assert bridge.finished and bridge.phase == "stopped"
    assert not any(call[0] in {"goto", "setup_advance", "model_step"} for call in integrated.calls)


def test_default_bridge_never_loads_autosave_module(integrated, monkeypatch):
    monkeypatch.setattr(
        "habfly.browser_autosave.autosave_manifest",
        lambda: pytest.fail("legacy must not load autosave metadata"),
    )
    bridge = start(integrated)
    bridge.step()
    assert not bridge.finished and "save_strategy" not in bridge.scope


@pytest.mark.parametrize("strategy", ["explicit", "autosave"])
def test_source_preflight_only_pins_new_helper_when_opted_in(source_options, strategy):
    source_options["project_save_strategy"] = strategy
    report = validate_autonomous_decision_sources(source_options)
    enabled = strategy == "autosave"
    assert ("autosave_manifest" in report["references"]) is enabled
    assert ("reference.autosave_implementation" in report["source_files"]) is enabled
    if enabled:
        from habfly.browser_autosave import autosave_manifest

        assert report["references"]["autosave_manifest"] == autosave_manifest()
        assert (
            report["references"]["autosave_implementation_sha256"]
            == report["source_files"]["reference.autosave_implementation"]["sha256"]
        )
    for key in (
        "model_loaded",
        "inference_executed",
        "training_executed",
        "browser_readiness_verified",
        "task_completed",
        "project_completed",
        "launch_authorized",
        "thirty_star_launch_authorized",
    ):
        assert report[key] is False
    assert not Path(source_options["artifact_dir"]).exists()


def test_preflight_rechecks_helper_after_first_hash(source_options, monkeypatch):
    import habfly.autonomous_validation as validation

    source_options["project_save_strategy"] = "autosave"
    original, seen = validation._Sources.file, []

    def changed(book, role, source, **kwargs):
        if role == "reference.autosave_implementation":
            seen.append(role)
            if len(seen) == 2:
                raise AutonomousValidationError("autonomous_validation_source_changed")
        return original(book, role, source, **kwargs)

    monkeypatch.setattr(validation._Sources, "file", changed)
    with pytest.raises(AutonomousValidationError, match="source_changed"):
        validate_autonomous_decision_sources(source_options)
    assert len(seen) == 2


def test_only_recommended_and_held_profiles_enable_autosave_with_unchanged_limits():
    expected = {
        Path("configs/browser_project_single_event_three_star.json"): (3, 5400),
        Path("configs/browser_project_thirty_star.json"): (30, "uncapped"),
    }
    for path in Path("configs").glob("browser_project*.json"):
        value = json.loads(path.read_bytes())
        assert value.get("project_save_strategy", "explicit") == (
            "autosave" if path in expected else "explicit"
        )
        if path in expected:
            assert (value["stars"], value["project_campaign_max_seconds"]) == expected[path]
            assert value["project_max_advances"] == 512 and value["project_max_seconds"] == 1800
            assert value["paused"] is True


@pytest.mark.parametrize("strategy", ["explicit", "autosave"])
def test_launcher_readonly_check_reports_assumption_not_save(launcher, capsys, strategy):
    launcher.test_payload["project_save_strategy"] = strategy
    launcher.test_profile.write_text(json.dumps(launcher.test_payload))
    assert launcher.main(["--check"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert ("save_strategy" in value) is (strategy == "autosave")
    if strategy == "autosave":
        assert "autosave assumed" in value["save_interpretation"]
        assert "persistence unverified" in value["save_interpretation"]
        assert value["persistence_verified"] is False
    assert value["browser_launched"] is value["credentials_requested"] is False
    assert value["launch_authorized"] is value["thirty_star_launch_authorized"] is False


@pytest.mark.parametrize("strategy", [None, "autosave"])
def test_compact_strategy_stays_parent_owned_and_v1_replay_is_exact(tmp_path, strategy, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("offline replay cannot start an environment")

    monkeypatch.setattr(Runtime, "start", forbidden)
    source = parent(window())
    if strategy:
        source.update(save_strategy=strategy, persistence_verified=False)
    compact = compact_project_event("state", source)
    assert ("save_strategy" in compact) is bool(strategy)
    assert compact["task_completed"] is compact["project_completed"] is False
    if strategy:
        assert compact["persistence_verified"] is False
    event, child = relay_window("state", window(save_strategy="autosave", persistence_verified=True))
    legacy = compact_project_event(event, parent(window(), child))
    assert "save_strategy" not in legacy and "persistence_verified" not in legacy
    raw = RuntimeEvent(event="state", sequence=0, run_id="fixture", payload=compact).model_dump_json() + "\n"
    path = tmp_path / "trace.jsonl"
    path.write_text(raw)
    runtime = Runtime(io.StringIO())
    runtime.command({"command": "replay", "payload": {"path": str(path)}})
    for _ in range(2):
        runtime.tick()
    played = [json.loads(line) for line in runtime.output.getvalue().splitlines()]
    assert played[-1]["payload"]["replay_finished"] is True
    assert played[-1]["payload"].get("save_strategy") == strategy
    assert played[-1]["payload"]["task_completed"] is False
    assert path.read_text() == raw and runtime.env is None
