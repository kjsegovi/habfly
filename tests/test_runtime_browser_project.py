"""Outer v1 integration with injected drivers only: never launches Chromium."""

import io
import json

import pytest
from test_browser_project_runtime import EMAIL, PASSWORD, write
from test_browser_project_runtime import rig as bridge_rig  # noqa: F401

import habfly.browser_project_runtime as bridge_module
from habfly.runtime import RunOptions, Runtime, parse_run_options, read_trace, serve


def options(tmp_path, **changes):
    for name in ("graph", "dataset", "color", "planet", "planet-final"):
        (tmp_path / name).mkdir(exist_ok=True)
    for name in ("checkpoint.pt", "browser.json"):
        (tmp_path / name).touch()
    (tmp_path / "browser.json").write_text(
        json.dumps(
            {
                "url": "http://localhost/activity",
                "frames": [{"name": "simulation", "url": "https://fixture.invalid"}],
            }
        )
    )
    return {
        "task": "browser_project",
        "environment": "browser",
        "policy": "checkpoint",
        "browser_setup": "automatic",
        "browser_execution": "autonomous",
        "stars": 1,
        "graph": str(tmp_path / "graph"),
        "dataset": str(tmp_path / "dataset"),
        "checkpoint": str(tmp_path / "checkpoint.pt"),
        "color_experiment": str(tmp_path / "color"),
        "browser_config": str(tmp_path / "browser.json"),
        "artifact_dir": str(tmp_path / "runs"),
        "paused": True,
        **changes,
    }


@pytest.fixture
def integrated(bridge_rig, monkeypatch):  # noqa: F811 - imported pytest fixture
    rig = bridge_rig
    real = bridge_module.BrowserProjectRuntime

    def injected(settings, output, **kwargs):
        return real(
            settings,
            output,
            config=rig.config,
            credentials=(EMAIL, PASSWORD),
            _driver_factory=rig.driver,
            _setup_factory=rig.setup,
            _capture_initial=rig.capture,
            _steps_factory=rig.steps,
            _clock=lambda: rig.clock.now,
            **kwargs,
        )

    monkeypatch.setattr(bridge_module, "BrowserProjectRuntime", injected)
    runtime = Runtime(io.StringIO())
    rig.runtime = runtime
    yield rig
    runtime.close()


def start(rig, **changes):
    rig.runtime.command({"command": "start", "payload": options(rig.root, **changes)})
    return rig.runtime.env


def command(rig, name, **payload):
    rig.runtime.command({"version": 1, "command": name, "payload": payload})


def awaiting_class(rig):
    bridge = start(rig)
    for _ in range(10):
        command(rig, "step")
        if bridge.phase == "awaiting_class_source":
            return bridge
    pytest.fail("Expected explicit class handoff")


def class_handoff(rig):
    bridge = awaiting_class(rig)
    write(bridge.output / "class/confirmed.json", {"star": "ALPHA"})
    command(
        rig,
        "step",
        class_source={
            "class_dir": "class",
            "selected_class": "main_sequence",
            "lifetime_prefix": "Ga",
        },
    )
    return bridge


@pytest.mark.parametrize(
    "change",
    [
        {"environment": "simulator"},
        {"policy": "expert"},
        {"browser_setup": "manual"},
        {"browser_execution": "supervised"},
        {"stars": 2},
        {"stars": 30},
        {"calculation_backend": "google_sheets"},
        {"spreadsheet_config": "sensitive.json"},
        {"knowledge_pack": "custom.json"},
        {"content_pack": "custom.json"},
        {"color_checkpoint": "old.pt"},
        {"color_dataset": "old"},
        {"planet_evaluation": "old"},
        {"planet_calibration": "old"},
        {"habitability_evaluation": "old"},
        {"browser_planet_pilot": "pilot"},
        {"browser_planet_final_evaluation": "final"},
        {"graph": None},
        {"dataset": None},
        {"checkpoint": None},
        {"color_experiment": None},
        {"browser_config": None},
        {"project_max_advances": 0},
        {"project_max_advances": 2049},
        {"project_max_advances": True},
        {"project_max_seconds": float("inf")},
        {"project_max_seconds": 3601},
        {"project_setup_max_advances": 1025},
        {"password": PASSWORD},
        {"url": "http://localhost?secret=private-session"},
        {"selected_class": "main_sequence"},
        {"assessment_enabled": True},
        {"submission_enabled": True},
    ],
)
def test_invalid_project_start_is_offline_and_sanitized(integrated, change):
    with pytest.raises(ValueError) as error:
        start(integrated, **change)
    assert PASSWORD not in str(error.value) and "private-session" not in str(error.value)
    assert not integrated.calls and not (integrated.root / "runs").exists()


def test_start_is_deferred_and_exact_bridge_events_are_retained(integrated):
    bridge = start(integrated)
    assert integrated.runtime.status == "paused" and bridge.phase == "launch_pending"
    assert not integrated.calls and not bridge.credentials_consumed
    outer = read_trace(integrated.runtime.trace_path)
    inner = read_trace(bridge.trace_path)
    assert len(outer) == len(inner) == 2
    for parent, child in zip(outer, inner):
        assert parent.version == child.version == 1 and parent.event == child.event
        for key, value in child.payload.items():
            if key != "trace_path":
                assert parent.payload[key] == value
        assert parent.payload["task"] == "browser_project"
        assert parent.payload["trace_path"] == str(integrated.runtime.trace_path)
        assert parent.payload["project_trace_path"] == str(bridge.trace_path)
    assert PASSWORD not in integrated.runtime.output.getvalue()
    assert "preview_sequence" not in integrated.runtime.output.getvalue()


@pytest.mark.parametrize("value", [1, "true", None])
def test_shallow_reference_is_strict_opt_in_without_launch(tmp_path, value):
    with pytest.raises(ValueError):
        parse_run_options(options(tmp_path, project_reference_shallow_transits=value))
    assert not (tmp_path / "runs").exists()


def test_shallow_reference_requires_explicit_planet_backend(tmp_path):
    with pytest.raises(ValueError, match="shallow_reference_requires"):
        parse_run_options(options(tmp_path, project_reference_shallow_transits=True))
    value = parse_run_options(
        options(
            tmp_path,
            project_reference_shallow_transits=True,
            project_reference_planet_continuation=True,
            browser_planet_pilot=str(tmp_path / "planet"),
            browser_planet_final_evaluation=str(tmp_path / "planet-final"),
        )
    )
    assert value.project_reference_shallow_transits
    assert not (tmp_path / "runs").exists()


def test_paused_scheduler_does_nothing_and_single_step_only_launches(integrated):
    bridge = start(integrated)
    for _ in range(4):
        integrated.runtime.advance_if_due()
    assert not integrated.calls
    command(integrated, "step")
    assert bridge.phase == "opening_preview"
    assert [x[0] for x in integrated.calls].count("launch") == 1
    assert not any(x[0] in {"goto", "setup_advance", "model_step"} for x in integrated.calls)
    integrated.runtime.advance_if_due()
    assert bridge.phase == "opening_preview"


def test_running_setup_can_pause_before_navigation_and_abort_is_sticky(integrated):
    bridge = start(integrated, paused=False)
    assert integrated.runtime.status == "running" and not integrated.calls
    integrated.runtime.advance_if_due()
    command(integrated, "pause")
    integrated.clock.now = 100
    integrated.runtime.advance_if_due()
    assert bridge.phase == "opening_preview"
    command(integrated, "abort")
    count = len(integrated.calls)
    command(integrated, "step")
    integrated.runtime.advance_if_due()
    assert integrated.runtime.status == "aborted" and len(integrated.calls) == count
    assert bridge.page is not None  # Aborting keeps the owned browser available for inspection.
    integrated.runtime.close()
    assert [x[0] for x in integrated.calls].count("context_close") == 1
    summaries = [e for e in read_trace(integrated.runtime.trace_path) if e.event == "episode_summary"]
    assert len(summaries) == 1 and not summaries[0].payload["completed"]


def test_resume_uses_bridge_scheduler_and_stops_at_explicit_class_handoff(integrated):
    bridge = start(integrated)
    command(integrated, "resume")
    for _ in range(10):
        integrated.runtime.advance_if_due()
        integrated.clock.now += 1
        if bridge.phase == "awaiting_class_source":
            break
    assert bridge.phase == "awaiting_class_source" and integrated.runtime.status == "paused"
    assert not any(x[0] == "model_step" for x in integrated.calls)
    count = len(integrated.calls)
    command(integrated, "step")
    assert len(integrated.calls) == count
    with pytest.raises(ValueError, match="explicit_handoff_required"):
        command(integrated, "resume")


def test_class_receipt_then_one_model_boundary_ends_truthful_handoff(integrated):
    bridge = class_handoff(integrated)
    assert bridge.phase == "ready" and integrated.runtime.status == "paused"
    assert not any(x[0] == "model_step" for x in integrated.calls)
    command(integrated, "step")
    assert integrated.runtime.status == "handoff" and bridge.phase == "planet_classification_required"
    events = read_trace(integrated.runtime.trace_path)
    summary = [e for e in events if e.event == "episode_summary"]
    assert len(summary) == 1 and summary[0].payload["completed"] is False
    assert summary[0].payload["project_completed"] is False
    nested = [e.payload["component_event"] for e in events if "component_event" in e.payload]
    assert any(e["event"] == "episode_summary" for e in nested)
    assert not any(e.payload.get("task_completed") for e in events)


def test_model_options_and_explicit_limits_are_forwarded(integrated):
    bridge = start(
        integrated,
        browser_planet_pilot=str(integrated.root / "planet"),
        browser_planet_final_evaluation=str(integrated.root / "planet-final"),
        browser_planet_seed=123,
        project_max_advances=17,
        project_max_seconds=42,
        project_setup_max_advances=12,
    )
    assert bridge.model_options["planet_seed"] == 123
    assert bridge.model_options["planet_pilot"] == integrated.root / "planet"
    assert bridge.model_options["graph_path"] == integrated.root / "graph"
    assert bridge.project_limits == {"max_advances": 17, "max_seconds": 42}
    assert bridge.scope["setup_max_advances"] == 12
    assert bridge.scope["observation_start_max_seconds"] == 60
    assert not bridge.scope["assessment_enabled"] and not bridge.scope["submission_enabled"]


@pytest.mark.parametrize(
    "payload",
    [
        {"approve_copy": True},
        {"browser_ready": True},
        {"class_source": PASSWORD},
        {"class_source": {"password": PASSWORD}},
        {"inventory": {"url": "http://localhost?secret=private-session"}},
    ],
)
def test_bad_handoff_never_leaks_or_dispatches(integrated, payload):
    bridge = awaiting_class(integrated)
    count = len(integrated.calls)
    with pytest.raises(ValueError) as error:
        command(integrated, "step", **payload)
    assert PASSWORD not in str(error.value) and "private-session" not in str(error.value)
    assert len(integrated.calls) == count and bridge.phase == "awaiting_class_source"


def test_serve_start_options_precedes_commands_and_redacts_protocol_errors(integrated):
    data = options(integrated.root)
    output = io.StringIO()
    messages = io.StringIO(
        json.dumps({"command": "bad-" + PASSWORD}) + "\n{\n" + json.dumps({"command": "abort"}) + "\n"
    )
    serve(messages, output, start_options=data)
    items = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [i["sequence"] for i in items] == list(range(len(items)))
    assert items[0]["event"] == "hello" and items[0]["payload"]["status"] == "idle"
    assert items[1]["event"] == "hello" and items[1]["payload"]["task"] == "browser_project"
    assert len([i for i in items if i["event"] == "error"]) == 2
    assert PASSWORD not in output.getvalue() and not integrated.calls


def test_save_and_offline_replay_use_outer_trace_not_relative_child_path(integrated, tmp_path):
    bridge = start(integrated)
    destination = tmp_path / "export.jsonl"
    command(integrated, "save_trace", path=str(destination))
    assert read_trace(destination)[-1].event == "state"
    command(integrated, "replay", path=str(destination))
    assert bridge.closed and not integrated.runtime.env
    command(integrated, "pause")
    command(integrated, "step")
    assert integrated.runtime.status == "paused" and not integrated.calls


def test_existing_defaults_and_missing_paths_are_unchanged(tmp_path):
    default = parse_run_options({})
    assert default == RunOptions() and default.task == "mini_habworlds"
    assert default.browser_setup == "manual" and default.browser_execution == "supervised"
    with pytest.raises(ValueError, match="missing_local"):
        parse_run_options(options(tmp_path, checkpoint=str(tmp_path / "absent.pt")))


def test_bridge_failure_reaches_outer_status_without_raw_exception(integrated):
    integrated.behaviour["launch_error"] = RuntimeError(PASSWORD + " http://localhost?private-session")
    bridge = start(integrated)
    command(integrated, "step")
    assert bridge.finished and integrated.runtime.status == "stopped"
    assert PASSWORD not in integrated.runtime.output.getvalue()
    assert "private-session" not in integrated.runtime.output.getvalue()


def test_read_only_config_validation_precedes_any_artifact_creation(integrated):
    payload = options(integrated.root)
    (integrated.root / "browser.json").write_text(json.dumps({"url": PASSWORD}))
    with pytest.raises(ValueError, match="invalid_browser_config"):
        integrated.runtime.command({"command": "start", "payload": payload})
    assert not integrated.calls and not (integrated.root / "runs").exists()


def test_inventory_is_explicit_and_unverified_completion_is_rejected(integrated, monkeypatch):
    bridge = class_handoff(integrated)
    bridge.project.phase = bridge.phase = "awaiting_inventory"
    command(integrated, "step")
    assert bridge.phase == "awaiting_inventory" and not bridge.finished
    forwarded = []

    def provide_inventory(**kwargs):
        forwarded.append(kwargs)
        bridge.project.finished, bridge.project.status = True, "completed"

    monkeypatch.setattr(bridge.project, "provide_inventory", provide_inventory, raising=False)
    command(integrated, "step", inventory={"inventory_dir": "inventory", "inventory_sha256": "a" * 64})
    assert forwarded == [{"inventory_dir": bridge.output / "inventory", "inventory_sha256": "a" * 64}]
    assert integrated.runtime.status == "stopped"
    assert bridge.report["failure_reason"] == "project_runtime_invalid_project_completion_claim"
    assert not bridge.report["task_completed"]


def test_changed_initial_capture_stops_before_model_step(integrated):
    bridge = class_handoff(integrated)
    (bridge.output / "initial-star/stellar/observation.json").write_text("changed")
    command(integrated, "step")
    assert integrated.runtime.status == "stopped"
    assert not any(call[0] == "model_step" for call in integrated.calls)


def test_class_receipt_cannot_escape_owned_artifacts(integrated):
    bridge = awaiting_class(integrated)
    command(
        integrated,
        "step",
        class_source={
            "class_dir": "../other-run",
            "selected_class": "main_sequence",
            "lifetime_prefix": "Ga",
        },
    )
    assert integrated.runtime.status == "stopped"
    assert bridge.failure == "project_runtime_invalid_owned_path"
    assert not any(call[0] == "class_handoff" for call in integrated.calls)


def test_first_project_command_error_is_private_even_before_options_installed(integrated):
    output = io.StringIO()
    payload = options(integrated.root, password=PASSWORD)
    serve(io.StringIO(json.dumps({"command": "start", "payload": payload}) + "\n"), output)
    assert PASSWORD not in output.getvalue()
    assert "browser_project_protocol_error" in output.getvalue()
    assert not integrated.calls
