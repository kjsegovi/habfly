"""Offline lifecycle injection only; never launches a real Chromium instance."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_project_next_star_steps import starfield_png

import habfly.browser_project_runtime as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_probe import BrowserProbeConfig
from habfly.browser_setup import SetupStop, visible_star_point
from habfly.runtime import RunOptions, read_trace

EMAIL, PASSWORD = "fixture-user@example.invalid", "fixture-only-secret"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    calls, events = [], []
    clock = SimpleNamespace(now=0)
    options = RunOptions(
        environment="browser", browser_setup="automatic", task="browser_numeric", paused=True
    )
    config = BrowserProbeConfig(
        url="http://localhost/activity?preview_sequence_id=q%3A123%3A946",
        frames=[
            {
                "name": "simulation",
                "url": "https://fixture.invalid/simulation",
            }
        ],
    )
    behaviour = {
        "setup_error": None,
        "setup_waits": 1,
        "launch_error": None,
        "child_phase": "planet_classification_required",
    }

    class Page:
        viewport_size = None

        def goto(self, url, **kwargs):
            calls.append(("goto", url, kwargs))

    page = Page()

    class Context:
        def new_page(self):
            calls.append(("new_page",))
            return page

        def close(self):
            calls.append(("context_close",))

    class Browser:
        def new_context(self, **kwargs):
            calls.append(("new_context", kwargs) if kwargs else ("new_context",))
            page.viewport_size = deepcopy(behaviour.get("context_viewport_override", kwargs.get("viewport")))
            return Context()

        def close(self):
            calls.append(("browser_close",))

    class Chromium:
        def launch(self, **kwargs):
            calls.append(("launch", kwargs))
            if behaviour["launch_error"]:
                raise behaviour["launch_error"]
            return Browser()

    class Driver:
        chromium = Chromium()

        def stop(self):
            calls.append(("driver_stop",))

    def driver():
        import os

        assert "HABFLY_LOGIN_EMAIL" not in os.environ and "HABFLY_LOGIN_PASSWORD" not in os.environ
        assert not any(name in os.environ for name in ("DEBUG", "PWDEBUG", "DEBUG_FILE"))
        calls.append(("driver_start",))
        return Driver()

    class Setup:
        def __init__(self, page, config, credentials, *, emit, output):
            assert credentials == (EMAIL, PASSWORD)
            calls.append(("setup_construct",))
            self.emit, self.output = emit, output
            self._email, self._password = credentials
            self.stage, self.closed, self.count = "opening_preview", False, 0

        def advance(self):
            if behaviour["setup_error"]:
                raise behaviour["setup_error"]
            self.stage = "signing_in" if self.count == 0 else "stellar_screen_ready"
            self.emit("state", {"setup_stage": self.stage, "action_source": "deterministic_setup"})
            if self.closed:
                raise SetupStop("setup_closed")
            calls.append(("setup_advance", self.count))
            self.count += 1
            if self.count > behaviour["setup_waits"]:
                self.close()
                self.stage = "stellar_screen_ready"
                return "stellar"
            return "waiting"

        def close(self):
            self._email = self._password = ""
            self.closed = True
            calls.append(("setup_close",))

    def capture(setup, output):
        assert setup.closed and setup.stage == "stellar_screen_ready"
        calls.append(("capture_initial",))
        png = starfield_png()
        (setup.output / "setup-starfield.png").write_bytes(png)
        receipt = {
            "star": "ALPHA",
            "stellar_observations_verified": True,
            "fresh_blank_numeric_answers_verified": True,
            "class_selection_verified": False,
            "action_source": "deterministic_navigation",
            "collection_count_verified": False,
            "answer_writes": 0,
            "task_completed": False,
            "selected_point": visible_star_point(png),
        }
        write(output / "confirmed.json", receipt)
        write(output / "stellar/observation.json", {"visible": "fixture measurements"})
        # Legacy default-only shape also remains supported by the new source pin.
        write(
            output / "scope.json",
            {
                "mode": "read_only_initial_setup_handoff",
                "starfield_sha256": hashlib.sha256(png).hexdigest(),
            },
        )
        return receipt

    class Steps:
        def __init__(self, page, config, output, *, star, emit, **kwargs):
            calls.append(("steps_construct", star))
            self.star, self.emit, self.output = star, emit, output
            self.status, self.phase, self.finished = "idle", "not_started", False
            output.mkdir()

        def state(self):
            return {
                "star": self.star,
                "status": self.status,
                "phase": self.phase,
                "task_completed": False,
                "artifact_paths": {"fixture": "active-star/fixture"},
                "reference_measurements": None,
                "failure_reason": None,
            }

        def start(self, *, paused):
            assert paused
            self.status, self.phase = "paused", "awaiting_class_source"
            self.emit("state", self.state())

        def provide_class(self, **kwargs):
            calls.append(("class_handoff", kwargs["selected_class"]))
            self.phase = "ready"
            self.emit("state", self.state())

        def step(self):
            calls.append(("model_step",))
            self.finished, self.status, self.phase = True, "handoff", behaviour["child_phase"]
            self.emit("episode_summary", {"completed": False})

        def tick(self):
            if self.status == "running":
                self.step()

        def resume(self):
            self.status = "running"

        def pause(self):
            self.status = "paused"

        def abort(self):
            self.finished, self.status, self.phase = True, "aborted", "aborted"
            calls.append(("steps_abort",))

        def close(self):
            if not self.finished:
                self.abort()

    def preflight(history, class_dir, star, selected):
        return {
            "star": star,
            "selected_class": selected,
            "fresh_star": "initial-star",
            "source_capture_sha256": sha(history / "initial-star/stellar/observation.json"),
        }

    monkeypatch.setattr(module, "validate_star_class_source", preflight)
    return SimpleNamespace(
        root=tmp_path,
        calls=calls,
        events=events,
        clock=clock,
        options=options,
        config=config,
        behaviour=behaviour,
        driver=driver,
        setup=Setup,
        capture=capture,
        steps=Steps,
        preflight=preflight,
    )


def create(rig, **kwargs):
    return module.BrowserProjectRuntime(
        rig.options,
        rig.root / "run",
        model_options={key: Path(key) for key in ("dataset", "checkpoint", "color_experiment", "graph_path")},
        config=rig.config,
        credentials=kwargs.pop("credentials", (EMAIL, PASSWORD)),
        emit=kwargs.pop("emit", lambda *e: rig.events.append(e)),
        _driver_factory=rig.driver,
        _setup_factory=rig.setup,
        _capture_initial=kwargs.pop("capture", rig.capture),
        _steps_factory=kwargs.pop("steps", rig.steps),
        _clock=lambda: rig.clock.now,
        **kwargs,
    )


def setup_ready(rig, **kwargs):
    bridge = create(rig, **kwargs)
    bridge.start()
    for _ in range(10):
        bridge.step()
        if bridge.phase == "awaiting_class_source" or bridge.finished:
            break
    return bridge


def ready(rig, **kwargs):
    bridge = setup_ready(rig, **kwargs)
    assert bridge.phase == "awaiting_class_source", bridge.state()
    write(bridge.output / "class/confirmed.json", {"star": "ALPHA"})
    bridge.provide_class(class_dir="class", selected_class="main_sequence", lifetime_prefix="Ga")
    assert bridge.phase == "ready", bridge.state()
    return bridge


def versioned_initial_capture(rig, *, mutate=lambda *_: None):
    def capture(setup, output):
        receipt = rig.capture(setup, output)
        png = (setup.output / "setup-starfield.png").read_bytes()
        receipt["selected_point"] = visible_star_point(png, anchor=(0.75, 0.4))
        write(output / "confirmed.json", receipt)
        scope = json.loads((output / "scope.json").read_bytes())
        scope.update(
            selection_schema_version=1,
            starfield_anchor=[0.75, 0.4],
            excluded_points=[],
            starfield_image="setup-starfield.png",
        )
        write(output / "scope.json", scope)
        (output / "setup-starfield.png").write_bytes(png)
        mutate(setup, output, receipt)
        return receipt

    return capture


def test_new_initial_capture_pins_exact_original_and_owned_png(rig):
    bridge = setup_ready(rig, capture=versioned_initial_capture(rig))
    assert bridge.phase == "awaiting_class_source", bridge.state()
    source, saved = (
        bridge.output / "setup/setup-starfield.png",
        bridge.output / "initial-star/setup-starfield.png",
    )
    assert bridge._initial_hashes["setup/setup-starfield.png"] == sha(source)
    assert bridge._initial_hashes["initial-star/setup-starfield.png"] == sha(saved) == sha(source)
    assert bridge.initial_star["selected_point"] == visible_star_point(
        source.read_bytes(), anchor=(0.75, 0.4)
    )
    assert not bridge.state()["task_completed"]
    bridge.close()


@pytest.mark.parametrize("path", ["setup/setup-starfield.png", "initial-star/setup-starfield.png"])
def test_pinned_initial_png_change_stops_before_class_handoff(rig, path):
    bridge = setup_ready(rig, capture=versioned_initial_capture(rig))
    (bridge.output / path).write_bytes(b"changed rendered selection")
    write(bridge.output / "class/confirmed.json", {"star": "ALPHA"})
    bridge.provide_class(class_dir="class", selected_class="main_sequence", lifetime_prefix="Ga")
    assert (
        bridge.status == "stopped"
        and bridge.report["failure_reason"] == "project_runtime_initial_source_changed"
    )
    assert not any(call[0] == "class_handoff" for call in rig.calls)
    bridge.close()


@pytest.mark.parametrize("change", ["anchor", "copy", "source", "version", "image_path"])
def test_new_selection_mismatch_stops_before_owner_construction(rig, change):
    def mutate(setup, output, receipt):
        if change in {"anchor", "version", "image_path"}:
            scope = json.loads((output / "scope.json").read_bytes())
            scope.update(
                {
                    {
                        "anchor": "starfield_anchor",
                        "version": "selection_schema_version",
                        "image_path": "starfield_image",
                    }[change]: {
                        "anchor": [0.4, 0.55],
                        "version": 2,
                        "image_path": "../setup/setup-starfield.png",
                    }[change]
                }
            )
            write(output / "scope.json", scope)
        elif change == "copy":
            (output / "setup-starfield.png").write_bytes(b"changed")
        else:
            (setup.output / "setup-starfield.png").write_bytes(b"changed")

    bridge = setup_ready(rig, capture=versioned_initial_capture(rig, mutate=mutate))
    assert bridge.status == "stopped" and not any(call[0] == "steps_construct" for call in rig.calls)
    bridge.close()


def test_old_nondefault_initial_receipt_is_not_retrofitted(rig):
    def capture(setup, output):
        receipt = rig.capture(setup, output)
        receipt["selected_point"] = visible_star_point(
            (setup.output / "setup-starfield.png").read_bytes(), anchor=(0.75, 0.4)
        )
        write(output / "confirmed.json", receipt)
        return receipt

    bridge = setup_ready(rig, capture=capture)
    assert bridge.status == "stopped" and not any(call[0] == "steps_construct" for call in rig.calls)
    assert "selection_schema_version" not in json.loads(
        (bridge.output / "initial-star/scope.json").read_bytes()
    )
    bridge.close()


def text_artifacts(output):
    # Include PNG bytes too: binary containers must not hide fixture credentials.
    return "\n".join(path.read_text(errors="replace") for path in output.rglob("*") if path.is_file())


def test_constructor_start_and_pause_defer_launch_and_credentials(rig, monkeypatch):
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", PASSWORD)
    bridge = create(rig, credentials=None)
    assert not rig.calls and not rig.events and not bridge.output.exists()
    assert not bridge.credentials_consumed
    bridge.start()
    bridge.advance_if_due()
    bridge.tick()
    assert not rig.calls and not bridge.credentials_consumed
    bridge.step()
    assert bridge.phase == "opening_preview" and bridge.credentials_consumed
    assert ("launch", {"headless": False, "timeout": 30000}) in rig.calls
    assert not any(c[0] == "goto" for c in rig.calls)
    bridge.close()


def test_setup_capture_and_project_construction_are_distinct_steps(rig):
    bridge = create(rig)
    bridge.start()
    for phase in (
        "opening_preview",
        "setting_up",
        "setting_up",
        "capturing_initial_star",
        "initializing_project",
        "awaiting_class_source",
    ):
        bridge.step()
        assert bridge.phase == phase
    assert not any(c[0] == "model_step" for c in rig.calls)
    assert bridge.initial_star_dir == bridge.output / "initial-star"
    assert bridge.status == "paused" and not bridge.state()["task_completed"]
    before = list(rig.calls)
    bridge.step()
    assert rig.calls == before and bridge.phase == "awaiting_class_source"
    with pytest.raises(BrowserSafetyStop, match="handoff"):
        bridge.resume()
    bridge.close()


def test_automatic_setup_always_pauses_for_class_source_without_guessing(rig):
    bridge = create(rig)
    bridge.start(paused=False)
    for _ in range(10):
        rig.clock.now += 1
        bridge.advance_if_due()
    assert bridge.phase == "awaiting_class_source" and bridge.status == "paused"
    assert not any(c[0] in {"model_step", "class_handoff"} for c in rig.calls)
    bridge.close()


def test_one_explicit_class_handoff_then_one_step_never_becomes_task_success(rig):
    bridge = ready(rig)
    assert not any(c[0] == "model_step" for c in rig.calls)
    bridge.step()
    assert bridge.status == "handoff" and bridge.phase == "planet_classification_required"
    assert not bridge.report["task_completed"]
    assert bridge.state()["browser_status"] == "visible"
    before = list(rig.calls)
    bridge.step()
    bridge.tick()
    bridge.abort()
    assert rig.calls == before
    bridge.close()
    assert [c[0] for c in rig.calls[-3:]] == ["context_close", "browser_close", "driver_stop"]
    assert bridge.state()["browser_status"] == "closed"


def test_credentials_debug_values_and_private_url_are_absent_from_artifacts(rig, monkeypatch):
    for key in ("DEBUG", "PWDEBUG", "DEBUG_FILE"):
        monkeypatch.setenv(key, PASSWORD)
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", PASSWORD)
    bridge = setup_ready(rig)
    assert bridge._credentials is None and bridge.setup._email == bridge.setup._password == ""
    bridge.abort()
    bridge.close()
    raw = text_artifacts(bridge.output)
    for secret in (EMAIL, PASSWORD, rig.config.url, "session-token", "private="):
        assert secret not in raw


def test_missing_credentials_never_launches_browser(rig, monkeypatch):
    monkeypatch.delenv("HABFLY_LOGIN_EMAIL", raising=False)
    monkeypatch.delenv("HABFLY_LOGIN_PASSWORD", raising=False)
    bridge = create(rig, credentials=None)
    bridge.start()
    bridge.step()
    assert bridge.failure == "setup_credentials_missing" and not rig.calls
    assert bridge.finished and bridge.closed


@pytest.mark.parametrize("steps", [0, 1, 2, 3, 4, 5, 6])
def test_abort_each_setup_boundary_is_sticky_and_close_is_idempotent(rig, steps):
    bridge = create(rig)
    bridge.start()
    for _ in range(steps):
        bridge.step()
    bridge.abort()
    assert bridge.status == "aborted" and bridge._credentials is None
    old = list(rig.calls)
    bridge.step()
    bridge.tick()
    bridge.advance_if_due()
    bridge.abort()
    assert rig.calls == old
    bridge.close()
    closed = list(rig.calls)
    bridge.close()
    assert rig.calls == closed


@pytest.mark.parametrize("effect", ["abort", "exception", "reentry", "pause"])
def test_setup_callback_cancellation_cannot_continue_to_later_action(rig, effect):
    bridge = None

    def callback(kind, payload):
        nested = payload.get("component_state", {})
        if nested.get("setup_stage") == "signing_in":
            if effect == "abort":
                bridge.abort()
            elif effect == "exception":
                raise RuntimeError(PASSWORD)
            elif effect == "reentry":
                bridge.step()
            else:
                bridge.pause()

    bridge = create(rig, emit=callback)
    bridge.start()
    bridge.step()
    bridge.step()
    bridge.step()
    assert sum(c[0] == "setup_advance" for c in rig.calls) == (1 if effect == "pause" else 0)
    assert bridge.status == ("paused" if effect == "pause" else "aborted" if effect == "abort" else "stopped")
    bridge.close()
    assert PASSWORD not in text_artifacts(bridge.output)


def test_setup_failures_are_sanitized_keep_browser_and_never_retry(rig):
    rig.behaviour["setup_error"] = RuntimeError("private " + PASSWORD + rig.config.url)
    bridge = setup_ready(rig)
    assert bridge.failure == "project_runtime_operation_failed" and bridge.finished
    assert not any(c[0] == "capture_initial" for c in rig.calls)
    assert bridge.state()["browser_status"] == "visible"
    assert PASSWORD not in text_artifacts(bridge.output)
    bridge.close()


def test_partial_launch_failure_releases_driver_without_exposing_error(rig):
    rig.behaviour["launch_error"] = RuntimeError(PASSWORD)
    bridge = create(rig)
    bridge.start()
    bridge.step()
    assert bridge.finished and bridge.closed and ("driver_stop",) in rig.calls
    assert PASSWORD not in text_artifacts(bridge.output)


def test_setup_action_count_is_fixed_no_retries_or_automatic_increase(rig):
    rig.behaviour["setup_waits"] = 999
    bridge = create(rig, setup_max_advances=1)
    bridge.start()
    bridge.step()
    bridge.step()
    bridge.step()
    bridge.step()
    assert bridge.failure == "project_runtime_setup_advance_limit" and bridge.setup_advances == 1
    assert bridge.scope["observation_start_max_seconds"] == 60
    assert bridge.scope["automatic_deadline_increase"] is False
    bridge.close()


@pytest.mark.parametrize("change", ["initial", "fresh_link", "capture_hash"])
def test_class_source_must_bind_unchanged_initial_star(rig, monkeypatch, change):
    bridge = setup_ready(rig)
    if change == "initial":
        write(bridge.output / "initial-star/stellar/observation.json", {"changed": True})
    else:
        original = rig.preflight

        def altered(*args):
            source = original(*args)
            source["fresh_star" if change == "fresh_link" else "source_capture_sha256"] = (
                "other" if change == "fresh_link" else "0" * 64
            )
            return source

        monkeypatch.setattr(module, "validate_star_class_source", altered)
    bridge.provide_class(class_dir="class", selected_class="main_sequence", lifetime_prefix="Ga")
    assert bridge.status == "stopped" and not any(c[0] == "class_handoff" for c in rig.calls)
    bridge.close()


def test_setup_state_payload_allowlist_blocks_accidental_secret_logging(rig):
    bridge = create(rig)
    bridge.start()
    bridge.step()
    with pytest.raises(BrowserSafetyStop):
        bridge._setup_event(
            "state",
            {"setup_stage": "signing_in", "action_source": "deterministic_setup", "password": PASSWORD},
        )
    assert PASSWORD not in text_artifacts(bridge.output)
    bridge.close()


@pytest.mark.parametrize(
    "stage", ["waiting_for_login_ui", "refreshing_authenticated_preview", "post_login_refresh_verified"]
)
def test_login_settling_and_refresh_stage_is_forwarded_without_new_task_authority(rig, stage):
    bridge = create(rig)
    bridge.start()
    bridge.step()
    bridge._setup_event("state", {"setup_stage": stage, "action_source": "deterministic_setup"})
    event = read_trace(bridge.trace_path)[-1]
    assert event.payload["component"] == "browser.setup"
    assert event.payload["component_state"]["setup_stage"] == stage
    assert not event.payload["task_completed"] and not event.payload["project_completed"]
    assert not any(call[0] in {"capture_initial", "model_step", "class_handoff"} for call in rig.calls)
    bridge.close()


def test_exact_v1_replay_has_one_outer_summary_and_nested_setup_child_events(rig):
    bridge = ready(rig)
    bridge.step()
    events = read_trace(bridge.output / "events.jsonl")
    assert [e.sequence for e in events] == list(range(len(events)))
    assert [(e.event, e.payload) for e in events] == rig.events
    assert sum(e.event == "episode_summary" for e in events) == 1
    assert any(e.payload.get("component") == "browser.setup" for e in events)
    assert any(e.payload.get("component") == "project.owner" for e in events)
    assert events[-2].event == "episode_summary" and events[-2].payload["completed"] is False
    assert events[-1].event == "state" and events[-1].payload == bridge.report
    assert events[-1].payload["task_completed"] is False
    bridge.close()


def test_v1_commands_do_not_auto_apply_class_or_replay(rig):
    bridge = create(rig)
    bridge.command({"command": "start", "payload": {}})
    bridge.command({"command": "step", "payload": {}})
    with pytest.raises(ValueError):
        bridge.command({"command": "replay", "payload": {"path": "not-supported-here"}})
    bridge.command({"command": "abort", "payload": {}})
    bridge.close()


@pytest.mark.parametrize(
    "invalid",
    [
        {"setup_max_advances": True},
        {"setup_max_advances": 0},
        {"project_limits": {"max_seconds": float("inf")}},
        {"project_limits": {"unknown": 1}},
        {"credentials": (EMAIL, "")},
        {"credentials": [EMAIL, PASSWORD]},
    ],
)
def test_invalid_configuration_never_creates_output_or_browser(rig, invalid):
    with pytest.raises(BrowserSafetyStop):
        create(rig, **invalid)
    assert not rig.calls and not (rig.root / "run").exists()


def test_pause_resume_interval_schedules_one_call_only(rig):
    bridge = create(rig)
    bridge.start(paused=False)
    bridge.advance_if_due()
    bridge.advance_if_due()
    assert bridge.phase == "opening_preview"
    bridge.pause()
    rig.clock.now = 50
    bridge.advance_if_due()
    assert bridge.phase == "opening_preview"
    bridge.resume()
    bridge.advance_if_due()
    bridge.advance_if_due()
    assert bridge.phase == "setting_up"
    bridge.close()


def test_actual_project_controller_composes_without_early_learned_steps(rig, monkeypatch):
    import habfly.browser_project_steps as steps_module

    monkeypatch.setattr(steps_module, "validate_star_class_source", rig.preflight)

    class Star:
        def __init__(self, page, config, output, *, emit, **kwargs):
            self.emit, self.output, self.finished = emit, output, False
            output.mkdir()
            rig.calls.append(("star_construct",))

        def state(self):
            return {
                "star": "ALPHA",
                "phase": "planet_measurement_required" if self.finished else "stellar_numeric",
                "task_completed": False,
                "artifact_paths": {"fixture": "active-star/star/fixture"},
            }

        def advance(self):
            self.emit("action_proposed", {"fixture": "one decision"})
            if self.finished:
                return
            rig.calls.append(("star_decision",))
            self.emit("action_result", {"fixture": "no native writes"})
            self.finished = True
            self.emit("episode_summary", {"completed": False})

        def abort(self):
            self.finished = True

    def steps(*args, **kwargs):
        return steps_module.BrowserProjectSteps(*args, **kwargs, component_factory=Star)

    bridge = ready(rig, steps=steps)
    assert not any(c[0] == "star_construct" for c in rig.calls)
    bridge.step()
    assert ("star_construct",) in rig.calls and not any(c[0] == "star_decision" for c in rig.calls)
    bridge.step()
    assert bridge.status == "handoff" and bridge.phase == "planet_measurement_required"
    assert not bridge.report["task_completed"]
    events = read_trace(bridge.trace_path)
    assert sum(event.event == "episode_summary" for event in events) == 1
    proposal = next(event for event in events if event.event == "action_proposed")
    assert proposal.payload["component_event"]["payload"]["component_event"]["event"] == "action_proposed"
    bridge.close()


@pytest.mark.parametrize("canonical_claim", [False, True])
def test_child_terminal_completion_requires_explicit_canonical_journal_receipt(
    rig, monkeypatch, canonical_claim
):
    bridge = ready(rig)
    state = bridge.project.state

    def bogus():
        return {**state(), "task_completed": canonical_claim}

    def step():
        bridge.project.finished = True
        bridge.project.status = "completed"
        bridge.project.phase = "verified_no_planet"

    monkeypatch.setattr(bridge.project, "step", step)
    monkeypatch.setattr(bridge.project, "state", bogus)
    bridge.step()
    assert bridge.status == "stopped" and not bridge.report["task_completed"]
    assert bridge.failure.endswith(
        "missing_canonical_task_receipt" if canonical_claim else "invalid_project_completion_claim"
    )
    bridge.close()


def test_final_callback_failure_is_terminal_and_replay_records_truthful_state(rig):
    def callback(kind, payload):
        if kind == "episode_summary":
            raise RuntimeError(PASSWORD)

    bridge = ready(rig, emit=callback)
    bridge.step()
    assert bridge.status == "stopped" and bridge.report["event_forwarding_failed"]
    assert read_trace(bridge.trace_path)[-1].payload["status"] == "stopped"
    assert PASSWORD not in text_artifacts(bridge.output)
    bridge.close()


def test_abort_during_read_only_capture_does_not_initialize_project(rig):
    bridge = None

    def capture(setup, output):
        receipt = rig.capture(setup, output)
        bridge.abort()
        return receipt

    bridge = create(rig, capture=capture)
    bridge.start()
    for _ in range(6):
        bridge.step()
    assert bridge.status == "aborted" and bridge.phase == "aborted"
    assert not any(c[0] == "steps_construct" for c in rig.calls)
    bridge.close()


def test_close_failure_still_releases_other_resources_and_never_retries(rig, monkeypatch):
    bridge = create(rig)
    bridge.start()
    bridge.step()
    monkeypatch.setattr(bridge.context, "close", lambda: (_ for _ in ()).throw(RuntimeError(PASSWORD)))
    bridge.close()
    assert bridge.state()["browser_status"] == "close_failed"
    assert ("browser_close",) in rig.calls and ("driver_stop",) in rig.calls
    assert json.loads((bridge.output / "browser-close.json").read_text())["failed_resources"] == ["context"]
    before = list(rig.calls)
    bridge.close()
    assert rig.calls == before and PASSWORD not in text_artifacts(bridge.output)
