"""Outer scheduling never turns component termination into task success."""
# ruff: noqa: F811

import hashlib
import json
from types import SimpleNamespace

import pytest
from test_browser_numeric import chromium, page  # noqa: F401
from test_browser_numeric import config as fixture_config
from test_browser_planet_steps import seed_sources
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_raster_planet_evidence import raster_page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_probe import BrowserProbeConfig
from habfly.browser_star_session import BrowserStarSession
from habfly.browser_stellar import SIMULATION_URL


@pytest.fixture
def rig(tmp_path, monkeypatch):
    import habfly.browser_star_session as module

    (tmp_path / "class").mkdir()
    (tmp_path / "class/confirmed.json").write_text(json.dumps({"star": "Fixture"}))
    calls, events, components = [], [], []
    outcome = {"window": "no_selected", "verified": True, "save_error": None, "workflow": True}
    monkeypatch.setattr(module, "validate_star_class_source", lambda *_: {"star": "Fixture"})
    monkeypatch.setattr(
        module, "matches_mapping", lambda source, mapping: mapping["star_name"] == source["star"]
    )

    class Component:
        def __init__(self, *args, emit, **kwargs):
            self.path, self.emit = args[2], emit
            self.kind = self.path.name
            self.finished, self.ticks, self.aborted = False, 0, False
            self.report = None
            self.session = SimpleNamespace(mapping={"star_name": "Fixture"})
            calls.append(("init", self.kind))
            self.path.mkdir()
            components.append(self)

        def state(self):
            return {
                "star": "Fixture",
                "phase": outcome["window"] if self.finished else "observing",
                "transport_verified": outcome["verified"] and self.finished and not self.aborted,
                "decision_readback_verified": outcome["verified"] and self.finished,
            }

        def advance(self):
            calls.append(("advance", self.kind))
            self.ticks += 1
            if self.kind == "window":
                (self.path / "choice").mkdir()
                (self.path / "choice/confirmed.json").write_text("{}")
                (self.path / "progress").mkdir()
                chart = self.path / "progress/chart.png"
                chart.write_bytes(b"fixture-only-chart")
                progress = self.path / "progress/report.json"
                progress.write_text(
                    json.dumps(
                        {
                            "star": "FIXTURE",
                            "requested_days": 5000,
                            "chart_sha256": hashlib.sha256(chart.read_bytes()).hexdigest(),
                        }
                    )
                )
                self.report = {
                    "star": "Fixture",
                    "phase": outcome["window"],
                    "progress_path": str(progress.relative_to(tmp_path)),
                    "progress_sha256": hashlib.sha256(progress.read_bytes()).hexdigest(),
                }
                (self.path / "report.json").write_text(json.dumps(self.report))
            else:
                self.emit(
                    {"event": "observation", "run_id": self.kind, "sequence": 1, "payload": {"visible": True}}
                )
                self.emit(
                    {
                        "event": "episode_summary",
                        "run_id": self.kind,
                        "sequence": 2,
                        "payload": {"completed": False},
                    }
                )
            self.finished = True

        def abort(self):
            self.aborted = self.finished = True
            calls.append(("abort", self.kind))

    def navigate(*args, **kwargs):
        calls.append(("navigate", args[3]))
        return {
            "same_star_verified": True,
            "destination_verified": True,
            "suggested_required_text": ["OBSERVATIONS"],
        }

    def save(*args, **kwargs):
        calls.append(("save", "No"))
        if outcome["save_error"]:
            raise BrowserSafetyStop(outcome["save_error"])
        return {"save_click_delivered": True}

    def verify(*args, **kwargs):
        calls.append(("verify", "No"))
        return {"task_completed": outcome["workflow"], "star": "Fixture"}

    for name in ("FullStellarNumericSteps", "FullStellarColorSteps", "PlanetWindowSession"):
        monkeypatch.setattr(module, name, Component)
    monkeypatch.setattr(module, "navigate_project", navigate)
    monkeypatch.setattr(module, "save_no_planet_work", save)
    monkeypatch.setattr(module, "verify_no_planet_workflow", verify)
    # Use the actual boundary contract, no actual page/network resources.
    from pathlib import Path

    config = BrowserProbeConfig.model_validate_json(Path("configs/browser_probe.example.json").read_text())
    return SimpleNamespace(
        root=tmp_path, calls=calls, events=events, components=components, outcome=outcome, config=config
    )


def create(rig, **kwargs):
    options = {
        "run_history": rig.root,
        "star": "Fixture",
        "class_dir": rig.root / "class",
        "selected_class": "main_sequence",
        "lifetime_prefix": "Ga",
        "dataset": "dataset",
        "checkpoint": "checkpoint",
        "color_experiment": "color",
        "graph_path": "graph",
        "emit": lambda *event: rig.events.append(event),
        **kwargs,
    }
    return BrowserStarSession(object(), rig.config, rig.root / "run", **options)


def complete(session):
    for _ in range(20):
        session.advance()
        if session.finished:
            return
    raise AssertionError("Scheduler did not finish within mocked bounded steps")


def test_no_planet_full_workflow_and_separate_component_events(rig):
    session = create(rig)
    assert not rig.calls
    complete(session)
    assert session.state()["task_completed"]
    assert session.phase == "verified_no_planet"
    assert not session.state()["project_completed"]
    assert rig.calls == [
        ("init", "numeric"),
        ("advance", "numeric"),
        ("init", "color"),
        ("advance", "color"),
        ("navigate", "planet"),
        ("init", "window"),
        ("advance", "window"),
        ("save", "No"),
        ("verify", "No"),
    ]
    summaries = [p for e, p in rig.events if e == "episode_summary"]
    assert len(summaries) == 1 and summaries[0]["completed"]
    observations = [p for e, p in rig.events if e == "observation"]
    assert observations[0]["component_event"] == {
        "run_id": "numeric",
        "sequence": 1,
        "event": "observation",
        "payload": {"visible": True},
    }
    assert observations[0]["component"] == "stellar.numeric"
    before = len(rig.calls), len(rig.events)
    session.advance()
    session.close()
    assert before == (len(rig.calls), len(rig.events))


def test_each_advance_is_a_pause_boundary_and_config_not_mutated(rig):
    session = create(rig)
    original = rig.config.model_dump()
    session.advance()
    for _ in range(10):
        session.state()
    assert rig.calls == [("init", "numeric")]
    assert rig.components[0].ticks == 0
    complete(session)
    assert rig.config.model_dump() == original
    assert next(f for f in session.config.frames if f.url == SIMULATION_URL).required_text == ["OBSERVATIONS"]


@pytest.mark.parametrize("stage", range(9))
def test_abort_never_schedules_followup(rig, stage):
    session = create(rig)
    for _ in range(stage):
        session.advance()
    session.abort()
    before = len(rig.calls), len(rig.events)
    for _ in range(3):
        session.advance()
        session.abort()
    assert session.finished and not session.state()["task_completed"]
    assert before == (len(rig.calls), len(rig.events))


def test_dip_handoff_never_saves_or_assumes_no(rig):
    rig.outcome["window"] = "dip_observed"
    session = create(rig)
    complete(session)
    assert session.phase == "planet_measurement_required"
    assert not session.state()["task_completed"]
    assert not any(c[0] in {"save", "verify"} for c in rig.calls)


def test_bad_stellar_transport_never_starts_window(rig):
    rig.outcome["verified"] = False
    session = create(rig)
    complete(session)
    assert session.failure == "star_session_stellar_transport_failed"
    assert rig.calls == [("init", "numeric"), ("advance", "numeric")]


def test_save_failure_never_retries_or_verifies(rig):
    rig.outcome["save_error"] = "no_planet_save_stale_acknowledgement"
    session = create(rig)
    complete(session)
    assert session.phase == "stopped" and not session.state()["task_completed"]
    for _ in range(4):
        session.advance()
    assert rig.calls.count(("save", "No")) == 1
    assert not any(c[0] == "verify" for c in rig.calls)


def test_false_workflow_result_never_means_completed(rig):
    rig.outcome["workflow"] = False
    session = create(rig)
    complete(session)
    assert session.failure == "star_session_workflow_unverified"
    assert not session.state()["task_completed"]


def test_other_star_class_receipt_rejected_before_run_creation(rig):
    (rig.root / "class/confirmed.json").write_text(json.dumps({"star": "Other"}))
    with pytest.raises(ValueError, match="different star"):
        create(rig)
    assert not (rig.root / "run").exists()
    assert not rig.calls


def test_exception_details_are_not_emitted(rig, monkeypatch):
    import habfly.browser_star_session as module

    def fail(*args, **kwargs):
        raise RuntimeError("https://private.invalid/?password=secret")

    monkeypatch.setattr(module, "FullStellarNumericSteps", fail)
    session = create(rig)
    session.advance()
    assert session.failure == "star_session_component_failed"
    assert "secret" not in json.dumps(rig.events)


def test_callback_abort_during_successful_component_summary_is_terminal(rig):
    session = create(rig)

    def receive(event, payload):
        if "component_summary" in payload:
            session.abort()

    session._callback = receive
    session.advance()
    session.advance()
    assert session.phase == "aborted"
    assert session.report["phase"] == "aborted"
    before = list(rig.calls)
    session.advance()
    assert rig.calls == before and session.finished


def test_reentrant_callback_cannot_advance_the_scheduler(rig):
    session = create(rig)
    session._callback = lambda *_: session.advance()
    session.advance()
    assert session.finished and session.phase == "stopped"
    assert not any(c[0] == "advance" for c in rig.calls)
    assert session.report is not None


def test_error_callback_failure_does_not_prevent_terminal_report(rig, monkeypatch):
    import habfly.browser_star_session as module

    def fail(*args, **kwargs):
        raise RuntimeError("private failure")

    monkeypatch.setattr(module, "FullStellarNumericSteps", fail)
    session = create(rig)
    session._callback = fail
    session.advance()
    assert session.finished and session.event_forwarding_failed
    assert (rig.root / "run/report.json").is_file()
    assert "private" not in (rig.root / "run/event-forwarding-failed.json").read_text()


def test_changed_current_star_is_rejected_before_first_model_decision(rig, monkeypatch):
    import habfly.browser_star_session as module

    factory = module.FullStellarNumericSteps

    def changed(*args, **kwargs):
        component = factory(*args, **kwargs)
        component.session.mapping["star_name"] = "Different"
        return component

    monkeypatch.setattr(module, "FullStellarNumericSteps", changed)
    session = create(rig)
    session.advance()
    assert session.failure == "star_session_stellar_star_changed"
    assert not any(c[0] == "advance" for c in rig.calls)


def test_class_source_change_stops_before_next_decision(rig, monkeypatch):
    import habfly.browser_star_session as module

    session = create(rig)
    session.advance()
    monkeypatch.setattr(module, "validate_star_class_source", lambda *_: {"star": "changed"})
    session.advance()
    assert session.failure == "star_session_class_source_changed"
    assert not any(c[0] == "advance" for c in rig.calls)


@pytest.fixture
def positive_rig(rig, monkeypatch):
    import habfly.browser_star_session as module
    from habfly.planet_charts import spectrum_excursion

    rig.outcome.update(
        window="dip_observed",
        spectrum_error=None,
        positive_phase="planet_classification_required",
        positive_verified=True,
        positive_star="Fixture",
        positive_error=None,
    )
    rig.positive_options = []
    rig.outcome["positive_init_callback"] = False

    def capture(page, config, output, *, expected_star, emit, **kwargs):
        rig.calls.append(("capture", "spectrum"))
        output.mkdir()
        if rig.outcome["spectrum_error"]:
            raise rig.outcome["spectrum_error"]
        rows = [
            {
                "kind": "observation",
                "payload": {"chart": {"source": "visible_tooltips", "star": expected_star.upper()}},
            }
        ]
        emit(rows[0]["kind"], rows[0]["payload"])
        readings = ("656.29995212nm", "656.30004788nm")
        for index, (marker, text) in enumerate(zip(("blue", "red"), readings, strict=True), 1):
            proposal = {"kind": "HOVER", "surface": "spectrum", "marker": marker, "sequence": index}
            result = {
                "sequence": index,
                "task_completed": False,
                "spectrum_sample": {
                    "marker": marker,
                    "source": "visible_spectrum_tooltip",
                    "wavelength_text": text,
                },
            }
            rows.extend(
                (
                    {"kind": "action_proposed", "payload": proposal},
                    {"kind": "action_result", "payload": result},
                )
            )
            emit("action_proposed", proposal)
            rig.calls.append(("spectrum_hover", marker))
            emit("action_result", result)
        result = {
            "events": rows,
            "result": spectrum_excursion("656.3nm", *readings),
            "visibility": "full_glyph_and_occlusion_checked",
        }
        (output / "spectrum.json").write_text(json.dumps(result))
        return result

    class Positive:
        def __init__(self, page, config, output, *, emit, **options):
            self.output, self.emit, self.finished, self.report = output, emit, False, None
            self.output.mkdir()
            self.phase = "presence"
            rig.calls.append(("init", "positive"))
            rig.positive_options.append(options)
            rig.components.append(self)
            if rig.outcome["positive_init_callback"]:
                emit("hello", {"task": "positive fixture"})

        def state(self):
            return {
                "star": rig.outcome["positive_star"],
                "phase": self.phase,
                "derived_transport_verified": self.finished and rig.outcome["positive_verified"],
                "task_completed": False,
                "reference_measurements": {"fixture": True},
            }

        def advance(self):
            self.emit("action_proposed", {"kind": "CLICK", "fixture": True})
            if rig.outcome["positive_error"]:
                raise rig.outcome["positive_error"]
            rig.calls.append(("advance", "positive"))
            self.finished, self.phase = True, rig.outcome["positive_phase"]
            self.report = self.state()
            self.emit("episode_summary", self.report)

        def abort(self):
            if not self.finished:
                self.finished, self.phase = True, "aborted"
                rig.calls.append(("abort", "positive"))

        def close(self):
            self.abort()

    monkeypatch.setattr(module, "capture_spectrum_excursion", capture)
    monkeypatch.setattr(module, "PositivePlanetSteps", Positive)
    monkeypatch.setattr(
        module, "load_graph", lambda path: (rig.calls.append(("load_graph", str(path))), object())[1]
    )
    return rig


def create_positive(rig, **kwargs):
    return create(rig, planet_pilot="planet-pilot", planet_final_evaluation="planet-final", **kwargs)


def advance_to(session, phase):
    for _ in range(20):
        if session.phase == phase:
            return
        assert not session.finished, session.state()
        session.advance()
    pytest.fail("Phase not reached")


def test_positive_continuation_is_separate_bounded_phases_and_not_completed(positive_rig):
    rig = positive_rig
    session = create_positive(rig)
    advance_to(session, "capture_spectrum")
    assert not any(c[0] == "capture" for c in rig.calls)
    assert session.window_evidence["progress_sha256"]
    session.advance()
    assert session.phase == "positive_planet" and not any(c == ("init", "positive") for c in rig.calls)
    assert rig.calls.count(("capture", "spectrum")) == 1
    session.advance()
    assert rig.calls[-1] == ("init", "positive")
    session.advance()
    assert session.phase == "planet_classification_required" and session.finished
    assert not session.state()["task_completed"] and not any(c[0] in {"save", "verify"} for c in rig.calls)
    assert session.report["reference_measurements"] == {"fixture": True}
    for field in (
        "class_dir",
        "numeric_dir",
        "color_dir",
        "window_dir",
        "spectrum_dir",
        "positive_dir",
        "presence_dir",
        "raw_dir",
        "derived_dir",
    ):
        assert field in session.report["artifact_paths"]
    option = rig.positive_options[0]
    assert option["window_report_sha256"] == session.window_evidence["progress_sha256"]
    assert option["spectrum_sha256"] == session.spectrum_evidence["sha256"]
    assert option["supplied_star_class"] == "main_sequence"
    summaries = [p for event, p in rig.events if event == "episode_summary"]
    assert len(summaries) == 1 and not summaries[0]["completed"]
    assert any(
        p.get("component") == "planet.positive" and "component_summary" in p
        for event, p in rig.events
        if event == "state"
    )


@pytest.mark.parametrize("phase", ["capture_spectrum", "positive_planet"])
def test_abort_at_new_phase_never_schedules_its_action(positive_rig, phase):
    session = create_positive(positive_rig)
    advance_to(session, phase)
    session.abort()
    before = list(positive_rig.calls)
    session.advance()
    session.close()
    assert positive_rig.calls == before and session.phase == "aborted"


@pytest.mark.parametrize("point", ["spectrum", "positive_init", "positive_action"])
@pytest.mark.parametrize("effect", ["abort", "failure", "reentry"])
def test_new_callback_boundaries_are_sticky(positive_rig, point, effect):
    rig = positive_rig
    session = create_positive(rig)
    rig.outcome["positive_init_callback"] = point == "positive_init"

    def receive(event, payload):
        name = payload.get("component")
        original = payload.get("component_event", {})
        match = (
            point == "spectrum"
            and name == "planet.spectrum"
            and event == "action_proposed"
            or point == "positive_init"
            and name == "planet.positive"
            and original.get("event") == "hello"
            or point == "positive_action"
            and name == "planet.positive"
            and event == "action_proposed"
        )
        if match:
            if effect == "abort":
                session.abort()
            elif effect == "reentry":
                session.advance()
            else:
                raise RuntimeError("private callback secret")

    session._callback = receive
    complete(session)
    assert session.phase in {"stopped", "aborted"} and not session.state()["task_completed"]
    assert ("advance", "positive") not in rig.calls
    if point == "spectrum":
        assert not any(c[0] == "spectrum_hover" for c in rig.calls)
    assert "private callback secret" not in json.dumps(session.report)
    before = list(rig.calls)
    session.advance()
    assert before == rig.calls


@pytest.mark.parametrize("source", ["window", "chart", "spectrum"])
def test_changed_pinned_source_blocks_new_positive_stages(positive_rig, source):
    rig = positive_rig
    session = create_positive(rig)
    advance_to(session, "positive_planet" if source == "spectrum" else "capture_spectrum")
    if source == "spectrum":
        path = rig.root / session.spectrum_evidence["path"]
    else:
        key = "progress_path" if source == "window" else "chart_path"
        path = rig.root / session.window_evidence[key]
    path.write_bytes(b"changed")
    before = list(rig.calls)
    session.advance()
    assert session.phase == "stopped" and session.failure == "star_session_source_hash_changed"
    assert rig.calls == before


@pytest.mark.parametrize("change", ["missing", "outside", "wrong_star", "bad_hash", "duration"])
def test_dip_without_valid_owned_capture_cannot_start_spectrum(positive_rig, monkeypatch, change):
    rig = positive_rig
    session = create_positive(rig)
    advance_to(session, "planet_window")
    session.advance()
    component = session.component
    advance = component.advance

    def invalid():
        advance()
        if change == "missing":
            component.report.pop("progress_path")
        elif change == "outside":
            component.report["progress_path"] = "../outside/report.json"
        elif change == "bad_hash":
            component.report["progress_sha256"] = "0" * 64
        else:
            path = rig.root / component.report["progress_path"]
            report = json.loads(path.read_text())
            report["star" if change == "wrong_star" else "requested_days"] = (
                "OTHER" if change == "wrong_star" else 10000
            )
            path.write_text(json.dumps(report))
            component.report["progress_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(component, "advance", invalid)
    session.advance()
    assert session.phase == "stopped" and ("capture", "spectrum") not in rig.calls


@pytest.mark.parametrize("selected", ["red_giant", "supergiant", "white_dwarf"])
@pytest.mark.parametrize("window", ["no_selected", "dip_observed"])
def test_non_main_no_verifies_but_positive_stops_before_calculations(positive_rig, selected, window):
    rig = positive_rig
    rig.outcome["window"] = window
    session = create_positive(rig, selected_class=selected, lifetime_prefix=None)
    complete(session)
    assert session.phase == (
        "verified_no_planet" if window == "no_selected" else "planet_calculation_unsupported"
    )
    assert (
        session.model_options["selected_class"] == selected
        and session.model_options["lifetime_prefix"] is None
    )
    assert session.state()["task_completed"] == (window == "no_selected")
    assert not any(c[0] in {"capture", "load_graph"} for c in rig.calls)
    assert (("save", "No") in rig.calls) == (window == "no_selected")
    assert (("verify", "No") in rig.calls) == (window == "no_selected")


@pytest.mark.parametrize(
    "options",
    [
        {"planet_pilot": "only"},
        {"planet_final_evaluation": "only"},
        {"planet_seed": True},
        {"selected_class": "giant", "lifetime_prefix": None},
        {"selected_class": "white_dwarf", "lifetime_prefix": "Ga"},
        {"lifetime_prefix": None},
        {"planet_preserve_painted_class": "unknown"},
        {"lifetime_prefix": "a"},
    ],
)
def test_incompatible_options_reject_before_run(rig, options):
    with pytest.raises(ValueError):
        create(rig, **options)
    assert not rig.calls and not (rig.root / "run").exists()


def test_supported_ta_prefix_is_preserved_without_selection(rig):
    session = create(rig, lifetime_prefix="Ta")
    assert session.model_options["lifetime_prefix"] == "Ta"
    assert not rig.calls


def test_spectrum_failure_is_redacted_and_never_retried(positive_rig):
    rig = positive_rig
    rig.outcome["spectrum_error"] = RuntimeError("private spectrum URL")
    session = create_positive(rig)
    complete(session)
    assert session.phase == "stopped" and rig.calls.count(("capture", "spectrum")) == 1
    assert "private spectrum URL" not in json.dumps(session.report)
    session.advance()
    assert rig.calls.count(("capture", "spectrum")) == 1


@pytest.mark.parametrize("change", ["not_verified", "wrong_phase", "other_star"])
def test_positive_termination_is_not_a_verified_handoff(positive_rig, change):
    rig = positive_rig
    rig.outcome.update(
        {
            "not_verified": {"positive_verified": False},
            "wrong_phase": {"positive_phase": "stopped"},
            "other_star": {"positive_star": "Other"},
        }[change]
    )
    session = create_positive(rig)
    complete(session)
    assert session.phase == "stopped" and not session.state()["task_completed"]


def test_terminal_callback_failure_cannot_persist_task_success(rig):
    session = create(rig)

    def emit(event, payload):
        if event == "episode_summary":
            raise RuntimeError("private final callback")

    session._callback = emit
    complete(session)
    assert session.phase == "stopped" and not session.report["task_completed"]


def test_intercepted_spectrum_positive_child_handoff_retains_evidence(rig, raster_page, monkeypatch):
    # Stellar decisions and window scheduling are fixture seams; spectrum,
    # positive coordinator, chart refreshes and native copies are real guarded
    # components. The fixture policy is scripted, not a learned score.
    import habfly.browser_planet_steps as planet_steps
    import habfly.browser_positive_steps as positive_steps
    import habfly.browser_star_session as module
    from habfly.browser_observation_progress import capture_observation_progress
    from habfly.environments.planet_calculations import FIELDS, planet_expert
    from habfly.planet_knowledge import PlanetCalculator

    page, frame = raster_page
    frame.locator("div").first.evaluate("e=>{e.textContent='Fixture';e.style.textTransform='uppercase'}")
    pilot, final, *_ = seed_sources(rig.root)
    graph = SimpleNamespace(body_ids=list(range(2000)))

    class Window:
        def __init__(self, page, config, output, *, run_history, select_no, emit):
            self.page, self.config, self.output = page, config, output
            self.finished, self.report = False, None

        def advance(self):
            path = self.output / "progress/report.json"
            capture_observation_progress(self.page, self.config, path.parent, requested_days=5000)
            self.report = {
                "progress_path": str(path.relative_to(rig.root)),
                "progress_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            self.finished = True

        def state(self):
            return {"star": "Fixture", "phase": "dip_observed" if self.finished else "observing"}

        def abort(self):
            self.finished = True

    class ReferencePolicy:
        hidden_size, observation_encoding = 16, "structured_planet_tool_v1"

        def act(self, observation, hidden):
            return planet_expert(observation, PlanetCalculator().pack), None, {"fixture": "reference_only"}

    for target in (positive_steps, planet_steps):
        monkeypatch.setattr(
            target, "load_checkpoint", lambda *a, **kw: (ReferencePolicy(), {"graph_hash": "fixture"})
        )
        monkeypatch.setattr(target, "graph_fingerprint", lambda _: "fixture")
    monkeypatch.setattr(module, "load_graph", lambda _: graph)
    monkeypatch.setattr(module, "PlanetWindowSession", Window)
    monkeypatch.setattr(
        module,
        "navigate_project",
        lambda *a, **kw: {
            "same_star_verified": True,
            "destination_verified": True,
            "suggested_required_text": ["Observations"],
        },
    )
    session = module.BrowserStarSession(
        page,
        fixture_config(),
        rig.root / "run",
        run_history=rig.root,
        star="Fixture",
        class_dir=rig.root / "class",
        selected_class="main_sequence",
        lifetime_prefix="Ga",
        dataset="fixture",
        checkpoint="fixture",
        color_experiment="fixture",
        graph_path="fixture",
        planet_pilot=pilot,
        planet_final_evaluation=final,
        emit=lambda *e: rig.events.append(e),
    )
    for _ in range(100):
        session.advance()
        if session.finished:
            break
    assert session.phase == "planet_classification_required", session.report
    assert not session.report["task_completed"]
    assert all(frame.locator("#" + name).input_value() for name in (*positive_steps.RAW, *FIELDS))
    assert not page.get_by_role("checkbox").is_checked()
    assert not any(c[0] in {"save", "verify"} for c in rig.calls)
    reference = session.report["reference_measurements"]
    assert reference["mode"] == "approximate_reference_raster" and reference["star"] == "Fixture"
    assert not reference["scientific_verified"] and not reference["learned_perception"]
    for key in ("raw_dir", "derived_dir"):
        assert (rig.root / session.report["artifact_paths"][key] / "report.json").exists()
    assert sum(event == "episode_summary" for event, _ in rig.events) == 1
