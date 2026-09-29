"""Injected scheduling gates; no native browser, trained score, or science claim."""
# ruff: noqa: F811

import hashlib
import io
import json
from copy import deepcopy

import pytest
from PIL import Image
from test_browser_planet_window import make, scheduled  # noqa: F401
from test_browser_star_session import (  # noqa: F401
    advance_to,
    complete,
    create,
    create_positive,
    positive_rig,
    rig,
)
from test_planet_window_policy import crop, labels

import habfly.browser_shallow_transit_steps as shallow_module
import habfly.browser_star_session as module
import habfly.planet_tooltip_reference as tooltip_module
from habfly.browser_observation_progress import trace_progress
from habfly.browser_shallow_transit_probe import FLAGS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return sha(path)


@pytest.fixture
def shallow_rig(positive_rig, monkeypatch):
    rig = positive_rig
    rig.outcome.update(window="shallow_reference_required", sensor_changes={}, sensor_callback=False)
    rig.sensor_options, rig.sensors, rig.window_options = [], [], []
    base_window = module.PlanetWindowSession

    def window(*args, **kwargs):
        rig.window_options.append(kwargs)
        return base_window(*args, **kwargs)

    class Sensor:
        def __init__(self, page, config, output, *, emit, **options):
            self.output, self.emit, self.options = output, emit, options
            self.phase, self.finished, self.report = "initializing", False, None
            self.ticks = 0
            output.mkdir()
            self.descriptor = {"expected_star": "Fixture", "diagnostics": []}
            sources = {}
            for index in range(3):
                directory = output / f"probe-{index}"
                checksum = write(directory / "fixture.json", {"injected": True, "index": index})
                relative = str(directory.relative_to(rig.root))
                sources[relative + "/fixture.json"] = checksum
                self.descriptor["diagnostics"].append(
                    {"directory": relative, "files": {"fixture.json": checksum}}
                )
            self.measurements = {
                "mode": tooltip_module.MODE,
                "star": "Fixture",
                "source_sha256": sources,
                "validated_directories": [row["directory"] for row in self.descriptor["diagnostics"]],
                "scientific_verified": False,
                "learned_perception": False,
                "training_label": False,
                "task_completed": False,
                "receipt_sha256": "a" * 64,
            }
            final = output / "final"
            final.mkdir()
            png = crop()
            times, fluxes = labels(5000)
            (final / "chart.png").write_bytes(png)
            progress = final / "report.json"
            write(
                progress,
                {
                    **trace_progress(png, times, requested_days=5000),
                    "star": "FIXTURE",
                    "chart_sha256": sha(final / "chart.png"),
                    "time_axis_labels": times,
                    "flux_axis_labels": fluxes,
                    "browser_actions": 0,
                    "answer_writes": 0,
                },
            )
            self.window = {
                "progress_path": str(progress.relative_to(rig.root)),
                "progress_sha256": sha(progress),
                "chart_path": str((final / "chart.png").relative_to(rig.root)),
                "chart_sha256": sha(final / "chart.png"),
            }
            write(output / "scope.json", {"mode": shallow_module.MODE, "star": "Fixture", **FLAGS})
            (output / "events.jsonl").write_text("injected source journal\n")
            rig.sensor_options.append(options)
            rig.sensors.append(self)
            rig.calls.append(("init", "shallow"))

        def state(self):
            return {
                "mode": shallow_module.MODE,
                "measurement_mode": tooltip_module.MODE,
                "star": "Fixture",
                "phase": self.phase,
                "status": "completed" if self.finished else "running",
                "finished": self.finished,
                "failure_reason": None,
                "native_action_outcome_uncertain": False,
                "event_forwarding_failed": False,
                "cleanup_failed": False,
                "window_evidence": deepcopy(self.window),
                "tooltip_reference": deepcopy(self.descriptor),
                "measurements": deepcopy(self.measurements),
                "project_completed": False,
                **FLAGS,
            }

        def advance(self):
            self.emit("action_proposed", {"kind": "HOVER", "surface": "chart", "injected": True})
            rig.calls.append(("advance", "shallow"))
            self.ticks += 1
            if self.ticks == 1:
                self.phase = "probing"
                # Child claims never promote the owner or bypass its final receipt.
                self.emit("episode_summary", {"task_completed": True, "finished": True})
                return
            self.phase, self.finished = "measurements_ready", True
            self.report = {**self.state(), **rig.outcome["sensor_changes"]}
            self.report["source_sha256"] = {
                str(path.relative_to(rig.root)): sha(path)
                for path in self.output.rglob("*")
                if path.is_file()
            }
            self.report["events_sha256"] = sha(self.output / "events.jsonl")
            write(self.output / "report.json", self.report)
            self.emit("episode_summary", deepcopy(self.report))

        def abort(self):
            if not self.finished:
                self.finished, self.phase = True, "aborted"
                rig.calls.append(("abort", "shallow"))

        close = abort

    def loaded_reference(history, diagnostics, *, expected_star):
        sensor = rig.sensors[-1]
        assert history == rig.root
        assert diagnostics == sensor.descriptor["diagnostics"]
        assert expected_star.casefold() == "fixture"
        return deepcopy(sensor.measurements)

    monkeypatch.setattr(module, "PlanetWindowSession", window)
    monkeypatch.setattr(shallow_module, "ShallowTransitSteps", Sensor)
    monkeypatch.setattr(tooltip_module, "load_tooltip_reference", loaded_reference)
    return rig


@pytest.mark.parametrize("enabled", [False, True])
def test_real_window_policy_stops_by_default_and_optin_only_hands_off(scheduled, enabled):
    image = Image.open(io.BytesIO(crop())).convert("RGB")
    image.putpixel((162, 20), (32, 39, 42))
    data = io.BytesIO()
    image.save(data, format="PNG")
    scheduled.images[:] = [data.getvalue()]
    session = make(scheduled, select_no=True, allow_shallow_reference=enabled)
    policy = deepcopy(session.scope["policy"])
    session.advance()
    scheduled.clock[0] += 5
    state = session.advance()
    assert state["phase"] == ("shallow_reference_required" if enabled else "stopped")
    assert session.analysis["reason"] == "unknown_plot_palette" and session.scope["policy"] == policy
    assert session.finished and session.receipt is None and not state["task_completed"]
    session.advance()
    assert scheduled.calls == [("start", 5000), ("capture", 5000)]


@pytest.mark.parametrize("options", [{"shallow_reference": 1}, {"shallow_reference": True}])
def test_invalid_optin_without_planet_policy_rejects_offline(rig, options):
    with pytest.raises(ValueError):
        create(rig, **options)
    assert not rig.calls and not (rig.root / "run").exists()


def test_old_star_default_cannot_adopt_shallow_handoff(shallow_rig):
    rig = shallow_rig
    session = create_positive(rig)
    complete(session)
    assert session.phase == "stopped" and not session.state()["task_completed"]
    assert "allow_shallow_reference" not in rig.window_options[0]
    assert not rig.sensors and not any(c[0] == "capture" for c in rig.calls)


@pytest.mark.parametrize("star_class", ["red_giant", "white_dwarf", "supergiant"])
def test_supplied_non_main_class_cannot_run_shallow_planet_calculations(shallow_rig, star_class):
    rig = shallow_rig
    session = create_positive(rig, shallow_reference=True, selected_class=star_class, lifetime_prefix=None)
    complete(session)
    assert session.phase == "planet_calculation_unsupported"
    assert not session.state()["task_completed"] and not rig.sensors


def test_cooperative_pending_child_never_promotes_task_and_final_source_is_forwarded(shallow_rig):
    rig = shallow_rig
    session = create_positive(rig, shallow_reference=True)
    advance_to(session, "shallow_reference")
    original = deepcopy(session.window_evidence)
    assert rig.window_options[0]["allow_shallow_reference"] is True
    session.advance()  # Constructor only.
    assert rig.sensors and not any(c == ("advance", "shallow") for c in rig.calls)
    session.advance()  # One injected chart action; misleading child summary is nested.
    assert not session.finished and not session.state()["task_completed"]
    assert session.phase == "shallow_reference" and not any(c[0] == "capture" for c in rig.calls)
    session.advance()
    assert session.phase == "capture_spectrum" and not session.state()["task_completed"]
    sensor = rig.sensors[0]
    assert sensor.options["source_report"] == rig.root / original["progress_path"]
    assert sensor.options["source_report_sha256"] == original["progress_sha256"]
    assert json.loads((session.output / "window-handoff.json").read_text()) == original
    assert session._original_window_evidence == original and session.window_evidence != original
    session.advance()
    session.advance()
    forwarded = rig.positive_options[0]
    assert forwarded["tooltip_reference"] == sensor.descriptor
    assert forwarded["window_report"] == rig.root / sensor.window["progress_path"]
    assert forwarded["window_report_sha256"] == sensor.window["progress_sha256"]
    session.advance()
    assert session.phase == "planet_classification_required" and not session.report["task_completed"]
    assert not any(c[0] in {"save", "verify"} for c in rig.calls)


@pytest.mark.parametrize("stage", ["before_init", "probing", "after_handoff"])
def test_abort_is_sticky_before_further_sensor_or_spectrum_actions(shallow_rig, stage):
    rig = shallow_rig
    session = create_positive(rig, shallow_reference=True)
    advance_to(session, "shallow_reference")
    if stage != "before_init":
        session.advance()
        session.advance()
    if stage == "after_handoff":
        session.advance()
    session.abort()
    calls = list(rig.calls)
    for _ in range(3):
        assert session.advance()["phase"] == "aborted"
    assert calls == rig.calls and not session.state()["task_completed"]


def test_callback_abort_prevents_next_sensor_action(shallow_rig):
    rig = shallow_rig
    session = create_positive(rig, shallow_reference=True)
    advance_to(session, "shallow_reference")
    session.advance()
    session._callback = lambda kind, payload: session.abort() if kind == "action_proposed" else None
    session.advance()
    assert session.phase == "aborted"
    assert not any(c == ("advance", "shallow") for c in rig.calls)
    assert not any(c[0] == "capture" for c in rig.calls)


@pytest.mark.parametrize(
    "changes",
    [
        {"phase": "stopped"},
        {"task_completed": True},
        {"star": "Other"},
        {"mode": "unknown_sensor"},
        {"scientific_verified": True},
        {"measurement_mode": "approximate_reference_raster"},
        {"learned_perception": True},
        {"native_action_outcome_uncertain": True},
    ],
)
def test_invalid_final_sensor_report_stops_before_spectrum(shallow_rig, changes):
    rig = shallow_rig
    rig.outcome["sensor_changes"] = changes
    session = create_positive(rig, shallow_reference=True)
    advance_to(session, "shallow_reference")
    session.advance()
    session.advance()
    session.advance()
    assert session.phase == "stopped" and not session.state()["task_completed"]
    assert not any(c[0] == "capture" for c in rig.calls)


@pytest.mark.parametrize(
    "change", ["report", "tree_leaf", "original_window", "descriptor_cache", "tree_addition"]
)
def test_mutated_handoff_sources_or_cache_stop_before_spectrum(shallow_rig, change):
    rig = shallow_rig
    session = create_positive(rig, shallow_reference=True)
    advance_to(session, "capture_spectrum")
    paths = {
        "report": session.output / "shallow/report.json",
        "tree_leaf": session.output / "shallow/probe-0/fixture.json",
        "original_window": rig.root / session._original_window_evidence["progress_path"],
    }
    if change in paths:
        paths[change].write_bytes(paths[change].read_bytes() + b" ")
    elif change == "descriptor_cache":
        session.tooltip_reference["expected_star"] = "Other"
    else:
        write(session.output / "shallow/unexpected.json", {"new": True})
    session.advance()
    assert session.phase == "stopped" and not any(c[0] == "capture" for c in rig.calls)


def test_final_window_elsewhere_in_history_is_not_a_child_source(shallow_rig):
    rig = shallow_rig
    session = create_positive(rig, shallow_reference=True)
    advance_to(session, "shallow_reference")
    session.advance()
    sensor = rig.sensors[-1]
    original = deepcopy(session.window_evidence)
    # Correct hashes and same star are insufficient: this is the OLD window.
    rig.outcome["sensor_changes"] = {"window_evidence": original}
    session.advance()
    session.advance()
    assert session.phase == "stopped"
    assert not any(c[0] == "capture" for c in rig.calls)
    assert sensor.window != original


@pytest.mark.parametrize("change", ["wrong_capture_star", "wrong_days", "outside_history", "symlink"])
def test_final_window_capture_identity_and_ownership_are_required(shallow_rig, tmp_path, change):
    rig = shallow_rig
    session = create_positive(rig, shallow_reference=True)
    advance_to(session, "shallow_reference")
    session.advance()
    sensor = rig.sensors[-1]
    path = rig.root / sensor.window["progress_path"]
    if change in {"wrong_capture_star", "wrong_days"}:
        value = json.loads(path.read_text())
        value["star" if change == "wrong_capture_star" else "requested_days"] = (
            "Other" if change == "wrong_capture_star" else 10000
        )
        write(path, value)
        sensor.window["progress_sha256"] = sha(path)
    elif change == "outside_history":
        # A sibling temporary directory is outside this run's owned history.
        path = tmp_path.parent / (tmp_path.name + "-outside.json")
        write(path, {"star": "Fixture"})
        sensor.window["progress_path"] = str(path)
        sensor.window["progress_sha256"] = sha(path)
    else:
        alias = sensor.output / "aliased-report.json"
        alias.symlink_to(path)
        sensor.window["progress_path"] = str(alias.relative_to(rig.root))
    session.advance()
    session.advance()
    assert session.phase == "stopped" and not session.state()["task_completed"]
    assert not any(c[0] == "capture" for c in rig.calls)


def test_persisted_sensor_report_changed_by_callback_is_rejected(shallow_rig):
    rig = shallow_rig
    session = create_positive(rig, shallow_reference=True)
    advance_to(session, "shallow_reference")
    session.advance()

    def emit(kind, payload):
        path = session.output / "shallow/report.json"
        if kind == "state" and path.exists():
            value = json.loads(path.read_text())
            value["star"] = "Changed by callback"
            write(path, value)

    session._callback = emit
    session.advance()
    session.advance()
    assert session.phase == "stopped" and not any(c[0] == "capture" for c in rig.calls)


def test_tooltip_descriptor_passed_to_positive_is_not_mutable_sensor_alias(shallow_rig):
    rig = shallow_rig
    session = create_positive(rig, shallow_reference=True)
    advance_to(session, "capture_spectrum")
    rig.sensors[0].descriptor["expected_star"] = "Later sensor mutation"
    assert session.tooltip_reference["expected_star"] == "Fixture"


@pytest.mark.parametrize("field", ["progress_sha256", "chart_sha256"])
@pytest.mark.parametrize("bad_hash", [None, "", "A" * 64, "g" * 64])
def test_final_window_requires_explicit_lowercase_hashes(shallow_rig, field, bad_hash):
    rig = shallow_rig
    session = create_positive(rig, shallow_reference=True)
    advance_to(session, "shallow_reference")
    session.advance()
    rig.sensors[-1].window[field] = bad_hash
    session.advance()
    session.advance()
    assert session.phase == "stopped" and not any(c[0] == "capture" for c in rig.calls)


@pytest.mark.parametrize("descriptor", [None, {}, {"expected_star": "Fixture", "diagnostics": None}])
def test_malformed_descriptor_stops_before_native_spectrum(shallow_rig, descriptor):
    rig = shallow_rig
    rig.outcome["sensor_changes"] = {"tooltip_reference": descriptor}
    session = create_positive(rig, shallow_reference=True)
    advance_to(session, "shallow_reference")
    session.advance()
    session.advance()
    session.advance()
    assert session.phase == "stopped" and not any(c[0] == "capture" for c in rig.calls)
