"""Injected star scheduler, real owned two-tooltip reader; no native/model runs.

The upstream stellar/window and downstream spectrum/policy components are the
existing scheduling seams. Only sensor scheduling is replaced here; its source
PNG, axes, six-zoom event bundles and independent reference reader are real
validation inputs with explicitly synthetic fixture origins.
"""
# ruff: noqa: F811

import json
from copy import deepcopy

import pytest
from test_browser_star_session import advance_to, create_positive, positive_rig, rig  # noqa: F401
from test_planet_two_tooltip_reference import read, repin, sha, two_bundle

import habfly.browser_shallow_transit_steps as shallow_module
from habfly.browser_shallow_transit_probe import FLAGS
from habfly.planet_tooltip_reference import MODE, TWO_MODE, load_tooltip_reference


def write(path, value):
    path.write_text(json.dumps(value))
    return sha(path.read_bytes())


def sources(history, output):
    specs = two_bundle(output)
    for spec in specs:
        directory = output / spec["directory"]
        scope, report = read(directory / "scope.json"), read(directory / "report.json")
        source = output / scope["source_dir"]
        original = read(source / "report.json")
        original["star"] = "FIXTURE"
        hashes = {**scope["source_hashes"], "report.json": write(source / "report.json", original)}
        scope.update(star="Fixture", source_hashes=hashes, source_dir=str(source.relative_to(history)))
        report.update(star="Fixture", source_hashes=hashes)
        write(directory / "scope.json", scope)
        write(directory / "report.json", report)
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        events[0]["payload"]["chart"]["star"] = "Fixture"
        events[-1]["payload"] = report
        (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
        repin(output, spec)
        spec["directory"] = str(directory.relative_to(history))
    first = history / read(history / specs[0]["directory"] / "scope.json")["source_dir"]
    window = {
        "progress_path": str((first / "report.json").relative_to(history)),
        "progress_sha256": sha((first / "report.json").read_bytes()),
        "chart_path": str((first / "chart.png").relative_to(history)),
        "chart_sha256": sha((first / "chart.png").read_bytes()),
    }
    descriptor = {"expected_star": "Fixture", "diagnostics": specs, "mode": TWO_MODE}
    measurements = load_tooltip_reference(history, specs, expected_star="Fixture", mode=TWO_MODE)
    return window, descriptor, measurements


@pytest.fixture
def two_rig(positive_rig, monkeypatch):
    subject = positive_rig
    subject.outcome.update(window="shallow_reference_required", sensor_changes={})
    subject.sensors, subject.sensor_options = [], []

    class Sensor:
        def __init__(self, page, config, output, *, emit, **options):
            self.output, self.emit, self.options = output, emit, options
            self.phase, self.finished, self.report = "initializing", False, None
            self.ticks = 0
            output.mkdir()
            self.window, self.descriptor, self.measurements = sources(subject.root, output)
            write(output / "scope.json", self.state())
            (output / "events.jsonl").write_text("synthetic scheduling seam; native bundles are separate\n")
            subject.sensors.append(self)
            subject.sensor_options.append(options)
            subject.calls.append(("init", "shallow"))

        def state(self):
            return {
                "mode": shallow_module.MODE,
                "measurement_mode": TWO_MODE,
                "star": "Fixture",
                "phase": self.phase,
                "status": "completed" if self.finished else "running",
                "finished": self.finished,
                "failure_reason": None,
                "native_action_outcome_uncertain": False,
                "event_forwarding_failed": False,
                "cleanup_failed": False,
                "two_event_reference_enabled": True,
                "probes": 2,
                "restorations": 2,
                "max_native_actions": 38,
                "window_evidence": deepcopy(self.window),
                "tooltip_reference": deepcopy(self.descriptor),
                "measurements": deepcopy(self.measurements),
                "project_completed": False,
                **FLAGS,
            }

        def advance(self):
            self.emit("action_proposed", {"kind": "HOVER", "surface": "chart", "injected": True})
            subject.calls.append(("advance", "shallow"))
            self.ticks += 1
            if self.ticks == 1:
                self.phase = "probing"
                return
            self.phase, self.finished = "measurements_ready", True
            self.report = {**self.state(), **subject.outcome["sensor_changes"]}
            write(self.output / "report.json", self.report)
            self.emit("episode_summary", deepcopy(self.report))

        def abort(self):
            if not self.finished:
                self.finished, self.phase = True, "aborted"
                subject.calls.append(("abort", "shallow"))

        close = abort

    monkeypatch.setattr(shallow_module, "ShallowTransitSteps", Sensor)
    return subject


def create_two(subject, **kwargs):
    return create_positive(subject, **{"shallow_reference": True, "two_event_reference": True, **kwargs})


def initialize_sensor(subject, **kwargs):
    session = create_two(subject, **kwargs)
    advance_to(session, "shallow_reference")
    session.advance()
    assert not session.finished, session.state()
    assert subject.sensors and subject.sensors[-1].ticks == 0
    return session


def assert_sticky_stop(session, subject):
    assert session.finished and session.phase in {"stopped", "aborted"}
    assert session.state()["task_completed"] is False
    assert session.state()["project_completed"] is False
    assert not any(call[0] in {"capture", "save", "verify"} for call in subject.calls)
    assert not subject.positive_options
    before = list(subject.calls)
    for _ in range(3):
        session.advance()
    assert subject.calls == before


def test_optin_adopts_real_two_bundle_and_forwards_exact_descriptor_without_success(two_rig):
    subject = two_rig
    session = initialize_sensor(subject)
    original = deepcopy(session.window_evidence)
    assert subject.sensor_options[0]["allow_two_events"] is True
    assert session.scope["two_event_reference_enabled"] is True
    session.advance()
    assert session.phase == "shallow_reference" and not session.state()["task_completed"]
    session.advance()
    assert session.phase == "capture_spectrum", session.state()
    sensor = subject.sensors[0]
    assert session._original_window_evidence == original
    assert session.tooltip_reference == sensor.descriptor
    assert session.window_evidence == sensor.window and session.window_evidence != original
    assert sensor.measurements["mode"] == TWO_MODE
    assert sensor.measurements["recurrence_confirmed"] is False
    assert len(sensor.measurements["features"]) == 2
    assert not any(call[0] == "capture" for call in subject.calls)
    session.advance()
    session.advance()
    assert subject.positive_options[0]["tooltip_reference"] == sensor.descriptor
    assert subject.positive_options[0]["window_report_sha256"] == sensor.window["progress_sha256"]
    session.advance()
    assert session.phase == "planet_classification_required"
    assert not session.report["task_completed"] and not session.report["project_completed"]


def test_no_optin_rejects_forged_successful_two_sensor_before_spectrum(two_rig):
    session = initialize_sensor(two_rig, two_event_reference=False)
    assert "allow_two_events" not in two_rig.sensor_options[0]
    assert "two_event_reference_enabled" not in session.scope
    session.advance()
    session.advance()
    assert_sticky_stop(session, two_rig)


@pytest.mark.parametrize(
    "options",
    [
        {"two_event_reference": 1},
        {"two_event_reference": None},
        {"two_event_reference": []},
        {"two_event_reference": True, "shallow_reference": False},
    ],
)
def test_invalid_permission_rejects_before_any_child_or_output(two_rig, options):
    with pytest.raises(ValueError):
        create_two(two_rig, **options)
    assert not two_rig.calls and not (two_rig.root / "run").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"measurement_mode": MODE},
        {"measurement_mode": "invented"},
        {"two_event_reference_enabled": False},
        {"two_event_reference_enabled": 1},
        {"probes": 3},
        {"probes": 2.0},
        {"restorations": 3},
        {"restorations": 2.0},
        {"max_native_actions": 64},
        {"max_native_actions": 38.0},
        {"task_completed": True},
        {"scientific_verified": True},
    ],
)
def test_forged_mode_counts_and_authority_fail_closed(two_rig, changes):
    session = initialize_sensor(two_rig)
    two_rig.outcome["sensor_changes"] = changes
    session.advance()
    session.advance()
    assert_sticky_stop(session, two_rig)


@pytest.mark.parametrize("change", ["missing_mode", "legacy_mode", "unknown_mode", "extra", "wrong_star"])
def test_descriptor_exact_shape_and_explicit_mode_are_required(two_rig, change):
    session = initialize_sensor(two_rig)
    descriptor = deepcopy(two_rig.sensors[0].descriptor)
    if change == "missing_mode":
        del descriptor["mode"]
    elif change == "extra":
        descriptor["fallback"] = True
    elif change == "wrong_star":
        descriptor["expected_star"] = "Other"
    else:
        descriptor["mode"] = MODE if change == "legacy_mode" else "unknown"
    two_rig.outcome["sensor_changes"] = {"tooltip_reference": descriptor}
    session.advance()
    session.advance()
    assert_sticky_stop(session, two_rig)


@pytest.mark.parametrize("change", ["measurement", "pinned_source", "descriptor_count"])
def test_actual_reader_not_report_claim_is_the_adoption_authority(two_rig, change):
    session = initialize_sensor(two_rig)
    sensor = two_rig.sensors[0]
    if change == "measurement":
        sensor.measurements["period_days"]["value"] = "1999"
    elif change == "pinned_source":
        path = two_rig.root / sensor.descriptor["diagnostics"][0]["directory"] / "events.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        sensor.descriptor["diagnostics"].append(deepcopy(sensor.descriptor["diagnostics"][0]))
    session.advance()
    session.advance()
    assert_sticky_stop(session, two_rig)


@pytest.mark.parametrize(
    "field,value",
    [
        ("option", False),
        ("option", 1),
        ("scope", False),
        ("scope", 1),
        ("disk_scope", False),
    ],
)
def test_permission_cannot_change_after_construction(two_rig, field, value):
    session = create_two(two_rig)
    if field == "option":
        session.two_event_reference = value
    elif field == "scope":
        session.scope["two_event_reference_enabled"] = value
    else:
        path = session.output / "scope.json"
        scope = read(path)
        scope["two_event_reference_enabled"] = value
        write(path, scope)
    session.advance()
    assert_sticky_stop(session, two_rig)
    assert not two_rig.calls


def test_permission_cannot_be_granted_after_default_construction(two_rig):
    session = create_two(two_rig, two_event_reference=False)
    session.two_event_reference = True
    session.scope["two_event_reference_enabled"] = True
    session.advance()
    assert_sticky_stop(session, two_rig)
    assert not two_rig.calls


def test_callback_permission_change_prevents_same_advance_sensor_action(two_rig):
    session = initialize_sensor(two_rig)

    def change(kind, payload):
        if kind == "action_proposed":
            session.two_event_reference = False

    session._callback = change
    before = list(two_rig.calls)
    session.advance()
    assert_sticky_stop(session, two_rig)
    assert two_rig.calls == before + [("abort", "shallow")]


@pytest.mark.parametrize("point", ["before_child", "child_callback", "after_adoption"])
def test_abort_never_continues_sensor_or_spectrum(two_rig, point):
    session = create_two(two_rig)
    advance_to(session, "shallow_reference")
    if point == "before_child":
        session.abort()
    else:
        session.advance()
        if point == "child_callback":
            session._callback = lambda kind, payload: session.abort() if kind == "action_proposed" else None
            session.advance()
        else:
            session.advance()
            session.advance()
            assert session.phase == "capture_spectrum"
            session.abort()
    assert_sticky_stop(session, two_rig)


@pytest.mark.parametrize("change", ["source", "descriptor", "permission"])
def test_adopted_source_or_permission_change_stops_before_spectrum(two_rig, change):
    session = initialize_sensor(two_rig)
    session.advance()
    session.advance()
    assert session.phase == "capture_spectrum", session.state()
    if change == "source":
        path = two_rig.root / session.tooltip_reference["diagnostics"][0]["directory"] / "readable.png"
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "descriptor":
        session.tooltip_reference["mode"] = MODE
    else:
        session.two_event_reference = False
    session.advance()
    assert_sticky_stop(session, two_rig)
