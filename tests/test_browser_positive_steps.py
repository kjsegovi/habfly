"""Bounded coordinator tests; scripted fixture children are not learned scores."""
# ruff: noqa: F811

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_steps import seed_sources
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_raster_planet_evidence import raster_page, sha, sources  # noqa: F401

import habfly.browser_positive_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.contracts import RuntimeEvent
from habfly.environments.planet_calculations import FIELDS, planet_expert
from habfly.planet_knowledge import PlanetCalculator


@pytest.fixture
def rig(tmp_path, monkeypatch):
    pilot, final, checkpoint, metadata, gate = seed_sources(tmp_path)
    window, spectrum = tmp_path / "window.json", tmp_path / "spectrum.json"
    window.write_text("{}")
    spectrum.write_text("{}")
    source_hashes = {window: sha(window), spectrum: sha(spectrum)}
    evidence = {
        "star": "DULAT",
        "measurements": {
            "line_shift": {"value": "0.00004788", "unit": "nm"},
            "brightness_drop": {"value": "7.0", "unit": "%", "lower": "6.0", "upper": "8.0"},
            "period_days": {"value": "195.0", "unit": "day", "lower": "193.0", "upper": "197.0"},
        },
    }
    flags = {
        "current_star": "Dulat",
        "source_star": "DULAT",
        "graph": "graph",
        "presence_error": None,
        "presence_hook": None,
        "child_star": "Dulat",
        "child_failed": None,
        "child_init_failed": None,
    }
    calls, received, children = [], [], []
    graph = SimpleNamespace(body_ids=list(range(2000)))
    monkeypatch.setattr(module, "graph_fingerprint", lambda _: flags["graph"])
    monkeypatch.setattr(
        module,
        "load_checkpoint",
        lambda *a, **k: (
            SimpleNamespace(hidden_size=16, observation_encoding="structured_planet_tool_v1"),
            {"graph_hash": "graph"},
        ),
    )

    def load_evidence(owner, report, report_hash, spectral, spectral_hash):
        owner.read(report, report_hash)
        owner.read(spectral, spectral_hash)
        return {**deepcopy(evidence), "star": flags["source_star"]}

    def reload_evidence(owner, value):
        for path, expected in source_hashes.items():
            owner.read(path, expected)
        assert value == {**evidence, "star": "DULAT"}

    def mapping(star, populated=False):
        fields = {"observation_days": {"current_value": "5000", "unit": "day"}}
        fields.update(
            {
                k: {"current_value": spec["value"] if populated else "", "unit": spec["unit"]}
                for k, spec in evidence["measurements"].items()
            }
        )
        return {"star_name": star, "observation": {"values": {"browser_field_map": fields}}}

    class ReadSession:
        def __init__(self, *args, **kwargs):
            calls.append("read_preflight")
            self.mapping, self.choices = mapping(flags["current_star"]), {"selected": None}

        def close(self):
            calls.append("close_preflight")

    def presence(page, config, output, **kwargs):
        calls.append("select_yes")
        if flags["presence_error"]:
            raise flags["presence_error"]
        output.mkdir()
        receipt = {"star": "DULAT", "value": "Yes", "readback_verified": True, "evidence": evidence}
        (output / "confirmed.json").write_text(json.dumps(receipt))
        if flags["presence_hook"]:
            flags["presence_hook"]()
        return receipt

    def get_presence(owner, path, expected):
        return json.loads(owner.read(path, expected)), {}, mapping("Dulat")

    class Child:
        def __init__(self, page, config, output, *, emit, **options):
            self.kind, self.output, self.emit = output.name, output, emit
            self.output.mkdir()
            self.session = SimpleNamespace(mapping=mapping(flags["child_star"], self.kind == "derived"))
            self.finished, self.report, self.ticks, self.aborted, self.sequence = False, None, 0, False, 0
            self.stream = (output / "events.jsonl").open("w")
            calls.append("init_" + self.kind)
            children.append(self)
            self.event("hello", {"fixture": True})
            if flags["child_init_failed"] == self.kind:
                self.finished, self.report = True, {"task_completed": False}
                self.session = None
                self.stream.close()

        def event(self, kind, payload):
            item = RuntimeEvent(
                event=kind, payload=payload, sequence=self.sequence, run_id="fixture-" + self.kind
            ).model_dump(mode="json")
            self.stream.write(json.dumps(item) + "\n")
            self.stream.flush()
            self.sequence += 1
            self.emit(item)

        def state(self):
            return {"kind": self.kind, "ticks": self.ticks, "finished": self.finished}

        def advance(self):
            assert not self.finished
            self.event("action_proposed", {"kind": "TYPE", "fixture": True})
            assert not self.finished
            calls.append("act_" + self.kind)
            self.ticks += 1
            self.event("action_result", {"task_completed": False})
            if self.ticks == (3 if self.kind == "raw" else 2):
                flag = (
                    "raw_measurement_transport_verified"
                    if self.kind == "raw"
                    else "planet_transport_verified"
                )
                self.report = {
                    flag: flags["child_failed"] != self.kind,
                    "task_completed": False,
                    "verified_fields": {name: {} for name in (module.RAW if self.kind == "raw" else FIELDS)},
                }
                self.event("episode_summary", self.report)
                (self.output / "report.json").write_text(json.dumps(self.report))
                self.stream.close()
                self.finished = True

        def abort(self):
            if not self.finished:
                self.finished = self.aborted = True
                calls.append("abort_" + self.kind)
                self.event("episode_summary", {"task_completed": False, "operator_aborted": True})
                self.stream.close()

        def close(self):
            self.abort()

    monkeypatch.setattr(module, "_evidence", load_evidence)
    monkeypatch.setattr(module, "_reload", reload_evidence)
    monkeypatch.setattr(module, "_presence", get_presence)
    monkeypatch.setattr(module, "_blank_current", lambda *a, **kw: None)
    monkeypatch.setattr(module, "PlanetNumericSession", ReadSession)
    monkeypatch.setattr(module, "select_raster_detected_planet", presence)
    monkeypatch.setattr(module, "RasterInputSteps", Child)
    monkeypatch.setattr(module, "PlanetDerivedSteps", Child)
    options = {
        "run_history": tmp_path,
        "star": "Dulat",
        "window_report": window,
        "window_report_sha256": sha(window),
        "spectrum_path": spectrum,
        "spectrum_sha256": sha(spectrum),
        "pilot": pilot,
        "final_evaluation": final,
        "graph": graph,
        "supplied_star_class": "main_sequence",
    }

    def create(**overrides):
        options2 = {**options, "emit": lambda *event: received.append(event), **overrides}
        return module.PositivePlanetSteps(None, config(), tmp_path / "run", **options2)

    return SimpleNamespace(
        create=create,
        root=tmp_path,
        options=options,
        flags=flags,
        calls=calls,
        received=received,
        children=children,
        window=window,
        checkpoint=checkpoint,
        metadata=metadata,
        gate=gate,
    )


def complete(component):
    for _ in range(20):
        component.advance()
        if component.finished:
            return
    pytest.fail("Mock bounded coordinator did not finish")


def test_tooltip_reference_display_never_fabricates_physical_depth_bounds():
    from habfly.planet_tooltip_reference import MODE

    component = module.PositivePlanetSteps.__new__(module.PositivePlanetSteps)
    component.star = "Fixture"
    component.evidence = {
        "mode": MODE,
        "measurements": {
            "period_days": {
                "value": "1178",
                "compatibility_interval": {"lower": "1177", "upper": "1179", "endpoints": "open"},
            },
            "brightness_drop": {"value": "0.762", "physical_bounds": None},
            "line_shift": {"value": "0.0001"},
        },
    }
    value = component._reference()
    assert value["mode"] == MODE
    assert value["period_days"]["compatibility_interval"] == {
        "lower": 1177.0,
        "upper": 1179.0,
        "endpoints": "open",
    }
    assert value["brightness_drop_percent"]["physical_bounds"] is None
    assert "lower" not in value["brightness_drop_percent"] and "upper" not in value["brightness_drop_percent"]
    assert not value["period_evidence_verified"] and not value["minimum_depth_verified"]


def test_handoff_is_not_completion_and_exact_children_keep_their_journals(rig):
    component = rig.create()
    assert rig.calls == ["read_preflight", "close_preflight"]
    old = (rig.root / "run/events.jsonl").read_bytes()
    component.state()
    assert (rig.root / "run/events.jsonl").read_bytes() == old
    complete(component)
    assert component.phase == "planet_classification_required" and component.finished
    assert component.report["derived_transport_verified"] and not component.report["task_completed"]
    assert rig.calls == [
        "read_preflight",
        "close_preflight",
        "select_yes",
        "init_raw",
        "act_raw",
        "act_raw",
        "act_raw",
        "init_derived",
        "act_derived",
        "act_derived",
    ]
    for name in ("presence", "raw", "derived"):
        assert name in component.report["child_artifacts"]
    outer = [json.loads(line) for line in (rig.root / "run/events.jsonl").read_text().splitlines()]
    for child in rig.children:
        original = [json.loads(line) for line in (child.output / "events.jsonl").read_text().splitlines()]
        forwarded = [
            e["payload"]["component_event"]
            for e in outer
            if e["payload"].get("component_identity") == child.kind
        ]
        assert original == forwarded
    assert [e["sequence"] for e in outer] == list(range(len(outer)))
    assert sum(e["event"] == "episode_summary" for e in outer) == 1
    assert outer[-1]["payload"] == component.report
    before = len(rig.calls)
    component.advance()
    component.close()
    assert len(rig.calls) == before
    reference = component.state()["reference_measurements"]
    assert reference["star"] == "Dulat" and reference["period_days"]["unit"] == "days"
    assert reference["brightness_drop_percent"]["unit"] == "percent"
    assert not reference["learned_perception"] and not reference["scientific_verified"]


@pytest.mark.parametrize("at", [0, 1, 2, 3, 5, 6, 7])
def test_abort_at_every_phase_is_sticky(rig, at):
    component = rig.create()
    for _ in range(at):
        component.advance()
    component.abort()
    count = len(rig.calls)
    component.advance()
    component.close()
    assert component.finished and component.phase == "aborted" and len(rig.calls) == count
    assert not component.report["task_completed"] and component.state()["reference_measurements"] is None
    assert all(child.finished for child in rig.children)


@pytest.mark.parametrize("phase", ["before_presence", "init_raw", "act_raw", "init_derived", "act_derived"])
@pytest.mark.parametrize("effect", ["abort", "failure", "reentry"])
def test_callback_boundaries_never_start_following_action(rig, phase, effect):
    holder = {}

    def emit(event, payload):
        envelope = payload.get("component_event", {})
        kind = payload.get("component_identity")
        match = (
            phase == "before_presence"
            and payload.get("stage") == "selecting_reference_presence"
            or phase == "init_" + str(kind)
            and envelope.get("event") == "hello"
            or phase == "act_" + str(kind)
            and envelope.get("event") == "action_proposed"
        )
        if match:
            if effect == "abort":
                holder["component"].abort()
            elif effect == "reentry":
                holder["component"].advance()
            else:
                raise RuntimeError("private callback details")

    component = rig.create(emit=emit)
    holder["component"] = component
    complete(component)
    assert component.finished and component.phase in {"stopped", "aborted"}
    forbidden = "select_yes" if phase == "before_presence" else phase.replace("init_", "act_")
    assert forbidden not in rig.calls
    assert "private callback details" not in (rig.root / "run/events.jsonl").read_text()
    count = len(rig.calls)
    component.advance()
    assert len(rig.calls) == count


def test_abort_during_blocking_presence_cannot_start_raw_child(rig):
    component = rig.create()
    rig.flags["presence_hook"] = component.abort
    component.advance()
    assert component.phase == "aborted" and not rig.children
    assert component.report["presence_stage_entered"] and component.report["native_actions_may_have_occurred"]


@pytest.mark.parametrize("source", ["window", "checkpoint", "metadata", "gate", "graph", "presence", "raw"])
def test_changed_sources_stop_before_another_action(rig, source):
    component = rig.create()
    component.advance()
    if source == "raw":
        while component.phase != "derived":
            component.advance()
        path = rig.root / "run/raw/report.json"
    elif source == "presence":
        path = rig.root / "run/presence/confirmed.json"
    elif source == "graph":
        rig.flags["graph"] = "changed"
        path = None
    else:
        path = getattr(rig, source)
    if path:
        path.write_text('{"changed":true}')
    count = len(rig.calls)
    component.advance()
    assert component.phase == "stopped" and not component.report["derived_transport_verified"]
    assert not any(c.startswith("act_") for c in rig.calls[count:])


@pytest.mark.parametrize("kind", ["raw", "derived"])
@pytest.mark.parametrize("when", ["init", "end"])
def test_failed_child_is_never_treated_as_verified(rig, kind, when):
    rig.flags["child_init_failed" if when == "init" else "child_failed"] = kind
    component = rig.create()
    complete(component)
    assert component.phase == "stopped" and not component.report["derived_transport_verified"]
    if kind == "raw":
        assert "init_derived" not in rig.calls


@pytest.mark.parametrize("which", ["source", "current", "child"])
def test_root_star_is_checked_without_guessing(rig, which):
    if which == "source":
        rig.flags["source_star"] = "OTHER"
        with pytest.raises(BrowserSafetyStop, match="source_star_changed"):
            rig.create()
        assert not (rig.root / "run").exists()
    else:
        rig.flags["current_star" if which == "current" else "child_star"] = "Other"
        component = rig.create()
        complete(component)
        assert component.phase == "stopped"
        assert not any(c.startswith("act_") for c in rig.calls)


@pytest.mark.parametrize(
    "option,value",
    [
        ("star", "Different"),
        ("supplied_star_class", "giant"),
        ("seed", True),
        ("presence_max_seconds", float("nan")),
        ("inputs_max_seconds", 901),
        ("preserve_painted_class", "unknown"),
        ("window_report_sha256", "0" * 64),
    ],
)
def test_bad_options_or_sources_reject_before_browser(rig, option, value):
    with pytest.raises(BrowserSafetyStop):
        rig.create(**{option: value})
    assert not rig.calls and not (rig.root / "run").exists()


def test_unknown_presence_error_is_redacted_and_does_not_retry(rig):
    component = rig.create()
    rig.flags["presence_error"] = RuntimeError("private address and password")
    component.advance()
    component.advance()
    assert rig.calls.count("select_yes") == 1 and component.finished
    assert "private address" not in (rig.root / "run/events.jsonl").read_text()


def test_callback_failure_on_final_summary_invalidates_handoff(rig):
    def emit(event, payload):
        if event == "episode_summary":
            raise RuntimeError("private terminal callback")

    component = rig.create(emit=emit)
    complete(component)
    assert component.phase == "stopped" and not component.report["derived_transport_verified"]
    assert (rig.root / "run/stopped.json").exists()


def test_intercepted_real_children_stop_before_planet_classification(raster_page, tmp_path, monkeypatch):
    import habfly.browser_planet_steps as planet_steps

    page, frame = raster_page
    frame.locator("div").first.evaluate("e=>{e.textContent='Dulat';e.style.textTransform='uppercase'}")
    source = sources(raster_page, tmp_path)
    pilot, final, *_ = seed_sources(tmp_path)
    graph = SimpleNamespace(body_ids=list(range(2000)))

    class ReferencePolicy:
        hidden_size, observation_encoding = 16, "structured_planet_tool_v1"

        def act(self, observation, hidden):
            return planet_expert(observation, PlanetCalculator().pack), None, {"fixture": "reference_only"}

    def load(*args, **kwargs):
        return ReferencePolicy(), {"graph_hash": "fixture"}

    for target in (module, planet_steps):
        monkeypatch.setattr(target, "load_checkpoint", load)
        monkeypatch.setattr(target, "graph_fingerprint", lambda _: "fixture")
    events = []
    component = module.PositivePlanetSteps(
        page,
        config(),
        tmp_path / "run",
        run_history=tmp_path,
        star="Dulat",
        **source,
        pilot=pilot,
        final_evaluation=final,
        graph=graph,
        supplied_star_class="main_sequence",
        emit=lambda *e: events.append(e),
    )
    assert not component.finished, component.report
    for _ in range(75):
        component.advance()
        if component.finished:
            break
    assert component.phase == "planet_classification_required", component.report
    assert component.report["derived_transport_verified"] and not component.report["task_completed"]
    assert all(frame.locator("#" + name).input_value() for name in (*module.RAW, *FIELDS))
    assert not page.get_by_role("checkbox").is_checked()
    assert not (tmp_path / "run/save").exists() and not (tmp_path / "run/planet-class").exists()
    assert len(list((tmp_path / "run/raw/native-copies").glob("*-confirmed.json"))) == 3
    assert len(list((tmp_path / "run/derived/native-copies").glob("*-confirmed.json"))) == 4
    derived = json.loads((tmp_path / "run/derived/report.json").read_text())
    assert derived["steps"] == 62 and derived["planet_transport_verified"]
    assert any(
        event == "neural_activity" and payload["component_identity"] == "derived" for event, payload in events
    )
