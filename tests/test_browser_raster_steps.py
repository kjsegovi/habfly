"""Cooperative reference transport fixtures, never scientific/learned scores."""
# ruff: noqa: F811

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_raster_planet_evidence import raster_page, select, sha, sources  # noqa: F401

import habfly.browser_raster_steps as module
from habfly.browser import BrowserSafetyStop


@pytest.fixture
def rig(tmp_path, monkeypatch):
    path = tmp_path / "presence.json"
    path.write_bytes(b"immutable fixture presence")
    checksum = sha(path)
    evidence = {
        "measurements": {
            "line_shift": {"value": "0.00004788", "unit": "nm"},
            "brightness_drop": {"value": "6.9615", "unit": "%"},
            "period_days": {"value": "195.3434", "unit": "day"},
        }
    }
    presence = {
        "star": "DULAT",
        "evidence": evidence,
        "painted_class_after": None,
        "painted_choices_after": {"selected": None},
    }
    flags = {
        "fresh": [],
        "writes": [],
        "fresh_hook": None,
        "copy_hook": None,
        "current_failed": False,
        "source_reads": 0,
    }

    def get_presence(owner, source, expected):
        flags["source_reads"] += 1
        if sha(source) != expected:
            raise BrowserSafetyStop("raster_planet_source_hash_mismatch")
        return deepcopy(presence), {"stable": True}, {"observation": {"values": {}}}

    def fresh(page, config, owner, directory, source):
        flags["fresh"].append(directory.name)
        directory.mkdir()
        if flags["fresh_hook"]:
            flags["fresh_hook"]()
        return {"fixture": directory.name}

    class Session:
        def __init__(self, page, config, output, emit, *, max_seconds):
            assert max_seconds == 900
            self.output = output
            output.mkdir()
            self.emit, self.attempted, self.verified, self.stopped = emit, set(), {}, False

        def current(self):
            assert not self.stopped
            if flags["current_failed"]:
                raise BrowserSafetyStop("stale_planet_observation")
            return {"stable": True}, {"observation": {"values": {}}}, {"selected": None}, {}

        def copy(self, name, text, unit, *, source):
            assert not self.stopped and source == "reference_diagnostic"
            self.attempted.add(name)
            intent = {
                "kind": "TYPE",
                "destination": name,
                "value": text,
                "unit": unit,
                "action_source": source,
                "task_completed": False,
            }
            module.persist_json(self.output / f"copy-{len(self.attempted):02d}-reserved.json", intent)
            self.emit("action_proposed", intent)
            self.current()
            if flags["copy_hook"]:
                flags["copy_hook"]()
            flags["writes"].append(name)
            receipt = {**intent, "readback_verified": True, "correctness_verified": False}
            self.verified[name] = receipt
            self.emit("action_result", receipt)

        def close(self):
            self.stopped = True

    monkeypatch.setattr(module, "_presence", get_presence)
    monkeypatch.setattr(module, "_reload", lambda owner, e: get_presence(owner, path, checksum))
    monkeypatch.setattr(module, "_fresh", fresh)
    monkeypatch.setattr(module, "_blank_current", lambda *a, **kw: None)
    monkeypatch.setattr(module, "planet_projection", lambda capture, mapping: capture)
    monkeypatch.setattr(module, "PlanetNumericSession", Session)

    def create(name="run", emit=lambda _: None):
        return module.RasterInputSteps(
            None,
            None,
            tmp_path / name,
            run_history=tmp_path,
            presence_path=path,
            presence_sha256=checksum,
            emit=emit,
        )

    return SimpleNamespace(create=create, root=tmp_path, source=path, flags=flags, evidence=evidence)


def journal(root):
    return [json.loads(line) for line in (root / "run/events.jsonl").read_text().splitlines()]


def test_constructor_zero_actions_pause_and_one_copy_per_advance(rig):
    received = []

    def emit(event):
        assert journal(rig.root)[-1] == event
        received.append(event)

    component = rig.create(emit=emit)
    assert not component.finished and not rig.flags["fresh"] and not rig.flags["writes"]
    assert not (rig.root / "run/reserved.json").exists()
    for count in range(1, 4):
        component.advance()
        assert rig.flags["writes"] == list(module.RAW[:count])
        old = (rig.root / "run/events.jsonl").read_bytes()
        for _ in range(3):
            component.state()
        assert (rig.root / "run/events.jsonl").read_bytes() == old
    assert component.finished and component.report["raw_measurement_transport_verified"]
    assert rig.flags["fresh"] == ["fresh", "precopy-1", "precopy-2", "precopy-3"]
    assert received == journal(rig.root)
    assert [e["sequence"] for e in received] == list(range(len(received)))
    assert len({e["run_id"] for e in received}) == 1
    assert received[-1]["event"] == "episode_summary" and received[-1]["payload"] == component.report
    report = json.loads((rig.root / "run/report.json").read_text())
    intent = json.loads((rig.root / "run/reserved.json").read_text())
    assert report == {
        **intent,
        "raw_measurement_transport_verified": True,
        "numeric_writes": 3,
        "verified_fields": report["verified_fields"],
        "precopy": report["precopy"],
        "native_events": report["native_events"],
        "saved": False,
        "assessed": False,
        "submitted": False,
    }
    assert len(report["native_events"]) == 6
    for index, event in enumerate(report["native_events"]):
        assert json.loads((rig.root / f"run/native-event-{index:02d}.json").read_text()) == event
    for index, recorded in enumerate(report["precopy"], 1):
        assert json.loads((rig.root / f"run/precopy-{index}/evidence.json").read_text()) == recorded
    runtime = json.loads((rig.root / "run/runtime-manifest.json").read_text())
    assert runtime["events_sha256"] == sha(rig.root / "run/events.jsonl")
    snapshot = (rig.root / "run/events.jsonl").read_bytes()
    assert component.close() is component.report
    component.advance()
    assert (rig.root / "run/events.jsonl").read_bytes() == snapshot
    assert not component.report["task_completed"] and component.session.stopped


@pytest.mark.parametrize("at", [0, 1, 2])
def test_abort_keeps_existing_writes_and_never_resumes(rig, at):
    component = rig.create()
    for _ in range(at):
        component.advance()
    report = component.abort()
    component.advance()
    assert component.finished and component.session.stopped and len(rig.flags["writes"]) == at
    assert report["numeric_writes"] == at and not report["raw_measurement_transport_verified"]
    assert not (rig.root / "run/report.json").exists()
    assert json.loads((rig.root / "run/stopped.json").read_text())["reservation_created"] == bool(at)
    if at:
        other = rig.create("retry")
        assert other.finished and other.report["outcome"] == "raster_planet_inputs_already_reserved"
        assert len(rig.flags["writes"]) == at


@pytest.mark.parametrize("event", ["reserved", "before_fresh", "after_fresh", "native"])
@pytest.mark.parametrize("effect", ["abort", "failure", "reentry", "source"])
def test_cooperative_boundaries_stop_before_any_native_fill(rig, event, effect):
    holder = {}

    def emit(message):
        stage = message["payload"].get("stage")
        matches = (
            event == "reserved"
            and stage == "inputs_reserved"
            or event == "before_fresh"
            and stage == "refreshing_reference_evidence"
            or event == "after_fresh"
            and stage == "reference_evidence_checked"
            or event == "native"
            and message["event"] == "action_proposed"
        )
        if matches:
            component = holder["component"]
            if effect == "abort":
                component.abort()
            elif effect == "reentry":
                component.advance()
            elif effect == "source":
                rig.source.write_bytes(b"changed")
            else:
                raise RuntimeError("private callback secret")

    component = rig.create(emit=emit)
    holder["component"] = component
    component.advance()
    assert component.finished and component.session.stopped and not rig.flags["writes"]
    assert not component.report["raw_measurement_transport_verified"]
    assert "private callback secret" not in (rig.root / "run/events.jsonl").read_text()
    component.advance()
    assert not rig.flags["writes"]


def test_cancellation_during_synchronous_freshness_waits_for_boundary_without_copy(rig):
    component = rig.create()
    rig.flags["fresh_hook"] = component.abort
    component.advance()
    assert component.finished and rig.flags["fresh"] == ["fresh"] and not rig.flags["writes"]
    assert component.report["outcome"] == "operator_aborted"


@pytest.mark.parametrize("mutation", ["source", "state", "claim"])
def test_changes_while_paused_never_allow_next_copy(rig, mutation):
    component = rig.create()
    component.advance()
    if mutation == "source":
        rig.source.write_bytes(b"changed")
    elif mutation == "state":
        rig.flags["current_failed"] = True
    else:
        (rig.root / "run/reserved.json").write_text("{}")
    component.advance()
    assert component.finished and len(rig.flags["writes"]) == 1
    assert not component.report["raw_measurement_transport_verified"]


@pytest.mark.parametrize("event", ["hello", "state", "episode_summary"])
def test_callback_failure_is_terminal_and_redacted(rig, event):
    def emit(message):
        if message["event"] == event:
            raise RuntimeError("private callback error")

    component = rig.create(emit=emit)
    while not component.finished:
        component.advance()
    assert not component.report["raw_measurement_transport_verified"]
    assert "private callback error" not in (rig.root / "run/events.jsonl").read_text()
    assert component._stream.closed and not (rig.root / "run/report.json").exists()


def test_runtime_manifest_failure_invalidates_previously_written_success(rig, monkeypatch):
    component = rig.create()
    persist = module.persist_json

    def write(path, value):
        if path.name == "runtime-manifest.json":
            raise OSError("private filesystem")
        return persist(path, value)

    monkeypatch.setattr(module, "persist_json", write)
    while not component.finished:
        component.advance()
    assert not component.report["raw_measurement_transport_verified"] and component.session.stopped
    assert (rig.root / "run/stopped.json").exists()
    assert "private filesystem" not in (rig.root / "run/events.jsonl").read_text()


def test_partial_canonical_reservation_failure_is_reported_as_reserved(rig, monkeypatch):
    component = rig.create()
    original = module._reserve

    def reserve(*args):
        original(*args)
        raise OSError("fixture after claim")

    monkeypatch.setattr(module, "_reserve", reserve)
    component.advance()
    assert component.finished and component.report["reservation_created"]
    assert json.loads((rig.root / "run/stopped.json").read_text())["reservation_created"]
    assert not rig.flags["writes"]


@pytest.mark.parametrize("value", [True, 0, 29, 901, float("inf"), float("nan")])
def test_bad_budget_rejects_before_browser_and_output(tmp_path, value):
    with pytest.raises(BrowserSafetyStop, match="invalid_time_budget"):
        module.RasterInputSteps(
            None,
            None,
            tmp_path / "run",
            run_history=tmp_path,
            presence_path=tmp_path / "missing",
            presence_sha256="0" * 64,
            max_seconds=value,
        )
    assert not (tmp_path / "run").exists()


def test_real_intercepted_titlecase_reference_copies_one_per_advance(raster_page, tmp_path):
    page, frame = raster_page
    frame.locator("div").first.evaluate("e=>{e.textContent='Dulat';e.style.textTransform='uppercase'}")
    source = sources(raster_page, tmp_path)
    select(raster_page, tmp_path, source)
    receipt = tmp_path / "presence/confirmed.json"
    component = module.RasterInputSteps(
        page,
        config(),
        tmp_path / "copy",
        run_history=tmp_path,
        presence_path=receipt,
        presence_sha256=sha(receipt),
    )
    assert not (tmp_path / "copy/fresh").exists() and not component.session.attempted
    for index in range(3):
        component.advance()
        assert len(component.session.verified) == index + 1, component.report
        for name in module.RAW[index + 1 :]:
            assert frame.locator("#" + name).input_value() == ""
    assert component.report["raw_measurement_transport_verified"] and component.finished
    assert not component.report["task_completed"] and component.report["numeric_writes"] == 3
    assert not page.get_by_role("checkbox").is_checked()
    assert len(list((tmp_path / "copy/native-copies").glob("copy-*-confirmed.json"))) == 3


def test_positive_workflow_loader_accepts_cooperative_raw_report_unchanged(
    raster_page, tmp_path, monkeypatch
):
    import test_browser_positive_planet_workflow as fixture

    def cooperative_copy(surface, history):
        receipt = history / "presence/confirmed.json"
        component = module.RasterInputSteps(
            surface[0],
            config(),
            history / "copy",
            run_history=history,
            presence_path=receipt,
            presence_sha256=sha(receipt),
        )
        while not component.finished:
            component.advance()
        assert component.report["raw_measurement_transport_verified"], component.report
        return component.report

    monkeypatch.setattr(fixture, "copy", cooperative_copy)
    fixture.positive.__wrapped__(raster_page, tmp_path, SimpleNamespace(param="mixed_case"))
    bundle = fixture.load(tmp_path)
    assert bundle["star"] == "Dulat" and bundle["planet_class"] == "gas_giant"


@pytest.mark.parametrize("effect", ["abort", "failure", "source"])
def test_real_native_reservation_callback_stops_before_fill(raster_page, tmp_path, effect):
    page, frame = raster_page
    source = sources(raster_page, tmp_path)
    select(raster_page, tmp_path, source)
    receipt = tmp_path / "presence/confirmed.json"
    holder = {}

    def emit(message):
        if message["event"] == "action_proposed":
            if effect == "abort":
                holder["component"].abort()
            elif effect == "source":
                receipt.write_text("{}")
            else:
                raise RuntimeError("private browser callback")

    component = module.RasterInputSteps(
        page,
        config(),
        tmp_path / "copy",
        run_history=tmp_path,
        presence_path=receipt,
        presence_sha256=sha(receipt),
        emit=emit,
    )
    holder["component"] = component
    component.advance()
    assert component.finished and len(component.session.attempted) == 1 and not component.session.verified
    assert all(frame.locator("#" + name).input_value() == "" for name in module.RAW)
    assert (tmp_path / "copy/native-copies/copy-01-reserved.json").exists()
    assert not list((tmp_path / "copy/native-copies").glob("copy-*-confirmed.json"))
    assert not component.report["raw_measurement_transport_verified"]


def test_tooltip_cooperative_copies_keep_exact_values_and_mode(rig):
    from habfly.browser_raster_planet_evidence import TOOLTIP_MODE

    rig.evidence["mode"] = TOOLTIP_MODE
    rig.evidence["measurements"]["brightness_drop"].update(value="0.762", physical_bounds=None)
    rig.evidence["measurements"]["period_days"].update(
        value="1000", compatibility_interval={"lower": "998", "upper": "1002", "endpoints": "open"}
    )
    component = rig.create()
    assert not rig.flags["writes"] and component.state()["provenance"] == TOOLTIP_MODE
    for _ in range(3):
        component.advance()
    report = component.report
    assert report["raw_measurement_transport_verified"] and report["mode"] == TOOLTIP_MODE
    assert report["action_source"] == "explicit_approximate_reference_visible_tooltips_v1_not_learned"
    assert report["verified_fields"]["brightness_drop"]["value"] == "0.762"
    assert report["verified_fields"]["period_days"]["value"] == "1000"
    assert all(event["provenance"] == TOOLTIP_MODE for event in report["native_events"])
    assert report["minimum_depth_verified"] is False and report["physical_period_verified"] is False
    assert not ({"lower", "upper"} & report["evidence"]["measurements"]["brightness_drop"].keys())
    rig.evidence.pop("mode")
    retry = rig.create("raster-mode-retry")
    assert retry.finished and retry.report["outcome"] == "raster_planet_inputs_already_reserved"
    assert rig.flags["writes"] == list(module.RAW)
