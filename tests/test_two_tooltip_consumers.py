"""Offline two-feature provenance: real source validators, no browser/model.

Native sensor records below are authored synthetic fixtures. Journal tests use
the pre-existing explicit upstream source seam; they do not prove native Save
or a scientific period. The transport tests rebuild genuine pinned bundles.
"""
# ruff: noqa: F811

from copy import deepcopy

import pytest
from test_browser_raster_planet_evidence import read, sha, tooltip_sources, write
from test_browser_raster_steps import rig as raw_rig  # noqa: F401
from test_planet_two_tooltip_reference import two_bundle
from test_project_tooltip_evidence import ingest, positive_offline, terrestrial_offline, tooltip  # noqa: F401

import habfly.browser_positive_planet_workflow as workflow
import habfly.browser_raster_planet_evidence as transport
from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_workflow import _Evidence
from habfly.browser_positive_steps import PositivePlanetSteps
from habfly.planet_tooltip_reference import MODE, TWO_MODE
from habfly.project_progress import ProgressError


def two_sources(history):
    """Synthetic public captures; the two-bundle and transport loaders are real."""
    source = tooltip_sources(history)
    diagnostics = two_bundle(history)
    source_dir = history / "two-0-overview"
    for name in ("report.json", "chart.png"):
        (source["window_report"].parent / name).write_bytes((source_dir / name).read_bytes())
    source["window_report_sha256"] = sha(source["window_report"])
    source["tooltip_reference"] = {
        "diagnostics": diagnostics,
        "expected_star": "Example",
        "mode": TWO_MODE,
    }
    return source


@pytest.fixture
def evidence(tmp_path):
    source = two_sources(tmp_path)
    owner = transport._Owned(tmp_path)
    return source, owner, transport._evidence(owner, **source)


def fresh_files(source, owner, evidence, directory):
    (directory / "progress").mkdir(parents=True)
    for name in ("report.json", "chart.png"):
        (directory / "progress" / name).write_bytes((source["window_report"].parent / name).read_bytes())
    (directory / "spectrum.json").write_bytes(source["spectrum_path"].read_bytes())
    path = directory / "progress/report.json"
    return {
        "window": transport._overview_window(owner, path, sha(path)),
        "spectrum_sha256": sha(directory / "spectrum.json"),
        **({"measurement_mode": TWO_MODE} if evidence.get("mode") == TWO_MODE else {}),
    }


def test_real_two_loader_pins_both_bundles_and_preserves_nonphysical_semantics(tmp_path, monkeypatch):
    source = two_sources(tmp_path)
    book = _Evidence(tmp_path)
    owner = workflow._RasterEvidence(book)
    monkeypatch.setattr(
        transport, "load_planet_window_measurements", lambda *a, **k: pytest.fail("raster fallback")
    )
    value = transport._evidence(owner, **source)
    assert value["mode"] == TWO_MODE == workflow._measurement_mode(value)
    assert set(value["tooltip_reference"]) == {"diagnostics", "expected_star", "mode"}
    assert len(value["tooltip_reference"]["diagnostics"]) == 2
    assert value["measurements"]["period_days"] == {
        "value": "2000",
        "unit": "day",
        "source": TWO_MODE,
        "compatibility_interval": {"lower": "1998", "upper": "2002", "endpoints": "open"},
        "exact_representative": {"numerator": 2000, "denominator": 1},
        "estimate_kind": "single_spacing",
    }
    assert value["measurements"]["brightness_drop"] == {
        "value": "0.762",
        "unit": "%",
        "source": TWO_MODE,
        "physical_bounds": None,
    }
    assert value["tooltip_measurements"]["answer_authorized"] is False
    for key, expected in transport.TWO_TOOLTIP_FLAGS.items():
        assert value["uncertainty"][key] == transport._flags_for(value)[key] == expected
        assert type(value["uncertainty"][key]) is type(expected)
    assert set(value["tooltip_measurements"]["source_sha256"]) <= book.hashes.keys()
    assert len(value["tooltip_measurements"]["features"]) == 2
    assert transport._action_source(value) == "explicit_" + TWO_MODE + "_not_learned"
    transport._reload(owner, value)
    book.unchanged()


@pytest.mark.parametrize("change", ["implicit_two", "explicit_legacy", "unknown", "extra", "three"])
def test_no_count_inference_or_cross_mode_descriptor_fallback(tmp_path, change):
    source = two_sources(tmp_path)
    descriptor = source["tooltip_reference"]
    if change == "implicit_two":
        descriptor.pop("mode")
    elif change == "explicit_legacy":
        descriptor["mode"] = MODE
    elif change == "unknown":
        descriptor["mode"] = "future_unverified_mode"
    elif change == "extra":
        descriptor["fallback"] = True
    else:
        descriptor["diagnostics"].append(deepcopy(descriptor["diagnostics"][0]))
    with pytest.raises(BrowserSafetyStop):
        transport._evidence(transport._Owned(tmp_path), **source)


def test_prevalidation_failure_does_not_claim_two_confirmed_features_or_touch_browser(tmp_path, monkeypatch):
    source = two_sources(tmp_path)
    source["tooltip_reference"]["unsupported_extra"] = True
    monkeypatch.setattr(transport, "PlanetNumericSession", lambda *a, **k: pytest.fail("browser touched"))
    output = tmp_path / "failed-presence"
    with pytest.raises(BrowserSafetyStop, match="invalid_tooltip_descriptor"):
        transport.select_raster_detected_planet(None, None, output, run_history=tmp_path, **source)
    failed = read(output / "stopped.json")
    assert failed["provenance"] == TWO_MODE and not failed["measurement_evidence_validated"]
    assert not (transport.TWO_TOOLTIP_FLAGS.keys() & failed.keys())
    assert not failed["write_may_have_occurred"] and not failed["reservation_created"]
    assert not list(tmp_path.glob("planet-raster-*-reservations"))


@pytest.mark.parametrize("change", ["mode", "descriptor", "bundle", "star", "count_alias", "assumption"])
def test_rebuild_rejects_modified_original_evidence(evidence, change):
    _source, owner, value = evidence
    if change == "mode":
        value["mode"] = MODE
    elif change == "descriptor":
        value["tooltip_reference"].pop("mode")
    elif change == "bundle":
        (owner.history / "two-1/readable.json").write_text("{}")
    elif change == "star":
        value["tooltip_reference"]["expected_star"] = "Other"
    elif change == "count_alias":
        value["tooltip_measurements"]["consistency_redundancy"] = False
    else:
        value["uncertainty"]["consecutive_events_assumed"] = 1
    with pytest.raises(BrowserSafetyStop):
        transport._reload(owner, value)


@pytest.mark.parametrize("stage", ["fresh", "preselect", "precopy-1", "precopy-2", "precopy-3"])
def test_each_fresh_boundary_binds_same_explicit_mode_and_original_capture(evidence, stage):
    source, owner, value = evidence
    directory = owner.history / stage
    recorded = fresh_files(source, owner, value, directory)
    transport._fresh_evidence(owner, directory, recorded, value)
    for replacement in (None, MODE, "unknown"):
        changed = deepcopy(recorded)
        if replacement is None:
            changed.pop("measurement_mode")
        else:
            changed["measurement_mode"] = replacement
        with pytest.raises(BrowserSafetyStop, match="fresh_measurement_mode_changed"):
            transport._fresh_evidence(owner, directory, changed, value)
    changed = deepcopy(recorded)
    changed["window"]["chart_sha256"] = "0" * 64
    with pytest.raises(BrowserSafetyStop, match="fresh_window_evidence_changed"):
        transport._fresh_evidence(owner, directory, changed, value)


@pytest.mark.parametrize("two", [False, True])
def test_fresh_writer_selects_explicit_mode_without_changing_legacy_shape(tmp_path, monkeypatch, two):
    source = (two_sources if two else tooltip_sources)(tmp_path)
    owner = transport._Owned(tmp_path)
    evidence = transport._evidence(owner, **source)
    source_events = read(source["spectrum_path"])["events"]
    moves = []

    class Chart:
        def __init__(self, page, config, emit, **limits):
            assert limits == {"max_actions": 2, "max_seconds": 120}
            self.star, self.emit = "EXAMPLE", emit
            emit(source_events[0]["kind"], source_events[0]["payload"])

        def close(self):
            moves.append("close")

    def hover(chart, side):
        index = {"blue": 1, "red": 3}[side]
        moves.append(side)
        for event in source_events[index : index + 2]:
            chart.emit(event["kind"], event["payload"])
        return source_events[index + 1]["payload"]["spectrum_sample"]

    def capture(page, config, output, **limits):
        assert limits == {"requested_days": 5000, "max_seconds": 120}
        output.mkdir()
        for name in ("report.json", "chart.png"):
            (output / name).write_bytes((source["window_report"].parent / name).read_bytes())

    # Only native capture/hover operations are injected; actual spectrum,
    # overview, source and fresh-evidence validators remain enabled.
    monkeypatch.setattr(transport, "FluxChartSession", Chart)
    monkeypatch.setattr(transport, "hover_spectrum_marker", hover)
    monkeypatch.setattr(transport, "capture_observation_progress", capture)
    directory = tmp_path / "fresh"
    result = transport._fresh(None, None, owner, directory, evidence)
    assert moves == ["blue", "red", "close"]
    assert set(result) == {"window", "spectrum_sha256"} | ({"measurement_mode"} if two else set())
    transport._fresh_evidence(owner, directory, result, evidence)


@pytest.mark.parametrize("change", [None, "precopy_mode", "second_bundle"])
def test_workflow_rebuilds_real_two_sources_and_all_fresh_records_before_native_chain(
    evidence, monkeypatch, change
):
    source, owner, value = evidence
    raw_dir = owner.history / "raw"
    raw_dir.mkdir()
    for name in ("derived", "planet-class"):
        (owner.history / name).mkdir()
    fresh = fresh_files(source, owner, value, raw_dir / "fresh")
    precopy = [fresh_files(source, owner, value, raw_dir / f"precopy-{i}") for i in range(1, 4)]
    if change == "precopy_mode":
        precopy[1]["measurement_mode"] = MODE
    intent = {
        **transport._flags_for(value),
        "schema_version": 1,
        "mode": TWO_MODE,
        "stage": "inputs",
        "star": "EXAMPLE",
        "evidence": value,
        "output": "raw",
        "maximum_numeric_writes": 3,
        "destinations": list(transport.RAW),
        "action_source": transport._action_source(value),
        "presence_path": "presence.json",
        "presence_sha256": "a" * 64,
        "fresh": fresh,
    }
    report = {
        **intent,
        "raw_measurement_transport_verified": True,
        "numeric_writes": 3,
        "verified_fields": {},
        "precopy": precopy,
        "native_events": [],
        "saved": False,
        "assessed": False,
        "submitted": False,
    }
    transport._reserve(owner, raw_dir, "EXAMPLE", "inputs", intent)
    write(raw_dir / "report.json", report)
    for index, recorded in enumerate(precopy, 1):
        write(raw_dir / f"precopy-{index}/evidence.json", recorded)
    if change == "second_bundle":
        (owner.history / "two-1/readable.json").write_text("{}")
    monkeypatch.setattr(workflow, "_presence", lambda *a: ({"star": "EXAMPLE", "evidence": value}, {}, {}))

    class NativeChainReached(Exception):
        pass

    def native(*args, **kwargs):
        assert change is None
        raise NativeChainReached

    monkeypatch.setattr(workflow, "_copy_chain", native)
    book = _Evidence(owner.history)
    with pytest.raises(BrowserSafetyStop if change else NativeChainReached):
        workflow._planet_sources(book, raw_dir, owner.history / "derived", owner.history / "planet-class")
    if change is None:
        assert set(value["tooltip_measurements"]["source_sha256"]) <= book.hashes.keys()
        assert all(f"raw/precopy-{index}/spectrum.json" in book.hashes for index in range(1, 4))
        book.unchanged()


def test_legacy_descriptor_flags_and_fresh_record_shape_unchanged(tmp_path):
    source = tooltip_sources(tmp_path)
    owner = transport._Owned(tmp_path)
    value = transport._evidence(owner, **source)
    assert set(value["tooltip_reference"]) == {"diagnostics", "expected_star"}
    assert transport._flags_for(value) == {
        **transport.FLAGS,
        "provenance": MODE,
        "physical_period_verified": False,
        "minimum_depth_verified": False,
    }
    assert not (transport.TWO_TOOLTIP_FLAGS.keys() & value["uncertainty"].keys())
    directory = tmp_path / "fresh"
    record = fresh_files(source, owner, value, directory)
    assert set(record) == {"window", "spectrum_sha256"}
    transport._fresh_evidence(owner, directory, record, value)
    record["measurement_mode"] = TWO_MODE
    with pytest.raises(BrowserSafetyStop, match="fresh_measurement_mode_changed"):
        transport._fresh_evidence(owner, directory, record, value)
    assert transport._flags_for({}) == transport.FLAGS


def test_display_records_zero_redundancy_and_no_physical_depth_bounds(evidence):
    _, _, value = evidence
    component = PositivePlanetSteps.__new__(PositivePlanetSteps)
    component.star, component.evidence = "EXAMPLE", value
    display = component._reference()
    for key, expected in transport.TWO_TOOLTIP_FLAGS.items():
        assert display[key] == expected and type(display[key]) is type(expected)
    assert display["period_days"]["estimate_kind"] == "single_spacing"
    assert "assumed-consecutive" in display["period_days"]["interpretation"]
    assert "recurrence_confirmed" in display and not display["recurrence_confirmed"]
    assert display["brightness_drop_percent"]["physical_bounds"] is None
    assert not ({"lower", "upper"} & display["brightness_drop_percent"].keys())
    assert not display["scientific_verified"] and not display["learned_perception"]


def test_two_mode_shares_canonical_reservations_with_old_modes(tmp_path):
    owner = transport._Owned(tmp_path)
    for stage in ("presence", "inputs"):
        directory = tmp_path / stage
        directory.mkdir()
        transport._reserve(owner, directory, "EXAMPLE", stage, {"mode": TWO_MODE})
        for index, mode in enumerate((MODE, transport.MODE, TWO_MODE)):
            retry = tmp_path / f"retry-{stage}-{index}"
            retry.mkdir()
            with pytest.raises(BrowserSafetyStop, match=stage + "_already_reserved"):
                transport._reserve(owner, retry, "Example", stage, {"mode": mode})


def test_cooperative_raw_copies_preserve_new_mode_flags_with_declared_native_seam(raw_rig):
    # Existing fake native session only: tests the cooperative journal contract,
    # not actual DOM actions. Real source validation is tested above separately.
    raw_rig.evidence["mode"] = TWO_MODE
    component = raw_rig.create()
    for _ in range(3):
        component.advance()
    assert component.finished and component.report["numeric_writes"] == 3
    assert component.report["mode"] == TWO_MODE
    for key, expected in transport.TWO_TOOLTIP_FLAGS.items():
        assert component.report[key] == expected
    assert all(event["provenance"] == TWO_MODE for event in component.report["native_events"])
    assert not component.report["task_completed"] and not component.report["scientific_verified"]


@pytest.fixture
def two_import(tooltip):
    """Declared existing upstream seam; do not call this a genuine native chain."""
    _, rig, _ = tooltip
    _, history, _, directory, registry, _ = rig
    hashes, bundle = next(iter(registry.values()))
    receipt = read(directory / "confirmed.json")
    value = deepcopy(bundle["raw"]["evidence"])
    value["mode"] = TWO_MODE
    value["tooltip_measurements"].update(
        mode=TWO_MODE, **transport.TWO_TOOLTIP_FLAGS, period_days={"estimate_kind": "single_spacing"}
    )
    value["uncertainty"].update(transport.TWO_TOOLTIP_FLAGS)
    value["uncertainty"]["period_days"]["estimate_kind"] = "single_spacing"
    value["uncertainty"]["period_days"]["interpretation"] = "One assumed-consecutive spacing; not recurrence."
    for field in ("brightness_drop", "period_days"):
        value["measurements"][field]["source"] = TWO_MODE
    # Restrict even this synthetic source seam to two declared feature leaves.
    third = next(p for p in value["tooltip_measurements"]["source_sha256"] if p.endswith("source-2.json"))
    value["tooltip_measurements"]["source_sha256"].pop(third)
    hashes.pop(third)
    raw_path = next(history / p for p in hashes if p.endswith("/raw/report.json"))
    raw = read(raw_path)
    raw.update(mode=TWO_MODE, evidence=value)
    write(raw_path, raw)
    hashes[str(raw_path.relative_to(history))] = sha(raw_path)
    bundle["raw"]["evidence"] = deepcopy(value)
    receipt["raw_measurement_evidence"] = value
    receipt["planet"]["measurement_provenance"] = TWO_MODE
    receipt["source_sha256"] = deepcopy(hashes)
    write(directory / "confirmed.json", receipt)
    return tooltip


def test_both_importers_preserve_single_spacing_evidence_and_no_scientific_claim(two_import):
    _, rig, _ = two_import
    journal, *_ = rig
    result = ingest(two_import)
    raw = result["evidence_scopes"]["raw_measurements"]
    assert raw["provenance"] == TWO_MODE
    for key, expected in transport.TWO_TOOLTIP_FLAGS.items():
        assert raw[key] == expected
    assert not raw["scientific_verified"] and not raw["training_label"]
    assert not result["progress"]["project_completed"] and not result["progress"]["submitted"]
    before = journal.path.read_bytes()
    assert ingest(two_import)["idempotent"] and journal.path.read_bytes() == before


@pytest.mark.parametrize("change", ["mode", "alias", "reference"])
def test_importers_reject_relabel_or_typed_metadata_drift_atomically(two_import, change):
    _, rig, _ = two_import
    journal, _, _, directory, _, _ = rig
    receipt = read(directory / "confirmed.json")
    if change == "mode":
        receipt["planet"]["measurement_provenance"] = MODE
    elif change == "alias":
        receipt["raw_measurement_evidence"]["tooltip_measurements"]["consistency_redundancy"] = False
    else:
        receipt["raw_measurement_evidence"]["tooltip_measurements"]["consecutive_events_assumed"] = False
    write(directory / "confirmed.json", receipt)
    before = journal.path.read_bytes()
    with pytest.raises((BrowserSafetyStop, ProgressError)):
        ingest(two_import)
    assert journal.path.read_bytes() == before
