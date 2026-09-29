"""Offline downstream mode/provenance tests behind an explicit source seam.

These reuse the existing synthetic upstream-validator seam; they do not certify
tooltip/native execution. Real tooltip and fresh-capture validation belongs to
the shared transport's separate tests. No browser or model is started here.
"""

from copy import deepcopy

import pytest
from test_browser_numeric import config
from test_project_evidence import read, sha, write
from test_project_positive_evidence import offline as positive_offline  # noqa: F401
from test_project_terrestrial_evidence import offline as terrestrial_offline  # noqa: F401

import habfly.browser_positive_planet_workflow as positive_browser
import habfly.browser_raster_planet_evidence as transport
import habfly.browser_terrestrial_workflow as terrestrial_browser
import habfly.project_positive_evidence as positive
import habfly.project_terrestrial_evidence as terrestrial
from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_workflow import _Evidence
from habfly.planet_tooltip_reference import MODE
from habfly.project_progress import ProgressError


@pytest.fixture(params=["positive", "terrestrial"])
def tooltip(request):
    branch = request.param
    rig = request.getfixturevalue(branch + "_offline")
    _journal, history, _inventory, workflow, registry, _ = rig
    original = read(workflow / "confirmed.json")
    key = next(iter(registry))
    hashes, bundle = registry[key]
    raw_path = next(history / p for p in hashes if p.endswith("/raw/report.json"))
    evidence = deepcopy(bundle["raw"]["evidence"])
    evidence["mode"] = MODE
    period = {
        "value": "1",
        "unit": "days",
        "exact_representative": {"numerator": 1, "denominator": 1},
        "compatibility_interval": {"lower": "0.9", "upper": "1.1", "endpoints": "open"},
        "interpretation": "Observed bracket recurrence, NOT physical uncertainty.",
    }
    decline = {
        "value": "1",
        "unit": "percent",
        "physical_bounds": None,
        "interpretation": "Maximum sampled decline, NOT a physical minimum.",
    }
    evidence["measurements"]["period_days"].update(
        source=MODE,
        compatibility_interval=period["compatibility_interval"],
        exact_representative=period["exact_representative"],
    )
    evidence["measurements"]["brightness_drop"].update(source=MODE, physical_bounds=None)
    evidence["uncertainty"] = {
        "kind": "sampled_tooltip_bracket_compatibility_not_physical_bounds",
        "period_days": period,
        "brightness_drop_percent": decline,
        "limitations": ["Missed/aliased events and unsampled minima remain unresolved."],
        "spectrum": "Visible spectrum marker spelling; physical uncertainty unverified.",
    }
    # Declared synthetic source-only files exercise hash retention/atomicity,
    # not the real offline tooltip loader or a completed browser hover chain.
    source_hashes = {}
    for index in range(3):
        path = raw_path.parent / f"tooltip-source-{index}.json"
        write(path, {"fixture_only": True, "bracket": index, "scientific_verified": False})
        source_hashes[str(path.relative_to(history))] = sha(path)
    evidence["tooltip_measurements"] = {
        "mode": MODE,
        "source_sha256": source_hashes,
        "learned_perception": False,
        "scientific_verified": False,
        "training_label": False,
    }
    raw = read(raw_path)
    raw.update(mode=MODE, evidence=evidence)
    write(raw_path, raw)
    hashes.update(source_hashes)
    hashes[str(raw_path.relative_to(history))] = sha(raw_path)
    bundle["raw"]["evidence"] = evidence
    original["raw_measurement_evidence"] = evidence
    original["planet"]["measurement_provenance"] = MODE
    original["source_sha256"] = deepcopy(hashes)
    write(workflow / "confirmed.json", original)
    return branch, rig, source_hashes


def ingest(subject):
    branch, rig, _ = subject
    module = positive if branch == "positive" else terrestrial
    method = (
        module.import_verified_positive_planet if branch == "positive" else module.import_verified_terrestrial
    )
    return method(*rig[:4])


def test_tooltip_import_preserves_distinct_nonphysical_semantics_and_all_source_hashes(tooltip):
    _, rig, hashes = tooltip
    journal, history, _, workflow, _, _ = rig
    result = ingest(tooltip)
    scopes = result["evidence_scopes"]
    raw = scopes["raw_measurements"]
    assert raw["provenance"] == scopes["derived_calculations"]["input_provenance"] == MODE
    assert raw["uncertainty"] == read(workflow / "confirmed.json")["raw_measurement_evidence"]["uncertainty"]
    assert raw["uncertainty"]["period_days"]["compatibility_interval"]["endpoints"] == "open"
    assert raw["uncertainty"]["brightness_drop_percent"]["physical_bounds"] is None
    for name in ("period_days", "brightness_drop"):
        assert not {"lower", "upper"} & raw["measurements"][name].keys()
    assert raw["learned_perception"] is False
    assert all(s["scientific_verified"] is False and s["training_label"] is False for s in scopes.values())
    state = journal.load().reduce()
    star = state.stars[result["star_id"]]
    assert star.planet.policy_label == star.habitability.policy_label == MODE
    assert not state.reservations and not state.receipts and not result["progress"]["project_completed"]
    binding = read(next(history.glob("project-evidence-*.json")))
    assert all(binding["validated_artifact_sha256"][p] == digest for p, digest in hashes.items())
    before = journal.path.read_bytes()
    assert ingest(tooltip)["idempotent"] and journal.path.read_bytes() == before


@pytest.mark.parametrize(
    "change", ["workflow_mode", "uncertainty", "source", "unknown_raw", "unknown_evidence"]
)
def test_tooltip_mismatch_never_partially_appends(tooltip, change):
    _, rig, source_hashes = tooltip
    journal, history, _, workflow, registry, _ = rig
    receipt = read(workflow / "confirmed.json")
    if change == "workflow_mode":
        receipt["planet"]["measurement_provenance"] = positive.MEASUREMENTS
    elif change == "uncertainty":
        receipt["raw_measurement_evidence"]["uncertainty"]["brightness_drop_percent"]["physical_bounds"] = [
            0,
            2,
        ]
    elif change == "source":
        write(history / next(iter(source_hashes)), {"mutated": True})
    else:
        hashes, bundle = next(iter(registry.values()))
        raw_path = next(history / p for p in hashes if p.endswith("/raw/report.json"))
        raw = read(raw_path)
        if change == "unknown_raw":
            raw["mode"] = "unrecognized_tooltip_mode"
        else:
            raw["evidence"]["mode"] = "unrecognized_tooltip_mode"
            bundle["raw"]["evidence"] = deepcopy(raw["evidence"])
        write(raw_path, raw)
        hashes[str(raw_path.relative_to(history))] = sha(raw_path)
        receipt["source_sha256"] = deepcopy(hashes)
    write(workflow / "confirmed.json", receipt)
    before = journal.path.read_bytes()
    with pytest.raises((BrowserSafetyStop, ProgressError)):
        ingest(tooltip)
    assert journal.path.read_bytes() == before
    assert not list(history.glob("project-evidence-*.json"))


def test_tooltip_source_edit_after_import_cannot_relabel_existing_journal(tooltip):
    _, rig, hashes = tooltip
    journal, history, *_ = rig
    ingest(tooltip)
    before = journal.path.read_bytes()
    write(history / next(iter(hashes)), {"changed_after_import": True})
    with pytest.raises((BrowserSafetyStop, ProgressError)):
        ingest(tooltip)
    assert journal.path.read_bytes() == before


def test_visible_workflow_producers_dispatch_provenance_from_validated_bundle(tooltip, monkeypatch):
    branch, rig, _ = tooltip
    _, history, _, workflow, _, _ = rig
    importer = positive if branch == "positive" else terrestrial
    producer = positive_browser if branch == "positive" else terrestrial_browser
    directories = importer._source_directories(
        _Evidence(history), read(workflow / "confirmed.json")["source_sha256"]
    )
    monkeypatch.setattr(producer, "_load_sources", importer._load_sources)
    # Pure navigation/readback seams; source and output metadata use the real
    # producer. These assertions do not claim a native browser pass.
    monkeypatch.setattr(
        producer,
        "navigate_project",
        lambda *a, **k: {
            "same_star_verified": True,
            "destination_verified": True,
            "suggested_required_text": [],
        },
    )
    sections = ["stellar", "planet"] + (["habitability"] if branch == "terrestrial" else [])
    for section in sections:
        report = read(workflow / f"{section}-readback/verified/observation.json")
        monkeypatch.setattr(producer, "_read_" + section, lambda *a, _report=report, **k: deepcopy(_report))
    method = (
        producer.verify_positive_planet_workflow
        if branch == "positive"
        else producer.verify_terrestrial_workflow
    )
    receipt = method(None, config(), history / "producer-fixture", run_history=history, **directories)
    assert receipt["planet"]["measurement_provenance"] == MODE
    assert (
        receipt["raw_measurement_evidence"] == read(workflow / "confirmed.json")["raw_measurement_evidence"]
    )
    assert receipt["scientific_verified"] is False and receipt["training_label"] is False


@pytest.mark.parametrize("mode", [transport.MODE, MODE])
@pytest.mark.parametrize("failed_boundary", [None, "precopy-2"])
def test_downstream_uses_shared_strict_rebuild_at_every_raw_boundary(
    tmp_path, monkeypatch, mode, failed_boundary
):
    """Delegation contract only; deliberately injected source/native boundaries."""
    evidence = {"star": "DULAT", "measurements": {}, **({"mode": mode} if mode == MODE else {})}
    flags = transport._flags_for(evidence)
    raw_dir = tmp_path / "raw"
    for name in ("raw", "derived", "planet-class"):
        (tmp_path / name).mkdir()
    fresh = {"fixture_boundary": "fresh"}
    precopy = [{"fixture_boundary": f"precopy-{i}"} for i in range(1, 4)]
    intent = {
        **flags,
        "schema_version": 1,
        "mode": mode,
        "stage": "inputs",
        "star": "Dulat",
        "evidence": evidence,
        "output": "raw",
        "maximum_numeric_writes": 3,
        "destinations": list(transport.RAW),
        "action_source": transport._action_source(evidence),
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
    write(raw_dir / "reserved.json", intent)
    write(raw_dir / "report.json", report)
    canonical = transport._reservation(transport._Owned(tmp_path), "DULAT", "inputs")
    write(canonical, intent)
    for index, value in enumerate(precopy, 1):
        write(raw_dir / f"precopy-{index}/evidence.json", value)
    calls = []
    monkeypatch.setattr(
        positive_browser,
        "_presence",
        lambda *args: ({"evidence": evidence, "star": "DULAT"}, {}, {}),
    )
    monkeypatch.setattr(positive_browser, "_reload", lambda owner, value: calls.append("reload"))

    def rebuild(owner, directory, recorded, original):
        assert isinstance(owner, positive_browser._RasterEvidence)
        assert original == evidence and recorded == {"fixture_boundary": directory.name}
        calls.append(directory.name)
        if directory.name == failed_boundary:
            raise BrowserSafetyStop("fixture_shared_rebuild_rejected")

    class ReachedNativeChain(Exception):
        pass

    def native(*args, **kwargs):
        calls.append("native_chain")
        raise ReachedNativeChain

    monkeypatch.setattr(positive_browser, "_fresh_evidence", rebuild)
    monkeypatch.setattr(positive_browser, "_copy_chain", native)
    expected = BrowserSafetyStop if failed_boundary else ReachedNativeChain
    with pytest.raises(expected):
        positive_browser._planet_sources(
            _Evidence(tmp_path), raw_dir, tmp_path / "derived", tmp_path / "planet-class"
        )
    assert calls == (
        ["reload", "fresh", "precopy-1", "precopy-2"]
        if failed_boundary
        else ["reload", "fresh", "precopy-1", "precopy-2", "precopy-3", "native_chain"]
    )
