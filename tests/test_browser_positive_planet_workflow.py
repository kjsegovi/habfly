"""Intercepted native receipts plus synthetic frozen-policy event streams.

These are transport fixtures, not learned performance or real course evidence.
No browser connects to HabWorlds; no tests read application state or grade truth.
"""
# ruff: noqa: F811

import hashlib
import json
from copy import deepcopy

import pytest
from test_browser_no_planet_workflow import read, snapshot, stellar_sources, stream, write
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_raster_planet_evidence import copy, raster_page, select, sources  # noqa: F401

import habfly.browser_positive_planet_workflow as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_classification import select_planet_class
from habfly.browser_planet_numeric import ANSWER_UNITS, PlanetNumericSession
from habfly.browser_planet_save import save_planet_work
from habfly.contracts import RuntimeEvent


def derived_stream(directory, report, payloads):
    rows = [("hello", {"synthetic_fixture": True}), *payloads, ("episode_summary", report)]
    raw = "".join(
        RuntimeEvent(event=kind, sequence=i, run_id="fixture-only", payload=payload).model_dump_json() + "\n"
        for i, (kind, payload) in enumerate(rows)
    )
    (directory / "events.jsonl").write_text(raw)
    write(directory / "report.json", {**report, "events_sha256": hashlib.sha256(raw.encode()).hexdigest()})


@pytest.fixture
def positive(raster_page, tmp_path, request):
    page, frame = raster_page
    setting = getattr(request, "param", "gas_giant")
    mixed_case = setting == "mixed_case"
    if mixed_case:
        frame.locator("div").first.evaluate("e=>{e.textContent='Dulat';e.style.textTransform='uppercase'}")
    frame.locator("#derived").evaluate(
        "e=>e.textContent='STAR MASS (Ms) 1 STAR RADIUS (Rs) 1 ORBIT (years) 0.000'"
    )
    frame.evaluate("""()=>{
      const style=document.createElement('style');
      style.textContent='.choice.selected label{border-color:white}.choice.selected label::after{opacity:1;background:white}';document.body.append(style);
      document.querySelectorAll('.choice label').forEach(label=>label.onclick=()=>label.parentElement.classList.add('selected'));
      const button=[...document.querySelectorAll('button')].find(e=>e.textContent==='Save');
      const footer=document.createElement('div'),notice=document.createElement('div');
      notice.id='save-notice';button.before(footer);footer.append(notice,button);
      button.onclick=()=>notice.textContent='Data saved';
    }""")
    source = sources(raster_page, tmp_path)
    select(raster_page, tmp_path, source)
    copy(raster_page, tmp_path)
    payloads = []
    directory = tmp_path / "derived"
    session = PlanetNumericSession(
        page, config(), directory / "native-copies", lambda *args: payloads.append(args)
    )
    try:
        for name in sorted(module.DERIVED):
            session.copy(name, "1", ANSWER_UNITS[name], source="checkpoint")
        verified = deepcopy(session.verified)
    finally:
        session.close()
    derived_stream(
        directory,
        {
            "scope": "four_derived_planet_browser_transport",
            "outcome": "planet_derived_transport_verified",
            "checkpoint_unchanged": True,
            "planet_transport_verified": True,
            "optimizer_updates": 0,
            "task_completed": False,
            "browser_acceptance_passed": False,
            "saved": False,
            "assessment_performed": False,
            "submitted": False,
            "write_attempts": sorted(module.DERIVED),
            "verified_fields": verified,
            "provenance": {
                "checkpoint_sha256": "a" * 64,
                "graph_hash": "b" * 64,
                "knowledge_pack_hash": "c" * 64,
                "optimizer_updates": 0,
                "supplied_star_class": "main_sequence",
                "classification_source": "supplied_not_learned",
                "fixture_only": True,
            },
        },
        payloads,
    )
    select_planet_class(
        page,
        config(),
        tmp_path / "planet-class",
        "gas_giant" if mixed_case else setting,
        source="reference_diagnostic",
    )
    save_planet_work(page, config(), tmp_path / "save")
    planet = snapshot(frame)
    # The shared stellar fixture reads only the star's name from this test seam.
    write(tmp_path / "choice/confirmed.json", {"star": "Dulat" if mixed_case else "JYREMIS"})
    stellar = stellar_sources(page, frame, tmp_path)
    return page, frame, {"stellar": stellar, "planet": planet}


def kwargs(history):
    return {
        key + "_dir": history / value
        for key, value in (
            ("numeric", "numeric"),
            ("color", "color"),
            ("class", "class"),
            ("raw", "copy"),
            ("derived", "derived"),
            ("planet_class", "planet-class"),
            ("save", "save"),
        )
    }


def load(history):
    return module._load_sources(module._Evidence(history), **kwargs(history))


def seam(monkeypatch, frame, screens, mutate=None):
    calls = []

    def navigate(page, config, output, destination, expected_star=None):
        calls.append(destination)
        frame.locator("body").evaluate("(e,h)=>e.innerHTML=h", screens[destination])
        if mutate:
            mutate(destination)
        return {"same_star_verified": True, "destination_verified": True, "suggested_required_text": []}

    monkeypatch.setattr(module, "navigate_project", navigate)
    return calls


def run(page, history, output="workflow"):
    return module.verify_positive_planet_workflow(
        page, config(), history / output, run_history=history, **kwargs(history)
    )


def test_positive_native_workflow_keeps_approximation_and_na_distinct(positive, tmp_path, monkeypatch):
    page, frame, screens = positive
    bundle = load(tmp_path)
    assert bundle["star"] == "JYREMIS" and bundle["planet_class"] == "gas_giant"
    calls = seam(monkeypatch, frame, screens)
    result = run(page, tmp_path)
    assert calls == ["stellar", "planet"]
    assert result["task_completed"] and result["authority"] == "visible_workflow_readback"
    assert result["planet"]["outcome"] == "planet" and result["planet"]["transport_verified"]
    assert result["planet"]["measurement_provenance"] == "approximate_reference_raster"
    assert (
        result["raw_measurement_evidence"]["uncertainty"]["kind"]
        == "conditional_pixel_bounds_not_statistical_confidence"
    )
    assert result["habitability"]["outcome"] == "not_applicable"
    assert result["habitability"]["applicability_reason"] == "non_terrestrial_planet"
    assert (
        result["habitability"]["branch_applicability_verified"]
        and not result["habitability"]["transport_verified"]
    )
    assert result["save_acknowledgement_verified"] and result["source_save_click_delivered"]
    assert result["source_sha256"] and set(result["current_screen_sha256"]) == {"stellar", "planet"}
    for key in (
        "scientific_verified",
        "correctness_verified",
        "browser_acceptance_passed",
        "course_completion_verified",
        "project_completed",
        "submitted",
        "training_label",
        "cross_session_persistence_verified",
    ):
        assert result[key] is False
    assert result["answer_writes"] == result["save_clicks"] == result["habitability_writes"] == 0
    assert not page.get_by_role("checkbox").is_checked()


def test_corrupt_or_uncertain_sources_fail_before_navigation(positive, tmp_path, monkeypatch):
    page, frame, screens = positive
    calls = seam(monkeypatch, frame, screens)
    names = (
        "source/chart.png",
        "presence/confirmed.json",
        "copy/precopy-2/progress/chart.png",
        "copy/native-event-02.json",
        "copy/native-copies/copy-01-confirmed.json",
        "derived/events.jsonl",
        "planet-class/reserved.json",
        "save/acknowledgement.json",
        "numeric/events.jsonl",
        "class/after/observation.json",
    )
    for i, name in enumerate(names):
        path = tmp_path / name
        original = path.read_bytes()
        path.write_bytes(b"{}")
        with pytest.raises(BrowserSafetyStop):
            run(page, tmp_path, output=f"invalid-{i}")
        assert not (tmp_path / f"invalid-{i}/confirmed.json").exists()
        path.write_bytes(original)
    assert calls == []
    for name in ("save/stopped.json", "derived/native-copies/copy-01-stopped.json", "copy/invalidated.json"):
        path = tmp_path / name
        write(path, {"write_may_have_occurred": True})
        with pytest.raises(BrowserSafetyStop):
            load(tmp_path)
        path.unlink()
    assert load(tmp_path)["star"] == "JYREMIS"


def test_approximate_flags_and_same_star_chain_cannot_be_promoted(positive, tmp_path):
    mutations = [
        ("copy/report.json", "schema_version", 2),
        ("copy/report.json", "raw_measurement_transport_verified", 1),
        ("copy/report.json", "training_label", True),
        ("copy/report.json", "provenance", "checkpoint"),
        ("copy/report.json", "daily_coverage_verified", True),
        ("copy/report.json", "numeric_writes", True),
        ("planet-class/confirmed.json", "star", "OTHER"),
        ("save/confirmed.json", "notice_was_already_present", True),
        ("save/confirmed.json", "task_completed", True),
        ("save/confirmed.json", "save_click_delivered", 1),
        ("planet-class/confirmed.json", "readback_verified", 1),
        ("copy/native-copies/copy-01-confirmed.json", "readback_verified", 1),
    ]
    for name, key, value in mutations:
        path = tmp_path / name
        original = path.read_bytes()
        data = json.loads(original)
        data[key] = value
        write(path, data)
        with pytest.raises(BrowserSafetyStop):
            load(tmp_path)
        path.write_bytes(original)


def test_terrestrial_is_pending_not_not_habitable(positive, tmp_path):
    for name in ("confirmed.json", "reserved.json"):
        path = tmp_path / "planet-class" / name
        data = read(path)
        data["value"] = "terrestrial"
        write(path, data)
    with pytest.raises(BrowserSafetyStop, match="terrestrial_completion_evidence_not_supported"):
        load(tmp_path)


@pytest.mark.parametrize("positive", ["ice_giant"], indirect=True)
def test_ice_giant_has_same_explicit_na_branch(positive, tmp_path, monkeypatch):
    page, frame, screens = positive
    seam(monkeypatch, frame, screens)
    receipt = run(page, tmp_path)
    assert receipt["planet"]["value"] == "ice_giant"
    assert receipt["habitability"]["outcome"] == "not_applicable"
    assert receipt["task_completed"] and not receipt["habitability"]["transport_verified"]


@pytest.mark.parametrize("positive", ["mixed_case"], indirect=True)
def test_title_case_native_identity_matches_uppercase_chart_without_rewriting_evidence(
    positive, tmp_path, monkeypatch
):
    page, frame, screens = positive
    raw = read(tmp_path / "copy/report.json")
    assert raw["star"] == raw["evidence"]["star"] == "DULAT"
    assert read(tmp_path / "planet-class/confirmed.json")["star"] == "Dulat"
    assert read(tmp_path / "save/confirmed.json")["star"] == "Dulat"
    before_hash = hashlib.sha256((tmp_path / "copy/report.json").read_bytes()).hexdigest()
    assert load(tmp_path)["star"] == "Dulat"
    seam(monkeypatch, frame, screens)
    receipt = run(page, tmp_path)
    assert receipt["task_completed"] and receipt["star"] == "Dulat"
    assert receipt["raw_measurement_evidence"]["star"] == "DULAT"
    assert hashlib.sha256((tmp_path / "copy/report.json").read_bytes()).hexdigest() == before_hash
    for directory in ("planet-class", "save"):
        paths = [tmp_path / directory / name for name in ("confirmed.json", "reserved.json")]
        originals = [path.read_bytes() for path in paths]
        for path, original in zip(paths, originals, strict=True):
            data = json.loads(original)
            data["star"] = "Other"
            write(path, data)
        with pytest.raises(BrowserSafetyStop):
            load(tmp_path)
        for path, original in zip(paths, originals, strict=True):
            path.write_bytes(original)
    manifest_path, events_path = tmp_path / "color/manifest.json", tmp_path / "color/events.jsonl"
    original_manifest, original_events = manifest_path.read_bytes(), events_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest.pop("events_sha256")
    events = [json.loads(line) for line in original_events.splitlines()]
    for event in events:
        if event["event"] == "action_result":
            event["payload"]["observation"]["values"]["star_name"] = "Other"
    stream(tmp_path / "color", manifest, [(e["event"], e["payload"]) for e in events[1:-1]])
    with pytest.raises(BrowserSafetyStop, match="star_mismatch"):
        load(tmp_path)
    manifest_path.write_bytes(original_manifest)
    events_path.write_bytes(original_events)


@pytest.mark.parametrize("value", [None, "", 0, True, "Other", " Dulat", "Dulat "])
def test_identity_comparison_only_accepts_case_difference(value):
    assert not module._identity_matches(value, "DULAT")
    assert module._identity_matches("Dulat", "DULAT")
    assert not module._identity_matches("ſtar", "STAR")


@pytest.mark.parametrize("change", ["number", "class", "star", "duration", "hidden", "modal", "source"])
def test_current_readback_must_still_match(positive, tmp_path, monkeypatch, change):
    page, frame, screens = positive

    def mutate(destination):
        if destination != "planet":
            return
        if change == "number":
            frame.locator("#planet_mass").fill("2")
        elif change == "class":
            frame.locator(".choice").first.evaluate("e=>e.classList.remove('selected')")
        elif change == "star":
            frame.locator("div").first.evaluate("e=>e.textContent='OTHER'")
        elif change == "duration":
            frame.get_by_role("textbox").first.fill("10000")
        elif change == "hidden":
            frame.locator("#line_shift").evaluate("e=>e.style.visibility='hidden'")
        elif change == "source":
            write(tmp_path / "save/acknowledgement.json", {})
        else:
            frame.locator("body").evaluate(
                "e=>e.insertAdjacentHTML('afterbegin','<div role=dialog>Unexpected</div>')"
            )

    seam(monkeypatch, frame, screens, mutate)
    with pytest.raises(BrowserSafetyStop):
        run(page, tmp_path)
    failure = read(tmp_path / "workflow/stopped.json")
    assert not failure["task_completed"] and failure["answer_writes"] == 0
    assert not (tmp_path / "workflow/confirmed.json").exists()
