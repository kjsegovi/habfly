"""Explicit stale paint recovery; all browser traffic is intercepted."""
# ruff: noqa: F811

import json

import pytest
from test_browser_full_stellar import full_page  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json
from habfly.browser_class_recovery import (
    recover_blank_main_sequence,
    synchronize_fresh_main_sequence,
    synchronize_fresh_stellar_class,
)
from habfly.browser_classification import StellarSelectionSession, read_class_choices
from habfly.browser_probe import inspect_page, save_probe
from habfly.browser_star_preflight import validate_star_class_source
from habfly.browser_stellar import CLASSES, SIMULATION_URL


@pytest.fixture
def stale(full_page, tmp_path):
    frame = full_page.frame(url=SIMULATION_URL)
    frame.locator(".choice label").first.evaluate("e=>e.classList.add('selected')")
    frame.evaluate("""() => {const previous=window.choose;window.choose=(e,main)=>{
        if(e.classList.contains('selected'))return;previous(e,main);
    }}""")

    def proof(name):
        source = tmp_path / name
        source.mkdir()
        save_probe(inspect_page(full_page, config()), source / "stellar")
        persist_json(
            source / "confirmed.json",
            {
                "star": "Althinagon",
                "painted_stellar_class": "main_sequence",
                "fresh_blank_numeric_answers_verified": True,
                "class_selection_verified": False,
                "answer_writes": 0,
                "action_source": "deterministic_navigation",
            },
        )
        return source

    failed = tmp_path / "failed"
    failed.mkdir()

    def record(event, payload):
        persist_json(failed / f"{event}.json", payload)

    old = StellarSelectionSession(full_page, config(), record, fresh_star=proof("old"))
    with pytest.raises(BrowserSafetyStop, match="conditional_fields_disagree"):
        old.select_class("main_sequence", source="reference_diagnostic")
    save_probe(inspect_page(full_page, config()), failed / "after")
    return full_page, frame, proof("reopened"), failed


def recover(state, output):
    page, _, fresh, failed = state
    return recover_blank_main_sequence(page, config(), output, fresh_star=fresh, failed_selection=failed)


def test_two_click_recovery_never_fills_or_grades_and_cannot_be_retried(stale, tmp_path):
    page, frame, _fresh, failed = stale
    original = (failed / "after/observation.json").read_bytes()
    receipt = recover(stale, tmp_path / "recovery")
    assert receipt["class_clicks"] == 2 and receipt["numeric_writes"] == 0
    assert receipt["original_failure_preserved"] and not receipt["correctness_verified"]
    assert not receipt["learned_classification"] and not receipt["intermediate_is_training_label"]
    assert frame.locator("#conditional").is_visible()
    assert all(e.input_value() == "" for e in frame.get_by_role("textbox").all() if e.is_visible())
    assert not page.get_by_role("checkbox").is_checked()
    assert (failed / "after/observation.json").read_bytes() == original
    assert not (failed / "action_result.json").exists()
    with pytest.raises(BrowserSafetyStop, match="unsupported_blank_class_recovery"):
        recover(stale, tmp_path / "retry")


@pytest.mark.parametrize("bad", ["changed_answer", "wrong_star", "tampered_capture", "successful_original"])
def test_rejects_changed_or_completed_evidence_before_writing(stale, tmp_path, bad):
    _page, frame, fresh, failed = stale
    if bad == "changed_answer":
        frame.locator("#distance").fill("77")
    elif bad == "wrong_star":
        frame.get_by_text("Althinagon", exact=True).evaluate("e=>e.innerText='Another'")
    elif bad == "tampered_capture":
        (failed / "after/observation.json").write_text("{}")
    else:
        persist_json(failed / "action_result.json", {"readback_verified": True})
    with pytest.raises(BrowserSafetyStop):
        recover(stale, tmp_path / "recovery")
    assert not (failed / "blank-recovery-reserved.json").exists()
    assert not (fresh / "class-selection-reserved.json").exists()


def test_interrupted_second_click_preserves_intermediate_and_stops(stale, tmp_path):
    _, frame, _, failed = stale
    frame.evaluate("() => {const p=window.choose;window.choose=(e,m)=>{if(!m)p(e,m)}}")
    with pytest.raises(BrowserSafetyStop, match="class_readback_mismatch"):
        recover(stale, tmp_path / "recovery")
    assert (tmp_path / "recovery/intermediate/observation.json").exists()
    stopped = json.loads((tmp_path / "recovery/stopped.json").read_text())
    assert stopped["class_clicks_may_have_occurred"] == 2 and not stopped["automatic_retry"]
    assert not (tmp_path / "recovery/confirmed.json").exists()
    assert (failed / "blank-recovery-reserved.json").exists()


def test_ordinary_revision_needs_explicit_previous_and_blank_answers(full_page):
    session = StellarSelectionSession(full_page, config(), lambda *_: None)
    session.select_class("white_dwarf", source="reference_diagnostic")
    revision = StellarSelectionSession(full_page, config(), lambda *_: None)
    with pytest.raises(BrowserSafetyStop, match="class_write_limit"):
        revision.select_class("main_sequence", source="reference_diagnostic")
    with pytest.raises(BrowserSafetyStop, match="invalid_blank_class_revision"):
        revision.select_class(
            "main_sequence",
            source="checkpoint",
            expected_previous="white_dwarf",
            revision_reason="Not learned",
        )
    full_page.frame(url=SIMULATION_URL).locator("#distance").fill("77")
    revision = StellarSelectionSession(full_page, config(), lambda *_: None)
    with pytest.raises(BrowserSafetyStop, match="invalid_blank_class_revision"):
        revision.select_class(
            "main_sequence",
            source="reference_diagnostic",
            expected_previous="white_dwarf",
            revision_reason="Cannot overwrite data",
        )


@pytest.mark.parametrize("paint", [None, "main_sequence", "white_dwarf"])
def test_fresh_setup_is_bounded_and_does_not_require_a_failed_click(full_page, tmp_path, paint):
    frame = full_page.frame(url=SIMULATION_URL)
    if paint:
        frame.locator(".choice label").nth(0 if paint == "main_sequence" else 3).evaluate(
            "e=>e.classList.add('selected')"
        )
    frame.evaluate(
        """()=>{const p=window.choose;window.choose=(e,m)=>{if(!e.classList.contains('selected'))p(e,m)}}"""
    )
    source = tmp_path / "fresh"
    source.mkdir()
    save_probe(inspect_page(full_page, config()), source / "stellar")
    persist_json(
        source / "confirmed.json",
        {
            "star": "Althinagon",
            "painted_stellar_class": paint,
            "fresh_blank_numeric_answers_verified": True,
            "class_selection_verified": False,
            "answer_writes": 0,
            "action_source": "deterministic_navigation",
        },
    )
    result = synchronize_fresh_main_sequence(full_page, config(), tmp_path / "setup", fresh_star=source)
    assert result["class_clicks"] == (2 if paint == "main_sequence" else 1)
    assert result["numeric_writes"] == 0 and not result["learned_classification"]
    assert not result["correctness_verified"] and not result["intermediate_is_training_label"]
    assert frame.locator("#conditional").is_visible()
    assert all(e.input_value() == "" for e in frame.get_by_role("textbox").all() if e.is_visible())
    with pytest.raises(BrowserSafetyStop, match="invalid_fresh_star_class_evidence"):
        synchronize_fresh_main_sequence(full_page, config(), tmp_path / "retry", fresh_star=source)


def test_fresh_setup_rejects_replaced_or_populated_evidence(stale, tmp_path):
    page, frame, source, _ = stale
    frame.locator("#distance").fill("44")
    with pytest.raises(BrowserSafetyStop):
        synchronize_fresh_main_sequence(page, config(), tmp_path / "setup", fresh_star=source)
    assert not (source / "class-selection-reserved.json").exists()
    assert frame.locator("#distance").input_value() == "44"


def test_fresh_setup_stops_on_failed_intermediate_without_retry(stale, tmp_path):
    page, frame, source, _ = stale
    frame.evaluate("()=>window.choose=()=>{}")
    with pytest.raises(BrowserSafetyStop):
        synchronize_fresh_main_sequence(page, config(), tmp_path / "setup", fresh_star=source)
    stopped = json.loads((tmp_path / "setup/stopped.json").read_text())
    assert stopped["class_clicks_may_have_occurred"] == 1 and not stopped["automatic_retry"]
    assert (source / "class-selection-reserved.json").exists()


@pytest.mark.parametrize("selected_class", CLASSES)
@pytest.mark.parametrize("paint", [None, *CLASSES])
def test_supplied_class_setup_and_offline_source_validation(full_page, tmp_path, selected_class, paint):
    frame = full_page.frame(url=SIMULATION_URL)
    if paint:
        frame.locator(".choice label").nth(CLASSES.index(paint)).evaluate("e=>e.classList.add('selected')")
    frame.evaluate(
        """()=>{const p=window.choose;window.choose=(e,m)=>{if(!e.classList.contains('selected'))p(e,m)}}"""
    )
    source = tmp_path / "fresh"
    source.mkdir()
    save_probe(inspect_page(full_page, config()), source / "stellar")
    persist_json(
        source / "confirmed.json",
        {
            "star": "Althinagon",
            "painted_stellar_class": paint,
            "fresh_blank_numeric_answers_verified": True,
            "class_selection_verified": False,
            "answer_writes": 0,
            "action_source": "deterministic_navigation",
        },
    )
    setup = tmp_path / "setup"
    result = synchronize_fresh_stellar_class(
        full_page, config(), setup, fresh_star=source, selected_class=selected_class
    )
    clicks = 2 if paint == selected_class else 1
    assert result["selected_class"] == selected_class and result["class_clicks"] == clicks
    assert result["readback_verified"] and result["numeric_writes"] == 0
    for key in (
        "correctness_verified",
        "learned_classification",
        "scientific_verified",
        "training_label",
        "task_completed",
    ):
        assert result[key] is False
    assert read_class_choices(frame)[0]["selected"] == selected_class
    assert frame.locator("#conditional").is_visible() is (selected_class == "main_sequence")
    assert all(e.input_value() == "" for e in frame.get_by_role("textbox").all() if e.is_visible())
    verified = validate_star_class_source(tmp_path, setup, "ALTHINAGON", selected_class)
    assert verified["selected_class"] == selected_class and verified["class_clicks"] == clicks
    if clicks == 2:
        first = json.loads((setup / "event-0.json").read_text())["payload"]["value"]
        assert first == ("red_giant" if selected_class == "white_dwarf" else "white_dwarf")
    with pytest.raises(BrowserSafetyStop, match="invalid_fresh_star_class_evidence"):
        synchronize_fresh_stellar_class(
            full_page, config(), tmp_path / "retry", fresh_star=source, selected_class=selected_class
        )


@pytest.mark.parametrize("selected_class", [None, "giant", "automatic", [], 7])
def test_unsupported_class_rejected_before_read_or_reservation(tmp_path, selected_class):
    with pytest.raises(BrowserSafetyStop, match="invalid_explicit_fresh_stellar_class"):
        synchronize_fresh_stellar_class(
            None, None, tmp_path / "setup", fresh_star=tmp_path / "missing", selected_class=selected_class
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("selected_class", ["red_giant", "supergiant", "white_dwarf"])
def test_non_main_setup_preserves_uncertain_first_dispatch_and_cannot_retry(stale, tmp_path, selected_class):
    page, frame, source, _ = stale
    frame.evaluate("()=>window.choose=()=>{}")
    with pytest.raises(BrowserSafetyStop):
        synchronize_fresh_stellar_class(
            page, config(), tmp_path / "setup", fresh_star=source, selected_class=selected_class
        )
    stopped = json.loads((tmp_path / "setup/stopped.json").read_text())
    assert stopped["class_clicks_may_have_occurred"] == 1 and stopped["automatic_retry"] is False
    assert not (tmp_path / "setup/confirmed.json").exists()
    with pytest.raises(BrowserSafetyStop, match="invalid_fresh_star_class_evidence"):
        synchronize_fresh_stellar_class(
            page, config(), tmp_path / "retry", fresh_star=source, selected_class=selected_class
        )
