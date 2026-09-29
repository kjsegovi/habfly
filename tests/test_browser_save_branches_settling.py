"""Opted-in branch integration; native cases require the sole browser slot."""
# ruff: noqa: F811

import json
import os

import pytest
import test_browser_terrestrial_workflow as terrestrial_fixture
from test_browser_habitability_actions import menu_page  # noqa: F401
from test_browser_habitability_numeric import habitat_page  # noqa: F401
from test_browser_habitability_save import ready, run  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_positive_finalize import injected  # noqa: F401
from test_browser_positive_finalize import (
    test_native_finalization_without_prior_save_preserves_import_contract as native_positive,
)
from test_browser_positive_planet_workflow import kwargs as positive_kwargs
from test_browser_raster_planet_evidence import raster_page  # noqa: F401
from test_browser_terrestrial_workflow import terrestrial  # noqa: F401
from test_browser_terrestrial_workflow import (
    test_native_three_tab_workflow_retains_reference_and_learned_scopes as native_terrestrial,
)

import habfly.browser_habitability_save as habitat
import habfly.browser_positive_finalize as positive
import habfly.browser_positive_planet_workflow as positive_workflow
import habfly.browser_terrestrial_workflow as terrestrial_workflow
from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_workflow import _Evidence
from habfly.browser_save_settlement import validate_reserved_phase

NATIVE = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Sole browser owner must authorize native fixtures",
)


def opt_in(monkeypatch):
    constructor = positive.PositiveFinalizationSteps

    def create(*args, **kwargs):
        # Admission includes the fixed 3s click allowance; this fixture's old
        # 2s total cannot admit any dispatch, independently of its fast reads.
        return constructor(*args, **{**kwargs, "settle_reserved_notice": True, "timeout_seconds": 5})

    monkeypatch.setattr(positive, "PositiveFinalizationSteps", create)


def test_injected_positive_waits_under_one_claim(injected, monkeypatch):
    state, make, history, _ = injected
    opt_in(monkeypatch)
    original = positive._notice
    seen = 0

    def notice(session, handle):
        nonlocal seen
        if positive._claim_path(history, "Dulat").exists() and state.clicked_at is None:
            seen += 1
            return seen in {1, 3}
        return original(session, handle)

    monkeypatch.setattr(positive, "_notice", notice)
    owner = make()
    owner.advance()
    assert owner.advance()["phase"] == "verify"
    assert state.clicks == 1
    output = history / "owner/save"
    intent = json.loads((output / "reserved.json").read_text())
    book = _Evidence(history)
    validate_reserved_phase(
        book,
        output,
        intent,
        after=book.capture(output / "after"),
        predispatched=book.capture(output / "pre-dispatch-observation"),
        same_view=lambda report: report["value"] == 42,
    )
    assert (output / "reserved-phase-confirmed.json").is_file()
    owner.close()


@pytest.mark.parametrize(
    "failure", ["source", "reservation", "answer", "button", "cancel", "never_clear", "late_ack", "late_read"]
)
def test_injected_positive_new_policy_stops_without_retry(injected, monkeypatch, failure):
    state, make, history, pin = injected
    opt_in(monkeypatch)
    original = positive._notice

    def notice(session, handle):
        claim = positive._claim_path(history, "Dulat")
        if claim.exists() and state.clicked_at is None:
            if failure == "source":
                pin.write_text("{}")
            if failure == "reservation":
                claim.write_text("{}")
            if failure == "answer":
                state.mode = "changed_answer"
            if failure == "button":
                state.mode = "replaced"
            if failure == "cancel":
                state.owner.abort()
            if failure not in {"late_ack", "late_read"}:
                return True
        result = original(session, handle)
        if failure == "late_ack" and result and state.clicked_at is not None:
            state.now += 6
        return result

    current = positive.PlanetNumericSession.current

    def read(session):
        result = current(session)
        if failure == "late_read" and state.clicked_at is not None:
            state.now += 6
        return result

    monkeypatch.setattr(positive, "_notice", notice)
    monkeypatch.setattr(positive.PlanetNumericSession, "current", read)
    owner = make()
    owner.advance()
    owner.advance()
    attempted = failure in {"late_ack", "late_read"}
    assert owner.finished and state.clicks == int(attempted)
    assert not owner.report["task_completed"] and owner.report["save_may_have_occurred"] is attempted
    assert not (history / "owner/save/confirmed.json").exists()
    owner.advance()
    assert state.clicks == int(attempted)
    owner.close()


@NATIVE
@pytest.mark.parametrize("planet_class", ["gas_giant", "ice_giant"])
def test_native_positive_optin_receipt_import(raster_page, tmp_path, monkeypatch, planet_class):
    constructor = positive.PositiveFinalizationSteps

    def create(*args, **kwargs):
        return constructor(*args, **kwargs, settle_reserved_notice=True)

    monkeypatch.setattr(positive, "PositiveFinalizationSteps", create)
    native_positive(raster_page, tmp_path, monkeypatch, planet_class)
    receipt = json.loads((tmp_path / "finalize/workflow/confirmed.json").read_text())
    assert "finalize/save/reserved-phase-confirmed.json" in receipt["source_sha256"]
    args = {**positive_kwargs(tmp_path), "save_dir": tmp_path / "finalize/save"}
    positive_workflow._load_sources(_Evidence(tmp_path), **args)
    claim = next((tmp_path / "positive-finalization-reservations").glob("*.json"))
    for path, field, alias in (
        (tmp_path / "finalize/save/confirmed.json", "settle_reserved_notice", 1),
        (claim, "schema_version", True),
        (claim, "maximum_save_dispatches", True),
        (tmp_path / "finalize/save/pre-dispatch-footer.json", "notice_present", 0),
        (tmp_path / "finalize/save/dispatch.json", "max_clicks", True),
    ):
        raw = path.read_bytes()
        value = json.loads(raw)
        value[field] = alias
        path.write_text(json.dumps(value))
        with pytest.raises(BrowserSafetyStop):
            positive_workflow._load_sources(_Evidence(tmp_path), **args)
        path.write_bytes(raw)


@NATIVE
def test_native_terrestrial_optin_full_workflow(request, tmp_path, monkeypatch):
    original = terrestrial_fixture.save_habitability_work

    def save(*args, **kwargs):
        return original(*args, **kwargs, settle_reserved_notice=True)

    monkeypatch.setattr(terrestrial_fixture, "save_habitability_work", save)
    case = request.getfixturevalue("terrestrial")
    native_terrestrial(case, tmp_path, monkeypatch)
    receipt = json.loads((tmp_path / "terrestrial-workflow/confirmed.json").read_text())
    assert "final-save/reserved-phase-confirmed.json" in receipt["source_sha256"]
    args = terrestrial_fixture.kwargs(tmp_path)
    terrestrial_workflow._load_sources(_Evidence(tmp_path), **args)
    claim = next((tmp_path / "habitability-save-reservations").glob("*.json"))
    for path in (tmp_path / "final-save/confirmed.json", claim):
        raw = path.read_bytes()
        value = json.loads(raw)
        value["settle_reserved_notice"] = 1
        path.write_text(json.dumps(value))
        with pytest.raises(BrowserSafetyStop):
            terrestrial_workflow._load_sources(_Evidence(tmp_path), **args)
        path.write_bytes(raw)


@NATIVE
def test_native_habitability_notice_reappears_then_settles(ready, tmp_path, monkeypatch):
    _, frame, _, _ = ready
    original = habitat._notice
    seen = 0

    def notice(session, handle):
        nonlocal seen
        if (tmp_path / "save/reserved.json").exists() and not (tmp_path / "save/dispatch.json").exists():
            seen += 1
            frame.locator("#save-notice").evaluate("(e,on)=>e.textContent=on?'Data saved':''", seen in {1, 3})
        return original(session, handle)

    monkeypatch.setattr(habitat, "_notice", notice)
    receipt = run(ready, tmp_path, settle_reserved_notice=True)
    assert frame.evaluate("window.saveClicks") == 1 and receipt["notice_was_already_present"] is False
    book = _Evidence(tmp_path)
    validate_reserved_phase(
        book,
        tmp_path / "save",
        json.loads((tmp_path / "save/reserved.json").read_text()),
        after=book.capture(tmp_path / "save/after"),
        predispatched=book.capture(tmp_path / "save/pre-dispatch-observation"),
        same_view=lambda report: habitat._same_view(report, book.capture(tmp_path / "save/before")),
    )


@NATIVE
@pytest.mark.parametrize(
    "failure", ["source", "reservation", "answer", "button", "cancel", "late_ack", "late_read"]
)
def test_native_habitability_optin_failure_retains_one_claim(ready, tmp_path, monkeypatch, failure):
    _, frame, _, choice = ready
    original, original_time = habitat._notice, habitat.time.monotonic
    cancelled, offset = False, 0

    def notice(session, handle):
        nonlocal cancelled, offset
        if (tmp_path / "save/reserved.json").exists() and not (tmp_path / "save/dispatch.json").exists():
            if failure == "source":
                (choice / "confirmed.json").write_text("{}")
            if failure == "reservation":
                (tmp_path / "save/reserved.json").write_text("{}")
            if failure == "answer":
                frame.locator("#temperature").evaluate("e=>e.value='300'")
            if failure == "button":
                frame.get_by_role("button", name="Save").evaluate("e=>e.replaceWith(e.cloneNode(true))")
            if failure == "cancel":
                cancelled = True
            if failure not in {"late_ack", "late_read"}:
                frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")
        result = original(session, handle)
        if failure == "late_ack" and result and (tmp_path / "save/dispatch.json").exists():
            offset = 31
        return result

    current = habitat.HabitabilityMenuSession.current

    def read(session):
        nonlocal offset
        result = current(session)
        if failure == "late_read" and (tmp_path / "save/acknowledgement.json").exists():
            offset = 31
        return result

    monkeypatch.setattr(habitat.time, "monotonic", lambda: original_time() + offset)
    monkeypatch.setattr(habitat, "_notice", notice)
    monkeypatch.setattr(habitat.HabitabilityMenuSession, "current", read)
    with pytest.raises(BrowserSafetyStop):
        run(ready, tmp_path, settle_reserved_notice=True, cancelled=lambda: cancelled)
    attempted = failure in {"late_ack", "late_read"}
    stopped = json.loads((tmp_path / "save/stopped.json").read_text())
    assert stopped["reservation_created"] and stopped["save_may_have_occurred"] is attempted
    assert frame.evaluate("window.saveClicks||0") == int(attempted)
    assert not (tmp_path / "save/confirmed.json").exists()
