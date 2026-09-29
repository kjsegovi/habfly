"""Explicit continuation fixtures: local interception, never the live preview."""
# ruff: noqa: F811

import hashlib
import json

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_observation import observation_page  # noqa: F401

import habfly.browser_planet_observation_recovery as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_observation import PlanetObservationStartSession


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def stopped(observation_page, tmp_path):
    page, frame = observation_page
    frame.evaluate("""()=>{
      document.querySelector('select').selectedIndex=-1;
      for(const name of ['orbital_radius','planet_mass','planet_radius','planet_density']){
        const field=document.getElementById(name);field.previousElementSibling.remove();field.remove();
      }
    }""")
    source = tmp_path / "start"
    session = PlanetObservationStartSession(page, config(), source, run_history=tmp_path)
    original = session._read

    def expire(**kwargs):
        if (source / "play-reserved.json").exists():
            session.guard.deadline = 0
        return original(**kwargs)

    session._read = expire
    try:
        with pytest.raises(BrowserSafetyStop, match="planet_copy_time_limit"):
            session.start(5000)
    finally:
        session.close()
    assert frame.evaluate("window.fixtureClicks") == 0
    assert frame.locator("#duration").input_value() == "5000"
    frame.locator("#duration").evaluate(
        "e=>e.oninput=()=>{window.extraDurationWrites=(window.extraDurationWrites||0)+1}"
    )
    return page, frame, source


def run(stopped, tmp_path, **kwargs):
    return module.continue_planet_observation(
        stopped[0], config(), stopped[2], run_history=tmp_path, **kwargs
    )


def test_exact_predispatch_continuation_does_not_rewrite_or_claim_science(stopped, tmp_path):
    page, frame, source = stopped
    original, _ = module.validate_observation_continuation_source(source, tmp_path)
    hashes = dict(original.hashes)
    result = run(stopped, tmp_path)
    assert result["mode"] == module.MODE and result["source_sha256"] == hashes
    assert len(hashes) == 11
    assert result["fixed_budget_seconds"] == 60
    assert result["max_duration_writes"] == 0 and result["max_total_play_clicks"] == 1
    assert result["original_reservation_retained"] and result["source_duration_committed"]
    assert result["current_class_paint_unchanged"] and result["play_click_dispatched_once"]
    assert result["answers_unchanged"] and result["readback_verified"]
    assert all(
        result[k] is False
        for k in (
            "source_play_click_dispatched",
            "original_class_paint_available",
            "historical_class_paint_equivalence_verified",
            "automatic_retry",
            "task_completed",
            "scientific_verified",
            "correctness_verified",
            "observation_completed",
            "observation_started_verified",
        )
    )
    assert result["planet_presence"] is None
    assert frame.evaluate("window.fixtureClicks") == 1
    assert frame.evaluate("window.extraDurationWrites||0") == 0
    assert not page.get_by_role("checkbox").is_checked()
    original.unchanged()
    assert read(source / "continuation-reserved.json") == read(source / "play-continuation/reserved.json")
    assert read(source / "play-continuation/dispatch.json") == {
        "kind": "CLICK",
        "visible_label": "Play",
        "max_clicks": 1,
    }
    with pytest.raises(BrowserSafetyStop, match="already_consumed"):
        run(stopped, tmp_path)
    assert frame.evaluate("window.fixtureClicks") == 1


@pytest.mark.parametrize(
    "change",
    [
        "reason",
        "attempted",
        "false_number",
        "duration_unconfirmed",
        "missing_play",
        "dispatched",
        "after",
        "confirmed",
        "invalidated",
        "claim",
        "capture_hash",
        "scope",
        "days",
        "legacy",
    ],
)
def test_unproven_original_never_creates_continuation(stopped, tmp_path, change):
    _, frame, source = stopped
    if change in {"reason", "attempted", "false_number"}:
        value = read(source / "stopped.json")
        key, replacement = {
            "reason": ("reason", "operator_aborted"),
            "attempted": ("play_click_may_have_occurred", True),
            "false_number": ("play_click_may_have_occurred", 0),
        }[change]
        value[key] = replacement
        (source / "stopped.json").write_text(json.dumps(value))
    elif change == "duration_unconfirmed":
        value = read(source / "play-reserved.json")
        value["duration_committed"] = False
        (source / "play-reserved.json").write_text(json.dumps(value))
    elif change == "missing_play":
        (source / "play-reserved.json").unlink()
    elif change in {"dispatched", "after", "confirmed", "invalidated"}:
        (source / (change if change == "after" else change + ".json")).write_text("{}")
    elif change == "claim":
        next((tmp_path / "observation-start-reservations").iterdir()).write_text("{}")
    elif change == "capture_hash":
        (source / "committed/observation.json").write_text("{}")
    elif change == "scope":
        value = read(source / "read-guard/scope.json")
        value["max_seconds"] = 1000
        (source / "read-guard/scope.json").write_text(json.dumps(value))
    elif change == "days":
        value = read(source / "reserved.json")
        value["days"] = 10000
        (source / "reserved.json").write_text(json.dumps(value))
    else:
        directory = tmp_path / "reference-observation-legacy"
        directory.mkdir()
        (directory / "reserved.json").write_text(json.dumps({"star": "jyremis", "days": 10000}))
    with pytest.raises(BrowserSafetyStop):
        run(stopped, tmp_path)
    assert not (source / "play-continuation").exists()
    assert frame.evaluate("window.fixtureClicks") == 0


@pytest.mark.parametrize("change", ["duration", "answer", "presence", "star", "pause", "focused", "outer"])
def test_changed_current_public_state_never_claims_or_clicks(stopped, tmp_path, change):
    page, frame, source = stopped
    if change == "outer":
        page.get_by_role("checkbox").check()
    elif change == "focused":
        frame.locator("#duration").focus()
    elif change == "pause":
        frame.locator("#play").evaluate(
            "e=>{e.setAttribute('aria-label','Pause');e.querySelector('svg').setAttribute('aria-label','Pause')}"
        )
    elif change == "presence":
        frame.get_by_role("combobox").select_option(label="No")
    elif change == "star":
        frame.locator("body").evaluate("e=>e.innerHTML=e.innerHTML.replace(/JYREMIS/gi,'OTHER')")
    else:
        frame.locator("#duration" if change == "duration" else "#period_days").fill("5")
    with pytest.raises(BrowserSafetyStop):
        run(stopped, tmp_path)
    assert frame.evaluate("window.fixtureClicks") == 0
    assert not (source / "continuation-reserved.json").exists()
    assert not read(source / "play-continuation/stopped.json")["play_click_may_have_occurred"]
    with pytest.raises(BrowserSafetyStop, match="already_consumed"):
        run(stopped, tmp_path)


@pytest.mark.parametrize(
    "change", ["play", "duration", "answer", "paint", "source", "cancel", "deadline", "popup"]
)
def test_postclaim_failure_is_consumed_without_play(stopped, tmp_path, monkeypatch, change):
    page, frame, source = stopped
    original, cancelled = module.persist_json, False

    def persist(path, value):
        nonlocal cancelled
        original(path, value)
        if path == source / "play-continuation/reserved.json":
            if change == "play":
                frame.locator("#play").evaluate("e=>e.replaceWith(e.cloneNode(true))")
            elif change == "duration":
                frame.locator("#duration").evaluate("e=>e.value='4999'")
            elif change == "answer":
                frame.locator("#line_shift").evaluate("e=>e.value='1'")
            elif change == "paint":
                paint_current_class(frame)
            elif change == "source":
                (source / "play-reserved.json").write_text("{}")
            elif change == "cancel":
                cancelled = True
            elif change == "deadline":
                monkeypatch.setattr(module.time, "monotonic", lambda: 1e20)
            else:
                page.evaluate("window.open('about:blank').close()")
                page.wait_for_timeout(50)

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop):
        run(stopped, tmp_path, cancelled=lambda: cancelled)
    stopped_record = read(source / "play-continuation/stopped.json")
    assert stopped_record["continuation_reserved"] and not stopped_record["play_click_may_have_occurred"]
    assert not stopped_record["duration_write_may_have_occurred"]
    assert not stopped_record["automatic_retry"]
    assert frame.evaluate("window.fixtureClicks") == 0
    assert not (source / "play-continuation/dispatch.json").exists()


@pytest.mark.parametrize("change", ["answer", "dialog", "cancel", "deadline", "source"])
def test_uncertain_play_never_retries_or_claims_completion(stopped, tmp_path, monkeypatch, change):
    _, frame, source = stopped
    if change in {"answer", "dialog"}:
        code = (
            "document.querySelector('#line_shift').value='1'"
            if change == "answer"
            else "confirm('private fixture message')"
        )
        frame.locator("#play").evaluate(
            "(e,code)=>{const prior=e.onclick;e.onclick=()=>{prior();new Function(code)()}}", code
        )
    original = module.persist_json

    def persist(path, value):
        original(path, value)
        if path == source / "play-continuation/dispatched.json":
            if change == "deadline":
                monkeypatch.setattr(module.time, "monotonic", lambda: 1e20)
            elif change == "source":
                (source / "play-reserved.json").write_text("{}")

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop):
        run(
            stopped,
            tmp_path,
            cancelled=lambda: change == "cancel" and (source / "play-continuation/dispatched.json").exists(),
        )
    record = read(source / "play-continuation/stopped.json")
    assert record["play_click_may_have_occurred"] and not record["automatic_retry"]
    assert not record["duration_write_may_have_occurred"] and not record["task_completed"]
    assert frame.evaluate("window.fixtureClicks") == 1
    assert frame.evaluate("window.extraDurationWrites||0") == 0
    assert not (source / "play-continuation/confirmed.json").exists()
    assert "private fixture message" not in json.dumps(record)


def test_idle_play_readback_is_not_animation_evidence(stopped, tmp_path):
    frame = stopped[1]
    frame.locator("#play").evaluate("e=>e.onclick=()=>{window.fixtureClicks++}")
    result = run(stopped, tmp_path)
    assert result["play_click_dispatched_once"] and not result["observation_started_verified"]
    assert frame.get_by_role("button", name="Play", exact=True).count() == 1


def paint_current_class(frame):
    frame.locator("body").evaluate("""e=>{
      const style=document.createElement('style');
      style.textContent='.choice.selected label{border-color:white}.choice.selected label::after{opacity:1;background:white}';
      e.append(style);document.querySelector('.choice').classList.add('selected');
    }""")


def test_fresh_paint_preserved_without_inventing_historical_paint(stopped, tmp_path):
    paint_current_class(stopped[1])
    result = run(stopped, tmp_path)
    assert result["current_class_paint"]["selected"] == "gas_giant"
    assert result["current_class_paint_unchanged"] is True
    assert result["original_class_paint_available"] is False
    assert result["historical_class_paint_equivalence_verified"] is False
    assert result["scientific_verified"] is False


def test_source_symlink_is_rejected(stopped, tmp_path):
    source = stopped[2]
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(BrowserSafetyStop, match="symlink"):
        module.validate_observation_continuation_source(alias, tmp_path)
    assert not (source / "play-continuation").exists()


def test_canonical_claim_hash_matches_original_record(stopped, tmp_path):
    book, source = module.validate_observation_continuation_source(stopped[2], tmp_path)
    star = source["star"]
    canonical = (
        "observation-start-reservations/" + hashlib.sha256(star.casefold().encode()).hexdigest() + ".json"
    )
    assert canonical in book.hashes


def test_success_uses_exactly_three_full_captures_under_same_fixed_budget(stopped, tmp_path, monkeypatch):
    original, reads = module.PlanetNumericSession.read, 0

    def read_full(self):
        nonlocal reads
        reads += 1
        return original(self)

    monkeypatch.setattr(module.PlanetNumericSession, "read", read_full)
    result = run(stopped, tmp_path)
    assert reads == 3  # Fresh baseline, final pre-dispatch guard, post-click readback.
    assert result["fixed_budget_seconds"] == 60
    assert result["max_duration_writes"] == 0 and result["max_total_play_clicks"] == 1
