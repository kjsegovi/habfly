"""Intercepted local fixtures only; no live observation or scientific oracle."""
# ruff: noqa: F811

import json

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_numeric import planet_html

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_observation import PlanetObservationStartSession, start_planet_observation
from habfly.browser_stellar import SIMULATION_URL


@pytest.fixture
def observation_page(page):
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.locator("iframe").first.evaluate("e=>{e.style.width='1200px';e.style.height='800px'}")
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    html = planet_html().replace('value="10000"', 'id="duration" value=""')
    html = html.replace(
        "<button>Play</button>",
        '<button id="play" aria-label="Play"><svg role="img" aria-label="Play" width="24" height="24"><rect width="24" height="24" fill="black"/></svg></button>',
    )
    frame.locator("body").evaluate("(e,html)=>e.innerHTML=html", html)
    # Fixture state exists only to assert native dispatch count. The adapter
    # never reads it. Blur replaces Play; clicking the old button while focused
    # loses that click, reproducing one possible UI race, not a live diagnosis.
    frame.evaluate("""()=>{
      window.fixtureClicks=0;
      function bind(b){b.onclick=()=>{window.fixtureClicks++;b.setAttribute('aria-label','Pause');b.querySelector('svg').setAttribute('aria-label','Pause');};}
      bind(document.querySelector('#play'));
      document.querySelector('#duration').onblur=()=>{
        const old=document.querySelector('#play'),fresh=old.cloneNode(true);
        old.replaceWith(fresh);bind(fresh);
        document.querySelector('svg:not(#play svg) text').textContent='Normalized Flux Days Observed 0 1000 5000';
      };
    }""")
    return page, frame


def session_for(observation_page, tmp_path):
    page, _ = observation_page
    return PlanetObservationStartSession(page, config(), tmp_path / "start", run_history=tmp_path)


def test_tab_commit_rebinds_play_and_dispatches_once(observation_page, tmp_path):
    page, frame = observation_page
    receipt = start_planet_observation(page, config(), tmp_path / "start", run_history=tmp_path)
    assert frame.evaluate("window.fixtureClicks") == 1
    assert frame.locator("#duration").input_value() == "5000"
    assert receipt["days"] == 5000
    assert frame.locator("#line_shift").input_value() == ""
    assert receipt["duration_committed"] and receipt["play_click_dispatched_once"]
    assert receipt["answers_unchanged"] and not receipt["observation_completed"]
    assert not receipt["observation_started_verified"] and receipt["planet_presence"] is None
    assert not receipt["task_completed"] and not page.get_by_role("checkbox").is_checked()
    assert (tmp_path / "start/committed/observation.json").exists()
    assert (tmp_path / "start/dispatched.json").exists()
    assert len(list((tmp_path / "observation-start-reservations").glob("*.json"))) == 1


@pytest.mark.parametrize(
    "days", [None, True, False, "5000", "1e4", "1+2", 0, -1, 5001, 10000, 1.0, float("nan"), float("inf")]
)
def test_invalid_budget_never_writes_or_reserves(observation_page, tmp_path, days):
    _, frame = observation_page
    session = session_for(observation_page, tmp_path)
    with pytest.raises(BrowserSafetyStop, match="invalid_observation_days"):
        session.start(days)
    assert frame.locator("#duration").input_value() == ""
    assert not session.reserved and frame.evaluate("window.fixtureClicks") == 0
    session.close()


@pytest.mark.parametrize("seconds", [True, 0, 4, 121, float("nan"), "60"])
def test_invalid_time_budget_no_writes(observation_page, tmp_path, seconds):
    page, frame = observation_page
    with pytest.raises(ValueError):
        PlanetObservationStartSession(
            page, config(), tmp_path / "start", run_history=tmp_path, max_seconds=seconds
        )
    assert frame.locator("#duration").input_value() == ""


@pytest.mark.parametrize(
    "change",
    ["answer", "duration", "replacement", "play", "reorder", "external", "modal", "auth", "frame", "overlay"],
)
def test_changed_context_fails_before_writes(observation_page, tmp_path, change):
    page, frame = observation_page
    session = session_for(observation_page, tmp_path)
    if change == "answer":
        frame.locator("#period_days").fill("5")
    elif change == "duration":
        frame.locator("#duration").fill("5")
    elif change == "replacement":
        frame.locator("#line_shift").evaluate("e=>e.replaceWith(e.cloneNode(true))")
    elif change == "play":
        frame.locator("#play").evaluate("e=>e.replaceWith(e.cloneNode(true))")
    elif change == "reorder":
        frame.locator("#line_shift").evaluate("e=>e.before(document.querySelector('#period_days'))")
    elif change == "external":
        page.get_by_role("checkbox").check()
    elif change == "frame":
        page.locator("iframe").nth(1).evaluate("e=>e.replaceWith(e.cloneNode(true))")
    elif change == "overlay":
        frame.locator("#play").evaluate(
            "e=>{const r=e.getBoundingClientRect(),o=document.createElement('div');o.style=`position:fixed;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;background:red;z-index:100`;document.body.append(o);}"
        )
    else:
        frame.locator("body").evaluate(
            "(e,h)=>e.insertAdjacentHTML('beforeend',h)",
            "<div role=dialog>stop</div>" if change == "modal" else "<input type=password>",
        )
    with pytest.raises(BrowserSafetyStop):
        session.start(5000)
    assert frame.evaluate("window.fixtureClicks") == 0
    assert not session.fill_attempted and session.stopped
    session.close()


@pytest.mark.parametrize(
    "change", ["answer", "duration", "replacement", "feedback", "dialog", "chart_words", "play_disabled"]
)
def test_commit_side_effects_stop_before_play(observation_page, tmp_path, change):
    _, frame = observation_page
    session = session_for(observation_page, tmp_path)
    script = {
        "answer": "document.querySelector('#period_days').value='5'",
        "duration": "e.value='9999'",
        "replacement": "e.replaceWith(e.cloneNode(true))",
        "feedback": "document.body.append('unexpected feedback')",
        "dialog": "confirm('private confirmation')",
        "chart_words": "document.querySelector('svg:not(#play svg) text').textContent='Normalized Flux Days Observed secret 100'",
        "play_disabled": "document.querySelector('#play').disabled=true",
    }[change]
    frame.locator("#duration").evaluate("(e,script)=>{e.onblur=()=>{new Function('e',script)(e)}}", script)
    with pytest.raises(BrowserSafetyStop):
        session.start(5000)
    assert frame.evaluate("window.fixtureClicks") == 0
    stopped = json.loads((tmp_path / "start/stopped.json").read_text())
    assert stopped["duration_write_may_have_occurred"]
    assert not stopped["play_click_may_have_occurred"]
    assert "private confirmation" not in (tmp_path / "start/stopped.json").read_text()
    with pytest.raises(BrowserSafetyStop, match="stopped"):
        session.start(5000)
    session.close()


def test_play_replacement_after_committed_reservation_is_not_clicked(observation_page, tmp_path, monkeypatch):
    import habfly.browser_planet_observation as module

    _, frame = observation_page
    original = module.persist_json

    def persist(path, value):
        original(path, value)
        if path.name == "play-reserved.json":
            frame.locator("#play").evaluate("e=>e.replaceWith(e.cloneNode(true))")

    monkeypatch.setattr(module, "persist_json", persist)
    session = session_for(observation_page, tmp_path)
    with pytest.raises(BrowserSafetyStop, match="replaced_before_dispatch"):
        session.start(5000)
    assert frame.evaluate("window.fixtureClicks") == 0
    session.close()


def test_dispatch_side_effect_preserved_and_never_retried(observation_page, tmp_path):
    page, frame = observation_page
    frame.locator("#duration").evaluate(
        "e=>{e.onblur=()=>{document.querySelector('#play').onclick=()=>{window.fixtureClicks++;document.querySelector('#line_shift').value='5';};};}"
    )
    session = session_for(observation_page, tmp_path)
    with pytest.raises(BrowserSafetyStop, match="side_effect"):
        session.start(5000)
    assert frame.evaluate("window.fixtureClicks") == 1
    assert json.loads((tmp_path / "start/stopped.json").read_text())["play_click_may_have_occurred"]
    session.close()
    # Even a separately requested session cannot replay the same star after a
    # human has cleared duration. Neither clearing nor rollback is in adapter.
    frame.locator("#duration").fill("")
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        PlanetObservationStartSession(page, config(), tmp_path / "again", run_history=tmp_path)
    assert frame.evaluate("window.fixtureClicks") == 1


def test_legacy_reservation_blocks_new_session(observation_page, tmp_path):
    page, frame = observation_page
    legacy = tmp_path / "reference-observation-10"
    legacy.mkdir()
    (legacy / "reserved.json").write_text(json.dumps({"star": "jyremis", "days": 10000}))
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        PlanetObservationStartSession(page, config(), tmp_path / "start", run_history=tmp_path)
    assert frame.locator("#duration").input_value() == ""


def test_prepopulated_zero_is_not_blank(observation_page, tmp_path):
    _, frame = observation_page
    frame.locator("#duration").fill("0")
    with pytest.raises(BrowserSafetyStop, match="already_populated"):
        session_for(observation_page, tmp_path)


def test_deadline_stops_before_fill(observation_page, tmp_path):
    _, frame = observation_page
    session = session_for(observation_page, tmp_path)
    session.guard.deadline = 0
    with pytest.raises(BrowserSafetyStop, match="time_limit"):
        session.start(5000)
    assert frame.locator("#duration").input_value() == ""
    session.close()


@pytest.mark.parametrize("boundary", ["duration", "play"])
def test_full_guard_expiring_deadline_never_dispatches_next_write(observation_page, tmp_path, boundary):
    _, frame = observation_page
    session = session_for(observation_page, tmp_path)
    original = session._read

    def read(**kwargs):
        result = original(**kwargs)
        marker = "reserved.json" if boundary == "duration" else "play-reserved.json"
        if (tmp_path / "start" / marker).exists():
            session.guard.deadline = 0
        return result

    session._read = read
    try:
        with pytest.raises(BrowserSafetyStop, match="time_limit"):
            session.start(5000)
        stopped = json.loads((tmp_path / "start/stopped.json").read_text())
        assert stopped["duration_write_may_have_occurred"] == (boundary == "play")
        assert stopped["play_click_may_have_occurred"] is False
        assert frame.evaluate("window.fixtureClicks") == 0
        assert not (tmp_path / "start/dispatched.json").exists()
    finally:
        session.close()


def test_only_committed_and_final_full_reads_follow_duration_write(observation_page, tmp_path):
    session = session_for(observation_page, tmp_path)
    original, reads = session._read, []

    def read(**kwargs):
        if kwargs["expected_duration"] == "5000":
            reads.append(kwargs.get("capture_name"))
        return original(**kwargs)

    session._read = read
    try:
        session.start(5000)
        assert reads == ["committed", None, "after"]
    finally:
        session.close()


def test_initial_answer_reordering_uses_visible_labels(observation_page, tmp_path):
    page, frame = observation_page
    frame.locator("#line_shift").evaluate("""e=>{
      const period=document.querySelector('#period_days'),label=period.previousElementSibling;
      e.previousElementSibling.before(label,period);
    }""")
    receipt = start_planet_observation(page, config(), tmp_path / "start", run_history=tmp_path, days=4500)
    assert receipt["answers_unchanged"] and frame.evaluate("window.fixtureClicks") == 1
    assert frame.locator("#period_days").input_value() == ""


@pytest.mark.parametrize(
    "change", ["wrong_label", "wrong_unit", "duplicate_play", "missing_chart", "blank_label"]
)
def test_unsupported_initial_controls_are_rejected(observation_page, tmp_path, change):
    _, frame = observation_page
    if change in {"wrong_label", "blank_label"}:
        frame.locator("#duration").evaluate(
            "(e,label)=>e.previousElementSibling.textContent=label",
            "Doppler shift (nm)" if change == "wrong_label" else "",
        )
    elif change == "wrong_unit":
        frame.locator("#duration").evaluate("e=>e.nextElementSibling.textContent='years'")
    elif change == "duplicate_play":
        frame.locator("#play").evaluate("e=>e.after(e.cloneNode(true))")
    else:
        frame.locator("svg:not(#play svg)").evaluate("e=>e.remove()")
    with pytest.raises((BrowserSafetyStop, ValueError)):
        session_for(observation_page, tmp_path)
    assert frame.locator("#duration").input_value() == ""
    assert frame.evaluate("window.fixtureClicks") == 0


@pytest.mark.parametrize("record", ["not-json", "{}", '{"star":null}'])
def test_malformed_legacy_history_fails_closed(observation_page, tmp_path, record):
    _, frame = observation_page
    legacy = tmp_path / "reference-observation-3"
    legacy.mkdir()
    (legacy / "reserved.json").write_text(record)
    with pytest.raises(BrowserSafetyStop, match="unreadable_observation_history"):
        session_for(observation_page, tmp_path)
    assert frame.locator("#duration").input_value() == ""


def test_concurrent_per_star_reservation_blocks_before_fill(observation_page, tmp_path):
    _, frame = observation_page
    session = session_for(observation_page, tmp_path)
    session.reservation.parent.mkdir()
    session.reservation.write_text("{}")
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        session.start(5000)
    assert not session.fill_attempted and frame.evaluate("window.fixtureClicks") == 0
    session.close()


def test_unknown_driver_failure_is_redacted_and_not_retried(observation_page, tmp_path, monkeypatch):
    _, frame = observation_page
    session = session_for(observation_page, tmp_path)

    def failed_press(*args, **kwargs):
        raise RuntimeError("private password in driver detail")

    monkeypatch.setattr(session.duration, "press", failed_press)
    with pytest.raises(BrowserSafetyStop, match="observation_start_failed") as caught:
        session.start(5000)
    assert "private password" not in str(caught.value)
    assert "private password" not in (tmp_path / "start/stopped.json").read_text()
    assert frame.evaluate("window.fixtureClicks") == 0 and session.fill_attempted
    session.close()


def test_idle_readback_does_not_assert_animation_started(observation_page, tmp_path):
    page, frame = observation_page
    frame.locator("#duration").evaluate(
        "e=>{e.onblur=()=>{document.querySelector('#play').onclick=()=>{window.fixtureClicks++;};};}"
    )
    receipt = start_planet_observation(page, config(), tmp_path / "start", run_history=tmp_path, days=5000)
    assert frame.get_by_role("button", name="Play", exact=True).count() == 1
    assert receipt["play_click_dispatched_once"] and not receipt["observation_started_verified"]
    assert not receipt["observation_completed"] and receipt["planet_presence"] is None


def test_output_must_belong_to_explicit_existing_history(observation_page, tmp_path):
    page, frame = observation_page
    with pytest.raises(ValueError):
        PlanetObservationStartSession(page, config(), tmp_path / "outside", run_history=tmp_path / "missing")
    history = tmp_path / "history"
    history.mkdir()
    with pytest.raises(ValueError):
        PlanetObservationStartSession(page, config(), tmp_path / "outside", run_history=history)
    assert frame.locator("#duration").input_value() == ""
