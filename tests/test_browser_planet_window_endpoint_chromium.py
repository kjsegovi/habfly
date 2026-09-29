"""Native window readiness over an explicitly synthetic growing SVG chart.

All network traffic is fulfilled/aborted by the imported fixture. Production
observation start, capture, analyzers and measurement reader are unchanged; no
browser/capture/clock/model seams are patched. Test-owned SVG updates stand for
rendering progress, not hidden course state or new observation requests.

This exercises the window's terminal handoff contract, not an executing stellar
or planet policy: an unfinished window cannot dispatch the parent's downstream
spectrum/positive stages. Native execution requires an explicitly released slot.
"""
# ruff: noqa: F811

import hashlib
import json
import os

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401

from habfly.browser_planet_window import PlanetWindowSession
from habfly.planet_window_measurements import load_planet_window_measurements

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must release the real browser before this intercepted Chromium gate",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def paint(frame, *, end, dips):
    """Fixture-only rendering progression, never called by a production owner."""
    frame.evaluate("shape=>window.fixturePaint(shape)", {"end": end, "dips": dips})


@pytest.fixture
def growing_window(window_page, tmp_path):
    page, frame = window_page
    frame.evaluate("""()=>{
      const chart=document.querySelector('#chart'),ns='http://www.w3.org/2000/svg';
      chart.querySelector('title').remove();
      const heading=document.createElementNS(ns,'text');
      heading.setAttribute('x',30);heading.setAttribute('y',10);
      heading.setAttribute('font-size',5);
      heading.textContent='Normalized Flux Days Observed';
      chart.prepend(heading);
      chart.querySelector('#trace').remove();
      const lines=document.createElementNS(ns,'g');lines.id='fixture-trace';
      chart.querySelector('rect').after(lines);
      window.fixturePaint=({end,dips})=>{
        lines.replaceChildren();
        function line(x1,y1,x2,y2){
          const e=document.createElementNS(ns,'line');
          for(const [k,v] of Object.entries({x1,y1,x2,y2,stroke:'rgb(80,132,154)','stroke-width':1}))
            e.setAttribute(k,v);
          lines.append(e);
        }
        if(end!==null)line(30.5,20.5,end+.5,20.5);
        for(const x of dips)line(x+.5,20.5,x+.5,30.5);
      };
      window.fixturePaint({end:null,dips:[]});
      const duration=document.querySelector('input');duration.id='duration';duration.value='';
      const play=[...document.querySelectorAll('button')].find(e=>e.textContent==='Play');
      play.outerHTML='<button id="play" aria-label="Play"><svg role="img" aria-label="Play" width="24" height="24"><rect width="24" height="24" fill="black"/></svg></button>';
      window.fixtureClicks=0;window.fixtureOtherClicks=0;window.fixtureAnswerInputs=0;
      function bind(button){button.onclick=()=>{
        window.fixtureClicks++;
        button.setAttribute('aria-label','Pause');button.querySelector('svg').setAttribute('aria-label','Pause');
        window.fixturePaint({end:110,dips:[60]});
      };}
      bind(document.querySelector('#play'));
      duration.onblur=()=>{
        const old=document.querySelector('#play'),current=old.cloneNode(true);
        old.replaceWith(current);bind(current);
      };
      for(const button of document.querySelectorAll('button:not(#play)'))
        button.addEventListener('click',()=>window.fixtureOtherClicks++);
      for(const input of document.querySelectorAll('input:not(#duration)'))
        input.addEventListener('input',()=>window.fixtureAnswerInputs++);
    }""")
    events = []
    session = PlanetWindowSession(
        page,
        config(),
        tmp_path / "window",
        run_history=tmp_path,
        select_no=True,
        # A shorter supported test interval only; no clock/deadline mutation.
        poll_interval=1,
        emit=lambda kind, payload: events.append((kind, payload)),
    )
    try:
        yield page, frame, session, events
    finally:
        session.abort()


def poll(page, session):
    page.wait_for_timeout(1050)
    return session.advance()


def assert_no_answers(page, frame, session):
    assert frame.evaluate("window.fixtureClicks") == 1
    assert frame.evaluate("window.fixtureSelections") == 0
    assert frame.evaluate("window.fixtureOtherClicks") == 0
    assert frame.evaluate("window.fixtureAnswerInputs") == 0
    assert frame.locator("#duration").input_value() == "5000"
    assert frame.get_by_role("combobox").input_value() == ""
    for name in ("line_shift", "brightness_drop", "period_days"):
        assert frame.locator("#" + name).input_value() == ""
    assert session.receipt is None and not session.state()["task_completed"]
    assert not page.get_by_role("checkbox").is_checked()


def test_native_early_dip_waits_for_endpoint_before_positive_handoff(growing_window, tmp_path):
    page, frame, session, events = growing_window
    assert session.advance()["phase"] == "observing"
    assert frame.evaluate("window.fixtureClicks") == 1
    started = read(tmp_path / "window/start/confirmed.json")
    assert started["play_click_dispatched_once"] and not started["observation_completed"]
    assert poll(page, session)["phase"] == "observing"
    assert not session.finished and session.analysis["status"] == "dip_observed"
    first = tmp_path / "window/progress-000"
    assert read(first / "report.json")["endpoint_visible"] is False
    pins = {str(path.relative_to(tmp_path)): sha(path) for path in first.iterdir() if path.is_file()}
    assert_no_answers(page, frame, session)
    assert not (tmp_path / "window/report.json").exists()

    # Enough dips alone are not enough: the immutable 5000-day endpoint is
    # required by the downstream measurement reader too.
    paint(frame, end=190, dips=[60, 120, 180])
    assert poll(page, session)["phase"] == "observing"
    assert not session.finished and session.analysis["status"] == "dip_observed"
    second = tmp_path / "window/progress-001"
    assert read(second / "report.json")["endpoint_visible"] is False
    pins.update({str(path.relative_to(tmp_path)): sha(path) for path in second.iterdir() if path.is_file()})
    assert_no_answers(page, frame, session)

    paint(frame, end=260, dips=[60, 120, 180])
    assert poll(page, session)["phase"] == "dip_observed"
    assert session.finished and session.polls == 3
    report = read(tmp_path / "window/report.json")
    assert report == session.report
    final_path = tmp_path / report["progress_path"]
    assert final_path == tmp_path / "window/progress-002/report.json"
    assert sha(final_path) == report["progress_sha256"]
    final = read(final_path)
    assert final["endpoint_visible"] is True
    assert final["status"] == "trace_reaches_requested_end"
    assert final["observation_completed"] is False
    measurement = load_planet_window_measurements(final_path, expected_report_sha256=sha(final_path))
    assert measurement["status"] == "approximate_reference_measurements"
    assert measurement["supported_dip_components"] == 3
    assert measurement["raster_endpoint_verified"] is True
    assert measurement["planet_decision"] is None
    for key in (
        "scientific_verified",
        "learned_perception",
        "training_label",
        "task_completed",
        "observation_completed",
    ):
        assert measurement[key] is False
    assert all(sha(tmp_path / name) == expected for name, expected in pins.items())
    assert session.scope["observation_limit_days"] == 5000
    assert session.scope["max_seconds"] == 600 and session.scope["max_polls"] == 60
    assert len(list((tmp_path / "observation-start-reservations").glob("*.json"))) == 1
    assert not any(kind == "action_proposed" for kind, _ in events)
    assert_no_answers(page, frame, session)
    before = len(events)
    session.advance()
    assert len(events) == before


def test_native_complete_window_with_two_dips_still_refuses_measurements(growing_window, tmp_path):
    page, frame, session, _ = growing_window
    assert session.advance()["phase"] == "observing"
    paint(frame, end=260, dips=[60, 150])
    assert poll(page, session)["phase"] == "dip_observed"
    assert session.finished
    report = session.report
    path = tmp_path / report["progress_path"]
    assert read(path)["endpoint_visible"] is True
    measurement = load_planet_window_measurements(path, expected_report_sha256=report["progress_sha256"])
    assert measurement["status"] == "measurement_error"
    assert measurement["error"]["code"] == "at_least_three_supported_dips_required"
    assert measurement["period_days"] is measurement["brightness_drop_percent"] is None
    assert measurement["planet_decision"] is None and measurement["task_completed"] is False
    assert_no_answers(page, frame, session)
