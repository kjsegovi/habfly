"""Intercepted fixture receipts only; no HabWorlds traffic or course labels."""
# ruff: noqa: F811

import json

import pytest
from test_browser_habitability_actions import menu_page  # noqa: F401
from test_browser_habitability_choice import RATIONALE
from test_browser_habitability_numeric import habitat_page  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401

import habfly.browser_habitability_save as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_habitability_actions import HabitabilityMenuSession
from habfly.browser_habitability_choice import select_habitability_reference
from habfly.browser_probe import inspect_page, save_probe


def read(path):
    return json.loads(path.read_text())


def prepare_phase_choice_sources(
    page, frame, root, *, phase="gas", choice="not_habitable", before_phase=None, before_choice=None
):
    """Synthetic chamber fixture plus real guarded native phase/choice adapters.

    Uses the fixture's existing gases, temperature, greenhouse and pressure;
    callers may first execute their own temperature/gas adapters. No real site
    is contacted and these synthetic phase observations are not course labels.
    """
    chamber = root / "chamber"
    capture = inspect_page(page, config())
    values = module._mapping(capture)["observation"]["values"]
    save_probe(capture, chamber / "task-before")
    save_probe(capture, chamber / "task-after")
    intent = {
        "pressure": values["measurements"]["pressure"]["display_text"],
        "pressure_unit": "atm",
        "temperature": values["readouts"]["surface_temp"]["display_text"],
        "temperature_unit": "K",
        "action_source": "reference_diagnostic",
        "task_answer_write": False,
    }
    result = {
        **intent,
        "phase": phase,
        "conditions_verified": True,
        "source": "visible_chamber_indicator",
        "task_completed": False,
        "visible_readback": {"atm": intent["pressure"], "K": intent["temperature"]},
        "icons": [
            {"phase": name, "paint": {"opacity": int(name == phase)}, "fully_exposed": True}
            for name in ("solid", "liquid", "gas")
        ],
    }
    for name, value in {
        "reserved": intent,
        "confirmed": result,
        "observed": result,
        "close-reserved": {"retry_allowed": False},
    }.items():
        (chamber / f"{name}.json").write_text(json.dumps(value))
    frame.locator("#surface").screenshot(path=str(chamber / "chamber.png"))
    phase_dir, choice_dir = root / "phase", root / "choice"
    if before_phase:
        before_phase()
    session = HabitabilityMenuSession(page, config(), phase_dir)
    try:
        session.water_phase_from_chamber(chamber)
    finally:
        session.close()
    if before_choice:
        before_choice()
    frame.locator("body").evaluate("""e=>{
      const style=document.createElement('style');style.textContent=`
        .choice label.picked{border:1px solid rgb(255,255,255)}
        .choice label.picked::after{opacity:1;background:rgb(255,255,255)}`;
      document.head.append(style);
      const labels=[...document.querySelectorAll('.choice label')];
      labels[0].classList.add('picked');labels.forEach(label=>label.onclick=()=>{
        window.choiceWrites=(window.choiceWrites||0)+1;
        labels.forEach(e=>e.classList.remove('picked'));label.classList.add('picked');
      });
    }""")
    select_habitability_reference(
        page, config(), choice_dir, phase_record=phase_dir, name=choice, rationale=RATIONALE
    )
    return phase_dir, choice_dir


@pytest.fixture
def ready(menu_page, tmp_path, request):
    page, frame = menu_page
    frame.locator("#temperature").fill("729.4")
    frame.locator("#temperature").press("Tab")
    frame.locator("#absorption").evaluate("e=>e.textContent='46.88'")
    frame.locator("#gases").evaluate("e=>e.innerHTML='<option selected>CO2</option>'")
    frame.locator("#greenhouse").select_option("Moderate")
    frame.get_by_role("button", name="Save", exact=True).evaluate("""e=>{
      const footer=document.createElement('div'),notice=document.createElement('div');
      notice.id='save-notice';e.before(footer);footer.append(notice,e);
      e.onclick=()=>{window.saveClicks=(window.saveClicks||0)+1;notice.textContent='Data saved'};
    }""")
    phase_name, decision, *footer_stage = getattr(request, "param", ("gas", "not_habitable"))
    callbacks = {}
    if footer_stage:
        callbacks[footer_stage[0]] = lambda: frame.locator("#save-notice").evaluate(
            "e=>e.textContent='Data saved'"
        )
    phase, choice = prepare_phase_choice_sources(
        page, frame, tmp_path, phase=phase_name, choice=decision, **callbacks
    )
    return page, frame, phase, choice


def run(ready, tmp_path, name="save", **kwargs):
    page, _, phase, choice = ready
    return module.save_habitability_work(
        page, config(), tmp_path / name, run_history=tmp_path, phase_dir=phase, choice_dir=choice, **kwargs
    )


def test_fresh_save_is_one_click_not_task_science_or_submission(ready, tmp_path):
    page, frame, phase, choice = ready
    result = run(ready, tmp_path)
    assert result["mode"] == module.MODE and result["choice"] == "not_habitable"
    assert result["choice_provenance"] == "reference_prediction"
    assert result["save_click_delivered"] and result["answers_unchanged"] and result["choice_paint_unchanged"]
    assert result["data_saved_notice_observed"] and not result["notice_was_already_present"]
    assert all(
        result[k] is False
        for k in (
            "task_completed",
            "scientific_verified",
            "correctness_verified",
            "cross_session_persistence_verified",
            "course_completion_verified",
            "assessed",
            "score_updated",
            "submitted",
            "hidden_values_inspected",
            "learned_habitability_decision",
            "automatic_retry",
        )
    )
    assert frame.evaluate("window.saveClicks") == 1
    assert frame.evaluate("window.choiceWrites") == 1
    assert not page.get_by_role("checkbox").is_checked()
    book, source = module._load_sources(tmp_path, phase, choice)
    assert book.hashes == result["source_sha256"] and source["choice"]["choice"] == result["choice"]
    assert (tmp_path / "save/pre-dispatch-observation/observation.json").exists()
    assert len(list((tmp_path / "habitability-save-reservations").iterdir())) == 1
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        run(ready, tmp_path, "another-output")
    assert frame.evaluate("window.saveClicks") == 1


@pytest.mark.parametrize(
    "ready", [("solid", "not_habitable"), ("liquid", "not_habitable"), ("liquid", "habitable")], indirect=True
)
def test_save_preserves_explicit_decision_not_inferred_from_water(ready, tmp_path):
    result = run(ready, tmp_path)
    assert result["choice"] == read(ready[3] / "confirmed.json")["choice"]
    assert not result["scientific_verified"] and not result["task_completed"]


@pytest.mark.parametrize(
    "ready",
    [("gas", "not_habitable", "before_phase"), ("gas", "not_habitable", "before_choice")],
    indirect=True,
)
def test_known_footer_transition_between_sources_is_not_answer_change(ready, tmp_path):
    ready[1].locator("#save-notice").evaluate("e=>e.textContent=''")
    result = run(ready, tmp_path)
    assert result["data_saved_notice_observed"] and result["answers_unchanged"]
    assert not result["notice_was_already_present"]


@pytest.mark.parametrize(
    "field,value",
    [("readback_verified", 1), ("max_choice_clicks", True), ("learned_habitability_decision", 0)],
)
def test_receipt_flags_are_not_coerced(ready, tmp_path, field, value):
    path = ready[3] / "confirmed.json"
    record = read(path)
    record[field] = value
    path.write_text(json.dumps(record))
    if field != "readback_verified":
        intent_path = ready[3] / "reserved.json"
        intent = read(intent_path)
        intent[field] = value
        intent_path.write_text(json.dumps(intent))
    with pytest.raises(BrowserSafetyStop):
        run(ready, tmp_path)
    assert not read(tmp_path / "save/stopped.json")["reservation_created"]


@pytest.mark.parametrize("source", ["choice", "phase", "chamber", "png", "close", "task_after", "stopped"])
def test_invalid_sources_never_reserve_or_click(ready, tmp_path, source):
    if source == "stopped":
        (tmp_path / "chamber/stopped.json").write_text("{}")
    else:
        paths = {
            "choice": "choice/confirmed.json",
            "phase": "phase/confirmed.json",
            "chamber": "chamber/confirmed.json",
            "png": "chamber/chamber.png",
            "close": "chamber/close-reserved.json",
            "task_after": "chamber/task-after/observation.json",
        }
        (tmp_path / paths[source]).write_text("{}")
    with pytest.raises(BrowserSafetyStop):
        run(ready, tmp_path)
    stopped = read(tmp_path / "save/stopped.json")
    assert not stopped["reservation_created"] and not stopped["save_may_have_occurred"]
    assert not (tmp_path / "habitability-save-reservations").exists()
    assert ready[1].evaluate("window.saveClicks||0") == 0


@pytest.mark.parametrize(
    "change",
    [
        "temperature",
        "surface",
        "gas",
        "empty_gas",
        "greenhouse",
        "empty_greenhouse",
        "phase",
        "pressure",
        "star",
        "paint",
        "outer",
    ],
)
def test_changed_live_fields_or_paint_never_click(ready, tmp_path, change):
    page, frame, _, _ = ready
    if change == "outer":
        page.get_by_role("checkbox").check()
    else:
        code = {
            "temperature": "document.querySelector('#temperature').value='300'",
            "surface": "document.querySelector('#surface').textContent='300'",
            "gas": "document.querySelector('#gases').innerHTML='<option selected>CH4</option>'",
            "empty_gas": "document.querySelector('#gases').innerHTML='<option selected></option>'",
            "greenhouse": "document.querySelector('#greenhouse').value='Weak'",
            "empty_greenhouse": "document.querySelector('#greenhouse').value=''",
            "phase": "document.querySelector('#phase').value='Liquid'",
            "pressure": "document.body.innerHTML=document.body.innerHTML.replace('atm) 9','atm) 8')",
            "star": "document.body.innerHTML=document.body.innerHTML.replace('JYREMIS','OTHER')",
            "paint": "document.querySelector('.picked').classList.remove('picked')",
        }[change]
        frame.evaluate(code)
    with pytest.raises(BrowserSafetyStop):
        run(ready, tmp_path)
    assert ready[1].evaluate("window.saveClicks||0") == 0
    assert not read(tmp_path / "save/stopped.json")["reservation_created"]


@pytest.mark.parametrize("behavior", ["clear", "never_clear", "reappear_during_guard"])
def test_existing_notice_settles_only_before_reservation(ready, tmp_path, monkeypatch, behavior):
    _, frame, _, _ = ready
    original, calls = HabitabilityMenuSession.current, 0
    frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")

    def current(self):
        nonlocal calls
        result = original(self)
        calls += 1
        if behavior == "clear" and calls == 1:
            frame.locator("#save-notice").evaluate("e=>setTimeout(()=>e.textContent='',100)")
        elif behavior == "reappear_during_guard":
            if calls == 1:
                frame.locator("#save-notice").evaluate("e=>e.textContent=''")
            elif calls == 2:
                frame.locator("#save-notice").evaluate(
                    "e=>{e.textContent='Data saved';setTimeout(()=>e.textContent='',150)}"
                )
        return result

    monkeypatch.setattr(HabitabilityMenuSession, "current", current)
    if behavior == "never_clear":
        with pytest.raises(BrowserSafetyStop, match="preflight_timeout"):
            run(ready, tmp_path, settle_timeout_seconds=0.5)
        assert not read(tmp_path / "save/stopped.json")["reservation_created"]
        assert not (tmp_path / "habitability-save-reservations").exists()
    else:
        assert run(ready, tmp_path)["data_saved_notice_observed"]
        assert frame.evaluate("window.saveClicks") == 1


@pytest.mark.parametrize("change", ["notice", "source", "control", "answer", "cancel"])
def test_postclaim_change_stops_once_and_retains_claim(ready, tmp_path, monkeypatch, change):
    _, frame, _, choice = ready
    original, done = module.persist_json, False
    cancelled = False

    def persist(path, value):
        nonlocal done, cancelled
        original(path, value)
        if path == tmp_path / "save/reserved.json" and not done:
            done = True
            if change == "notice":
                frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")
            elif change == "source":
                (choice / "confirmed.json").write_text("{}")
            elif change == "control":
                frame.get_by_role("button", name="Save").evaluate("e=>e.replaceWith(e.cloneNode(true))")
            elif change == "answer":
                frame.locator("#temperature").evaluate("e=>e.value='300'")
            else:
                cancelled = True

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop):
        run(ready, tmp_path, cancelled=lambda: cancelled)
    stopped = read(tmp_path / "save/stopped.json")
    assert stopped["reservation_created"] and not stopped["save_may_have_occurred"]
    assert not stopped["automatic_retry"] and not (tmp_path / "save/dispatch.json").exists()
    assert len(list((tmp_path / "habitability-save-reservations").iterdir())) == 1
    assert frame.evaluate("window.saveClicks||0") == 0


@pytest.mark.parametrize("change", ["no_ack", "surface", "phase", "paint", "dialog", "source", "cancel"])
def test_uncertain_dispatch_never_retries_or_completes(ready, tmp_path, monkeypatch, change):
    _, frame, _, choice = ready
    body = {
        "no_ack": "",
        "surface": "document.querySelector('#surface').textContent='99'",
        "phase": "document.querySelector('#phase').value='Solid'",
        "paint": "document.querySelector('.picked').classList.remove('picked')",
        "dialog": "confirm('private fixture message')",
        "source": "",
        "cancel": "",
    }[change]
    frame.get_by_role("button", name="Save").evaluate(
        """(e,body)=>e.onclick=()=>{
        window.saveClicks=(window.saveClicks||0)+1;
        if(body!=='no_ack')document.querySelector('#save-notice').textContent='Data saved';
        if(body!=='no_ack')new Function(body)();
    }""",
        "no_ack" if change == "no_ack" else body,
    )
    original = module._notice

    def notice(session, handle):
        value = original(session, handle)
        if change == "source" and (tmp_path / "save/dispatch.json").exists():
            (choice / "confirmed.json").write_text("{}")
        return value

    monkeypatch.setattr(module, "_notice", notice)
    with pytest.raises(BrowserSafetyStop):
        run(
            ready,
            tmp_path,
            timeout_seconds=0.1,
            cancelled=lambda: change == "cancel" and (tmp_path / "save/dispatch.json").exists(),
        )
    stopped = read(tmp_path / "save/stopped.json")
    assert stopped["reservation_created"] and stopped["save_may_have_occurred"]
    assert not stopped["task_completed"] and not stopped["automatic_retry"]
    assert not (tmp_path / "save/confirmed.json").exists()
    assert frame.evaluate("window.saveClicks") == 1
    assert "private fixture" not in json.dumps(stopped)


def test_brief_fresh_notice_is_recorded_before_full_capture(ready, tmp_path, monkeypatch):
    _, frame, _, _ = ready
    frame.get_by_role("button", name="Save").evaluate("""e=>{
      const prior=e.onclick;e.onclick=()=>{prior();setTimeout(()=>document.querySelector('#save-notice').textContent='',150)};
    }""")
    original = HabitabilityMenuSession.current

    def current(self):
        if (tmp_path / "save/acknowledgement.json").exists():
            self.page.wait_for_timeout(250)
        return original(self)

    monkeypatch.setattr(HabitabilityMenuSession, "current", current)
    assert run(ready, tmp_path)["data_saved_notice_observed"]
    assert frame.locator("#save-notice").inner_text() == ""


@pytest.mark.parametrize("kind", ["duplicate", "foreign", "transparent"])
def test_acknowledgement_must_be_one_exposed_own_footer(ready, tmp_path, kind):
    code = {
        "duplicate": "document.body.insertAdjacentHTML('beforeend','<div>Data saved</div>')",
        "foreign": "document.querySelector('#save-notice').textContent='';document.body.insertAdjacentHTML('beforeend','<div>Data saved</div>')",
        "transparent": "document.querySelector('#save-notice').style.opacity='0'",
    }[kind]
    ready[1].get_by_role("button", name="Save").evaluate(
        """(e,code)=>{
      const before=e.onclick;e.onclick=()=>{before();new Function(code)()};
    }""",
        code,
    )
    with pytest.raises(BrowserSafetyStop, match="unverified_acknowledgement"):
        run(ready, tmp_path)
    assert read(tmp_path / "save/stopped.json")["save_may_have_occurred"]
    assert not (tmp_path / "save/acknowledgement.json").exists()
    assert not (tmp_path / "save/confirmed.json").exists()


@pytest.mark.parametrize("when", ["before_sources", "preclaim", "last_probe"])
def test_preclaim_cancellation_never_consumes_save_attempt(ready, tmp_path, monkeypatch, when):
    original, cancelled = module.persist_json, when == "before_sources"

    def persist(path, value):
        nonlocal cancelled
        original(path, value)
        if when == "preclaim" and path == tmp_path / "save/pre-reservation-settled.json":
            cancelled = True

    monkeypatch.setattr(module, "persist_json", persist)
    original_notice = module._notice

    def notice(session, handle):
        nonlocal cancelled
        value = original_notice(session, handle)
        if when == "last_probe" and (tmp_path / "save/pre-reservation-settled.json").exists():
            cancelled = True
        return value

    monkeypatch.setattr(module, "_notice", notice)
    with pytest.raises(BrowserSafetyStop, match="cancelled"):
        run(ready, tmp_path, cancelled=lambda: cancelled)
    assert not read(tmp_path / "save/stopped.json")["save_may_have_occurred"]
    # A cancellation during the last native probe must be checked again before
    # creating the durable claim, not merely at the top of the settle cycle.
    assert not list((tmp_path / "habitability-save-reservations").glob("*.json"))


def test_covered_save_control_never_claims(ready, tmp_path):
    ready[1].get_by_role("button", name="Save").evaluate("""e=>{
      const r=e.getBoundingClientRect(),cover=document.createElement('div');
      cover.style=`position:fixed;left:${r.x}px;top:${r.y}px;width:${r.width}px;height:${r.height}px;background:black;z-index:999`;
      e.after(cover);
    }""")
    with pytest.raises(BrowserSafetyStop, match="control_unavailable"):
        run(ready, tmp_path)
    assert not read(tmp_path / "save/stopped.json")["reservation_created"]


def test_changed_source_during_settling_never_claims(ready, tmp_path, monkeypatch):
    original, calls = HabitabilityMenuSession.current, 0

    def current(self):
        nonlocal calls
        result = original(self)
        calls += 1
        if calls == 2:
            (ready[3] / "confirmed.json").write_text("{}")
        return result

    monkeypatch.setattr(HabitabilityMenuSession, "current", current)
    with pytest.raises(BrowserSafetyStop):
        run(ready, tmp_path)
    assert not read(tmp_path / "save/stopped.json")["reservation_created"]


@pytest.mark.parametrize("when", ["before_claim", "after_claim"])
def test_opened_and_closed_popup_cannot_escape_last_guard(ready, tmp_path, monkeypatch, when):
    original, triggered = module._notice, False

    def notice(session, handle):
        nonlocal triggered
        result = original(session, handle)
        claimed = (tmp_path / "save/reserved.json").exists()
        if not triggered and claimed == (when == "after_claim"):
            triggered = True
            session.page.evaluate("window.open('about:blank').close()")
            session.page.wait_for_timeout(50)
        return result

    monkeypatch.setattr(module, "_notice", notice)
    with pytest.raises(BrowserSafetyStop, match="unexpected_popup"):
        run(ready, tmp_path)
    assert triggered
    stopped = read(tmp_path / "save/stopped.json")
    assert stopped["reservation_created"] == (when == "after_claim")
    assert not stopped["save_may_have_occurred"] and not (tmp_path / "save/dispatch.json").exists()


@pytest.mark.parametrize(
    "kwargs", [{"timeout_seconds": True}, {"settle_timeout_seconds": 31}, {"cancelled": False}]
)
def test_invalid_budget_or_callback_never_creates_output(ready, tmp_path, kwargs):
    with pytest.raises(ValueError):
        run(ready, tmp_path, **kwargs)
    assert not (tmp_path / "save").exists()
