"""Intercepted Chromium fixtures; no live browser or project completion claims."""
# ruff: noqa: F811

import hashlib
import json
from copy import deepcopy

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import choose, source, window_page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_save import (
    CONDITIONAL_FIELDS,
    RAW_FIELDS,
    no_planet_visit_equivalence,
    resume_no_planet_save,
    save_no_planet_work,
)
from habfly.browser_numeric import screen_identity
from habfly.browser_planet import map_planet_capture
from habfly.browser_planet_numeric import PlanetNumericSession
from habfly.browser_stellar import SIMULATION_URL


def visit_capture():
    """Synthetic public capture; exact layout, no real project/answer source."""
    import yaml

    time = list(map(str, range(0, 10001, 1000)))
    flux = list(map(str, range(0, 101, 10)))
    atoms = [
        "img",
        "img",
        "button",
        "button [disabled]",
        {"text": "Fixture"},
        {"img": "1"},
        {"img": "2"},
        {"img": "3"},
        "img",
        {"text": "Observations Spectrum"},
        "button",
        "button",
        {"text": "656.3nm 656.2997nm 656.3003nm observe for"},
        {'textbox "0"': "10000"},
        {"text": "days"},
        'button "Play"',
        {"img": "Normalized Flux Days Observed " + " ".join(time + flux)},
        {"button": ["img"]},
        {"text": "doppler shift (nm)"},
        'textbox "0"',
        {"text": "brightness drop (%)"},
        'textbox "0"',
        {"text": "brightness drop period (days)"},
        'textbox "0"',
        {"text": "Your Reconstruction has planet?"},
        {"combobox": ['option "Yes"', 'option "No" [selected]']},
        {"text": "gas giant ice giant terrestrial"},
        "button",
        "button",
        'button "+"',
        'button "-"',
        "button",
        {"text": "STAR MASS (Ms) 2.118 STAR RADIUS (Rs) 1.793 ORBIT (years) 0.000"},
        'button "Save"',
    ]
    controls = []
    for role in ("button", "textbox", "combobox"):
        for atom in atoms:
            key = atom if isinstance(atom, str) else next(iter(atom))
            if key.split()[0] != role:
                continue
            control = {
                "id": f"simulation-0:c{len(controls)}",
                "role": role,
                "accessibility": yaml.safe_dump([atom], sort_keys=False).strip(),
                "enabled": "[disabled]" not in key,
                "actions": [],
                "protected": False,
            }
            if role != "button":
                control["value"] = "No" if role == "combobox" else atom[key] if isinstance(atom, dict) else ""
            controls.append(control)
    return {
        "schema_version": 1,
        "mode": "read_only_browser_preflight",
        "allow_submission": False,
        "actions_executed": 0,
        "ignored_frame_urls": [],
        "outer_controls": [],
        "frames": [
            {
                "id": "simulation-0",
                "url": SIMULATION_URL,
                "accessibility": yaml.safe_dump(atoms, sort_keys=False, width=100000),
                "controls": controls,
                "text": "FIXTURE\n1\n2\n3\nFIXTURE\nSTAR COLLECTED\nVIEW STAR DATA\nOBSERVATIONS\n"
                "SPECTRUM\nNORMALIZED FLUX\nDAYS OBSERVED\n" + "\n".join(time + flux) + "\nSave",
            },
            {
                "id": "widget",
                "url": "https://fixture.invalid",
                "text": "Score",
                "accessibility": "- text: Score",
                "controls": [],
            },
        ],
    }


def revisit_capture(original):
    current = deepcopy(original)
    frame = current["frames"][0]
    old = " ".join(map(str, range(0, 10001, 1000)))
    new = " ".join(map(str, range(0, 5001, 500)))
    frame["accessibility"] = (
        frame["accessibility"].replace(old, new).replace("- button [disabled]", "- button", 1)
    )
    frame["text"] = frame["text"].replace(old.replace(" ", "\n"), new.replace(" ", "\n"))
    frame["text"] = frame["text"].replace("3\nFIXTURE\nSTAR COLLECTED", "3\nOTHER\nSTAR COLLECTED")
    frame["controls"][1].update(enabled=True, accessibility="- button")
    return current


def compare_visit(original, current):
    return no_planet_visit_equivalence(
        original,
        map_planet_capture(original, capture_sha256=screen_identity(original)),
        current,
        map_planet_capture(current, capture_sha256=screen_identity(current)),
    )


def test_cross_visit_accepts_only_explicit_view_differences():
    original = visit_capture()
    before = deepcopy(original)
    current = revisit_capture(original)
    result = compare_visit(original, current)
    assert [change["view"] for change in result["differences"]] == [
        "second_header_navigation_enabled",
        "known_time_axis_reset",
        "ax_absent_collected_tooltip",
    ]
    assert result["original_screen_sha256"] == screen_identity(original)
    assert result["current_screen_sha256"] == screen_identity(current)
    assert result["current_chart_measurements_verified"] is result["scientific_verified"] is False
    assert original == before
    assert compare_visit(original, deepcopy(original)) is None


@pytest.mark.parametrize(
    "change",
    [
        "flux",
        "panned",
        "nonlinear",
        "duration",
        "answer_zero",
        "unit",
        "mass",
        "radius",
        "star",
        "has_planet",
        "header_label",
        "header_protected",
        "other_button",
        "extra_control",
        "visible_tooltip",
        "raw_feedback",
        "raw_answer",
        "outer",
        "widget",
        "tooltip_markup",
        "tooltip_removed",
        "reverse",
    ],
)
def test_cross_visit_rejects_every_other_difference(change):
    original = visit_capture()
    current = revisit_capture(original)
    frame = current["frames"][0]
    if change in {"flux", "panned", "nonlinear"}:
        old, new = {
            "flux": ("90 100", "90 101"),
            "panned": ("0 500 1000", "5 505 1005"),
            "nonlinear": ("0 500 1000", "0 501 1000"),
        }[change]
        frame["accessibility"] = frame["accessibility"].replace(old, new)
    elif change in {"duration", "answer_zero", "has_planet"}:
        control = frame["controls"][12 if change == "duration" else 13 if change == "answer_zero" else 16]
        control["value"] = "5000" if change == "duration" else "0" if change == "answer_zero" else "Yes"
    elif change in {"unit", "mass", "radius", "star"}:
        old, new = {
            "unit": ("shift (nm)", "shift (m)"),
            "mass": ("2.118", "2.119"),
            "radius": ("1.793", "1.794"),
            "star": ("Fixture", "Other"),
        }[change]
        frame["accessibility"] = frame["accessibility"].replace(old, new)
    elif change == "header_label":
        frame["controls"][1]["accessibility"] = '- button "Delete"'
    elif change == "header_protected":
        frame["controls"][1]["protected"] = True
    elif change == "other_button":
        frame["controls"][2]["enabled"] = False
    elif change == "extra_control":
        frame["controls"].append({**frame["controls"][0], "id": "simulation-0:c17"})
    elif change == "visible_tooltip":
        frame["accessibility"] += "- text: OTHER STAR COLLECTED VIEW STAR DATA\n"
    elif change in {"raw_feedback", "raw_answer"}:
        frame["text"] += "\n" + ("unexpected feedback" if change == "raw_feedback" else "9")
    elif change == "outer":
        current["outer_controls"].append({"role": "checkbox", "checked": True})
    elif change == "widget":
        current["frames"][1]["text"] = "Assessed"
    elif change == "tooltip_markup":
        frame["text"] = frame["text"].replace("STAR COLLECTED", "STAR DELETED")
    elif change == "tooltip_removed":
        frame["text"] = frame["text"].replace("OTHER\nSTAR COLLECTED\nVIEW STAR DATA\n", "")
    else:
        original, current = current, original
    with pytest.raises((BrowserSafetyStop, ValueError)):
        compare_visit(original, current)


@pytest.fixture
def no_save_page(window_page, tmp_path):
    page, frame = window_page
    frame.get_by_role("button", name="Save", exact=True).evaluate("""e=>{
      const notice=document.createElement('div');notice.id='save-notice';
      const footer=document.createElement('div');e.before(footer);footer.append(notice,e);
      window.fixtureSaves=0;e.onclick=()=>{window.fixtureSaves++;notice.textContent='Data saved'};
    }""")
    choose(window_page, tmp_path, source(window_page, tmp_path))
    path = tmp_path / "choice/confirmed.json"
    options = {
        "run_history": tmp_path,
        "choice_path": path,
        "choice_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return page, frame, options


@pytest.fixture
def revisit_no_save_page(window_page, tmp_path):
    """Intercepted screen with public header/covered tooltip and real native inputs."""
    _, frame = window_page
    frame.evaluate("""()=>{
      const star=document.body.firstElementChild;
      star.insertAdjacentHTML('beforebegin',
        '<svg role=img width=1 height=1></svg><svg role=img width=1 height=1></svg>'+
        '<button id=previous></button><button id=next disabled></button>');
      star.nextElementSibling.outerHTML='<svg role=img height=10><text y=9>1</text></svg>'+
        '<svg role=img height=10><text y=9>2</text></svg>'+
        '<svg role=img height=10><text y=9>3</text></svg><svg role=img width=1 height=1></svg>'+
        '<div aria-hidden=true id=covered-tip>JYREMIS<br>STAR COLLECTED<br>VIEW STAR DATA</div>';
      const observations=[...document.querySelectorAll('div')].find(e=>e.textContent==='Observations Spectrum');
      observations.innerHTML='Observations<br>Spectrum';observations.style.textTransform='uppercase';
      const chart=document.querySelector('#chart');chart.querySelectorAll('text').forEach(e=>e.remove());
      chart.querySelector('title').remove();
      for(const [label,y] of [['Normalized Flux',10],['Days Observed',15]]){
        const title=document.createElementNS('http://www.w3.org/2000/svg','text');
        title.textContent=label;title.setAttribute('y',y);title.style.textTransform='uppercase';
        title.setAttribute('font-size',4);chart.append(title);
      }
      for(const [className,max] of [['time',10000],['flux',100]]){
        for(let i=0;i<=10;i++){
          const t=document.createElementNS('http://www.w3.org/2000/svg','text');
          t.setAttribute('class',className);t.textContent=String(max*i/10);
          t.setAttribute('x',className==='time'?30.5+23*i:25);
          t.setAttribute('y',className==='time'?165:151-13.05*i);
          t.setAttribute('text-anchor',className==='time'?'middle':'end');
          t.setAttribute('font-size',4);t.setAttribute('fill','white');chart.append(t);
          const top=chart.getBoundingClientRect().top,desired=Number(t.getAttribute('y')),b=t.getBoundingClientRect();
          t.setAttribute('y',desired+desired-(b.top+b.height/2-top));
        }
      }
      document.querySelector('input').value='10000';
    }""")
    return no_save_page.__wrapped__(window_page, tmp_path)


def apply_revisit(frame):
    frame.evaluate("""()=>{
      document.querySelector('#next').disabled=false;
      document.querySelector('#covered-tip').innerHTML='OTHER<br>STAR COLLECTED<br>VIEW STAR DATA';
      document.querySelectorAll('#chart .time').forEach((e,i)=>e.textContent=String(i*500));
    }""")


@pytest.mark.parametrize("banner", [False, True])
def test_read_only_revisit_reconciliation_preserves_source_and_no_click_claims(
    revisit_no_save_page, tmp_path, monkeypatch, banner
):
    from playwright.sync_api import ElementHandle, Locator

    from habfly.browser_no_planet_save import reconcile_no_planet_autosave

    page, frame, _ = revisit_no_save_page
    predispatched(revisit_no_save_page, tmp_path, monkeypatch)
    original = {p: p.read_bytes() for p in (tmp_path / "save").rglob("*.json")}
    apply_revisit(frame)
    if not banner:
        frame.locator("#save-notice").evaluate("e=>e.textContent=''")

    def forbidden(*_, **__):
        raise AssertionError("Read-only reconciliation attempted a browser action")

    with monkeypatch.context() as context:
        for cls in (ElementHandle, Locator):
            for method in ("click", "fill", "select_option", "press"):
                context.setattr(cls, method, forbidden)
        receipt = reconcile_no_planet_autosave(
            page,
            config(),
            tmp_path / "save",
            tmp_path / "review",
            run_history=tmp_path,
            timeout_seconds=0,
        )
    assert len(receipt["historical_view_equivalence"]["differences"]) == 3
    assert receipt["save_acknowledgement_verified"] is banner
    assert receipt["save_click_delivered"] is receipt["task_completed"] is False
    assert all(path.read_bytes() == raw for path, raw in original.items())
    assert frame.evaluate("window.fixtureSaves") == 0


@pytest.mark.parametrize("change", ["time_axis", "header", "covered_tooltip", "answer", "class_paint"])
def test_revisit_never_relaxes_fresh_same_screen_guard(revisit_no_save_page, tmp_path, monkeypatch, change):
    import habfly.browser_no_planet_save as module

    page, frame, _ = revisit_no_save_page
    predispatched(revisit_no_save_page, tmp_path, monkeypatch)
    apply_revisit(frame)
    original = module.persist_json

    def persist(path, payload):
        original(path, payload)
        if path.name == "acknowledgement.json":
            script = {
                "time_axis": "document.querySelectorAll('#chart .time').forEach((e,i)=>e.textContent=String(i*1000))",
                "header": "document.querySelector('#next').disabled=true",
                "covered_tooltip": "document.querySelector('#covered-tip').innerHTML='THIRD<br>STAR COLLECTED<br>VIEW STAR DATA'",
                "answer": "document.querySelector('#period_days').value='0'",
                "class_paint": "document.querySelector('.choice label').style.borderColor='white'",
            }[change]
            frame.evaluate("()=>{" + script + "}")

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop):
        module.reconcile_no_planet_autosave(
            page,
            config(),
            tmp_path / "save",
            tmp_path / "review",
            run_history=tmp_path,
            timeout_seconds=0,
        )
    assert not (tmp_path / "review/confirmed.json").exists()
    assert frame.evaluate("window.fixtureSaves") == 0


def save(fixture, tmp_path, **kwargs):
    page, _, options = fixture
    return save_no_planet_work(page, config(), tmp_path / "save", **options, **kwargs)


def test_explicit_no_save_ack_preserves_blank_raw_fields_and_conditional_absence(no_save_page, tmp_path):
    page, frame, options = no_save_page
    receipt = save(no_save_page, tmp_path)
    assert frame.evaluate("window.fixtureSaves") == 1
    assert frame.get_by_role("combobox").input_value() == "No"
    assert all(frame.locator("#" + name).input_value() == "" for name in RAW_FIELDS)
    assert all(not frame.locator("#" + name).is_visible() for name in CONDITIONAL_FIELDS)
    assert receipt["visible_blank_raw_fields"] == list(RAW_FIELDS)
    assert receipt["conditionally_absent_derived_fields"] == list(CONDITIONAL_FIELDS)
    assert receipt["data_saved_notice_observed"] and receipt["answers_unchanged"]
    assert receipt["duration_and_class_paint_unchanged"]
    for name in (
        "notice_was_already_present",
        "hidden_values_inspected",
        "hidden_values_blank_verified",
        "cross_session_persistence_verified",
        "correctness_verified",
        "course_completion_verified",
        "task_completed",
        "automatic_retry",
        "assessed",
        "score_updated",
        "submitted",
    ):
        assert receipt[name] is False
    assert not page.get_by_role("checkbox").is_checked()
    assert len(list((tmp_path / "no-planet-save-reservations").glob("*.json"))) == 1
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        save_no_planet_work(page, config(), tmp_path / "new-output", **options)
    assert frame.evaluate("window.fixtureSaves") == 1


@pytest.mark.parametrize("when", ["initial", "after_full_guard"])
def test_fresh_save_waits_only_before_claim_for_notice_to_clear(no_save_page, tmp_path, monkeypatch, when):
    import habfly.browser_no_planet_save as module

    _, frame, _ = no_save_page
    original_probe, original_capture = module._record_notice_probe, module.save_probe

    def probe(session, handle, directory, stage, report):
        if when == "initial" and stage == "before-reservation":
            frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")
        if stage == "settle-wait-001":
            assert not list((tmp_path / "no-planet-save-reservations").glob("*.json"))
            frame.locator("#save-notice").evaluate("e=>e.textContent=''")
        return original_probe(session, handle, directory, stage, report)

    def capture(report, path, *args, **kwargs):
        result = original_capture(report, path, *args, **kwargs)
        if when == "after_full_guard" and path == tmp_path / "save/pre-reservation-observations/000":
            assert not (tmp_path / "save/reserved.json").exists()
            frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")
        return result

    monkeypatch.setattr(module, "_record_notice_probe", probe)
    monkeypatch.setattr(module, "save_probe", capture)
    receipt = save(no_save_page, tmp_path)
    settled = json.loads((tmp_path / "save/pre-reservation-settled.json").read_text())
    assert settled["read_only_cycles"] == 2 and settled["notice_present"] is False
    assert settled["reservation_created"] is False and settled["save_click_dispatched"] is False
    assert not settled["later_notice_excluded"] and not settled["fresh_save_acknowledgement_verified"]
    assert receipt["data_saved_notice_observed"] and receipt["notice_was_already_present"] is False
    assert frame.evaluate("window.fixtureSaves") == 1
    assert len(list((tmp_path / "no-planet-save-reservations").glob("*.json"))) == 1


def test_fresh_save_never_clearing_notice_times_out_without_claim(no_save_page, tmp_path, monkeypatch):
    import habfly.browser_no_planet_save as module

    _, frame, _ = no_save_page
    original_probe, original_time = module._record_notice_probe, module.time.monotonic
    expired = False

    def probe(session, handle, directory, stage, report):
        nonlocal expired
        if stage == "before-reservation":
            frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")
        value = original_probe(session, handle, directory, stage, report)
        expired = True
        return value

    monkeypatch.setattr(module, "_record_notice_probe", probe)
    monkeypatch.setattr(module.time, "monotonic", lambda: original_time() + (31 if expired else 0))
    with pytest.raises(BrowserSafetyStop, match="preflight_timeout"):
        save(no_save_page, tmp_path)
    assert frame.evaluate("window.fixtureSaves") == 0
    assert not list((tmp_path / "no-planet-save-reservations").glob("*.json"))
    assert not (tmp_path / "save/reserved.json").exists()
    stop = json.loads((tmp_path / "save/stopped.json").read_text())
    assert not stop["reservation_created"] and not stop["save_may_have_occurred"]


@pytest.mark.parametrize("change", ["answer", "duration", "button", "source", "cancel", "callback_error"])
def test_read_only_settling_rejects_changes_and_cancellation_before_claim(
    no_save_page, tmp_path, monkeypatch, change
):
    import habfly.browser_no_planet_save as module

    _, frame, options = no_save_page
    original_probe = module._record_notice_probe
    stop_requested = False

    def cancelled():
        if stop_requested and change == "callback_error":
            raise RuntimeError("PRIVATE callback details")
        return stop_requested and change == "cancel"

    def probe(session, handle, directory, stage, report):
        nonlocal stop_requested
        if stage == "before-reservation":
            frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")
        elif stage == "settle-wait-001":
            frame.locator("#save-notice").evaluate("e=>e.textContent=''")
            if change == "answer":
                frame.locator("#period_days").fill("7")
            elif change == "duration":
                frame.get_by_role("textbox").first.fill("3000")
            elif change == "button":
                frame.get_by_role("button", name="Save").evaluate("e=>e.replaceWith(e.cloneNode(true))")
            elif change == "source":
                options["choice_path"].write_text("{}")
            stop_requested = True
        return original_probe(session, handle, directory, stage, report)

    monkeypatch.setattr(module, "_record_notice_probe", probe)
    with pytest.raises(BrowserSafetyStop):
        save(no_save_page, tmp_path, cancelled=cancelled)
    assert frame.evaluate("window.fixtureSaves") == 0
    assert not list((tmp_path / "no-planet-save-reservations").glob("*.json"))
    assert not (tmp_path / "save/reserved.json").exists()
    assert "PRIVATE" not in (tmp_path / "save/stopped.json").read_text()


def test_cancellation_after_reservation_is_permanent_without_dispatch(no_save_page, tmp_path, monkeypatch):
    import habfly.browser_no_planet_save as module

    original = module.persist_json
    cancel = False

    def persist(path, payload):
        nonlocal cancel
        original(path, payload)
        if path == tmp_path / "save/reserved.json":
            cancel = True

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop, match="save_cancelled"):
        save(no_save_page, tmp_path, cancelled=lambda: cancel)
    assert (tmp_path / "save/reserved.json").exists()
    assert not (tmp_path / "save/dispatch.json").exists()
    page, frame, options = no_save_page
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        save_no_planet_work(page, config(), tmp_path / "other-output", **options)
    assert frame.evaluate("window.fixtureSaves") == 0


@pytest.mark.parametrize("value", [0, 31, True, "20", float("nan"), float("inf")])
def test_invalid_settlement_budget_is_rejected_before_any_output(no_save_page, tmp_path, value):
    with pytest.raises(ValueError):
        save(no_save_page, tmp_path, settle_timeout_seconds=value)
    assert not (tmp_path / "save").exists()


def test_allow_no_read_guard_never_enables_numeric_copy(no_save_page, tmp_path):
    page, frame, _ = no_save_page
    with pytest.raises(BrowserSafetyStop, match="yes_selection_required"):
        PlanetNumericSession(page, config(), tmp_path / "default")
    session = PlanetNumericSession(page, config(), tmp_path / "read", _allow_no_planet=True)
    try:
        with pytest.raises(BrowserSafetyStop, match="yes_selection_required"):
            session.copy("line_shift", "1", "nm", source="checkpoint")
        assert not session.attempted and frame.locator("#line_shift").input_value() == ""
    finally:
        session.close()


@pytest.mark.parametrize(
    "kind",
    [
        "stale_notice",
        "raw_zero",
        "raw_answer",
        "duration",
        "yes",
        "unset",
        "star",
        "paint",
        "disabled",
        "overlay",
        "modal",
        "auth",
        "external",
        "conditional_visible",
    ],
)
def test_incompatible_current_state_never_saves(no_save_page, tmp_path, kind):
    page, frame, _ = no_save_page
    if kind == "stale_notice":
        frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")
    elif kind in {"raw_zero", "raw_answer"}:
        frame.locator("#period_days").fill("0" if kind == "raw_zero" else "42")
    elif kind == "duration":
        frame.get_by_role("textbox").first.fill("4000")
    elif kind in {"yes", "unset"}:
        frame.get_by_role("combobox").evaluate("(e,v)=>e.value=v", "Yes" if kind == "yes" else "")
    elif kind == "star":
        frame.locator("div").first.evaluate("e=>e.innerText='OTHERSTAR'")
    elif kind == "paint":
        frame.locator("body").evaluate("""e=>e.insertAdjacentHTML('beforeend',
          '<style>.choice:first-of-type label::after{opacity:1;background:white}</style>')""")
        # Select one painted choice without relying on the relative sibling type.
        frame.locator(".choice").first.evaluate("""e=>{
          e.classList.add('selected');const s=document.createElement('style');
          s.textContent='.choice.selected label{border-color:white}.choice.selected label::after{opacity:1;background:white}';
          document.body.append(s)}""")
    elif kind == "disabled":
        frame.get_by_role("button", name="Save").evaluate("e=>e.disabled=true")
    elif kind == "overlay":
        frame.get_by_role("button", name="Save").evaluate("""e=>{
          const r=e.getBoundingClientRect(),d=document.createElement('div');
          d.style=`position:fixed;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;background:red`;
          e.after(d)}""")
    elif kind == "conditional_visible":
        frame.locator("#orbital_radius").evaluate("e=>e.parentElement.hidden=false")
    elif kind == "external":
        page.get_by_role("checkbox").check()
    else:
        frame.locator("body").evaluate(
            "(e,h)=>e.insertAdjacentHTML('beforeend',h)",
            "<div role=dialog>stop</div>" if kind == "modal" else "<input type=password>",
        )
    with pytest.raises(BrowserSafetyStop):
        save(
            no_save_page, tmp_path, settle_timeout_seconds=0.1 if kind in {"stale_notice", "disabled"} else 20
        )
    assert frame.evaluate("window.fixtureSaves") == 0
    assert not list((tmp_path / "no-planet-save-reservations").glob("*.json"))
    assert not json.loads((tmp_path / "save/stopped.json").read_text())["save_may_have_occurred"]


@pytest.mark.parametrize("kind", ["hash", "receipt", "intent", "canonical", "png", "after", "invalidated"])
def test_choice_provenance_is_hash_verified_before_save(no_save_page, tmp_path, kind):
    _, frame, options = no_save_page
    if kind == "hash":
        options["choice_sha256"] = "bad"
    elif kind == "receipt":
        options["choice_path"].write_text(options["choice_path"].read_text() + " ")
    elif kind == "intent":
        path = tmp_path / "choice/reserved.json"
        payload = json.loads(path.read_text())
        payload["numeric_writes"] = 1
        path.write_text(json.dumps(payload))
    elif kind == "canonical":
        next((tmp_path / "planet-window-choice-reservations").glob("*.json")).write_text("{}")
    elif kind == "png":
        (tmp_path / "choice/preselect-progress/chart.png").write_bytes(b"changed")
    elif kind == "after":
        (tmp_path / "choice/after/observation.json").write_text("{}")
    else:
        (tmp_path / "choice/invalidated.json").write_text("{}")
    with pytest.raises(BrowserSafetyStop):
        save(no_save_page, tmp_path)
    assert frame.evaluate("window.fixtureSaves") == 0
    assert not (tmp_path / "save/confirmed.json").exists()


@pytest.mark.parametrize("kind", ["answer", "duration", "replacement", "choice", "stale_notice"])
def test_reserved_preclick_change_stops_without_retry_in_another_output(
    no_save_page, tmp_path, monkeypatch, kind
):
    import habfly.browser_no_planet_save as module

    page, frame, options = no_save_page
    original = module.persist_json

    def persist(path, payload):
        original(path, payload)
        if path == tmp_path / "save/reserved.json":
            if kind == "answer":
                frame.locator("#period_days").fill("3")
            elif kind == "duration":
                frame.get_by_role("textbox").first.fill("3000")
            elif kind == "replacement":
                frame.get_by_role("button", name="Save").evaluate("e=>e.replaceWith(e.cloneNode(true))")
            elif kind == "choice":
                options["choice_path"].write_text("{}")
            else:
                frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop):
        save(no_save_page, tmp_path)
    stopped = json.loads((tmp_path / "save/stopped.json").read_text())
    assert stopped["reservation_created"] and not stopped["save_may_have_occurred"]
    with pytest.raises(BrowserSafetyStop):
        save_no_planet_work(page, config(), tmp_path / "different-output", **options)
    assert frame.evaluate("window.fixtureSaves") == 0


@pytest.mark.parametrize("kind", ["no_ack", "answer", "axis", "dialog", "replacement", "feedback", "escape"])
def test_uncertain_save_stays_reserved_and_never_claims_completion(no_save_page, tmp_path, kind):
    page, frame, options = no_save_page
    code = {
        "no_ack": "",
        "answer": "document.getElementById('period_days').value='5'",
        "axis": "document.querySelector('.time').textContent='5'",
        "dialog": "confirm('private dialog content')",
        "replacement": "e.replaceWith(e.cloneNode(true))",
        "feedback": "document.body.append('Unexpected failure')",
        "escape": "history.replaceState(null,'','/escaped')",
    }[kind]
    frame.get_by_role("button", name="Save").evaluate(
        """(e,body)=>e.onclick=()=>{
      window.fixtureSaves++;new Function('e',body)(e);
    }""",
        code,
    )
    with pytest.raises(BrowserSafetyStop):
        save(no_save_page, tmp_path, timeout_seconds=0.1)
    text = (tmp_path / "save/stopped.json").read_text()
    stopped = json.loads(text)
    assert stopped["reservation_created"] and stopped["save_may_have_occurred"]
    assert not stopped["automatic_retry"] and not stopped["task_completed"]
    assert "private dialog" not in text and not (tmp_path / "save/confirmed.json").exists()
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        save_no_planet_work(page, config(), tmp_path / "different-output", **options)
    assert frame.evaluate("window.fixtureSaves") == 1


def test_hidden_derived_values_are_not_read_or_certified_blank(no_save_page, tmp_path, monkeypatch):
    from playwright.sync_api import ElementHandle, Locator

    page, frame, _ = no_save_page
    frame.locator("#planet_mass").evaluate("e=>e.value='this is hidden and must not be inspected'")
    for cls in (ElementHandle, Locator):
        original = cls.input_value

        def visible_only(self, *args, original=original, **kwargs):
            assert self.is_visible(), "Hidden input values must not be read"
            return original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "input_value", visible_only)
    result = save(no_save_page, tmp_path)
    assert not result["hidden_values_inspected"] and not result["hidden_values_blank_verified"]
    assert page.get_by_role("checkbox").is_checked() is False


@pytest.mark.parametrize("delay_ms", [0, 80])
def test_brief_fresh_notice_is_persisted_before_slower_capture(no_save_page, tmp_path, monkeypatch, delay_ms):
    _, frame, _ = no_save_page
    frame.get_by_role("button", name="Save").evaluate(
        """(e,delay)=>e.onclick=()=>{
      window.fixtureSaves++;const n=document.getElementById('save-notice');
      setTimeout(()=>{n.textContent='Data saved';setTimeout(()=>n.textContent='',200)},delay);
    }""",
        delay_ms,
    )
    original = PlanetNumericSession.current

    def slow_current(self):
        if (tmp_path / "save/dispatch.json").exists():
            self.page.wait_for_timeout(350)
        return original(self)

    monkeypatch.setattr(PlanetNumericSession, "current", slow_current)
    result = save(no_save_page, tmp_path)
    assert result["data_saved_notice_observed"]
    assert (tmp_path / "save/acknowledgement.json").exists()
    assert frame.locator("#save-notice").inner_text() == ""


def predispatched(no_save_page, tmp_path, monkeypatch):
    import habfly.browser_no_planet_save as module

    _, frame, _ = no_save_page
    original = module.persist_json

    def persist(path, payload):
        original(path, payload)
        if path == tmp_path / "save/reserved.json":
            frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")

    with monkeypatch.context() as context:
        context.setattr(module, "persist_json", persist)
        with pytest.raises(BrowserSafetyStop, match="stale_acknowledgement"):
            save(no_save_page, tmp_path)
    assert frame.evaluate("window.fixtureSaves") == 0


def test_transient_preclick_notice_is_recorded_even_when_full_captures_have_no_banner(
    no_save_page, tmp_path, monkeypatch
):
    import habfly.browser_no_planet_save as module

    _, frame, _ = no_save_page
    original = module._notice
    injected = False

    def transient_notice(session, handle):
        nonlocal injected
        inject = (tmp_path / "save/reserved.json").exists() and not injected
        if inject:
            frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")
        present = original(session, handle)
        if inject:
            injected = True
            frame.locator("#save-notice").evaluate("e=>e.textContent=''")
        return present

    monkeypatch.setattr(module, "_notice", transient_notice)
    with pytest.raises(BrowserSafetyStop, match="stale_acknowledgement"):
        save(no_save_page, tmp_path)
    directory = tmp_path / "save"
    before = json.loads((directory / "before/observation.json").read_text())
    latest = json.loads((directory / "pre-dispatch-observation/observation.json").read_text())
    assert screen_identity(before) == screen_identity(latest)
    assert all("Data saved" not in report["frames"][0]["accessibility"] for report in (before, latest))
    prior = json.loads((directory / "footer-probes/before-reservation.json").read_text())
    actual = json.loads((directory / "footer-probes/after-reservation.json").read_text())
    assert prior["verdict"] == "absent" and actual["verdict"] == "present"
    assert actual["notice_present"] and actual["same_footer_and_exposure_verified"]
    assert actual["visible_text"] == "Data saved"
    assert actual["compared_capture_sha256"] == screen_identity(latest)
    assert actual["probe_started_monotonic_seconds"] <= actual["probe_finished_monotonic_seconds"]
    assert actual["diagnostic_only"] and not actual["compared_capture_is_simultaneous"]
    for flag in (
        "fresh_save_acknowledgement_verified",
        "save_click_dispatched",
        "retry_authorized",
        "task_completed",
    ):
        assert actual[flag] is False
    assert not (directory / "dispatch.json").exists() and not (directory / "acknowledgement.json").exists()
    module._predispatch_source(directory, tmp_path)  # Extra diagnostics do not turn it into a Save receipt.
    assert frame.evaluate("window.fixtureSaves") == 0


def test_last_verified_capture_is_the_actual_post_reservation_capture(no_save_page, tmp_path, monkeypatch):
    predispatched(no_save_page, tmp_path, monkeypatch)
    directory = tmp_path / "save"
    before = json.loads((directory / "before/observation.json").read_text())
    final_guard = json.loads((directory / "pre-dispatch-observation/observation.json").read_text())
    saved_last = json.loads((directory / "last-verified-observation/observation.json").read_text())
    assert screen_identity(before) != screen_identity(final_guard)
    assert screen_identity(saved_last) == screen_identity(final_guard)
    frame = next(row for row in final_guard["frames"] if row["url"] == SIMULATION_URL)
    assert "Data saved" in frame["accessibility"]
    assert json.loads((directory / "footer-probes/after-reservation.json").read_text())["notice_present"]


def test_footer_probe_rejection_records_no_acknowledgement_or_dispatch(no_save_page, tmp_path):
    import habfly.browser_no_planet_save as module

    _, frame, _ = no_save_page
    # The preexisting unknown visible text is normally rejected by the broader
    # source guard first. Exercise only the diagnostic wrapper's rejection seam.
    session = PlanetNumericSession(no_save_page[0], config(), tmp_path / "guard", _allow_no_planet=True)
    output = tmp_path / "diagnostic"
    output.mkdir()
    try:
        frame.locator("body").evaluate("e=>e.insertAdjacentHTML('beforeend','<div>Data saved</div>')")
        handle = frame.get_by_role("button", name="Save").element_handle()
        with pytest.raises(BrowserSafetyStop, match="unverified_no_planet_save_acknowledgement"):
            module._record_notice_probe(session, handle, output, "fixture-rejected", session.report)
        record = json.loads((output / "footer-probes/fixture-rejected.json").read_text())
        assert record["verdict"] == "rejected" and record["notice_present"] is None
        assert not record["save_click_dispatched"] and not record["fresh_save_acknowledgement_verified"]
        assert frame.evaluate("window.fixtureSaves") == 0
    finally:
        session.close()


@pytest.mark.parametrize("banner", [False, True])
def test_read_only_predispatch_reconciliation_needs_current_banner(
    no_save_page, tmp_path, monkeypatch, banner
):
    from playwright.sync_api import ElementHandle, Locator

    from habfly.browser_no_planet_save import reconcile_no_planet_autosave

    page, frame, options = no_save_page
    predispatched(no_save_page, tmp_path, monkeypatch)
    source = {p: p.read_bytes() for p in (tmp_path / "save").rglob("*.json")}
    if not banner:
        frame.locator("#save-notice").evaluate("e=>e.textContent=''")

    def forbidden(*_, **__):
        raise AssertionError("Read-only reconciliation must never act on the browser")

    with monkeypatch.context() as context:
        for cls in (ElementHandle, Locator):
            for method in ("click", "fill", "select_option", "press"):
                context.setattr(cls, method, forbidden)
        report = reconcile_no_planet_autosave(
            page, config(), tmp_path / "save", tmp_path / "review", run_history=tmp_path, timeout_seconds=0
        )
    assert report["save_acknowledgement_verified"] == banner
    assert report["data_saved_notice_observed"] == banner
    assert report["browser_actions"] == 0 and report["reservation_retained"]
    assert not report["save_click_delivered"] and not report["site_autosave_trigger_verified"]
    assert not report["task_completed"] and not report["cross_session_persistence_verified"]
    assert (tmp_path / "review/confirmed.json").exists() == banner
    assert (tmp_path / "review/blocked.json").exists() != banner
    assert all(p.read_bytes() == raw for p, raw in source.items())
    assert frame.evaluate("window.fixtureSaves") == 0
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        save_no_planet_work(page, config(), tmp_path / "cannot-retry", **options)


@pytest.mark.parametrize("change", ["dispatched", "truthy", "reason", "reservation", "answer", "star"])
def test_read_only_review_rejects_uncertain_or_changed_source(no_save_page, tmp_path, monkeypatch, change):
    from habfly.browser_no_planet_save import reconcile_no_planet_autosave

    page, frame, _ = no_save_page
    predispatched(no_save_page, tmp_path, monkeypatch)
    if change in {"dispatched", "truthy", "reason"}:
        path = tmp_path / "save/stopped.json"
        payload = json.loads(path.read_text())
        if change == "dispatched":
            payload["save_may_have_occurred"] = True
        elif change == "truthy":
            payload["save_may_have_occurred"] = 0
        else:
            payload["reason"] = "no_planet_save_acknowledgement_timeout"
        path.write_text(json.dumps(payload))
    elif change == "reservation":
        next((tmp_path / "no-planet-save-reservations").glob("*.json")).write_text("{}")
    elif change == "answer":
        frame.locator("#period_days").fill("5")
    else:
        frame.locator("div").first.evaluate("e=>e.innerText='OTHERSTAR'")
    with pytest.raises(BrowserSafetyStop):
        reconcile_no_planet_autosave(
            page, config(), tmp_path / "save", tmp_path / "review", run_history=tmp_path
        )
    assert not (tmp_path / "review/confirmed.json").exists()
    assert frame.evaluate("window.fixtureSaves") == 0


def resume(fixture, tmp_path, **kwargs):
    return resume_no_planet_save(fixture[0], config(), tmp_path / "save", run_history=tmp_path, **kwargs)


def test_same_intent_continuation_dispatches_once_and_retains_original_evidence(
    no_save_page, tmp_path, monkeypatch
):
    import habfly.browser_no_planet_save as module

    page, frame, options = no_save_page
    predispatched(no_save_page, tmp_path, monkeypatch)
    originals = {p: p.read_bytes() for p in (tmp_path / "save").rglob("*.json")}
    canonical = next((tmp_path / "no-planet-save-reservations").glob("*.json"))
    originals[canonical] = canonical.read_bytes()
    original_poll = module._poll_button
    polls = 0

    def clear_after_poll(session, handle):
        nonlocal polls
        polls += 1
        if polls == 2:
            frame.locator("#save-notice").evaluate("e=>e.textContent=''")
        return original_poll(session, handle)

    monkeypatch.setattr(module, "_poll_button", clear_after_poll)
    receipt = resume(no_save_page, tmp_path)
    directory = tmp_path / "save/resume"
    assert frame.evaluate("window.fixtureSaves") == 1 and polls >= 3
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    assert receipt["resumed_same_intent"] and receipt["reservation_retained"]
    assert receipt["max_total_save_clicks"] == 1 and receipt["source_output"] == "save"
    assert receipt["output"] == "save/resume"
    assert receipt["save_click_delivered"] and receipt["data_saved_notice_observed"]
    assert not receipt["source_save_click_dispatched"] and not receipt["notice_was_already_present"]
    assert not receipt["further_continuation_allowed"] and not receipt["automatic_retry"]
    assert not receipt["task_completed"] and not receipt["correctness_verified"]
    assert not receipt["cross_session_persistence_verified"]
    assert len(receipt["source_sha256"]) == 6
    for relative, digest in receipt["source_sha256"].items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == digest
    assert (directory / "dispatch.json").exists() and (directory / "acknowledgement.json").exists()
    assert json.loads((tmp_path / "save/resume-reserved.json").read_text()) == json.loads(
        (directory / "reserved.json").read_text()
    )
    with pytest.raises(BrowserSafetyStop, match="continuation_already_reserved"):
        resume(no_save_page, tmp_path)
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        save_no_planet_work(page, config(), tmp_path / "different-output", **options)
    assert frame.evaluate("window.fixtureSaves") == 1


@pytest.mark.parametrize(
    "change", ["dispatched", "missing", "truthy", "reason", "dispatch_marker", "ack", "confirmed"]
)
def test_resume_rejects_any_ambiguous_or_previously_dispatched_source(
    no_save_page, tmp_path, monkeypatch, change
):
    _, frame, _ = no_save_page
    predispatched(no_save_page, tmp_path, monkeypatch)
    if change in {"dispatched", "missing", "truthy", "reason"}:
        path = tmp_path / "save/stopped.json"
        payload = json.loads(path.read_text())
        if change == "missing":
            del payload["save_may_have_occurred"]
        elif change == "reason":
            payload["reason"] = "no_planet_save_acknowledgement_timeout"
        else:
            payload["save_may_have_occurred"] = True if change == "dispatched" else 0
        path.write_text(json.dumps(payload))
    else:
        filename = {"dispatch_marker": "dispatch", "ack": "acknowledgement", "confirmed": "confirmed"}[change]
        (tmp_path / f"save/{filename}.json").write_text("{}")
    with pytest.raises(BrowserSafetyStop, match="predispatch"):
        resume(no_save_page, tmp_path)
    assert frame.evaluate("window.fixtureSaves") == 0
    assert not (tmp_path / "save/resume-reserved.json").exists()


@pytest.mark.parametrize("change", ["answer", "duration", "star", "paint", "choice"])
def test_resume_rejects_changed_visible_work_or_source(no_save_page, tmp_path, monkeypatch, change):
    _, frame, options = no_save_page
    predispatched(no_save_page, tmp_path, monkeypatch)
    frame.locator("#save-notice").evaluate("e=>e.textContent=''")
    if change == "answer":
        frame.locator("#period_days").fill("5")
    elif change == "duration":
        frame.get_by_role("textbox").first.fill("4000")
    elif change == "star":
        frame.locator("div").first.evaluate("e=>e.innerText='OTHERSTAR'")
    elif change == "paint":
        frame.locator(".choice").first.evaluate("""e=>{
          e.classList.add('selected');const s=document.createElement('style');
          s.textContent='.choice.selected label{border-color:white}.choice.selected label::after{opacity:1;background:white}';
          document.body.append(s)}""")
    else:
        options["choice_path"].write_text("{}")
    with pytest.raises(BrowserSafetyStop):
        resume(no_save_page, tmp_path)
    assert frame.evaluate("window.fixtureSaves") == 0
    assert not (tmp_path / "save/resume/dispatch.json").exists()
    assert not (tmp_path / "save/resume/confirmed.json").exists()


def test_resume_stale_notice_never_clears_consumes_continuation_without_click(
    no_save_page, tmp_path, monkeypatch
):
    _, frame, _ = no_save_page
    predispatched(no_save_page, tmp_path, monkeypatch)
    with pytest.raises(BrowserSafetyStop, match="stale_acknowledgement_timeout"):
        resume(no_save_page, tmp_path, timeout_seconds=0.1)
    stopped = json.loads((tmp_path / "save/resume/stopped.json").read_text())
    assert stopped["reservation_retained"] and stopped["resume_claim_created"]
    assert not stopped["save_may_have_occurred"] and not stopped["further_continuation_allowed"]
    assert not (tmp_path / "save/resume/dispatch.json").exists()
    frame.locator("#save-notice").evaluate("e=>e.textContent=''")
    with pytest.raises(BrowserSafetyStop, match="continuation_already_reserved"):
        resume(no_save_page, tmp_path)
    assert frame.evaluate("window.fixtureSaves") == 0


@pytest.mark.parametrize("change", ["answer", "source", "claim", "replacement", "stale_notice"])
def test_resume_final_guards_reject_changes_after_exclusive_claim(
    no_save_page, tmp_path, monkeypatch, change
):
    import habfly.browser_no_planet_save as module

    _, frame, _ = no_save_page
    predispatched(no_save_page, tmp_path, monkeypatch)
    frame.locator("#save-notice").evaluate("e=>e.textContent=''")
    original = module.persist_json

    def persist(path, payload):
        original(path, payload)
        if path == tmp_path / "save/resume/stale-notice-cleared.json":
            if change == "answer":
                frame.locator("#period_days").fill("5")
            elif change == "source":
                source = tmp_path / "save/stopped.json"
                source.write_text(source.read_text() + " ")
            elif change == "claim":
                (tmp_path / "save/resume-reserved.json").write_text("{}")
            elif change == "replacement":
                frame.get_by_role("button", name="Save").evaluate("e=>e.replaceWith(e.cloneNode(true))")
            else:
                frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop):
        resume(no_save_page, tmp_path)
    assert frame.evaluate("window.fixtureSaves") == 0
    assert not json.loads((tmp_path / "save/resume/stopped.json").read_text())["save_may_have_occurred"]
    with pytest.raises(BrowserSafetyStop, match="continuation_already_reserved"):
        resume(no_save_page, tmp_path)


def test_resume_late_brief_ack_survives_slow_full_capture(no_save_page, tmp_path, monkeypatch):
    _, frame, _ = no_save_page
    predispatched(no_save_page, tmp_path, monkeypatch)
    frame.locator("#save-notice").evaluate("e=>e.textContent=''")
    frame.get_by_role("button", name="Save").evaluate("""e=>e.onclick=()=>{
      window.fixtureSaves++;const n=document.getElementById('save-notice');
      setTimeout(()=>{n.textContent='Data saved';setTimeout(()=>n.textContent='',200)},80);
    }""")
    original = PlanetNumericSession.current

    def slow_after_dispatch(self):
        if (tmp_path / "save/resume/dispatch.json").exists():
            assert (tmp_path / "save/resume/acknowledgement.json").exists()
            self.page.wait_for_timeout(350)
        return original(self)

    monkeypatch.setattr(PlanetNumericSession, "current", slow_after_dispatch)
    receipt = resume(no_save_page, tmp_path)
    assert receipt["data_saved_notice_observed"] and frame.evaluate("window.fixtureSaves") == 1
    assert frame.locator("#save-notice").inner_text() == ""


@pytest.mark.parametrize("kind", ["timeout", "uncertain_click", "changed_answer"])
def test_resume_possible_dispatch_is_permanently_reserved(no_save_page, tmp_path, monkeypatch, kind):
    from playwright.sync_api import ElementHandle

    _, frame, _ = no_save_page
    predispatched(no_save_page, tmp_path, monkeypatch)
    frame.locator("#save-notice").evaluate("e=>e.textContent=''")
    if kind == "uncertain_click":
        original = ElementHandle.click

        def uncertain(self, *args, **kwargs):
            original(self, *args, **kwargs)
            raise RuntimeError("private transport detail")

        monkeypatch.setattr(ElementHandle, "click", uncertain)
    else:
        frame.get_by_role("button", name="Save").evaluate(
            """(e,change)=>e.onclick=()=>{
              window.fixtureSaves++;
              if(change){document.getElementById('period_days').value='8';
                document.getElementById('save-notice').textContent='Data saved'}
            }""",
            kind == "changed_answer",
        )
    with pytest.raises(BrowserSafetyStop):
        resume(no_save_page, tmp_path, timeout_seconds=2)
    stopped_text = (tmp_path / "save/resume/stopped.json").read_text()
    stopped = json.loads(stopped_text)
    assert stopped["save_may_have_occurred"] and (tmp_path / "save/resume/dispatch.json").exists()
    assert not stopped["automatic_retry"] and "private transport detail" not in stopped_text
    assert not (tmp_path / "save/resume/confirmed.json").exists()
    with pytest.raises(BrowserSafetyStop, match="continuation_already_reserved"):
        resume(no_save_page, tmp_path)
    assert frame.evaluate("window.fixtureSaves") == 1


def consumed_predispatched(no_save_page, tmp_path, monkeypatch):
    """Model a second independently appearing site notice, not a Save click."""
    import habfly.browser_no_planet_save as module

    _, frame, _ = no_save_page
    predispatched(no_save_page, tmp_path, monkeypatch)
    frame.locator("#save-notice").evaluate("e=>e.textContent=''")
    original = module.persist_json

    def persist(path, payload):
        original(path, payload)
        if path == tmp_path / "save/resume/stale-notice-cleared.json":
            frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")

    with monkeypatch.context() as context:
        context.setattr(module, "persist_json", persist)
        with pytest.raises(BrowserSafetyStop, match="stale_acknowledgement"):
            resume(no_save_page, tmp_path)
    assert frame.evaluate("window.fixtureSaves") == 0


def test_resume_records_cleared_notice_then_new_predispatch_notice_without_click(
    no_save_page, tmp_path, monkeypatch
):
    consumed_predispatched(no_save_page, tmp_path, monkeypatch)
    directory = tmp_path / "save/resume"
    cleared = json.loads((directory / "footer-probes/clear-wait-000.json").read_text())
    later = json.loads((directory / "footer-probes/after-clear-and-full-guard.json").read_text())
    assert cleared["notice_present"] is False and later["notice_present"] is True
    assert cleared["probe_finished_monotonic_seconds"] <= later["probe_started_monotonic_seconds"]
    latest = json.loads((directory / "pre-dispatch-observation/observation.json").read_text())
    assert later["compared_capture_sha256"] == screen_identity(latest)
    assert not later["fresh_save_acknowledgement_verified"]
    assert not (directory / "dispatch.json").exists()
    assert not (directory / "acknowledgement.json").exists()
    assert no_save_page[1].evaluate("window.fixtureSaves") == 0


@pytest.mark.parametrize("banner", [False, True])
def test_reconciliation_accounts_for_both_consumed_no_click_claims(
    no_save_page, tmp_path, monkeypatch, banner
):
    from playwright.sync_api import ElementHandle, Locator

    from habfly.browser_no_planet_save import reconcile_no_planet_autosave

    page, frame, options = no_save_page
    consumed_predispatched(no_save_page, tmp_path, monkeypatch)
    originals = {p: p.read_bytes() for p in (tmp_path / "save").rglob("*.json")}
    if not banner:
        frame.locator("#save-notice").evaluate("e=>e.textContent=''")

    def forbidden(*_, **__):
        raise AssertionError("Reconciliation must perform no browser action")

    with monkeypatch.context() as context:
        for cls in (ElementHandle, Locator):
            for method in ("click", "fill", "select_option", "press"):
                context.setattr(cls, method, forbidden)
        receipt = reconcile_no_planet_autosave(
            page, config(), tmp_path / "save", tmp_path / "review", run_history=tmp_path, timeout_seconds=0
        )
    assert receipt["data_saved_notice_observed"] == banner
    assert receipt["save_acknowledgement_verified"] == banner
    assert receipt["browser_actions"] == 0 and not receipt["save_click_delivered"]
    assert not receipt["site_autosave_trigger_verified"] and not receipt["task_completed"]
    continuation = receipt["continuation_disposition"]
    assert continuation["output"] == "save/resume"
    assert continuation["reservation_retained"] and not continuation["save_click_dispatched"]
    assert not continuation["further_continuation_allowed"]
    assert len(continuation["source_sha256"]) == 8 and len(receipt["source_sha256"]) == 6
    for relative, checksum in continuation["source_sha256"].items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == checksum
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    assert (tmp_path / "review/confirmed.json").exists() == banner
    assert not (tmp_path / "save/resume/dispatch.json").exists()
    with pytest.raises(BrowserSafetyStop, match="continuation_already_reserved"):
        resume(no_save_page, tmp_path)
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        save_no_planet_work(page, config(), tmp_path / "new-save", **options)
    assert frame.evaluate("window.fixtureSaves") == 0


@pytest.mark.parametrize(
    "change",
    [
        "dispatched",
        "missing",
        "truthy",
        "reason",
        "claim",
        "partial",
        "dispatch",
        "acknowledgement",
        "confirmed",
        "cleared",
        "capture",
    ],
)
def test_reconciliation_never_ignores_partial_uncertain_or_changed_continuation(
    no_save_page, tmp_path, monkeypatch, change
):
    from habfly.browser_no_planet_save import reconcile_no_planet_autosave

    page, frame, _ = no_save_page
    consumed_predispatched(no_save_page, tmp_path, monkeypatch)
    if change in {"dispatched", "missing", "truthy", "reason"}:
        path = tmp_path / "save/resume/stopped.json"
        payload = json.loads(path.read_text())
        if change == "missing":
            del payload["save_may_have_occurred"]
        elif change == "reason":
            payload["reason"] = "no_planet_save_acknowledgement_timeout"
        else:
            payload["save_may_have_occurred"] = True if change == "dispatched" else 0
        path.write_text(json.dumps(payload))
    elif change == "claim":
        (tmp_path / "save/resume-reserved.json").write_text("{}")
    elif change == "partial":
        (tmp_path / "save/resume/stopped.json").unlink()
    elif change == "cleared":
        (tmp_path / "save/resume/stale-notice-cleared.json").write_text('{"data_saved_notice_present":0}')
    elif change == "capture":
        (tmp_path / "save/resume/last-verified-observation/observation.json").write_text("{}")
    else:
        (tmp_path / f"save/resume/{change}.json").write_text("{}")
    with pytest.raises(BrowserSafetyStop):
        reconcile_no_planet_autosave(
            page, config(), tmp_path / "save", tmp_path / "review", run_history=tmp_path, timeout_seconds=0
        )
    assert not (tmp_path / "review/confirmed.json").exists()
    assert frame.evaluate("window.fixtureSaves") == 0


def test_read_only_poll_captures_late_brief_notice_before_slow_final_guard(
    no_save_page, tmp_path, monkeypatch
):
    import habfly.browser_no_planet_save as module

    page, frame, _ = no_save_page
    consumed_predispatched(no_save_page, tmp_path, monkeypatch)
    frame.locator("#save-notice").evaluate("e=>e.textContent=''")
    original_poll = module._poll_button
    original_current = PlanetNumericSession.current
    polls = 0

    def scheduled_notice(session, handle):
        nonlocal polls
        polls += 1
        if polls == 1:
            frame.locator("#save-notice").evaluate("""e=>setTimeout(()=>{
              e.textContent='Data saved';setTimeout(()=>e.textContent='',200)},80)""")
        return original_poll(session, handle)

    def slow_final_guard(self):
        assert (tmp_path / "review/acknowledgement.json").exists()
        self.page.wait_for_timeout(350)
        return original_current(self)

    monkeypatch.setattr(module, "_poll_button", scheduled_notice)
    monkeypatch.setattr(PlanetNumericSession, "current", slow_final_guard)
    receipt = module.reconcile_no_planet_autosave(
        page, config(), tmp_path / "save", tmp_path / "review", run_history=tmp_path, timeout_seconds=3
    )
    assert polls >= 2 and receipt["save_acknowledgement_verified"]
    assert not receipt["save_click_delivered"] and not receipt["site_autosave_trigger_verified"]
    assert frame.locator("#save-notice").inner_text() == "" and frame.evaluate("window.fixtureSaves") == 0


def test_read_only_reconciliation_rechecks_consumed_claim_after_current_ack(
    no_save_page, tmp_path, monkeypatch
):
    import habfly.browser_no_planet_save as module

    page, frame, _ = no_save_page
    consumed_predispatched(no_save_page, tmp_path, monkeypatch)
    original = module.persist_json

    def persist(path, payload):
        original(path, payload)
        if path == tmp_path / "review/acknowledgement.json":
            claim = tmp_path / "save/resume-reserved.json"
            claim.write_text(claim.read_text() + " ")

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop, match="sources_changed"):
        module.reconcile_no_planet_autosave(
            page, config(), tmp_path / "save", tmp_path / "review", run_history=tmp_path, timeout_seconds=0
        )
    assert not (tmp_path / "review/confirmed.json").exists()
    assert frame.evaluate("window.fixtureSaves") == 0


@pytest.mark.parametrize("timeout", [-1, 30, float("inf"), float("nan"), True, "1"])
def test_reconciliation_wait_must_be_bounded_below_thirty_seconds(tmp_path, timeout):
    from habfly.browser_no_planet_save import reconcile_no_planet_autosave

    with pytest.raises(ValueError, match="below 30 seconds"):
        reconcile_no_planet_autosave(
            None, None, tmp_path / "save", tmp_path / "review", run_history=tmp_path, timeout_seconds=timeout
        )
    assert not (tmp_path / "review").exists()
