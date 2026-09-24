"""All browser traffic is fulfilled locally; never contacts HabWorlds."""

import hashlib
import json
import time
from copy import deepcopy

import pytest
from playwright.sync_api import sync_playwright
from typer.testing import CliRunner

from habfly.browser import BrowserSafetyStop
from habfly.browser_numeric import (
    NumericJournal,
    NumericSession,
    committed_display,
    run_numeric_diagnostic,
    screen_changes,
    screen_identity,
)
from habfly.browser_probe import BrowserProbeConfig, ProbeFrame
from habfly.browser_stellar import COLOR_OPTIONS, SIMULATION_URL, StellarMappingError
from habfly.cli import app
from habfly.contracts import Action, Observation, RuntimeEvent, validate_action
from habfly.knowledge import LocalCalculator, load_knowledge_pack

OUTER = "http://localhost/activity?preview_sequence_id=q%3A123%3A946"
WIDGET = "https://fixture.invalid/widget"


def config():
    return BrowserProbeConfig(
        url=OUTER,
        frames=[
            ProbeFrame(name="simulation", url=SIMULATION_URL),
            ProbeFrame(name="widgets", url=WIDGET, count=2),
        ],
    )


def stellar_html(order=("distance", "luminosity", "temperature")):
    labels = {
        "distance": "distance (ly)",
        "luminosity": "luminosity (L<sub>s</sub>)",
        "temperature": "temperature (K)",
    }
    fields = "".join(
        f'<span>{labels[name]}</span><input placeholder="0" value="" id="{name}">' for name in order
    )
    return (
        '<div style="text-transform:uppercase">Althinagon</div><img alt="1">'
        "<div>Observations parallax (&quot;) 0.045 peak wavelength (nm) 370 Flux 5.68E-10 "
        "Spectrum metallicity 0.21 Your Reconstruction " + fields + "peak λ color<select>"
        # Empty hidden option reproduces the observed menu with no selected visible color.
        "<option hidden selected></option>"
        + "".join(f"<option>{color}</option>" for color in COLOR_OPTIONS)
        + "</select>mass, radius and lifetime are only relevant for main sequence stars "
        "main sequence red giant supergiant white dwarf 1 Rs<button>Save</button></div>"
        '<div hidden>Hidden answers<input value="999"></div>'
    )


def stellar_autosave_html():
    return stellar_html().replace(
        "1 Rs<button>Save</button>",
        '<div>1 Rs</div><div id="autosave"></div><button>Save</button>',
    )


@pytest.fixture(scope="module")
def chromium():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(chromium):
    context = chromium.new_context()
    pages = {
        OUTER: f'<label><input type=checkbox>I am ready to submit project.</label><iframe src="{SIMULATION_URL}"></iframe><iframe src="{WIDGET}"></iframe><iframe src="{WIDGET}"></iframe>',
        SIMULATION_URL: stellar_html(),
        WIDGET: "<button>Update Score</button><button>Submit Project</button>",
    }
    context.route(
        "**/*",
        lambda route: (
            route.fulfill(status=200, content_type="text/html; charset=utf-8", body=pages[route.request.url])
            if route.request.url in pages
            else route.abort()
        ),
    )
    current = context.new_page()
    current.goto(OUTER, wait_until="load")
    yield current
    context.close()


@pytest.fixture
def session(page, tmp_path):
    journal = NumericJournal(tmp_path / "session")
    current = NumericSession(page, config(), journal)
    current.start()
    yield current
    journal.finish(
        outcome="fixture_only",
        attempts=current.attempts,
        verified=current.verified,
        pack_hash=current.calculator.pack.checksum,
    )


def distance(session):
    return session.calculate("distance", {"parallax": "browser_parallax"})[0]


def test_full_manual_transport_offline_events_and_no_protected_actions(page, tmp_path):
    # Buttons remove the document if touched; any unexpected click fails the run.
    for frame in page.frames:
        for button in frame.get_by_role("button").all():
            button.evaluate("b => b.onclick = () => document.body.replaceChildren()")
    answers = iter(
        [
            "distance",
            "browser_parallax",
            "distance",
            "luminosity",
            "browser_flux",
            "result:1",
            "luminosity",
            "temperature",
            "browser_wavelength",
            "temperature",
        ]
    )
    report = run_numeric_diagnostic(
        page,
        config(),
        tmp_path / "run",
        prompt=lambda _: next(answers),
        confirm=lambda _: True,
        echo=lambda _: None,
    )
    assert report["numeric_transport_passed"], report
    assert report["write_attempts"] == 3
    assert not report["task_completed"] and not report["browser_acceptance_passed"]
    assert not page.get_by_role("checkbox").is_checked()
    raw = (tmp_path / "run/events.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == report["events_sha256"]
    assert b"preview_sequence_id" not in raw and b"Hidden answers" not in raw
    events = [RuntimeEvent.model_validate_json(line) for line in raw.splitlines()]
    assert [event.sequence for event in events] == list(range(len(events)))
    assert sum(e.event == "action_result" for e in events) == 3
    assert all(e.payload["action"]["kind"] == "TYPE" for e in events if e.event == "action_result")
    observation = None
    for event in events:
        if event.event == "observation":
            observation = Observation.model_validate(event.payload)
        if event.event == "action_proposed":
            validate_action(observation, Action.model_validate(event.payload["action"]))
    replay = CliRunner().invoke(app, ["replay", str(tmp_path / "run/events.jsonl")])
    assert replay.exit_code == 0, replay.output
    assert '"star_class":null' in raw.decode()


def test_reordered_inputs_bind_by_visible_label_not_index(page, tmp_path):
    page.frames[1].set_content(stellar_html(("temperature", "distance", "luminosity")))
    journal = NumericJournal(tmp_path / "reordered")
    session = NumericSession(page, config(), journal)
    try:
        session.start()
        ident = distance(session)
        assert session.copy(ident, "distance", confirm=lambda _: True)
        assert page.frames[1].locator("#distance").input_value() == repr(3.26 / 0.045)
        assert page.frames[1].locator("#temperature").input_value() == ""
    finally:
        journal.finish(
            outcome="fixture",
            attempts=session.attempts,
            verified=session.verified,
            pack_hash=session.calculator.pack.checksum,
        )


def test_jena_rounding_after_tab_completes_and_retains_exact_calculation_sources(page, tmp_path):
    frame = page.frames[1]
    # Student-visible values transcribed from browser-numeric-002, not answer keys.
    frame.set_content(
        stellar_html()
        .replace("Althinagon", "Jena")
        .replace("0.045", "0.023")
        .replace("370", "359")
        .replace("5.68E-10", "1.85E-10")
    )
    for field in ("distance", "luminosity", "temperature"):
        frame.locator(f"#{field}").evaluate(
            "e => e.onblur = () => { e.value = Number(e.value).toPrecision(4); }"
        )
    answers = iter(
        [
            "distance",
            "browser_parallax",
            "distance",
            "luminosity",
            "browser_flux",
            "result:1",
            "luminosity",
            "temperature",
            "browser_wavelength",
            "temperature",
        ]
    )
    report = run_numeric_diagnostic(
        page,
        config(),
        tmp_path / "rounded",
        prompt=lambda _: next(answers),
        confirm=lambda _: True,
        echo=lambda _: None,
    )
    assert report["numeric_transport_passed"], report
    assert report["write_attempts"] == 3
    assert not report["task_completed"] and not report["browser_acceptance_passed"]
    assert report["numeric_readbacks"]["distance"]["exact_copied"] == "141.7391304347826"
    assert report["numeric_readbacks"]["distance"]["display_value"] == "141.7"
    assert report["numeric_readbacks"]["luminosity"]["exact_copied"] == "10.922720945303972"
    assert report["numeric_readbacks"]["luminosity"]["display_value"] == "10.92"
    events = [
        RuntimeEvent.model_validate_json(line)
        for line in (tmp_path / "rounded/events.jsonl").read_text().splitlines()
    ]
    final = [e.payload for e in events if e.event == "observation"][-1]
    assert final["values"]["diagnostic_results"]["result:1"]["result"]["value"] == 141.7391304347826
    assert final["controls"][0]["value"] == "141.7"
    assert final["values"]["numeric_readbacks"] == report["numeric_readbacks"]
    commits = [
        e.payload["commit_action"] for e in events if e.event == "state" and "commit_action" in e.payload
    ]
    assert len(commits) == 3 and all(c["kind"] == "KEYPRESS" and c["value"] == "Tab" for c in commits)
    assert all(
        e.payload["numeric_readback"]["exact_input_verified"] for e in events if e.event == "action_result"
    )
    assert not page.get_by_role("checkbox").is_checked()
    assert CliRunner().invoke(app, ["replay", str(tmp_path / "rounded/events.jsonl")]).exit_code == 0


@pytest.mark.parametrize(
    "exact,display,kind",
    [
        ("141.7391304347826", "141.7", "decimal_rounding"),
        ("10.922720945303972", "10.92", "decimal_rounding"),
        ("0.000123456789", "0.0001235", "decimal_rounding"),
        ("1234567.89", "1.235e+6", "decimal_rounding"),
        ("1000.0001", "1000", "decimal_rounding"),
        ("1.2345", "1.234", "decimal_rounding"),
        ("1.2345", "1.235", "decimal_rounding"),
        ("1.0", "1.000", "equivalent_numeric_spelling"),
        ("0.000001", "1e-6", "equivalent_numeric_spelling"),
        ("141.7391304347826", "141.7391304347826", "exact"),
    ],
)
def test_commit_display_has_explicit_decimal_rules_not_numeric_tolerance(exact, display, kind):
    receipt = committed_display(exact, display)
    assert receipt["format"] == kind
    assert receipt["exact_copied"] == exact and receipt["display_value"] == display


@pytest.mark.parametrize(
    "display", ["141.8", "142", "100", "141.73", "0", "-141.7", "NaN", "", "141.7 ly", " 141.7", "1e9999"]
)
def test_unapproved_or_too_coarse_display_rejects(display):
    with pytest.raises(BrowserSafetyStop, match="unapproved_committed_display"):
        committed_display("141.7391304347826", display)


@pytest.mark.parametrize("invalid", ["", "0", "NaN", "Infinity", "-1"])
def test_display_validator_rejects_invalid_even_when_equal(invalid):
    with pytest.raises(BrowserSafetyStop, match="unapproved_committed_display"):
        committed_display(invalid, invalid)


@pytest.mark.parametrize(
    "handler,reason",
    [
        ("e.value='72.5'", "unapproved_committed_display"),
        ("e.value='72'", "unapproved_committed_display"),
        ("document.querySelector('#temperature').value='1'", "unrelated_field_changed"),
        ("document.querySelector('button').disabled=true", "unrelated_state_changed_during_commit"),
        ("document.body.insertAdjacentHTML('beforeend','<div role=dialog>Stop</div>')", "unexpected_modal"),
        ("e.outerHTML=e.outerHTML", "numeric_control_replaced_after_write"),
    ],
)
def test_bad_blur_changes_remain_fail_closed(session, handler, reason):
    ident = distance(session)
    session.handles["distance"].evaluate(f"e => e.onblur = () => {{ {handler}; }}")
    with pytest.raises(BrowserSafetyStop, match=reason):
        session.copy(ident, "distance", confirm=lambda _: True)
    assert session.stopped and session.attempts == 1 and session.verified == []
    assert session.journal.numeric_readbacks == {}


def test_rounding_on_input_before_exact_readback_remains_rejected(session):
    ident = distance(session)
    session.handles["distance"].evaluate("e => e.oninput = () => { e.value=Number(e.value).toPrecision(4); }")
    with pytest.raises(BrowserSafetyStop, match="numeric_readback_mismatch"):
        session.copy(ident, "distance", confirm=lambda _: True)
    events = [
        RuntimeEvent.model_validate_json(line)
        for line in (session.journal.output / "events.jsonl").read_text().splitlines()
    ]
    assert not any(e.event == "state" and "commit_action" in e.payload for e in events)


def test_rounding_later_outside_controlled_commit_is_not_whitelisted(session):
    ident = distance(session)
    assert session.copy(ident, "distance", confirm=lambda _: True)
    session.handles["distance"].evaluate("e => e.value=Number(e.value).toPrecision(4)")
    with pytest.raises(BrowserSafetyStop, match="stale_numeric_observation"):
        session.calculate("temperature", {"wavelength": "browser_wavelength"})
    assert session.attempts == 1


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("document.querySelector('#distance').value='12'", "stale_numeric_observation"),
        (
            "document.querySelector('#distance').outerHTML=document.querySelector('#distance').outerHTML",
            "numeric_control_replaced",
        ),
        ("document.body.firstElementChild.textContent='Another Star'", "stale_numeric_observation"),
        ("document.body.insertAdjacentHTML('beforeend','<div role=dialog>Stop</div>')", "unexpected_modal"),
        ("document.body.insertAdjacentHTML('beforeend','<input type=password>')", "authentication_required"),
        ("document.querySelector('#distance').readOnly=true", "unsupported_or_readonly_field"),
    ],
)
def test_changes_during_confirmation_reject_without_write(session, mutation, reason):
    ident = distance(session)

    def confirm(_):
        session.frame.evaluate(mutation)
        return True

    with pytest.raises(BrowserSafetyStop, match=reason):
        session.copy(ident, "distance", confirm=confirm)
    assert session.attempts == 0
    if reason == "stale_numeric_observation":
        reference = session.journal.screen_diagnostics[-1]
        evidence = json.loads((session.journal.output / reference["path"]).read_text())
        assert evidence["reason"] == reason and evidence["change_count"] > 0
        assert evidence["before_screen_sha256"] == screen_identity(session.report)
        assert evidence["before_screen_sha256"] != evidence["after_screen_sha256"]
    else:
        assert session.journal.screen_diagnostics == []


def test_cancel_no_write_and_no_repeat_destination(session):
    ident = distance(session)
    assert not session.copy(ident, "distance", confirm=lambda _: False)
    assert session.attempts == 0
    assert session.copy(ident, "distance", confirm=lambda _: True)
    with pytest.raises(BrowserSafetyStop, match="numeric_write_limit"):
        session.copy(ident, "distance", confirm=lambda _: True)
    assert session.attempts == 1


@pytest.mark.parametrize(
    "handler,reason",
    [
        ("e.target.value='72.4'", "numeric_readback_mismatch"),
        ("document.querySelector('#temperature').value='1'", "unrelated_field_changed"),
        ("document.body.firstElementChild.textContent='Another Star'", "star_changed_after_write"),
        ("document.body.insertAdjacentHTML('beforeend','<div role=dialog>Stop</div>')", "unexpected_modal"),
    ],
)
def test_post_write_failures_preserved_and_never_retried(session, handler, reason):
    ident = distance(session)
    session.handles["distance"].evaluate(
        f"element => element.addEventListener('input', e => {{ {handler}; }})"
    )
    with pytest.raises(BrowserSafetyStop, match=reason):
        session.copy(ident, "distance", confirm=lambda _: True)
    assert session.attempts == 1 and session.verified == [] and session.stopped
    with pytest.raises(BrowserSafetyStop, match="diagnostic_stopped"):
        session.copy(ident, "distance", confirm=lambda _: True)


def test_binding_units_and_missing_or_conditional_operations_not_repaired(session):
    for operation, bindings, reason in [
        ("distance", {"parallax": "browser_wavelength"}, "incompatible_unit"),
        ("distance", {}, "missing_or_extra_inputs"),
        ("mass", {"luminosity": "browser_flux"}, "operation_not_applicable"),
    ]:
        with pytest.raises(StellarMappingError, match=reason):
            session.calculate(operation, bindings)
    assert session.attempts == 0


def test_wrong_numeric_value_is_not_corrected_and_wrong_destination_unit_rejects(session):
    ident = distance(session)
    # API transport test deliberately substitutes a wrong same-unit result.
    session.results[ident]["result"]["value"] = 1.2345678901234567
    with pytest.raises(StellarMappingError, match="incompatible_destination_unit"):
        session.copy(ident, "temperature", confirm=lambda _: True)
    assert session.copy(ident, "distance", confirm=lambda _: True)
    assert session.handles["distance"].input_value() == "1.2345678901234567"


def test_native_label_evidence_required_even_if_ax_can_be_faked(page, tmp_path):
    # AX text can be present while the DOM visible label is different.
    page.frames[1].locator("#distance").evaluate(
        "e => e.previousElementSibling.setAttribute('aria-label', 'distance (ly)')"
    )
    # Existing strict AX mapping rejects unsupported semantics instead of guessing.
    journal = NumericJournal(tmp_path / "labels")
    session = NumericSession(page, config(), journal)
    try:
        page.frames[1].locator("#distance").evaluate(
            "e => e.previousElementSibling.textContent='distance (pc)'"
        )
        with pytest.raises((BrowserSafetyStop, StellarMappingError)):
            session.start()
    finally:
        journal.finish(outcome="fixture", attempts=0, verified=[], pack_hash=session.calculator.pack.checksum)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("history.replaceState(null, '', '/outside')", "navigation_outside_activity"),
        (
            "document.body.insertAdjacentHTML('beforeend', '<iframe src=about:blank></iframe>')",
            "unknown_visible_frame",
        ),
    ],
)
def test_cached_navigation_or_new_frame_during_human_pause_stops(session, change, reason):
    ident = distance(session)

    def confirm(_):
        session.page.evaluate(f"setTimeout(() => {{ {change}; }}, 10)")
        time.sleep(0.1)  # Emulate blocking terminal confirmation; driver cache is stale.
        return True

    with pytest.raises(BrowserSafetyStop, match=reason):
        session.copy(ident, "distance", confirm=confirm)
    assert session.attempts == 0
    assert session.journal.screen_diagnostics == []


def test_disappearing_visible_status_still_stops_and_records_full_pair(page, tmp_path):
    frame = page.frames[1]
    frame.locator("body").evaluate(
        "body => body.insertAdjacentHTML('beforeend', '<div id=status>STAR COLLECTED</div>')"
    )
    answers = iter(["distance", "browser_parallax", "distance", "luminosity", "browser_flux", "result:1"])
    choices, confirmations, messages = [], [], []

    def prompt(_):
        value = next(answers)
        choices.append(value)
        if value == "luminosity":
            frame.locator("#status").evaluate("node => node.remove()")
        return value

    report = run_numeric_diagnostic(
        page,
        config(),
        tmp_path / "disappeared",
        prompt=prompt,
        confirm=lambda question: confirmations.append(question) or True,
        echo=messages.append,
    )
    assert report["outcome"] == "stale_numeric_observation"
    assert report["verified_fields"] == ["distance"] and report["write_attempts"] == 1
    assert not report["numeric_transport_passed"] and len(confirmations) == 1
    assert frame.locator("#luminosity").input_value() == ""
    assert frame.locator("#temperature").input_value() == ""
    reference = report["screen_diagnostics"][0]
    raw = (tmp_path / "disappeared" / reference["path"]).read_bytes()
    evidence = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
    assert "STAR COLLECTED" in evidence["before"]["frames"][0]["text"]
    assert "STAR COLLECTED" not in evidence["after"]["frames"][0]["text"]
    assert {change["path"] for change in evidence["changes"]} == {"/frames/0/text", "/frames/0/accessibility"}
    # Pair uses the latest verified screen, not the empty initial capture.
    distance_before = next(c for c in evidence["before"]["frames"][0]["controls"] if c["role"] == "textbox")
    assert distance_before["value"] == repr(3.26 / 0.045)
    assert b"preview_sequence_id" not in raw and b"Hidden answers" not in raw
    assert any("/frames/0/text" in message for message in messages)
    events = [
        RuntimeEvent.model_validate_json(line)
        for line in (tmp_path / "disappeared/events.jsonl").read_text().splitlines()
    ]
    diagnostic = next(e for e in events if "screen_change" in e.payload)
    assert diagnostic.sequence < next(e.sequence for e in events if e.event == "error")
    assert diagnostic.payload["screen_change"]["sha256"] == reference["sha256"]
    replay = CliRunner().invoke(app, ["replay", str(tmp_path / "disappeared/events.jsonl")])
    assert replay.exit_code == 0 and "screen_change" in replay.output


def test_change_during_binding_records_the_two_checked_snapshots(session, monkeypatch):
    from habfly import browser_numeric

    original = browser_numeric.inspect_page
    calls = 0

    def changed_snapshot(*args):
        nonlocal calls
        report = original(*args)
        calls += 1
        if calls == 2:
            report["frames"][0]["text"] += "\nUpdated status"
        return report

    monkeypatch.setattr(browser_numeric, "inspect_page", changed_snapshot)
    with pytest.raises(BrowserSafetyStop, match="screen_changed_during_binding"):
        distance(session)
    assert session.attempts == 0 and calls == 2  # Logging adds no browser reads.
    reference = session.journal.screen_diagnostics[-1]
    evidence = json.loads((session.journal.output / reference["path"]).read_text())
    assert evidence["reason"] == "screen_changed_during_binding"
    assert evidence["changes"][0]["path"] == "/frames/0/text"


def test_screen_diff_ignores_only_existing_metadata_and_preserves_presence():
    before = {"captured_at": "old", "setup_mode": "existing_state", "frames": [{"value": ""}]}
    after = deepcopy(before)
    after.update(captured_at="new", setup_mode="independent_test_session")
    assert screen_changes(before, after)["changes"] == []
    assert screen_identity(before) == screen_identity(after)
    after["frames"][0]["value"] = "0"
    result = screen_changes(before, after)
    assert result["changes"] == [
        {"path": "/frames/0/value", "before_present": True, "after_present": True, "before": "", "after": "0"}
    ]
    after["frames"][0]["missing"] = None
    result = screen_changes(before, after)
    assert result["change_count"] == 2
    added = next(c for c in result["changes"] if c["path"].endswith("/missing"))
    assert not added["before_present"] and added["after_present"]
    assert screen_identity(before) != screen_identity(after)
    assert screen_changes(after, before)["changes"][0]["after_present"] is False


def autosave_report(*, notice):
    text_notice, ax_notice = ("Data saved\n", " Data saved") if notice else ("", "")
    return {
        "frames": [
            {
                "url": SIMULATION_URL,
                "text": f"FIXTURE STAR\nPARALLAX 0.045\n1 Rs\n{text_notice}Save",
                "accessibility": (
                    '- text: Fixture Star\n- textbox "distance": "123"\n'
                    "- text: mass, radius and lifetime are only relevant for main sequence stars "
                    f'main sequence red giant supergiant white dwarf 1 Rs{ax_notice}\n- button "Save"'
                ),
                "controls": [{"role": "textbox", "value": "123", "enabled": True}],
            }
        ]
    }


def test_exact_autosave_footer_can_appear_and_disappear_without_mutating_raw_evidence():
    before, after = autosave_report(notice=False), autosave_report(notice=True)
    original_before, original_after = deepcopy(before), deepcopy(after)
    assert screen_identity(before) == screen_identity(after)
    assert screen_changes(before, after)["change_count"] == 0
    assert screen_changes(after, before)["change_count"] == 0
    assert before == original_before and after == original_after


@pytest.mark.parametrize(
    "change",
    ["notice_error", "unpaired", "duplicate", "other_frame", "measurement", "star", "value", "disabled"],
)
def test_autosave_allowance_does_not_hide_other_changes(change):
    before, after = autosave_report(notice=False), autosave_report(notice=True)
    frame = after["frames"][0]
    if change == "notice_error":
        frame["text"] = frame["text"].replace("Data saved", "Data saved with errors")
        frame["accessibility"] = frame["accessibility"].replace("Data saved", "Data saved with errors")
    elif change == "unpaired":
        frame["accessibility"] = before["frames"][0]["accessibility"]
    elif change == "duplicate":
        frame["text"] = "Data saved\n" + frame["text"]
    elif change == "other_frame":
        before["frames"][0]["url"] = frame["url"] = WIDGET
    elif change == "measurement":
        frame["text"] = frame["text"].replace("0.045", "0.046")
    elif change == "star":
        frame["text"] = frame["text"].replace("FIXTURE STAR", "ANOTHER STAR")
    elif change == "value":
        frame["controls"][0]["value"] = "456"
    elif change == "disabled":
        frame["controls"][0]["enabled"] = False
    assert screen_identity(before) != screen_identity(after)
    assert screen_changes(before, after)["change_count"] > 0


def test_autosave_between_actions_and_during_commit_preserves_exact_copy_checks(page, tmp_path):
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    frame.set_content(stellar_autosave_html())
    journal = NumericJournal(tmp_path / "autosave")
    current = NumericSession(page, config(), journal)
    try:
        current.start()
        result = distance(current)
        frame.locator("#autosave").evaluate("e => e.textContent='Data saved'")
        # Emulate the periodic notification disappearing on focus/appearing on
        # controlled Tab. Never click Save or change any other field.
        frame.locator("#distance").evaluate("""e => {
            e.onfocus=()=>document.querySelector('#autosave').textContent='';
            e.onblur=()=>{e.value=Number(e.value).toPrecision(4);
                document.querySelector('#autosave').textContent='Data saved';};
        }""")
        assert current.copy(result, "distance", confirm=lambda _: True)
        assert current.verified == ["distance"] and current.attempts == 1
        assert "Data saved" in current.report["frames"][0]["text"]  # Raw report retained.
        frame.locator("#autosave").evaluate("e => e.textContent=''")
        _, luminosity = current.calculate("luminosity", {"flux": "browser_flux", "distance": result})
        assert luminosity.ok
        frame.locator("#autosave").evaluate("e => e.textContent='Data saved'")
        frame.locator("#luminosity").fill("456")
        with pytest.raises(BrowserSafetyStop, match="stale_numeric_observation"):
            current._current()
        assert current.attempts == 1  # Simultaneous answer edit is NOT ignored.
    finally:
        report = journal.finish(
            outcome="fixture_only",
            attempts=current.attempts,
            verified=current.verified,
            pack_hash=current.calculator.pack.checksum,
        )
    assert report["screen_comparison_policy"]["verifies_persistence"] is False


def test_screen_diff_clips_summary_at_change_location_but_artifact_retains_exact_text(tmp_path):
    before = {"frames": [{"text": "same " * 200 + "BEFORE"} for _ in range(50)]}
    after = {"frames": [{"text": "same " * 200 + "AFTER"} for _ in range(50)]}
    original = deepcopy(before)
    differences = screen_changes(before, after)
    assert differences["change_count"] == 50 and len(differences["changes"]) == 40
    assert differences["truncated"]
    assert "BEFORE" in differences["changes"][0]["before"]["excerpt"]
    assert "AFTER" in differences["changes"][0]["after"]["excerpt"]
    assert before == original
    journal = NumericJournal(tmp_path / "bounded")
    journal.record_screen_change("stale_numeric_observation", before, after)
    report = journal.finish(outcome="stale_numeric_observation", attempts=0, verified=[], pack_hash="fixture")
    reference = report["screen_diagnostics"][0]
    evidence = json.loads((tmp_path / "bounded" / reference["path"]).read_text())
    assert evidence["before"] == before and evidence["after"] == after
    assert evidence["before_screen_sha256"] == screen_identity(before)
    assert evidence["after_screen_sha256"] == screen_identity(after)


def test_screen_diff_handles_reordering_types_and_escaped_json_pointers():
    before = {"items": ["a", "b"], "key/~": False}
    after = {"items": ["b"], "key/~": 0}
    result = screen_changes(before, after)
    assert {c["path"] for c in result["changes"]} == {"/items/0", "/items/1", "/key~1~0"}
    assert result["change_count"] == 3
    with pytest.raises(ValueError):
        screen_changes(before, after, limit=0)


def test_limits_and_failure_artifacts(page, tmp_path):
    answers = iter(["distance", "browser_wavelength"])
    report = run_numeric_diagnostic(
        page,
        config(),
        tmp_path / "failure",
        prompt=lambda _: next(answers),
        confirm=lambda _: True,
        echo=lambda _: None,
    )
    assert report["outcome"] == "incompatible_unit" and report["write_attempts"] == 0
    assert (tmp_path / "failure/manifest.json").exists()
    assert '"event":"error"' in (tmp_path / "failure/events.jsonl").read_text()
    with pytest.raises(FileExistsError):
        NumericJournal(tmp_path / "failure")


def test_calculation_and_time_budget(session):
    session.calculations = session.MAX_CALCULATIONS
    with pytest.raises(BrowserSafetyStop, match="calculation_limit"):
        distance(session)
    session.started -= session.MAX_SECONDS + 1
    with pytest.raises(BrowserSafetyStop, match="diagnostic_time_limit"):
        distance(session)


def test_final_safety_stop_overrides_individually_verified_fields(tmp_path):
    journal = NumericJournal(tmp_path / "final-stop")
    report = journal.finish(
        outcome="stale_numeric_observation",
        attempts=3,
        verified=["distance", "luminosity", "temperature"],
        pack_hash="fixture",
    )
    assert not report["numeric_transport_passed"]


def test_popup_and_native_dialog_are_fail_closed(session):
    ident = distance(session)

    def confirm(_):
        session.page.evaluate("alert('private dialog text')")
        return True

    with pytest.raises(BrowserSafetyStop, match="unexpected_browser_dialog"):
        session.copy(ident, "distance", confirm=confirm)
    assert session.attempts == 0
    session.unexpected_dialog = False
    popup = session.page.context.new_page()
    try:
        with pytest.raises(BrowserSafetyStop, match="unexpected_popup"):
            session.copy(ident, "distance", confirm=lambda _: True)
    finally:
        popup.close()


def test_interrupt_after_a_write_reports_interruption_not_success(page, tmp_path):
    answers = iter(["distance", "browser_parallax", "distance"])

    def prompt(_):
        try:
            return next(answers)
        except StopIteration:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_numeric_diagnostic(
            page, config(), tmp_path / "interrupt", prompt=prompt, confirm=lambda _: True, echo=lambda _: None
        )
    report = json.loads((tmp_path / "interrupt/manifest.json").read_text())
    assert report["outcome"] == "interrupted" and report["write_attempts"] == 1
    assert not report["numeric_transport_passed"] and not report["task_completed"]


def test_readback_failure_keeps_actual_value_evidence(page, tmp_path):
    page.frames[1].locator("#distance").evaluate("el => el.oninput = () => el.value='72.4'")
    answers = iter(["distance", "browser_parallax", "distance"])
    report = run_numeric_diagnostic(
        page,
        config(),
        tmp_path / "readback",
        prompt=lambda _: next(answers),
        confirm=lambda _: True,
        echo=lambda _: None,
    )
    assert report["outcome"] == "numeric_readback_mismatch" and report["write_attempts"] == 1
    events = [
        RuntimeEvent.model_validate_json(line)
        for line in (tmp_path / "readback/events.jsonl").read_text().splitlines()
    ]
    state = next(e.payload["post_write_observation"] for e in events if "post_write_observation" in e.payload)
    assert state["values"]["browser_field_map"]["distance"]["current_value"] == "72.4"


def test_unknown_class_scope_does_not_change_training_api_or_pack_hash():
    calculator = LocalCalculator(load_knowledge_pack())
    before = calculator.pack.checksum
    bindings = {"parallax": {"value": 0.032, "unit": "arcsec"}}
    assert not calculator.execute("distance", bindings, None).ok
    assert calculator.execute_unclassified_common("distance", bindings).value == pytest.approx(101.875)
    assert not calculator.execute_unclassified_common("mass", {"luminosity": {"value": 1, "unit": "Lsun"}}).ok
    for value in (0, -1, float("nan"), float("inf"), None):
        assert not calculator.execute_unclassified_common(
            "distance", {"parallax": {"value": value, "unit": "arcsec"}}
        ).ok
    assert calculator.pack.checksum == before


def test_check_command_is_offline_and_live_requires_explicit_test_mode(tmp_path, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("network not allowed")

    monkeypatch.setattr(socket, "socket", forbidden)
    args = ["browser", "test-numeric", "configs/browser_probe.example.json", str(tmp_path / "unused")]
    result = CliRunner().invoke(app, args + ["--check"])
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0 and "--new-test-session" in result.output
    assert not (tmp_path / "unused").exists()
