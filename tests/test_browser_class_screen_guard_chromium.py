"""Strict footer races on isolated, fully intercepted native class fixtures."""
# ruff: noqa: F401,F811 -- imported pytest fixtures

import pytest
from test_browser_class_setup_steps_intercepted import (
    create_steps,
    isolated_chromium,
    native_fresh_star,
    pytestmark,
)
from test_browser_numeric import config

import habfly.browser_classification as module
from habfly.browser import BrowserSafetyStop


def test_native_busy_save_changes_do_not_repeat_class_or_prefix(native_fresh_star, monkeypatch):
    page, frame, history, fresh, actions = native_fresh_star(inherited_main=True)
    original = module.inspect_page

    def inspect(*args):
        report = original(*args)
        frame.get_by_role("button", name="Save", exact=True).evaluate("e=>e.disabled=!e.disabled")
        return report

    monkeypatch.setattr(module, "inspect_page", inspect)
    owner = create_steps(page, history, fresh, "main_sequence", "Ga", [])
    while not owner.finished:
        owner.advance()
    assert owner.report["setup_verified"], owner.report
    assert actions == [
        {"target": "stellar_class", "value": "white_dwarf"},
        {"target": "stellar_class", "value": "main_sequence"},
        {"target": "lifetime_prefix", "value": "Ga"},
    ]
    assert not (owner.output / "class-screen-failure").exists()


@pytest.mark.parametrize("message", ["Data saved", "Saving failed"])
def test_native_exact_notice_only_race_uses_existing_footer_rule(native_fresh_star, monkeypatch, message):
    page, frame, _, _, actions = native_fresh_star(inherited_main=False)
    frame.evaluate("""() => {
      const save=Array.from(document.querySelectorAll('button')).find(e=>e.innerText==='Save');
      const text=save.previousSibling;
      if(text.nodeType!==Node.TEXT_NODE || !text.textContent.endsWith('1 Rs')) throw Error('fixture footer');
      const br=document.createElement('br'), tail=document.createTextNode('1 Rs');
      text.textContent=text.textContent.slice(0,-4);
      text.after(br,tail,document.createElement('br'));
      const notice=document.createElement('span'); notice.id='fixtureNotice';
      save.before(notice);
    }""")
    original = module.inspect_page
    toggle = [False]

    def inspect(*args):
        report = original(*args)
        toggle[0] = not toggle[0]
        frame.locator("#fixtureNotice").evaluate(
            "(e,value)=>{e.replaceChildren();if(value){e.append(document.createTextNode(value),document.createElement('br'));}}",
            message if toggle[0] else "",
        )
        return report

    monkeypatch.setattr(module, "inspect_page", inspect)
    if message != "Data saved":
        with pytest.raises(module.ClassScreenMismatch, match="screen_changed_during_class_read"):
            module.StellarSelectionSession(page, config(), lambda *_: None)
        assert actions == []
        return
    session = module.StellarSelectionSession(page, config(), lambda *_: None)
    assert session.select_class("main_sequence", source="reference_diagnostic")["readback_verified"]
    assert session.select_prefix("Ga")["readback_verified"]
    assert actions == [
        {"target": "stellar_class", "value": "main_sequence"},
        {"target": "lifetime_prefix", "value": "Ga"},
    ]


def test_native_prefix_pair_preserves_mutation_and_exact_dispatch_flags(native_fresh_star, monkeypatch):
    page, frame, _, _, actions = native_fresh_star(inherited_main=False)
    session = module.StellarSelectionSession(page, config(), lambda *_: None)
    session.select_class("main_sequence", source="reference_diagnostic")
    original = module.inspect_page
    changed = [False]

    def inspect(*args):
        report = original(*args)
        if session._prefix_selection_returned and not changed[0]:
            changed[0] = True
            frame.locator("#distance").evaluate("e=>e.value='1'")
        return report

    monkeypatch.setattr(module, "inspect_page", inspect)
    with pytest.raises(module.ClassScreenMismatch) as caught:
        session.select_prefix("Ga")
    error = caught.value
    assert error.diagnostic["failure_phase"] == "post_prefix_read"
    assert error.diagnostic["native_prefix_selection_invoked"] is True
    assert error.diagnostic["native_prefix_selection_returned"] is True
    assert error.diagnostic["native_class_click_invoked"] is False
    assert error.diagnostic["native_class_click_returned"] is False
    assert error.before != error.after and error.diagnostic["projected_screen_changes"]["change_count"] > 0
    assert actions == [
        {"target": "stellar_class", "value": "main_sequence"},
        {"target": "lifetime_prefix", "value": "Ga"},
    ]
    with pytest.raises(BrowserSafetyStop):
        session.select_prefix("Ga")
    assert len(actions) == 2
