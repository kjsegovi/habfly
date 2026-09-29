"""Real native form handling; every request is fulfilled by the fixture."""

# ruff: noqa: F811

import json

import pytest
from test_browser_setup import (  # noqa: F401
    EMAIL,
    PASSWORD,
    begin,
    chromium,
    setup_page,
)

from habfly.browser_setup import BrowserSetup, SetupStop

COOKIE = """<section role=dialog style='position:fixed;inset:0;background:white;z-index:99'>
<h2>We use cookies</h2><button onclick='this.parentNode.remove()'>Close</button></section>"""


def test_cookie_appears_after_initial_check_and_is_closed_before_any_fill(setup_page, tmp_path, monkeypatch):
    page, state = setup_page
    monkeypatch.setattr(BrowserSetup, "LOGIN_UI_SETTLE_SECONDS", 1.5)
    events = []
    setup = begin(page, tmp_path, events)
    setup.advance()  # Existing immediate fixture notice.
    page.evaluate("markup=>setTimeout(()=>document.body.insertAdjacentHTML('beforeend',markup),800)", COOKIE)
    setup.advance()
    assert setup.stage == "waiting_for_login_ui"
    assert page.get_by_placeholder("Email").input_value() == ""
    page.wait_for_timeout(1000)
    setup.advance()
    assert setup.stage == "cookie_notice_closed"
    assert page.get_by_placeholder("Email").input_value() == ""
    assert page.get_by_placeholder("Password").input_value() == ""
    assert state["posts"] == 0 and setup.login_ui_since is None
    setup.advance()
    page.wait_for_timeout(1550)
    setup.advance()
    assert state["posts"] == 1 and setup.submitted_login
    assert not setup.visited and not setup.star_clicked
    assert not list(tmp_path.iterdir())
    assert EMAIL not in json.dumps(events) and PASSWORD not in json.dumps(events)
    setup.close()


def test_cookie_arriving_from_password_fill_still_precedes_submit(setup_page, tmp_path):
    page, state = setup_page
    setup = begin(page, tmp_path)
    setup.advance()
    page.get_by_placeholder("Password").evaluate(
        "(e,markup)=>e.addEventListener('input',()=>document.body.insertAdjacentHTML('beforeend',markup),{once:true})",
        COOKIE,
    )
    setup.advance()
    assert state["posts"] == 0 and not setup.submitted_login
    assert setup.stage == "cookie_notice_closed" and setup.login_ui_since is None
    setup.advance()
    assert state["posts"] == 1 and setup.submitted_login
    setup.close()


@pytest.mark.parametrize("modal", [True, False])
def test_unknown_overlay_never_gets_dismissed_or_force_clicked(setup_page, tmp_path, modal):
    page, state = setup_page
    setup = begin(page, tmp_path)
    setup.advance()
    page.evaluate(
        "modal=>document.body.insertAdjacentHTML('beforeend',`<div ${modal?'role=dialog':''} style='position:fixed;inset:0;background:white;z-index:99'>Unrecognized panel</div>`)",
        modal,
    )
    if not modal:
        assert setup.advance() == "waiting"
        page.wait_for_timeout(3050)
    with pytest.raises(SetupStop, match="setup_unexpected_modal|setup_login_control_not_exposed"):
        setup.advance()
    assert state["posts"] == 0 and setup.closed
    assert page.get_by_placeholder("Email").input_value() == ""
    assert page.get_by_placeholder("Password").input_value() == ""
    assert not list(tmp_path.iterdir())


def test_nonmodal_backdrop_precedes_cookie_heading_and_close_animates(setup_page, tmp_path, monkeypatch):
    page, state = setup_page
    monkeypatch.setattr(BrowserSetup, "LOGIN_UI_SETTLE_SECONDS", 1.5)
    setup = begin(page, tmp_path)
    setup.advance()
    # This models the observed public paint sequence, not hidden application data.
    page.evaluate("""() => {
      setTimeout(()=>{
        const e=document.createElement('div');
        e.id='fixture-backdrop';e.style='position:fixed;inset:0;background:white;z-index:99';
        document.body.append(e);
        setTimeout(()=>{
          e.setAttribute('role','dialog');
          e.innerHTML='<h2>We use cookies</h2><button>Close</button>';
          e.querySelector('button').onclick=()=>{
            document.body.dataset.fixtureCloseCount=String(Number(document.body.dataset.fixtureCloseCount||0)+1);
            setTimeout(()=>e.remove(),400);
          };
        },200);
      },500);
    }""")
    setup.advance()
    page.wait_for_timeout(550)
    setup.advance()  # Unknown nonmodal cover: no dismissal and no credential fill.
    assert setup.stage == "waiting_for_login_ui"
    assert page.get_by_placeholder("Email").input_value() == ""
    assert state["posts"] == 0
    page.wait_for_timeout(250)
    setup.advance()  # The known notice is now visible; Close exactly once.
    assert setup.stage == "cookie_notice_closed"
    for _ in range(4):
        page.wait_for_timeout(50)
        setup.advance()
    assert page.locator("body").get_attribute("data-fixture-close-count") == "1"
    assert page.get_by_placeholder("Password").input_value() == ""
    page.wait_for_timeout(250)
    setup.advance()  # Original notice disappears, starts a new clean interval.
    page.wait_for_timeout(1550)
    setup.advance()
    assert state["posts"] == 1 and setup.submitted_login
    setup.close()


def test_replaced_cookie_notice_stops_instead_of_clicking_replacement(setup_page, tmp_path):
    page, state = setup_page
    setup = begin(page, tmp_path)
    page.get_by_role("heading", name="We use cookies").locator("..").get_by_role(
        "button", name="Close", exact=True
    ).evaluate("e=>e.onclick=()=>e.parentNode.outerHTML=e.parentNode.outerHTML")
    setup.advance()
    with pytest.raises(SetupStop, match="^setup_cookie_notice_changed_during_close$"):
        setup.advance()
    assert state["posts"] == 0 and setup.closed
    assert page.get_by_placeholder("Email").input_value() == ""
