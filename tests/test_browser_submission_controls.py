"""Binding tests; native geometry cases require root's explicit idle-browser gate.

Injected tests cover paired-handle lifecycle and the real submission owner. The
opt-in browser cases separately exercise the DOM/style/hit-test predicates; no
network request is continued and no course URL/account is opened.
"""
# ruff: noqa: F811

import os
from types import SimpleNamespace

import pytest
from test_browser_project_scoring_steps import case  # noqa: F401
from test_browser_project_submission_preflight import subject  # noqa: F401
from test_browser_project_submission_steps import create, setup  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_project_submission_preflight import _EXPOSED
from habfly.browser_submission_controls import (
    _IDENTITY,
    _LABEL_PROOF,
    READY,
    SUBMIT_EXPOSED,
    bind_readiness,
    outer_submit_box,
    submit_outer_exposed,
)


def _label_fixture(setup):
    """Inject only DOM proof/handles; real owner reservations/guards stay intact."""
    state, page = setup.state, setup.page
    ready = state.ready
    ready.exposed = False
    label = SimpleNamespace()
    text = SimpleNamespace()
    proof = {"x": 20, "y": 20, "width": 229, "height": 56}
    state.proof_valid = True
    state.label_clicks = 0
    text.count = lambda: 1
    text.is_visible = lambda: True
    text.element_handle = lambda **_: text
    text.evaluate = lambda script, other: script == _IDENTITY and other is text

    def evaluate(script, arg):
        if script == _IDENTITY:
            return arg is label
        assert script == _LABEL_PROOF
        assert arg["ready"] == READY
        return dict(proof) if state.proof_valid and arg["input"] is state.ready else None

    def click(**options):
        state.label_clicks += 1
        ready.click(**options)

    label.evaluate, label.click = evaluate, click
    ready.evaluate_handle = lambda *_: SimpleNamespace(as_element=lambda: label)
    page.get_by_text = lambda name, exact: text if name == READY and exact else None
    return label, text


def test_label_owner_reserves_once_and_reads_native_checked_state(setup):
    _label_fixture(setup)
    owner = create(setup)
    assert owner.advance()["phase"] == "selecting_readiness", owner.state()
    preflight = owner.book.json(owner.output / "preflight/confirmed.json")
    assert preflight["controls"]["readiness"]["click_surface"] == "associated_painted_label"
    assert preflight["controls"]["readiness"]["checked_state_source"] == "native_checkbox"
    assert setup.state.native == [] and setup.state.label_clicks == 0
    assert owner.advance()["phase"] == "submitting", owner.state()
    assert setup.state.native == ["readiness"] and setup.state.label_clicks == 1
    assert setup.state.ready.checked is True
    assert (owner.output / "readiness-dispatch-reserved.json").exists()
    assert owner.advance()["phase"] == "capturing_outcome", owner.state()
    assert setup.state.native == ["readiness", "submit"]
    owner.close()
    assert not owner.report["submitted"] and not owner.report["project_completed"]
    assert setup.case.journal.load().reduce().receipt("submission") is None


def test_binding_identity_rejects_input_replaced_under_same_label(setup):
    _label_fixture(setup)
    before = bind_readiness(setup.page, setup.state.ready, checked=False, exposed=_EXPOSED, prefix="test_")
    native = setup.state.ready
    replacement = type(native)(READY, "checkbox")
    replacement.exposed = False
    replacement.evaluate_handle = native.evaluate_handle
    setup.state.ready = replacement
    after = bind_readiness(setup.page, replacement, checked=False, exposed=_EXPOSED, prefix="test_")
    assert not before.evaluate(_IDENTITY, after)
    with pytest.raises(BrowserSafetyStop, match="unexposed|binding_changed"):
        before.click(timeout=100)
    assert setup.state.native == []


def test_lost_label_proof_after_reservation_never_clicks(setup):
    _label_fixture(setup)
    owner = create(setup)
    owner.advance()
    assert owner.phase == "selecting_readiness"
    setup.state.proof_valid = False
    owner.advance()
    assert owner.finished and owner.failure == "project_submission_readiness_control_unexposed"
    assert setup.state.native == [] and setup.state.label_clicks == 0
    assert len(setup.case.journal.load().reduce().pending) == 1
    assert not (owner.output / "readiness-dispatch-reserved.json").exists()


def test_rounded_outer_grid_uses_same_interior_points():
    box = {"x": 0, "y": 0, "width": 260, "height": 40, "hit_inset": 5}
    outer = outer_submit_box(box)
    assert [outer["x"] + 0.5, outer["x"] + outer["width"] / 2, outer["x"] + outer["width"] - 0.5] == [
        5,
        130,
        255,
    ]
    assert [outer["y"] + 0.5, outer["y"] + outer["height"] / 2, outer["y"] + outer["height"] - 0.5] == [
        5,
        20,
        35,
    ]
    assert box["width"] == 260


_NATIVE = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must grant an idle browser window before native fixtures",
)


@pytest.fixture(scope="module")
def controls_chromium():
    if os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1":
        pytest.skip("No native window granted")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def native_controls(controls_chromium):
    context = controls_chromium.new_context(viewport={"width": 1000, "height": 500})
    page = context.new_page()
    requests = []
    outer = "http://submission-controls.invalid/fixture"
    widget = "http://submission-controls.invalid/widget"
    body = f"""<style>
      body{{margin:0;background:#303038;color:white}}
      label{{position:absolute;left:20px;top:20px;width:229px;height:56px;cursor:pointer}}
      label:before{{content:'';position:absolute;left:0;top:0;width:20px;height:20px;background:rgba(255,255,255,.4)}}
      input{{position:absolute;left:0;top:5px;width:16px;height:16px;margin:0;opacity:0}}
      p{{margin:0 0 0 30px;width:190px;font:18px/28px sans-serif}}
      iframe{{position:absolute;left:280px;top:20px;width:260px;height:40px;border:0;border-radius:4px}}
      </style><label><input type=checkbox><p><span>{READY}</span></p></label>
      <iframe src='{widget}'></iframe>"""
    button = """<style>html,body{margin:0;overflow:hidden}button{width:260px;height:40px;
      border:0;border-radius:4px;background:#3565a5;color:white}</style><button>Submit Project</button>"""

    def route(request):
        requests.append(request.request.url)
        if request.request.url not in {outer, widget}:
            request.abort()
        else:
            request.fulfill(
                status=200, content_type="text/html", body=body if request.request.url == outer else button
            )

    context.route("**/*", route)
    try:
        page.goto(outer, wait_until="load")
        yield page, page.frame(url=widget)
        assert set(requests) <= {outer, widget}
    finally:
        context.close()


def _native_binding(page):
    checkbox = page.get_by_role("checkbox", name=READY, exact=True)
    return bind_readiness(page, checkbox, checked=False, exposed=_EXPOSED, prefix="test_")


@_NATIVE
def test_native_painted_label_click_and_rounded_iframe_button(native_controls):
    page, frame = native_controls
    binding = _native_binding(page)
    assert binding.mode == "associated_painted_label"
    assert not page.get_by_role("checkbox").is_checked()
    binding.click(timeout=1000)
    assert page.get_by_role("checkbox").is_checked()
    button = frame.get_by_role("button", name="Submit Project", exact=True)
    box = button.evaluate(SUBMIT_EXPOSED)
    assert box["hit_inset"] == 5
    assert submit_outer_exposed(frame, [outer_submit_box(box)])


@_NATIVE
@pytest.mark.parametrize(
    "mutation", ["hidden", "clipped", "covered", "wrong_association", "unpainted", "duplicate_text"]
)
def test_native_label_rejects_unproved_surfaces(native_controls, mutation):
    page, _ = native_controls
    scripts = {
        "hidden": "document.querySelector('label').style.opacity='0'",
        "clipped": "document.querySelector('label').style.left='-1px'",
        "covered": "document.body.insertAdjacentHTML('beforeend','<div style=\"position:absolute;left:20px;top:20px;width:229px;height:56px;background:red\"></div>')",
        "wrong_association": f"document.querySelector('input').setAttribute('aria-label',{READY!r});document.querySelector('label').htmlFor='missing-native-input'",
        "unpainted": "document.head.insertAdjacentHTML('beforeend','<style>label:before{background:transparent}</style>')",
        "duplicate_text": f"document.body.insertAdjacentHTML('beforeend','<div style=\"position:absolute;top:100px\">{READY}</div>')",
    }
    page.evaluate(scripts[mutation])
    with pytest.raises(BrowserSafetyStop, match="unexposed"):
        _native_binding(page)
    assert not page.locator("input").is_checked()


@_NATIVE
def test_native_original_input_and_label_replacement_cannot_reuse_binding(native_controls):
    page, _ = native_controls
    old = _native_binding(page)
    page.locator("input").evaluate("e=>e.replaceWith(e.cloneNode(true))")
    new = _native_binding(page)
    assert not old.evaluate(_IDENTITY, new)
    with pytest.raises(BrowserSafetyStop, match="binding_changed"):
        old.click(timeout=1000)
    assert not page.locator("input").is_checked()


@_NATIVE
@pytest.mark.parametrize("mutation", ["replaced_label", "covered_label", "extra_control"])
def test_native_label_final_guard_stops_changed_click_surface(native_controls, mutation):
    page, _ = native_controls
    binding = _native_binding(page)
    if mutation == "replaced_label":
        page.locator("label").evaluate("e=>e.replaceWith(e.cloneNode(true))")
    elif mutation == "covered_label":
        page.evaluate(
            "document.body.insertAdjacentHTML('beforeend','<div style=\"position:absolute;left:20px;top:20px;width:229px;height:56px;background:red\"></div>')"
        )
    else:
        page.locator("label").evaluate("e=>e.insertAdjacentHTML('beforeend','<button>Other action</button>')")
    with pytest.raises(BrowserSafetyStop, match="binding_changed|unexposed"):
        binding.click(timeout=1000)
    assert not page.locator("input").is_checked()


@_NATIVE
def test_native_button_center_or_outer_frame_overlay_refused(native_controls):
    page, frame = native_controls
    button = frame.get_by_role("button")
    frame.evaluate(
        "document.body.insertAdjacentHTML('beforeend','<div style=\"position:absolute;left:100px;top:10px;width:60px;height:20px;background:red\"></div>')"
    )
    assert button.evaluate(SUBMIT_EXPOSED) is None
    frame.locator("div").evaluate("e=>e.remove()")
    box = button.evaluate(SUBMIT_EXPOSED)
    page.evaluate(
        "document.body.insertAdjacentHTML('beforeend','<div style=\"position:absolute;left:280px;top:20px;width:260px;height:40px;background:red\"></div>')"
    )
    assert not submit_outer_exposed(frame, [outer_submit_box(box)])
    page.locator("body>div").evaluate("e=>e.remove()")
    page.locator("body").evaluate("e=>e.style.opacity='0'")
    assert not submit_outer_exposed(frame, [outer_submit_box(box)])
    page.locator("body").evaluate("e=>e.style.opacity='1'")
    page.locator("iframe").evaluate("e=>e.style.left='-1px'")
    assert not submit_outer_exposed(frame, [outer_submit_box(box)])


@_NATIVE
def test_native_exposed_checkbox_keeps_legacy_native_path(native_controls):
    page, _ = native_controls
    page.locator("input").evaluate("e=>e.style.opacity='1'")
    binding = _native_binding(page)
    assert binding.mode == "native_input"
    binding.click(timeout=1000)
    assert page.locator("input").is_checked()
