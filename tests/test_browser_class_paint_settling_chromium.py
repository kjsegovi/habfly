"""Explicit idle-slot native CSS animation fixtures; all traffic intercepted."""
# ruff: noqa: F401,F811 -- imported pytest fixtures

import json

import pytest
from test_browser_class_setup_steps_intercepted import (
    assert_exact_events,
    create_steps,
    isolated_chromium,
    native_fresh_star,
    pytestmark,
)

import habfly.browser_classification as module


def test_native_cssom_serializes_tiny_opacity_as_exponent(isolated_chromium):
    context = isolated_chromium.new_context()
    context.route("**/*", lambda route: route.abort())
    try:
        page = context.new_page()
        samples = {}
        for value in ("0.0000001", "1e-12", "1e-38", "1e-45"):
            page.set_content(
                '<style>label::after {content:"";opacity:' + value + "}</style><label>Fixture</label>"
            )
            samples[value] = page.locator("label").evaluate(module.RENDERING)["dotOpacity"]
        print("Native fixture computed opacity:", json.dumps(samples, sort_keys=True))
        assert float(samples["0.0000001"]) == 1e-7
        assert "e" in samples["0.0000001"].casefold()
        assert all(module._diagnostic_property("dotOpacity", value) == value for value in samples.values())
    finally:
        context.close()


def install_animation(frame, mode):
    frame.add_style_tag(
        content="""
      .choice label::after {transition:opacity 1.2s linear !important}
      @keyframes unsettled {from {opacity:0.2} to {opacity:0.8}}
      body.permanent .choice label.selected::after {animation:unsettled 1s linear infinite alternate}
    """
    )
    frame.evaluate(
        """mode => {
      const old=window.choose;
      window.choose=(element,main)=>{
        old(element,main);
        if(mode==='permanent') document.body.classList.add('permanent');
      };
    }""",
        mode,
    )


def record_settling(monkeypatch, frame=None):
    calls = []
    original = module.StellarSelectionSession._settle_post_click_paint

    def settling(session, error):
        calls.append(error.diagnostic.copy())
        if frame is not None:
            # Fixture-only external mutation, never a production hook.
            frame.evaluate("() => document.querySelector('#distance').value='1'")
        return original(session, error)

    monkeypatch.setattr(module.StellarSelectionSession, "_settle_post_click_paint", settling)
    return calls


def test_native_inherited_main_animation_reaches_exact_paint_with_one_click_each(
    native_fresh_star,
    monkeypatch,
):
    page, frame, history, fresh, actions = native_fresh_star(inherited_main=True)
    install_animation(frame, "finite")
    calls = record_settling(monkeypatch)
    events = []
    owner = create_steps(page, history, fresh, "main_sequence", "Ga", events)
    while not owner.finished:
        owner.advance()
    assert owner.report["setup_verified"], owner.report
    assert len(calls) >= 1
    assert all(call["validation_stage"] == "paint_state" for call in calls)
    assert actions == [
        {"target": "stellar_class", "value": "white_dwarf"},
        {"target": "stellar_class", "value": "main_sequence"},
        {"target": "lifetime_prefix", "value": "Ga"},
    ]
    assert not (owner.output / "class-rendering-failure.json").exists()
    assert_exact_events(owner, events)


def test_native_tiny_opacity_readback_still_requires_exact_final_paint(native_fresh_star, monkeypatch):
    page, frame, history, fresh, actions = native_fresh_star(inherited_main=True)
    frame.add_style_tag(
        content="""
      .choice label.tiny::after {opacity:0.0000001 !important}
    """
    )
    frame.evaluate("""() => {
      const old=window.choose;
      window.choose=(element,main)=>{
        const previous=document.querySelector('.choice label.selected');
        old(element,main);
        previous.classList.add('tiny');
        setTimeout(()=>previous.classList.remove('tiny'),1200);
      };
    }""")
    calls = record_settling(monkeypatch)
    events = []
    owner = create_steps(page, history, fresh, "main_sequence", "Ga", events)
    while not owner.finished:
        owner.advance()
    assert owner.report["setup_verified"], owner.report
    assert any(call["visible_rendering"]["dotOpacity"] == "1e-07" for call in calls)
    assert actions == [
        {"target": "stellar_class", "value": "white_dwarf"},
        {"target": "stellar_class", "value": "main_sequence"},
        {"target": "lifetime_prefix", "value": "Ga"},
    ]
    assert not (owner.output / "class-rendering-failure.json").exists()
    assert_exact_events(owner, events)


@pytest.mark.parametrize("mode", ["permanent", "changed_answer"])
def test_native_animation_never_retries_click_or_accepts_changed_data(native_fresh_star, monkeypatch, mode):
    page, frame, history, fresh, actions = native_fresh_star(inherited_main=True)
    install_animation(frame, "permanent" if mode == "permanent" else "finite")
    calls = record_settling(monkeypatch, frame if mode == "changed_answer" else None)
    owner = create_steps(page, history, fresh, "main_sequence", "Ga", [])
    owner.advance()
    assert calls and owner.report["status"] == "stopped"
    assert not owner.report["setup_verified"]
    assert actions == [{"target": "stellar_class", "value": "white_dwarf"}]
    if mode == "permanent":
        assert owner.report["failure_reason"] == "unsupported_class_circle_rendering"
        diagnostic = json.loads((owner.output / "class-rendering-failure.json").read_bytes())
        assert diagnostic["post_click_settling"]["settled"] is False
        assert diagnostic["native_class_click_invoked"] is True
        assert diagnostic["native_class_click_returned"] is True
    else:
        assert owner.report["failure_reason"] in {
            "screen_changed_during_class_read",
            "stellar_data_changed_by_class",
        }
    owner.advance()
    assert len(actions) == 1
    assert not (owner.output / "class/confirmed.json").exists()
