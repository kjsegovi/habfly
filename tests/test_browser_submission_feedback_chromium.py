"""Intercepted native feedback geometry, not real course completion evidence."""

import html
import os

import pytest

from habfly.browser_submission_feedback import (
    READY,
    REFUSAL,
    classify_submission_feedback,
    read_submission_feedback_exposure,
    submission_refusal_candidate,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must authorize intercepted native browser fixtures",
)


@pytest.fixture(scope="module")
def feedback_chromium():
    if os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1":
        pytest.skip("No native gate authorized")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.mark.parametrize(
    "variant,accepted",
    [
        ("direct", True),
        ("span", True),
        ("nested_span", True),
        ("sibling_overlay", False),
        ("transparent_span", False),
        ("offscreen_text", False),
        ("interactive_span", False),
        ("unrelated_span_overlay", False),
    ],
)
def test_native_paragraph_wrapper_is_not_mistaken_for_sibling_occlusion(feedback_chromium, variant, accepted):
    context = feedback_chromium.new_context(viewport={"width": 1000, "height": 800}, service_workers="block")
    page = context.new_page()
    requests = []
    url = "http://submission-feedback.invalid/fixture"
    text = html.escape(REFUSAL)
    wrapped = text if variant == "direct" else f"<span>{text}</span>"
    if variant == "nested_span":
        wrapped = f"<span><span>{text}</span></span>"
    if variant == "interactive_span":
        wrapped = f"<span tabindex='0'>{text}</span>"
    feedback = (
        '<button id="ok">OK</button>'
        '<button aria-label="Close feedback">Open/Close Feedback</button>'
        f'<p id="refusal">{wrapped}</p>'
        '<button aria-label="Close feedback">&#xf00d;</button>'
    )
    body = f"""<style>
      body{{margin:20px;background:#1c1c1c;color:white;font:18px/24px sans-serif}}
      button{{display:block;min-width:130px;height:44px;border-radius:4px;border:0;margin:8px}}
      #refusal{{width:650px;line-height:24px;margin:16px 0}}
      </style><main><div role="group" aria-label="Select all that apply">
      <input type="checkbox" checked aria-label="{READY}"><p>{READY}</p></div>
      <button id="submit">Submit Project</button></main>
      <template>{feedback}</template><script>
      document.querySelector('#submit').onclick=()=>document.querySelector('main').append(
        document.querySelector('template').content.cloneNode(true));
      </script>"""

    def route(request):
        requests.append(request.request.url)
        request.fulfill(status=200, content_type="text/html", body=body if request.request.url == url else "")

    context.route("**/*", route)

    def snapshot():
        return {
            "text": page.locator("body").inner_text(),
            "accessibility": page.locator("body").aria_snapshot(),
        }

    try:
        page.goto(url)
        before = snapshot()
        page.get_by_role("button", name="Submit Project", exact=True).click()
        # Test-owned fixture mutations; no real application or hidden state read.
        if variant == "transparent_span":
            page.locator("#refusal span").evaluate("e=>e.style.opacity='0'")
        elif variant == "offscreen_text":
            page.locator("#refusal").evaluate("e=>{e.style.position='fixed';e.style.top='-25px'}")
        elif variant in {"sibling_overlay", "unrelated_span_overlay"}:
            page.locator("#refusal").evaluate(
                "(e,tag)=>{const r=e.getBoundingClientRect(),o=document.createElement(tag);"
                "Object.assign(o.style,{position:'fixed',left:r.x+'px',top:r.y+'px',"
                "width:r.width+'px',height:r.height+'px',background:'red',zIndex:100});"
                "document.body.append(o)}",
                "span" if variant == "unrelated_span_overlay" else "div",
            )
        after = snapshot()
        assert submission_refusal_candidate(before, after)
        guards = []

        def guard():
            assert page.url == url and page.context.pages == [page] and page.frames == [page.main_frame]
            guards.append(True)

        proof = read_submission_feedback_exposure(page, after, guard=guard, timeout=lambda: 2000)
        result = classify_submission_feedback(before, after, submit_dispatched=True, exposure=proof)
        assert (result["status"] == "course_refusal") is accepted
        assert all(
            result[k] is False
            for k in (
                "submitted",
                "task_completed",
                "project_completed",
                "canonical_receipt",
                "automatic_retry",
            )
        )
        assert requests and set(requests) <= {url, "http://submission-feedback.invalid/favicon.ico"}
        assert guards
    finally:
        context.close()
