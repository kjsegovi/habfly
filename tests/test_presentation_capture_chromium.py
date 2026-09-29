"""Isolated intercepted pages only: the top-level mascot never enters crops."""

import io
import os

import pytest
from PIL import Image
from playwright.sync_api import sync_playwright

from habfly.presentation_capture import (
    MASCOT_OVERLAY_ATTRIBUTE,
    MASCOT_OVERLAY_ID,
    MASCOT_OVERLAY_MARKER,
    evidence_screenshot,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Coordinate the isolated Chromium slot before these synthetic fixtures",
)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 700, "height": 500}, service_workers="block")
    seen = []

    def intercept(route):
        url = route.request.url
        seen.append(url)
        if url in {"http://fixture.test/frame", "http://other-fixture.test/frame"}:
            body = '<style>body{margin:0}#chart{width:280px;height:195px;background:rgb(80,132,154)}</style><div id="chart"></div>'
        elif url == "http://fixture.test/":
            body = '<style>body{margin:0;background:white}iframe{position:absolute;left:40px;top:80px;width:300px;height:240px;border:0}</style><button>Student button</button><iframe title="Fixture chart"></iframe>'
        else:
            route.abort()
            return
        route.fulfill(status=200, content_type="text/html", body=body)

    context.route("**/*", intercept)
    page = context.new_page()
    page.goto("http://fixture.test/")
    yield page
    assert all(
        url in {"http://fixture.test/", "http://fixture.test/frame", "http://other-fixture.test/frame"}
        for url in seen
    )
    context.close()


def overlay(page):
    page.evaluate(
        """({id,attribute,marker}) => {
          const host=document.createElement('div');host.id=id;host.setAttribute(attribute,marker);
          host.setAttribute('aria-hidden','true');host.setAttribute('role','presentation');host.inert=true;
          host.style='position:fixed;inset:0;z-index:2147483647;pointer-events:none';
          const shadow=host.attachShadow({mode:'closed'});
          const canvas=document.createElement('canvas');
          canvas.width=700;canvas.height=500;
          canvas.getContext('2d').fillStyle='red';canvas.getContext('2d').fillRect(0,0,700,500);
          shadow.append(canvas);document.body.append(host);
        }""",
        {"id": MASCOT_OVERLAY_ID, "attribute": MASCOT_OVERLAY_ATTRIBUTE, "marker": MASCOT_OVERLAY_MARKER},
    )


@pytest.mark.parametrize("frame_url", ["http://fixture.test/frame", "http://other-fixture.test/frame"])
@pytest.mark.parametrize("kind", ["page", "locator", "handle"])
def test_parent_overlay_excluded_from_every_capture_surface(page, frame_url, kind):
    page.locator("iframe").evaluate("(e,url)=>e.src=url", frame_url)
    frame = page.frame(url=frame_url)
    if frame is None:
        page.locator("iframe").element_handle().content_frame().wait_for_url(frame_url)
        frame = page.frame(url=frame_url)
    chart = frame.locator("#chart")
    chart.wait_for(state="visible")
    target = page if kind == "page" else chart if kind == "locator" else chart.element_handle()
    options = {"full_page": False} if kind == "page" else {}
    original_text = page.locator("body").inner_text()
    original_ax = page.locator("body").aria_snapshot()
    baseline = target.screenshot(**options)
    assert evidence_screenshot(target, **options) == baseline
    overlay(page)
    assert page.locator("body").inner_text() == original_text
    assert page.locator("body").aria_snapshot() == original_ax
    assert target.screenshot(**options) != baseline
    assert evidence_screenshot(target, **options) == baseline
    assert page.locator(f"#{MASCOT_OVERLAY_ID}").evaluate("e=>getComputedStyle(e).visibility") == "visible"
    assert target.screenshot(**options) != baseline


def test_existing_style_applies_and_unrelated_overlay_is_not_hidden(page):
    page.evaluate(
        """() => {
          const e=document.createElement('div');e.id='student-overlay';
          e.style='position:fixed;left:0;top:0;width:10px;height:10px;background:blue;z-index:100';
          document.body.append(e);
        }"""
    )
    style = "body { background: rgb(0, 255, 0) !important; }"
    baseline = page.screenshot(style=style)
    overlay(page)
    actual = evidence_screenshot(page, style=style)
    assert actual == baseline
    pixels = Image.open(io.BytesIO(actual)).convert("RGB")
    assert pixels.getpixel((2, 2)) == (0, 0, 255)
    assert pixels.getpixel((650, 450)) == (0, 255, 0)


@pytest.mark.parametrize("tag,marker", [("div", None), ("div", "not-mascot"), ("section", "mascot-v1")])
def test_same_id_without_exact_owned_host_marker_is_not_hidden(page, tag, marker):
    baseline = page.screenshot()
    page.evaluate(
        """({tag,id,attribute,marker}) => {
          const e=document.createElement(tag);e.id=id;
          if(marker!==null)e.setAttribute(attribute,marker);
          e.style='position:fixed;inset:0;z-index:2147483647;background:red;pointer-events:none';
          document.body.append(e);
        }""",
        {"tag": tag, "id": MASCOT_OVERLAY_ID, "attribute": MASCOT_OVERLAY_ATTRIBUTE, "marker": marker},
    )
    unrelated = page.screenshot()
    assert unrelated != baseline
    assert evidence_screenshot(page) == unrelated
