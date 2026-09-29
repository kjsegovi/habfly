"""Opt-in real Chromium pagination over wholly intercepted fixture documents.

No parser, exposure check, source validator or handle comparison is replaced.
The notice-fade test uses a short fixture DOM timer and observes the real notice
predicate without altering its result. Only settled raw captures may persist. This is
transport testing, not evidence of a live HabWorlds collection or assessment.

Root must grant an idle browser window before HABFLY_INTERCEPTED_BROWSER_IDLE=1
is set. Collection is safe without that opt-in; every network request is either
fulfilled locally or aborted and no real preview/account URL is opened.
"""

import hashlib
import html
import json
import os
from types import SimpleNamespace

import pytest
from playwright.sync_api import sync_playwright
from test_browser_numeric import OUTER, WIDGET, config

import habfly.browser_project_paginated_inventory_steps as module
from habfly.browser_no_planet_workflow import _Evidence
from habfly.browser_project_navigation import LIST_LABELS
from habfly.browser_stellar import SIMULATION_URL
from habfly.project_inventory_source import load_inventory_source

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must grant an idle window for intercepted native pagination fixtures",
)


def _document(total, *, notice=False):
    header = "Funding $50000 Data Quality 0% Scavenger Hunt 0/8 Observations Analyzed Data Star " + " ".join(
        LIST_LABELS["stellar"]
    )
    return (
        "<!doctype html><meta charset=utf-8><style>"
        "html,body{margin:0;padding:0;width:950px;height:600px;overflow:hidden;font:14px Arial}"
        "#header{position:absolute;left:30px;top:25px;width:885px}"
        "#rows{position:absolute;left:30px;top:150px;width:890px}"
        ".star-row{display:flex;width:890px;height:33px;align-items:flex-start}"
        ".eye,.delete{display:block;width:22px;height:25px;flex:none}"
        ".row-data{display:flex}.name{width:85px}.values{white-space:nowrap}"
        "#notice{position:absolute;left:320px;top:525px;height:20px}"
        ".pager{position:absolute;top:555px;width:30px;height:30px;padding:0;border:1px solid #444}"
        "#previous{background:linear-gradient(45deg,#ddd,#bbb)}"
        "#next{background:linear-gradient(-45deg,#ddd,#bbb)}"
        "#counts{position:absolute;left:420px;top:538px;text-transform:uppercase;font-size:10px;line-height:11px}"
        "#counts span{display:block}"
        "#save{position:absolute;left:700px;top:555px;height:30px;width:70px}"
        "</style>"
        f'<div id="header">{html.escape(header)}</div><div id="rows"></div>'
        f'<div id="notice"{"" if notice else " hidden"}>Data saved</div>'
        '<button id="previous" class="pager" type="button"></button>'
        '<button id="next" class="pager" type="button"></button>'
        '<div id="counts"></div><button id="save" type="button">Save</button>'
        "<script>"
        f"const total={total};let start=0;"
        "function render(){"
        "const end=Math.min(start+10,total), rows=document.getElementById('rows');"
        "rows.replaceChildren();"
        "for(let i=start;i<end;i++){const row=document.createElement('div');row.className='star-row';"
        "const name='Star'+String(i).padStart(2,'0');"
        'row.innerHTML=\'<svg class="eye" role="img"><rect width="22" height="25" fill="#444"/></svg>\''
        "+'<div class=\"row-data\"><div><div class=\"name\">'+name+'</div></div>'"
        "+'<div class=\"values\"> 0.045 370 5.68E-10 1 1 UV 1 Main Sequence 1 1 1 Ga</div></div>'"
        # The real list has unnamed trailing row icons. Besides matching that
        # public structure, this keeps a footer notice separate in native AX.
        '+\'<svg class="delete" role="img"><rect width="22" height="25" fill="#777"/></svg>\';'
        "rows.append(row);}"
        "const left=start===0?323.3359375:318.8828125;"
        "document.getElementById('previous').style.left=left+'px';"
        "document.getElementById('next').style.left=(left+34.453125)+'px';"
        "document.getElementById('previous').disabled=start===0;"
        "document.getElementById('next').disabled=end===total;"
        "document.getElementById('counts').innerHTML='<span>viewing</span><span>'+(start+1)+'-'+end+' of '+total+'</span><span>total collected</span><span>'+total+'</span>';"
        "}"
        "for(const [id,delta] of [['previous',-10],['next',10]]){"
        "document.getElementById(id).addEventListener('click',async()=>{"
        "await fixturePagerDispatch(id);start+=delta;render();});}"
        "document.getElementById('save').addEventListener('click',()=>fixtureForbiddenAction('Save'));"
        "render();</script>"
    )


@pytest.fixture(scope="module")
def pager_chromium():
    if os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1":
        pytest.skip("No idle native browser window granted")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def native_pager(pager_chromium, tmp_path):
    contexts, subjects = [], []

    def create(total=11, *, notice=False, emit=lambda _: None, viewport=None, outer_margin="20px"):
        context = pager_chromium.new_context(
            viewport=viewport or {"width": 1400, "height": 950}, service_workers="block"
        )
        contexts.append(context)
        page = context.new_page()
        requests, clicks, forbidden = [], [], []
        root = tmp_path / f"native-pager-{len(subjects)}"
        root.mkdir()
        subject = SimpleNamespace(
            root=root,
            page=page,
            requests=requests,
            clicks=clicks,
            forbidden=forbidden,
            owner=None,
            diagnostics=[],
        )
        subjects.append(subject)

        def dispatch(direction):
            owner = subject.owner
            assert owner is not None
            directory = owner.output / "transitions" / f"{owner.attempts:02d}"
            reserved = json.loads((directory / "reserved.json").read_bytes())
            assert reserved["direction"] == ("next" if direction == "next" else "previous")
            assert reserved["max_clicks"] == 1 and reserved["automatic_retry"] is False
            clicks.append(direction)

        page.expose_function("fixturePagerDispatch", dispatch)
        page.expose_function("fixtureForbiddenAction", lambda value: forbidden.append(value))
        outer = (
            "<!doctype html><style>body{margin:" + outer_margin + "}iframe{display:block;border:0;margin:0}"
            "#simulation{width:950px;height:600px}.widget{width:950px;height:65px}</style>"
            f'<iframe id="simulation" name="simulation" src="{SIMULATION_URL}"></iframe>'
            f'<iframe class="widget" src="{WIDGET}"></iframe>'
            f'<iframe class="widget" src="{WIDGET}"></iframe>'
        )
        documents = {
            OUTER: outer,
            SIMULATION_URL: _document(total, notice=notice),
            WIDGET: '<button onclick="fixtureForbiddenAction(this.textContent)">Update Score</button>'
            '<button onclick="fixtureForbiddenAction(this.textContent)">Submit Project</button>',
        }

        def route(request):
            requests.append(request.request.url)
            document = documents.get(request.request.url)
            if document is None:
                request.abort()
            else:
                request.fulfill(status=200, content_type="text/html; charset=utf-8", body=document)

        context.route("**/*", route)
        page.goto(OUTER, wait_until="load")
        subject.frame = page.frame(name="simulation")
        subject.frame.locator(".name").first.wait_for(state="visible", timeout=5000)
        subject.names = [f"Star{i:02d}" for i in range(total)]
        subject.owner = module.PaginatedInventorySteps(
            page,
            config(),
            root / "inventory",
            run_history=root,
            expected_stars=subject.names,
            max_seconds=240,
            max_clicks=4,
            max_advances=8,
            emit=emit,
        )
        original_read = subject.owner._read

        def diagnostic_read():
            try:
                return original_read()
            except Exception as exc:
                # Fixture-only diagnostics preserve the exact failure and only
                # read public body text/AX. No validator result is replaced.
                failure = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "text": subject.frame.locator("body").inner_text(),
                    "accessibility": subject.frame.locator("body").aria_snapshot(),
                }
                subject.diagnostics.append(failure)
                (root / "fixture-read-failure.json").write_text(json.dumps(failure, indent=2))
                raise

        subject.owner._read = diagnostic_read
        assert not clicks and not forbidden
        return subject

    try:
        yield create
    finally:
        for subject in subjects:
            if subject.owner is not None:
                subject.owner.close()
            assert not subject.forbidden
            assert set(subject.requests) <= {OUTER, SIMULATION_URL, WIDGET}
        for context in contexts:
            context.close()


def finish(subject):
    for _ in range(8):
        if subject.owner.finished:
            break
        old = len(subject.clicks)
        subject.owner.advance()
        assert len(subject.clicks) - old <= 1
    return subject.owner.state()


def assert_stopped(subject):
    assert subject.owner.finished and subject.owner.status == "stopped", subject.owner.state()
    assert not (subject.owner.output / "confirmed.json").exists()
    assert not subject.clicks
    for key in ("task_completed", "project_completed", "scientific_verified", "training_label"):
        assert subject.owner.state()[key] is False
    previous = list(subject.clicks)
    subject.owner.advance()
    assert subject.clicks == previous


@pytest.mark.parametrize("total", [11, 30])
def test_native_full_forward_reverse_and_actual_anchor(native_pager, total):
    subject = native_pager(total)
    state = finish(subject)
    assert state["status"] == "completed", (state, subject.diagnostics)
    assert subject.clicks == ["next"] * ((total - 1) // 10) + ["previous"] * ((total - 1) // 10)
    receipt, rows, anchor = module.load_live_paginated_inventory(subject.root, subject.owner.output)
    assert [row["name"] for row in rows] == subject.names
    assert module.parse_inventory_page(anchor)[0] == {"start": 1, "end": 10, "total": total}
    source = load_inventory_source(_Evidence(subject.root), subject.owner.output)
    assert source.anchor == anchor and source.kind == "live_paginated"
    assert receipt["complete_collection_traversal_verified"] is True
    assert receipt["complete_visible_list_verified"] is False
    assert receipt["whole_collection_sha256"] != receipt["visible_assessment_anchor_sha256"]
    for path, digest in receipt["source_sha256"].items():
        assert hashlib.sha256((subject.root / path).read_bytes()).hexdigest() == digest
    assert all(receipt[key] is False for key in module.FALSE)
    assert all(receipt[key] == 0 for key in module.ZERO)


@pytest.mark.parametrize(
    "viewport,top,complete",
    [
        ({"width": 1600, "height": 1100}, 100, True),
        ({"width": 1280, "height": 720}, 100, True),
        ({"width": 1280, "height": 720}, 200, False),
    ],
)
def test_explicit_project_viewport_and_clipped_legacy_size(native_pager, viewport, top, complete):
    # Match the diagnostic preview's exposed embed position without reading
    # or changing any actual preview; move it down only in the clipped case.
    # The smaller viewport alone is not evidence of clipping.
    subject = native_pager(viewport=viewport, outer_margin=f"{top}px 0 0 325px")
    state = finish(subject)
    if complete:
        assert state["status"] == "completed", (state, subject.diagnostics)
        assert subject.clicks == ["next", "previous"]
    else:
        assert_stopped(subject)
        assert subject.owner.attempts == 0


def test_native_paired_footer_notice_settles_before_raw_source_captures(native_pager, monkeypatch):
    subject = native_pager(notice=True)
    original = module.paired_footer_autosave
    observed = []

    def observe(report):
        result = original(report)
        observed.append(result)
        return result

    initial = module.inspect_page(subject.page, config())
    (subject.root / "fixture-initial-notice.json").write_text(json.dumps(initial, indent=2))
    assert original(initial) is True, initial["frames"][0]
    monkeypatch.setattr(module, "paired_footer_autosave", observe)
    subject.frame.locator("#notice").evaluate("e=>setTimeout(()=>{e.hidden=true},1000)")
    state = finish(subject)
    assert state["status"] == "completed", (state, subject.diagnostics)
    assert True in observed and False in observed
    for path in subject.owner.output.rglob("observation.json"):
        capture = json.loads(path.read_bytes())
        frame = next(f for f in capture["frames"] if f["url"] == SIMULATION_URL)
        assert "Data saved" not in frame["text"] and "Data saved" not in frame["accessibility"]
    assert not subject.owner.report["cross_session_persistence_verified"]


@pytest.mark.parametrize("fault", ["unavailable", "pager_occluded", "row_occluded"])
def test_unavailable_or_occluded_native_surface_stops_before_any_click(native_pager, fault):
    subject = native_pager()
    if fault == "unavailable":
        subject.frame.locator("#next").evaluate("e=>e.disabled=true")
    else:
        target = "#next" if fault == "pager_occluded" else ".name"
        subject.frame.locator(target).first.evaluate("""e=>{
            const r=e.getBoundingClientRect(), cover=document.createElement('div');
            cover.style=`position:fixed;left:${r.x}px;top:${r.y}px;width:${r.width}px;height:${r.height}px;background:#fff;z-index:99`;
            document.body.append(cover);
        }""")
    subject.owner.advance()
    assert_stopped(subject)
    assert subject.owner.attempts == 0


@pytest.mark.parametrize("target", ["#next", ".name"])
def test_changed_identical_native_handle_stops_before_dispatch(native_pager, target):
    subjects = []

    def emit(event):
        if event["event"] == "action_proposed":
            subjects[0].frame.locator(target).first.evaluate("e=>e.replaceWith(e.cloneNode(true))")

    subject = native_pager(emit=emit)
    subjects.append(subject)
    subject.owner.advance()
    assert subject.owner.phase == "forward_next", subject.owner.state()
    subject.owner.advance()
    assert_stopped(subject)
    assert subject.owner.failure == "paginated_inventory_live_native_binding_changed"
    assert subject.owner.attempts == 0
