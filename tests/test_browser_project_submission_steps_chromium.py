"""Opt-in intercepted native submission transport, not course acceptance.

Only ``_sources`` (the offline prerequisite loader, in its two importing modules)
is injected. Synthetic thirty-task/assessment records explicitly bypass evidence
eligibility for this transport test. Native preflight, captures, exposure, stable
handles, reservation, checked readback, screenshots and dispatch remain real.

Root must grant an idle native-browser window before setting
HABFLY_INTERCEPTED_BROWSER_IDLE=1. Collection never launches a browser. No request
is continued to a network origin and no actual preview/account is opened.
"""

import hashlib
import html
import json
import os
from types import SimpleNamespace

import pytest
from playwright.sync_api import sync_playwright
from test_browser_numeric import OUTER, WIDGET, config
from test_browser_project_scoring_steps import thirty_tasks
from test_project_assessment import visible
from test_project_progress import receipt_for

import habfly.browser_project_submission_preflight as preflight
import habfly.browser_project_submission_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json, rows_hash
from habfly.browser_stellar import SIMULATION_URL
from habfly.project_assessment import parse_assessment_text
from habfly.project_progress import ProjectJournal, WriteReserved, _json

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must grant an idle window for intercepted native submission fixtures",
)

ESCAPE = "http://localhost/outside-submission-fixture"


@pytest.fixture(scope="module")
def submission_chromium():
    # A second guard at the actual launch site prevents an accidental marker bypass.
    if os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1":
        pytest.skip("No idle browser window granted")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def native_submission(submission_chromium, tmp_path, monkeypatch):
    root = tmp_path / "synthetic-submission-only"
    root.mkdir()
    text = visible(mode="scavenger_hunt", funding=49800, overall=31, collected=30)
    text = text.replace(
        "ANALYZED DATA STAR\n", "ANALYZED DATA STAR\n" + " ".join(f"Star{i:02d}" for i in range(30)) + "\n"
    )
    row_sha = rows_hash(text)
    progress = thirty_tasks()
    for kind in ("assessment_data_quality", "assessment_scavenger_hunt", "score_transfer"):
        revision = progress.reduce().revision
        progress = progress.append(
            WriteReserved(
                action_id="fixture-" + kind,
                write_kind=kind,
                revision=revision,
                project_rows_sha256=row_sha,
                before_sha256=hashlib.sha256((kind + "-fixture-before").encode()).hexdigest(),
            )
        )
        progress = progress.append(receipt_for(progress))
    journal = ProjectJournal(root, project_id=progress.project_id, attempt_id=progress.attempt_id).create()
    journal.path.write_text(
        _json(progress.header())
        + "\n"
        + "".join(_json(r.model_dump(mode="json")) + "\n" for r in progress.records)
    )
    scoring_dir = root / "injected-offline-scoring"
    scoring_dir.mkdir()
    source = scoring_dir / "synthetic-source.json"
    synthetic = {
        "test_only": True,
        "real_course_eligibility_verified": False,
        "project_rows_sha256": row_sha,
        "assessment": parse_assessment_text(text).model_dump(mode="json"),
        "score": 43.4,
    }
    persist_json(source, synthetic)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    def offline_sources(book, actual_journal, actual_directory):
        """The sole injected seam. Never certifies a real artifact chain."""
        assert actual_journal is journal and book.clean(actual_directory) == scoring_dir
        assert hashlib.sha256(book.read(source)).hexdigest() == source_sha
        assert book.json(source) == synthetic
        book.read(journal.path)
        state = journal.load().reduce()
        assert len(state.stars) == 30 and all(s.task_completed for s in state.stars.values())
        if state.pending or any(r.write_kind == "submission" for r in state.reservations):
            raise BrowserSafetyStop("project_submission_preflight_pending_or_submitted_write")
        assert all(state.receipt(k) is not None for k in preflight.KINDS)
        return state, row_sha, 43.4, parse_assessment_text(text)

    monkeypatch.setattr(module, "_sources", offline_sources)
    monkeypatch.setattr(preflight, "_sources", offline_sources)
    context = submission_chromium.new_context(
        viewport={"width": 1400, "height": 1100}, service_workers="block"
    )
    requests, native = [], []
    page = context.new_page()
    captured = SimpleNamespace(
        root=root,
        journal=journal,
        scoring_dir=scoring_dir,
        source=source,
        page=page,
        native=native,
        requests=requests,
        effect="none",
        owners=[],
    )

    def record(target):
        pending = journal.load().reduce().pending
        # Native fixture handlers observe the reservation already on disk.
        assert len(pending) == 1 and pending[0].write_kind == "submission"
        native.append(target)
        return captured.effect

    page.expose_function("fixtureNativeSubmission", record)
    outer = (
        "<style>body{margin:20px}iframe{display:block;width:1000px;border:0;margin:10px 0}"
        "input[type=checkbox]{width:24px;height:24px}</style>"
        "<div>Score: 43.4</div>"
        f'<label><input id="ready" type="checkbox">{preflight.READY}</label>'
        f'<iframe name="simulation" style="height:350px" src="{SIMULATION_URL}"></iframe>'
        f'<iframe name="update" style="height:90px" src="{WIDGET}"></iframe>'
        f'<iframe name="submit" id="submission-widget" style="height:90px" src="{WIDGET}"></iframe>'
        "<script>document.querySelector('#ready').addEventListener('click',()=>fixtureNativeSubmission('readiness'));"
        "addEventListener('message', event=>{if(event.data?.fixture!=='submission')return;"
        f"if(event.data.effect==='navigation')location.href={json.dumps(ESCAPE)};"
        "if(event.data.effect==='modal'){const d=document.createElement('div');d.role='dialog';"
        "d.textContent='Fixture modal: do not acknowledge';document.body.append(d);}"
        "if(event.data.effect==='auth'){const p=document.createElement('input');p.type='password';"
        "document.body.append(p);}"
        "});</script>"
    )
    submit_html = (
        f'<button id="update">Update Score</button><button id="submit">{preflight.SUBMIT}</button>'
        # The frame name is not populated on Playwright's initial navigation
        # request. Use the fixture document's browsing-context name at load.
        "<script>document.getElementById(window.name==='submit'?'update':'submit').remove();"
        "const submit=document.getElementById('submit');if(submit)submit.addEventListener('click',async()=>{"
        "const effect=await fixtureNativeSubmission('submit');"
        "if(effect==='generic_notice'){const p=document.createElement('p');"
        "p.textContent='Project submitted';document.body.append(p);}"
        "parent.postMessage({fixture:'submission',effect},'*');});</script>"
    )

    def intercept(route):
        requests.append(route.request.url)
        documents = {
            OUTER: outer,
            SIMULATION_URL: '<div style="white-space:pre-wrap">' + html.escape(text) + "</div>",
            ESCAPE: "<p>Fixture-only disallowed destination</p>",
        }
        if route.request.url == WIDGET:
            body = submit_html
        else:
            body = documents.get(route.request.url)
        if body is None:
            route.abort()
        else:
            route.fulfill(status=200, content_type="text/html; charset=utf-8", body=body)

    context.route("**/*", intercept)
    page.goto(OUTER, wait_until="load")
    captured.widget = page.frame(name="submit")
    captured.widget.get_by_role("button", name=preflight.SUBMIT, exact=True).wait_for(
        state="visible", timeout=5000
    )
    try:
        yield captured
    finally:
        for owner in captured.owners:
            owner.close()
        # Fixture assertions never use hidden app state; the recorded events are
        # test-only instrumentation, unavailable to the production adapter.
        assert set(requests) <= {OUTER, SIMULATION_URL, WIDGET, ESCAPE}
        context.close()


def create(subject, *, emit=lambda _: None, output="submission"):
    owner = module.BrowserProjectSubmissionSteps(
        subject.page,
        config(),
        subject.root / output,
        run_history=subject.root,
        journal=subject.journal,
        scoring_dir=subject.scoring_dir,
        allow_submission=True,
        max_seconds=180,
        emit=emit,
    )
    subject.owners.append(owner)
    return owner


def prepare(subject, **kwargs):
    owner = create(subject, **kwargs)
    assert not subject.native
    assert owner.advance()["phase"] == "selecting_readiness", owner.state()
    assert not subject.native and len(subject.journal.load().reduce().pending) == 1
    return owner


def assert_no_confirmation(subject, owner, *, pending):
    state = subject.journal.load().reduce()
    assert bool(state.pending) is pending and state.receipt("submission") is None
    assert not (owner.output / "confirmed.json").exists()
    assert all(
        owner.state()[key] is False
        for key in ("submitted", "submission_verified", "task_completed", "project_completed")
    )


@pytest.mark.parametrize("effect", ["none", "generic_notice"])
def test_real_native_clicks_end_unknown_even_with_generic_submitted_text(native_submission, effect):
    subject, events = native_submission, []
    subject.effect = effect
    owner = prepare(subject, emit=events.append)
    assert owner.advance()["phase"] == "submitting", owner.state()
    assert subject.native == ["readiness"] and subject.page.get_by_role("checkbox").is_checked()
    assert owner.advance()["phase"] == "capturing_outcome", owner.state()
    subject.page.wait_for_timeout(0)  # Drain the fixture's exposed-function callback.
    assert subject.native == ["readiness", "submit"]
    if effect == "generic_notice":
        subject.widget.get_by_text("Project submitted", exact=True).wait_for(state="visible")
    owner.advance()
    assert owner.phase == "unknown_pending" and owner.status == "stopped", owner.state()
    assert owner.failure == "project_submission_acknowledgement_not_grounded"
    assert owner.report["submit_click_returned"] and owner.report["submission_outcome"] == "unknown"
    assert_no_confirmation(subject, owner, pending=True)
    for location in ("before", "post-submit"):
        image = owner.output / location / "viewport.png"
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        diagnostic = json.loads((image.parent / "diagnostic.json").read_bytes())
        assert diagnostic["viewport_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
        assert not diagnostic["submission_verified"] and not diagnostic["acknowledgement_interpreted"]
    saved = [json.loads(line) for line in (owner.output / "events.jsonl").read_text().splitlines()]
    assert saved == events
    assert [e["sequence"] for e in saved] == list(range(len(saved)))
    for _ in range(3):
        owner.advance()
        owner.resume()
        owner.tick()
    assert subject.native == ["readiness", "submit"]
    with pytest.raises(BrowserSafetyStop, match="pending_or_submitted"):
        create(subject, output="forbidden-retry")


@pytest.mark.parametrize("target", ["readiness", "submit"])
def test_real_overlay_blocks_reserved_dispatch(native_submission, target):
    subject = native_submission
    owner = prepare(subject)
    if target == "submit":
        assert owner.advance()["phase"] == "submitting"
    subject.page.evaluate(
        """selector=>{const r=document.querySelector(selector).getBoundingClientRect();
        const cover=document.createElement('div');Object.assign(cover.style,{position:'fixed',
        left:r.left+'px',top:r.top+'px',width:r.width+'px',height:r.height+'px',
        zIndex:'9999',background:'white'});document.body.append(cover)}""",
        "#ready" if target == "readiness" else "#submission-widget",
    )
    owner.advance()
    assert owner.finished and owner.phase == "unknown_pending"
    assert "unexposed" in owner.failure
    assert subject.native == ([] if target == "readiness" else ["readiness"])
    assert_no_confirmation(subject, owner, pending=True)


@pytest.mark.parametrize("target", ["readiness", "submit"])
def test_real_replaced_native_handle_is_not_rebound_for_dispatch(native_submission, target):
    subject = native_submission

    def replace(event):
        if event["event"] == "action_proposed" and event["payload"].get("target") == target:
            surface = subject.page if target == "readiness" else subject.widget
            surface.locator("#ready" if target == "readiness" else "#submit").evaluate(
                "e=>e.replaceWith(e.cloneNode(true))"
            )

    owner = prepare(subject, emit=replace)
    if target == "submit":
        assert owner.advance()["phase"] == "submitting"
    owner.advance()
    assert owner.finished and owner.failure == "project_submission_native_binding_changed"
    assert subject.native == ([] if target == "readiness" else ["readiness"])
    assert_no_confirmation(subject, owner, pending=True)


@pytest.mark.parametrize("target", ["readiness", "submit"])
def test_native_callback_abort_prevents_next_dispatch(native_submission, target):
    subject, holder = native_submission, {}

    def cancel(event):
        if event["event"] == "action_proposed" and event["payload"].get("target") == target:
            holder["owner"].abort()

    owner = holder["owner"] = prepare(subject, emit=cancel)
    if target == "submit":
        assert owner.advance()["phase"] == "submitting"
    owner.advance()
    assert owner.finished and owner.status == "aborted"
    assert subject.native == ([] if target == "readiness" else ["readiness"])
    assert_no_confirmation(subject, owner, pending=True)


@pytest.mark.parametrize(
    "effect,reason",
    [
        ("navigation", "project_submission_context_changed"),
        ("modal", "project_submission_unexpected_modal"),
        ("auth", "project_submission_authentication_required"),
    ],
)
def test_post_submit_diagnostics_never_bypass_native_boundary(native_submission, effect, reason):
    subject = native_submission
    subject.effect = effect
    owner = prepare(subject)
    assert owner.advance()["phase"] == "submitting"
    owner.advance()
    if effect == "navigation":
        subject.page.wait_for_url(ESCAPE, wait_until="load")
    elif effect == "modal":
        subject.page.get_by_role("dialog").wait_for(state="visible")
    else:
        subject.page.locator('input[type="password"]').wait_for(state="visible")
    owner.advance()
    assert owner.finished and owner.phase == "unknown_pending", owner.state()
    assert owner.failure == reason, owner.state()
    assert subject.native == ["readiness", "submit"]
    assert owner.report["submit_write_may_have_occurred"]
    assert (owner.output / "before/viewport.png").exists()
    assert not (owner.output / "post-submit").exists()
    assert_no_confirmation(subject, owner, pending=True)


def test_mutated_offline_source_blocks_native_write(native_submission):
    subject = native_submission
    owner = prepare(subject)
    subject.source.write_text('{"test_only":true,"mutated":true}\n')
    owner.advance()
    assert owner.finished and not subject.native
    assert_no_confirmation(subject, owner, pending=True)


def test_disabled_native_submit_is_not_assumed_enabled_after_checkbox(native_submission):
    subject = native_submission
    subject.widget.get_by_role("button", name=preflight.SUBMIT, exact=True).evaluate("e=>e.disabled=true")
    owner = create(subject)
    owner.advance()
    assert owner.finished and owner.failure == "project_submission_preflight_not_eligible"
    assert not subject.native and not subject.page.get_by_role("checkbox").is_checked()
    assert_no_confirmation(subject, owner, pending=False)
