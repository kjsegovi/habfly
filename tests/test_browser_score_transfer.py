"""All browser requests are intercepted; score transfer is never submission."""
# ruff: noqa: F811

import json

import pytest
from test_browser_numeric import WIDGET, chromium, config, page  # noqa: F401
from test_project_assessment import visible

from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json, rows_hash
from habfly.browser_probe import inspect_page, save_probe
from habfly.browser_score_transfer import load_assessed_revision, transfer_assessed_score
from habfly.browser_stellar import SIMULATION_URL
from habfly.project_assessment import AssessmentLedger, parse_assessment_text


@pytest.fixture
def transfer_page(page, tmp_path):
    page.set_viewport_size({"width": 1600, "height": 1100})
    page.locator("body").evaluate("""e=>{
        e.insertAdjacentHTML('afterbegin','<div id="score">Score: 0.00</div><div id="notice"></div>');
        window.addEventListener('message',event=>{
            if(event.data==='fixture-update'){
                document.querySelector('#score').textContent='Score: 56.59';
                document.querySelector('#notice').textContent='Score updated.';
            }
        });
    }""")
    page.locator("iframe").first.evaluate("e=>{e.style.width='950px';e.style.height='600px'}")
    widgets = [f for f in page.frames if f.url == WIDGET]
    widgets[0].locator("body").evaluate("""e=>{
        e.innerHTML='<button id="update">Update Score</button>';
        window.clicks=0;document.querySelector('button').onclick=()=>{
            window.clicks++;parent.postMessage('fixture-update','*');
        };
    }""")
    widgets[1].locator("body").evaluate("e=>e.innerHTML='<button>Submit Project</button>'")
    frame = page.frame(url=SIMULATION_URL)
    frame.locator("body").evaluate("(e,html)=>e.innerHTML=html", '<pre id="status"></pre>')
    evidence_dir = tmp_path / "assessment"
    evidence_dir.mkdir()
    ledger = AssessmentLedger(budget=200, max_attempts=2)
    for index, kind in enumerate(("data_quality", "scavenger_hunt")):
        before = visible(mode=kind, funding=50000 - index * 100, stars=99.8, overall=33.3)
        after = visible(
            mode=kind, funding=49900 - index * 100, stars=99.8, overall=33.3, found=index, ack=True
        )
        ledger.reserve(kind, parse_assessment_text(before), data_revision=1)
        ledger.confirm(index, parse_assessment_text(after), data_revision=1)
        frame.locator("#status").evaluate("(e,text)=>e.textContent=text", after)
        save_probe(inspect_page(page, config()), evidence_dir / f"attempt-{index:03d}-after")
    path = evidence_dir / "attempt-001-confirmed.json"
    persist_json(
        path,
        {
            "ledger": ledger.model_dump(mode="json"),
            "receipt": ledger.attempts[-1].model_dump(mode="json"),
            "project_rows_sha256": rows_hash(after),
            "report": ledger.report(data_revision=1),
        },
    )
    frame.locator("#status").evaluate(
        "(e,text)=>e.textContent=text", after.removesuffix("SCAVENGER HUNT UPDATED OK")
    )
    return page, frame, widgets, path


def transfer(fixture, tmp_path, **kwargs):
    page, _, _, path = fixture
    return transfer_assessed_score(
        page,
        config(),
        tmp_path / "transfer",
        assessment_receipt=path,
        allow_score_transfer=True,
        timeout_seconds=0.2,
        **kwargs,
    )


def test_explicit_transfer_requires_both_receipts_and_never_submits(transfer_page, tmp_path):
    page, _, widgets, _ = transfer_page
    before = []
    page.expose_function(
        "checkReserved", lambda: before.append(json.loads((tmp_path / "transfer/reserved.json").read_text()))
    )
    widgets[0].locator("button").evaluate(
        "e=>{const old=e.onclick;e.onclick=async()=>{await window.checkReserved();old()}}"
    )
    receipt = transfer(transfer_page, tmp_path)
    assert before[0]["submission_enabled"] is False
    assert receipt["score_before"] == "0.00" and receipt["score_after"] == "56.59"
    assert receipt["score_transfer_verified"] and not receipt["score_computed_locally"]
    assert not any(
        receipt[k] for k in ("submitted", "task_completed", "browser_acceptance_passed", "save_verified")
    )
    assert widgets[0].evaluate("window.clicks") == 1
    assert not page.get_by_role("checkbox").is_checked()
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        transfer(transfer_page, tmp_path)
    assert widgets[0].evaluate("window.clicks") == 1


def test_disabled_by_default(transfer_page, tmp_path):
    page, _, widgets, path = transfer_page
    with pytest.raises(BrowserSafetyStop, match="not_enabled"):
        transfer_assessed_score(page, config(), tmp_path / "transfer", assessment_receipt=path)
    assert widgets[0].evaluate("window.clicks") == 0 and not (tmp_path / "transfer").exists()


@pytest.mark.parametrize(
    "fault", ["rows", "funding", "checkbox", "stale_ack", "duplicate_button", "missing_score", "modal"]
)
def test_stale_or_unsafe_transfer_never_clicks(transfer_page, tmp_path, fault):
    page, frame, widgets, _ = transfer_page
    if fault in {"rows", "funding"}:
        frame.locator("#status").evaluate(
            "(e,k)=>e.textContent=e.textContent.replace(k[0],k[1])",
            [
                "ANALYZED DATA STAR" if fault == "rows" else "$49800",
                "ANALYZED DATA STAR CHANGED" if fault == "rows" else "$49700",
            ],
        )
    elif fault == "checkbox":
        page.get_by_role("checkbox").check()
    elif fault == "stale_ack":
        page.locator("#notice").evaluate("e=>e.textContent='Score updated.'")
    elif fault == "duplicate_button":
        widgets[1].locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<button>Update Score</button>')"
        )
    elif fault == "missing_score":
        page.locator("#score").evaluate("e=>e.remove()")
    else:
        page.locator("body").evaluate(
            "e=>{const d=document.createElement('dialog');e.append(d);d.showModal()}"
        )
    with pytest.raises(BrowserSafetyStop):
        transfer(transfer_page, tmp_path)
    assert widgets[0].evaluate("window.clicks") == 0
    assert not json.loads((tmp_path / "transfer/stopped.json").read_text())["write_may_have_occurred"]


@pytest.mark.parametrize("fault", ["no_ack", "hidden_ack", "changed_answer", "submission_checked"])
def test_uncertain_or_unexpected_effect_is_not_retried(transfer_page, tmp_path, fault):
    page, frame, widgets, _ = transfer_page
    if fault == "no_ack":
        widgets[0].locator("button").evaluate("e=>e.onclick=()=>window.clicks++")
    elif fault == "hidden_ack":
        page.locator("#notice").evaluate("e=>e.style.opacity='0'")
    elif fault == "changed_answer":
        frame.evaluate("""()=>window.addEventListener('message',event=>{
            if(event.data==='fixture-change'){
                const e=document.querySelector('#status');
                e.textContent=e.textContent.replace('ANALYZED DATA STAR','ANALYZED DATA STAR CHANGED');
                parent.postMessage('fixture-update','*');
            }
        })""")
        page.evaluate("""()=>window.addEventListener('message',event=>{
            if(event.data==='fixture-change')document.querySelector('iframe').contentWindow.postMessage('fixture-change','*');
        })""")
        widgets[0].locator("button").evaluate(
            "e=>e.onclick=()=>{window.clicks++;parent.postMessage('fixture-change','*')}"
        )
    else:
        page.evaluate(
            "()=>window.addEventListener('message',()=>document.querySelector('input').checked=true)"
        )
    with pytest.raises(BrowserSafetyStop):
        transfer(transfer_page, tmp_path)
    assert widgets[0].evaluate("window.clicks") == 1
    assert json.loads((tmp_path / "transfer/stopped.json").read_text())["write_may_have_occurred"]
    assert not (tmp_path / "transfer/confirmed.json").exists()
    with pytest.raises(BrowserSafetyStop, match="already_reserved"):
        transfer(transfer_page, tmp_path / "different-output")
    assert widgets[0].evaluate("window.clicks") == 1


def test_source_capture_tampering_blocks_before_operation(transfer_page, tmp_path):
    _, _, widgets, path = transfer_page
    (path.parent / "attempt-000-after/observation.json").write_text("{}")
    with pytest.raises(BrowserSafetyStop, match="hash_mismatch"):
        load_assessed_revision(path)
    assert widgets[0].evaluate("window.clicks") == 0


def add_course_feedback(page, *, unexpected=False):
    page.locator("body").evaluate(
        """e=>e.insertAdjacentHTML('beforeend','<button id="previous">Previous screen</button>')"""
    )
    page.evaluate(
        """unexpected=>window.addEventListener('message',event=>{
        if(event.data!=='fixture-update')return;
        document.querySelector('#previous').insertAdjacentHTML('beforebegin',
            '<button>OK</button><button aria-label="Close feedback">Open/Close Feedback</button><button aria-label="Close feedback">&#xf00d;</button>'
            +(unexpected?'<button>Unexpected action</button>':''));
    })""",
        unexpected,
    )


@pytest.mark.parametrize("unexpected", [False, True])
def test_exact_known_feedback_controls_are_allowed_but_extras_fail(transfer_page, tmp_path, unexpected):
    page, _, widgets, _ = transfer_page
    add_course_feedback(page, unexpected=unexpected)
    if unexpected:
        with pytest.raises(BrowserSafetyStop, match="side_effect"):
            transfer(transfer_page, tmp_path)
    else:
        assert transfer(transfer_page, tmp_path)["score_transfer_verified"]
    assert widgets[0].evaluate("window.clicks") == 1


def test_read_only_score_reconciliation_preserves_failure_and_never_reclicks(
    transfer_page, tmp_path, monkeypatch
):
    import habfly.browser_score_transfer as module
    from habfly.browser_numeric import comparable_screen

    page, frame, widgets, _ = transfer_page
    add_course_feedback(page)
    original = module.score_projection
    monkeypatch.setattr(module, "score_projection", lambda report, **kwargs: comparable_screen(report))
    with pytest.raises(BrowserSafetyStop, match="side_effect"):
        transfer(transfer_page, tmp_path)
    monkeypatch.setattr(module, "score_projection", original)
    failure = (tmp_path / "transfer/stopped.json").read_bytes()
    receipt = module.reconcile_score_transfer(page, config(), tmp_path / "transfer", tmp_path / "recovered")
    assert receipt["score_transfer_verified"] and receipt["browser_actions_executed"] == 0
    assert receipt["score_after"] == "56.59" and receipt["original_failure_preserved"]
    assert widgets[0].evaluate("window.clicks") == 1
    assert (tmp_path / "transfer/stopped.json").read_bytes() == failure
    frame.locator("#status").evaluate("e=>e.textContent=e.textContent.replace('$49800','$49700')")
    with pytest.raises(BrowserSafetyStop, match="live_state_changed"):
        module.reconcile_score_transfer(page, config(), tmp_path / "transfer", tmp_path / "stale")
