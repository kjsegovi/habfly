"""Explicit new revision budgets; never repeat a pending/confirmed assessment."""
# ruff: noqa: F811

import json

import pytest
from test_browser_assessment_actions import assessment_page  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_project_assessment import visible

from habfly.browser_assessment_actions import record_undispatched_assessment
from habfly.browser_probe import inspect_page, save_probe
from habfly.browser_project_assessment import assessment_history, begin_assessment_stage
from habfly.browser_stellar import SIMULATION_URL
from habfly.project_assessment import AssessmentError


def stage(page, root, name, revision, collected=2):
    return begin_assessment_stage(
        page,
        config(),
        root / name,
        history_root=root,
        data_revision=revision,
        collected=collected,
        reason="Explicit fixture revision",
    )


def test_new_stage_journals_budget_and_prior_confirmed_history(assessment_page, tmp_path):
    page = assessment_page
    first = stage(page, tmp_path, "first", 1)
    first.assess("data_quality", data_revision=1)
    first.dismiss_receipt()
    with pytest.raises(AssessmentError, match="new_project_revision_required"):
        stage(page, tmp_path, "duplicate", 1)
    with pytest.raises(AssessmentError, match="rows_unchanged"):
        stage(page, tmp_path, "unchanged", 2)
    frame = page.frame(url=SIMULATION_URL)
    frame.locator("#status").evaluate("(e,s)=>e.textContent=s", visible(funding=49900, collected=3))
    with pytest.raises(AssessmentError, match="collection_or_receipt"):
        stage(page, tmp_path, "wrong-count", 2)
    second = stage(page, tmp_path, "second", 2, collected=3)
    assert not second.ledger.attempts and second.ledger.budget == 200
    record = json.loads((tmp_path / "second/stage.json").read_text())
    assert record["history"][0]["outcome"] == "confirmed"
    assert not record["same_revision_retry_allowed"] and not record["real_money"]
    assert frame.evaluate("window.assessmentClicks") == 1


def test_dispatched_unknown_attempt_blocks_all_later_revisions(assessment_page, tmp_path):
    first = stage(assessment_page, tmp_path, "first", 1)
    frame = assessment_page.frame(url=SIMULATION_URL)
    frame.locator("#assess").evaluate("e=>e.onclick=()=>{}")
    first.timeout_seconds = 0.1
    with pytest.raises(AssessmentError, match="receipt_timeout"):
        first.assess("data_quality", data_revision=1)
    frame.locator("#status").evaluate("(e,s)=>e.textContent=s", visible(collected=3))
    with pytest.raises(AssessmentError, match="unresolved_project_assessment_reservation"):
        stage(assessment_page, tmp_path, "new", 2, collected=3)
    assert not (tmp_path / "new").exists()


def test_reviewed_predispatch_stop_allows_only_changed_later_data(assessment_page, tmp_path, monkeypatch):
    import habfly.browser_assessment_actions as module

    page, frame = assessment_page, assessment_page.frame(url=SIMULATION_URL)
    first = stage(page, tmp_path, "first", 1)
    original = module.persist_json

    def replace(path, payload):
        original(path, payload)
        if path.name.endswith("-reserved.json"):
            frame.locator("#assess").evaluate("e=>e.replaceWith(e.cloneNode(true))")

    monkeypatch.setattr(module, "persist_json", replace)
    with pytest.raises(AssessmentError, match="changed_after_reservation"):
        first.assess("data_quality", data_revision=1)
    save_probe(inspect_page(page, config()), tmp_path / "readback")
    record_undispatched_assessment(tmp_path / "first", tmp_path / "readback")
    history = assessment_history(tmp_path)
    assert history[0]["outcome"] == "not_dispatched_original_reservation_retained"
    assert first.ledger.pending
    with pytest.raises(AssessmentError, match="new_project_revision_required"):
        stage(page, tmp_path, "retry", 1)
    with pytest.raises(AssessmentError, match="rows_unchanged"):
        stage(page, tmp_path, "same-data", 2)
    frame.locator("#status").evaluate("(e,s)=>e.textContent=s", visible(collected=3))
    next_stage = stage(page, tmp_path, "next", 2, collected=3)
    assert not next_stage.ledger.attempts and frame.evaluate("window.assessmentClicks") == 0
    path = tmp_path / "readback/observation.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(AssessmentError, match="source_changed"):
        assessment_history(tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data_revision": 0},
        {"data_revision": True},
        {"collected": 0},
        {"collected": 501},
        {"reason": ""},
        {"max_attempts": 3},
    ],
)
def test_bad_stage_budget_or_identity_is_rejected_before_browser(tmp_path, kwargs):
    values = {"history_root": tmp_path, "data_revision": 1, "collected": 2, "reason": "Explicit test"}
    with pytest.raises(ValueError):
        begin_assessment_stage(None, None, tmp_path / "new", **(values | kwargs))
    assert not (tmp_path / "new").exists()
