"""No external traffic or real-money charges in these Chromium fixtures."""

import json
from html import escape

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_project_assessment import visible

from habfly.browser_assessment_actions import AssessmentActuator
from habfly.browser_stellar import SIMULATION_URL
from habfly.project_assessment import AssessmentError, AssessmentLedger


@pytest.fixture
def assessment_page(page):  # noqa: F811
    page.locator("iframe").first.evaluate("e=>{e.style.width='950px';e.style.height='600px'}")
    frame = page.frame(url=SIMULATION_URL)
    frame.locator("body").evaluate(
        "(e,html)=>e.innerHTML=html",
        f'<pre id="status">{escape(visible())}</pre><button id="assess">ASSESS</button>'
        '<button id="ok" hidden>OK</button><div hidden>private answers 999999</div>',
    )
    frame.evaluate(
        """values=>{
        window.assessmentClicks=0;
        document.querySelector('#assess').onclick=()=>{
            window.assessmentClicks++;
            document.querySelector('#status').textContent=values[0];
            document.querySelector('#ok').hidden=false;
        };
        document.querySelector('#ok').onclick=()=>{
            document.querySelector('#status').textContent=values[1];
            document.querySelector('#ok').hidden=true;
        };
    }""",
        [
            visible(funding=49900, stars=99.8, overall=33.3, ack=True),
            visible(funding=49900, stars=99.8, overall=33.3),
        ],
    )
    return page


def test_reservation_precedes_click_and_receipt_is_independent(assessment_page, tmp_path):
    output = tmp_path / "assessment"
    actuator = AssessmentActuator(assessment_page, config(), output, budget=100, max_attempts=1)
    frame = assessment_page.frame(url=SIMULATION_URL)
    checked = []
    assessment_page.expose_function(
        "checkReservation",
        lambda: checked.append(json.loads((output / "attempt-000-reserved.json").read_text())["ledger"]),
    )
    frame.evaluate(
        "()=>{const old=document.querySelector('#assess').onclick;document.querySelector('#assess').onclick=async()=>{await window.checkReservation();old()}}"
    )
    report = actuator.assess("data_quality", data_revision=8)
    assert checked[0]["attempts"][0]["after"] is None
    assert report["quality"]["stars"] == 99.8
    assert report["data_quality_assessed"]
    assert not any(
        report[k] for k in ["task_completed", "submitted", "save_verified", "score_transfer_verified"]
    )
    assert frame.evaluate("window.assessmentClicks") == 1
    assert (output / "attempt-000-confirmed.json").is_file()
    assert "private answers" not in (output / "attempt-000-after/observation.json").read_text()
    actuator.dismiss_receipt()
    with pytest.raises(AssessmentError, match="already_assessed_revision"):
        actuator.assess("data_quality", data_revision=8)
    assert frame.evaluate("window.assessmentClicks") == 1


def test_zero_budget_never_clicks(assessment_page, tmp_path):
    actuator = AssessmentActuator(assessment_page, config(), tmp_path / "run")
    with pytest.raises(AssessmentError, match="budget_exhausted"):
        actuator.assess("data_quality", data_revision=0)
    assert assessment_page.frame(url=SIMULATION_URL).evaluate("window.assessmentClicks") == 0


def test_acknowledgement_is_journaled_once_without_another_assessment(assessment_page, tmp_path):
    output = tmp_path / "run"
    actuator = AssessmentActuator(assessment_page, config(), output, budget=100, max_attempts=1)
    frame = assessment_page.frame(url=SIMULATION_URL)
    actuator.assess("data_quality", data_revision=1)
    frame.locator("#ok").evaluate(
        "e=>{const old=e.onclick;e.onclick=()=>{window.ackClicks=(window.ackClicks||0)+1;old()}}"
    )
    actuator.dismiss_receipt()
    receipt = json.loads((output / "ack-000-confirmed.json").read_text())
    assert receipt["acknowledgement_dismissed"] and receipt["assessment_clicks"] == 0
    assert not receipt["task_completed"]
    with pytest.raises(AssessmentError, match="dismissal_already_attempted"):
        actuator.dismiss_receipt()
    assert frame.evaluate("window.ackClicks") == 1 and frame.evaluate("window.assessmentClicks") == 1


def test_stale_acknowledgement_preserves_rejected_capture_without_click(assessment_page, tmp_path):
    output = tmp_path / "run"
    actuator = AssessmentActuator(assessment_page, config(), output, budget=100, max_attempts=1)
    frame = assessment_page.frame(url=SIMULATION_URL)
    actuator.assess("data_quality", data_revision=1)
    frame.locator("#status").evaluate("e=>e.textContent+=' Visible changed state'")
    with pytest.raises(AssessmentError, match="receipt_changed_before_dismiss"):
        actuator.dismiss_receipt()
    record = json.loads((output / "ack-000-stale.json").read_text())
    assert not record["click_dispatch_started"]
    assert "Visible changed state" in (output / "ack-000-stale-capture/observation.json").read_text()
    assert not (output / "ack-000-reserved.json").exists()
    assert frame.locator("#ok").is_visible()


@pytest.mark.parametrize("failure", ["during", "before"])
def test_uncertain_acknowledgement_is_never_retried(assessment_page, tmp_path, monkeypatch, failure):
    import habfly.browser_assessment_actions as module

    output = tmp_path / "run"
    actuator = AssessmentActuator(assessment_page, config(), output, budget=100, max_attempts=1)
    frame = assessment_page.frame(url=SIMULATION_URL)
    actuator.assess("data_quality", data_revision=1)
    if failure == "during":
        frame.locator("#ok").evaluate(
            "e=>e.onclick=()=>{window.ackClicks=(window.ackClicks||0)+1;document.querySelector('#status').textContent+=' unexpected'}"
        )
    else:
        original = module.persist_json

        def replace(path, value):
            original(path, value)
            if path.name == "ack-000-reserved.json":
                frame.locator("#ok").evaluate("e=>e.replaceWith(e.cloneNode(true))")

        monkeypatch.setattr(module, "persist_json", replace)
    with pytest.raises(AssessmentError):
        actuator.dismiss_receipt()
    record = json.loads((output / "ack-000-stopped.json").read_text())
    assert record["click_dispatch_started"] == (failure == "during")
    with pytest.raises(AssessmentError, match="stopped"):
        actuator.dismiss_receipt()
    assert frame.evaluate("window.ackClicks||0") == (1 if failure == "during" else 0)
    assert frame.evaluate("window.assessmentClicks") == 1


@pytest.mark.parametrize(
    "payload",
    [
        {
            "reason": "assessment_receipt_timeout",
            "click_dispatch_started": False,
            "writes_may_have_occurred": False,
        },
        {
            "reason": "assessment_changed_after_reservation",
            "click_dispatch_started": True,
            "writes_may_have_occurred": True,
        },
        {
            "reason": "assessment_changed_after_reservation",
            "click_dispatch_started": False,
            "writes_may_have_occurred": True,
        },
        {
            "reason": "assessment_changed_after_reservation",
            "click_dispatch_started": "false",
            "writes_may_have_occurred": False,
        },
    ],
)
def test_disposition_never_infers_no_click_from_ambiguous_or_delivered_failures(tmp_path, payload):
    from habfly.browser_assessment_actions import record_undispatched_assessment

    (tmp_path / "attempt-000-stopped.json").write_text(json.dumps(payload))
    with pytest.raises(AssessmentError, match="dispatch_not_proven_absent"):
        record_undispatched_assessment(tmp_path, tmp_path / "absent", accept_legacy_preclick_guard=True)
    assert not list(tmp_path.glob("*-undispatched.json"))


@pytest.mark.parametrize("fault", ["timeout", "funding", "rows", "collected", "mode"])
def test_uncertain_or_inconsistent_receipt_stays_pending(assessment_page, tmp_path, fault):
    frame = assessment_page.frame(url=SIMULATION_URL)
    value = visible(funding=49900, ack=True)
    if fault == "timeout":
        value = visible()
    elif fault == "funding":
        value = visible(funding=49800, ack=True)
    elif fault == "rows":
        value = value.replace("ANALYZED DATA STAR", "ANALYZED DATA STAR CHANGED ANSWER")
    elif fault == "collected":
        value = visible(funding=49900, collected=3, ack=True)
    else:
        value = visible(mode="scavenger_hunt", funding=49900, ack=True)
    frame.evaluate(
        "text=>document.querySelector('#assess').onclick=()=>{window.assessmentClicks++;document.querySelector('#status').textContent=text}",
        value,
    )
    output = tmp_path / "uncertain"
    actuator = AssessmentActuator(
        assessment_page, config(), output, budget=200, max_attempts=2, timeout_seconds=0.1
    )
    with pytest.raises(AssessmentError):
        actuator.assess("data_quality", data_revision=1)
    assert actuator.ledger.pending
    restored = AssessmentLedger.model_validate(
        json.loads((output / "attempt-000-stopped.json").read_text())["ledger"]
    )
    assert restored.pending and restored.reserved == 100
    if fault == "rows":
        observed = json.loads((output / "attempt-000-changed-rows/observation.json").read_text())
        assert "CHANGED ANSWER" in next(f["text"] for f in observed["frames"] if f["url"] == SIMULATION_URL)
        difference = json.loads((output / "attempt-000-row-difference.json").read_text())
        assert difference["before_sha256"] != difference["after_sha256"]
        assert difference["click_dispatch_started"] and not difference["automatic_retry"]
    with pytest.raises(AssessmentError, match="stopped"):
        actuator.assess("data_quality", data_revision=2)
    assert frame.evaluate("window.assessmentClicks") == 1


def test_disk_failure_before_reservation_prevents_click(assessment_page, tmp_path, monkeypatch):
    import habfly.browser_assessment_actions as module

    actuator = AssessmentActuator(assessment_page, config(), tmp_path / "run", budget=100, max_attempts=1)
    original = module.persist_json

    def fail(path, value):
        if path.name.endswith("-reserved.json"):
            raise OSError("fixture full disk")
        original(path, value)

    monkeypatch.setattr(module, "persist_json", fail)
    with pytest.raises(OSError):
        actuator.assess("data_quality", data_revision=0)
    assert actuator.ledger.pending
    assert assessment_page.frame(url=SIMULATION_URL).evaluate("window.assessmentClicks") == 0
    stopped = json.loads((tmp_path / "run/attempt-000-stopped.json").read_text())
    assert not stopped["click_dispatch_started"] and not stopped["writes_may_have_occurred"]


@pytest.mark.parametrize("fault", ["replaced_button", "changed_text"])
def test_predispatch_stop_records_exact_checks_and_never_clicks(
    assessment_page, tmp_path, monkeypatch, fault
):
    import habfly.browser_assessment_actions as module

    frame = assessment_page.frame(url=SIMULATION_URL)
    output = tmp_path / "predispatch"
    actuator = AssessmentActuator(assessment_page, config(), output, budget=100, max_attempts=1)
    original = module.persist_json

    def mutate_after_reservation(path, value):
        original(path, value)
        if path.name.endswith("-reserved.json"):
            if fault == "replaced_button":
                frame.locator("#assess").evaluate(
                    "e=>{const n=e.cloneNode(true);n.onclick=e.onclick;e.replaceWith(n)}"
                )
            else:
                frame.locator("#status").evaluate("e=>e.textContent+=' visible feedback'")

    monkeypatch.setattr(module, "persist_json", mutate_after_reservation)
    with pytest.raises(AssessmentError, match="assessment_changed_after_reservation"):
        actuator.assess("data_quality", data_revision=3)
    stopped = json.loads((output / "attempt-000-stopped.json").read_text())
    assert not stopped["click_dispatch_started"] and not stopped["writes_may_have_occurred"]
    assert not stopped["retry_allowed"] and actuator.ledger.pending
    assert stopped["predispatch_checks"]["rows_unchanged"]
    assert stopped["predispatch_checks"]["native_control_unchanged"] == (fault != "replaced_button")
    assert stopped["predispatch_checks"]["snapshot_unchanged"] == (fault != "changed_text")
    assert (output / "attempt-000-predispatch/manifest.json").exists()
    assert frame.evaluate("window.assessmentClicks") == 0


@pytest.mark.parametrize("legacy", [False, True])
def test_offline_predispatch_disposition_preserves_pending_and_cannot_authorize_retry(
    assessment_page, tmp_path, monkeypatch, legacy
):
    import habfly.browser_assessment_actions as module
    from habfly.browser_probe import inspect_page, save_probe

    frame = assessment_page.frame(url=SIMULATION_URL)
    source = tmp_path / "source"
    actuator = AssessmentActuator(assessment_page, config(), source, budget=100, max_attempts=1)
    original = module.persist_json

    def replace(path, value):
        original(path, value)
        if path.name.endswith("-reserved.json"):
            frame.locator("#assess").evaluate("e=>e.replaceWith(e.cloneNode(true))")

    monkeypatch.setattr(module, "persist_json", replace)
    with pytest.raises(AssessmentError, match="assessment_changed_after_reservation"):
        actuator.assess("data_quality", data_revision=9)
    save_probe(inspect_page(assessment_page, config()), tmp_path / "observed")
    if legacy:
        path = source / "attempt-000-stopped.json"
        payload = json.loads(path.read_text())
        del payload["click_dispatch_started"]
        del payload["predispatch_checks"]
        payload["writes_may_have_occurred"] = True
        path.write_text(json.dumps(payload))
        with pytest.raises(AssessmentError, match="legacy_predispatch_review_required"):
            module.record_undispatched_assessment(source, tmp_path / "observed")
    before = (source / "attempt-000-stopped.json").read_bytes()
    receipt = module.record_undispatched_assessment(
        source, tmp_path / "observed", accept_legacy_preclick_guard=legacy
    )
    assert receipt["clicks_executed_by_reconciliation"] == 0 and not receipt["automatic_retry"]
    assert not receipt["assessment_confirmed"] and receipt["legacy_guard_review"] == legacy
    assert actuator.ledger.pending and actuator.stopped
    assert (source / "attempt-000-stopped.json").read_bytes() == before
    assert frame.evaluate("window.assessmentClicks") == 0
    with pytest.raises(FileExistsError):
        module.record_undispatched_assessment(
            source, tmp_path / "observed", accept_legacy_preclick_guard=legacy
        )
    with pytest.raises(AssessmentError, match="stopped"):
        actuator.assess("data_quality", data_revision=9)


def test_automation_unlock_is_not_allowed(assessment_page, tmp_path):
    actuator = AssessmentActuator(assessment_page, config(), tmp_path / "run", budget=100, max_attempts=1)
    with pytest.raises(AssessmentError, match="unsupported_assessment_kind"):
        actuator.assess("automation", data_revision=0)


@pytest.mark.parametrize("covered", [False, True])
def test_visible_button_child_is_allowed_but_overlay_is_not(assessment_page, tmp_path, covered):
    frame = assessment_page.frame(url=SIMULATION_URL)
    frame.locator("#assess").evaluate(
        "e=>{e.innerHTML='<span>Assess</span>';e.style.textTransform='uppercase'}"
    )
    if covered:
        frame.locator("body").evaluate(
            "e=>e.insertAdjacentHTML('beforeend','<div style=\"position:fixed;inset:0;z-index:999\"></div>')"
        )
    actuator = AssessmentActuator(assessment_page, config(), tmp_path / "run", budget=100, max_attempts=1)
    if covered:
        with pytest.raises(AssessmentError, match="unavailable"):
            actuator.assess("data_quality", data_revision=0)
        assert not actuator.ledger.attempts
    else:
        assert actuator.assess("data_quality", data_revision=0)["data_quality_assessed"]


def test_observed_custom_acknowledgement_is_scoped_to_receipt(assessment_page, tmp_path):
    actuator = AssessmentActuator(assessment_page, config(), tmp_path / "run", budget=100, max_attempts=1)
    frame = assessment_page.frame(url=SIMULATION_URL)
    frame.evaluate("""()=>{
      const old=document.querySelector('#assess').onclick;
      document.querySelector('#assess').onclick=()=>{
        old();
        const button=document.querySelector('#ok'), replacement=document.createElement('button-popup');
        replacement.textContent='Ok';replacement.onclick=button.onclick;
        const panel=document.createElement('div');panel.append('DATA QUALITY UPDATED',replacement);
        document.querySelector('#status').textContent=document.querySelector('#status').textContent.replace('DATA QUALITY UPDATED OK','');
        button.hidden=true;document.body.append(panel);
        replacement.style.cssText='display:block;width:50px;height:30px';
        replacement.onclick=()=>panel.remove();
      };
    }""")
    actuator.assess("data_quality", data_revision=1)
    actuator.dismiss_receipt()
    assert frame.get_by_text("Ok", exact=True).count() == 0
