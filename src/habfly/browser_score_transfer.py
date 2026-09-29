"""One explicit transfer of confirmed assessment results; never submission.

The immutable latest assessment receipt binds both assessment kinds to the same
data revision and visible project rows. A fresh UI acknowledgement and visible
outer score are required. There is no inferred score formula or automatic retry.
"""

import hashlib
import json
import re
import time
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import assessment_config, persist_json, rows_hash
from .browser_numeric import comparable_screen
from .browser_probe import inspect_page, save_probe
from .browser_setup import RENDER_VISIBILITY_JS
from .browser_stellar import SIMULATION_URL
from .project_assessment import AssessmentLedger, parse_assessment_text


def score_projection(report, *, feedback=False):
    value = comparable_screen(report)
    controls = value["outer_controls"]
    if feedback:
        expected = [
            '- button "OK"',
            '- button "Close feedback": Open/Close Feedback',
            '- button "Close feedback": \uf00d',
        ]
        matches = [
            i
            for i in range(len(controls) - 3)
            if [c["accessibility"] for c in controls[i : i + 3]] == expected
            and controls[i + 3]["accessibility"] == '- button "Previous screen"'
        ]
        if len(matches) > 1:
            raise BrowserSafetyStop("ambiguous_score_feedback_controls")
        if matches:
            start = matches[0]
            if any(
                c["role"] != "button" or not c["enabled"] or c["protected"]
                for c in controls[start : start + 3]
            ):
                raise BrowserSafetyStop("unexpected_score_feedback_controls")
            del controls[start : start + 3]
    # Probe IDs are observation-local, so insertion of the known feedback block
    # renumbers later controls. All labels, roles, values and states still match.
    for index, control in enumerate(controls):
        control["id"] = f"outer:projected:{index}"
    return value


def load_assessed_revision(path):
    raw = Path(path).read_bytes()
    evidence = json.loads(raw)
    ledger = AssessmentLedger.model_validate(evidence["ledger"])
    receipt = evidence["receipt"]
    if not ledger.attempts or ledger.pending or receipt != ledger.attempts[-1].model_dump(mode="json"):
        raise BrowserSafetyStop("unconfirmed_score_source")
    revision = receipt["data_revision"]
    report = ledger.report(data_revision=revision)
    if (
        report != evidence["report"]
        or not report["data_quality_assessed"]
        or not report["scavenger_hunt_assessed"]
        or not re.fullmatch(r"[0-9a-f]{64}", evidence.get("project_rows_sha256", ""))
    ):
        raise BrowserSafetyStop("both_current_assessments_required")
    # Bind each accepted assessment's original row snapshot, not just the last
    # receipt's caller-provided hash. No changed/recollected data is permitted.
    row_hash = evidence["project_rows_sha256"]
    for attempt in ledger.attempts:
        if attempt.data_revision != revision:
            continue
        prefix = Path(path).parent / f"attempt-{attempt.sequence:03d}-after"
        data = (prefix / "observation.json").read_bytes()
        manifest = json.loads((prefix / "manifest.json").read_text())
        if hashlib.sha256(data).hexdigest() != manifest["observation_sha256"]:
            raise BrowserSafetyStop("score_source_capture_hash_mismatch")
        capture = json.loads(data)
        frames = [f for f in capture["frames"] if f["url"] == SIMULATION_URL]
        if len(frames) != 1 or rows_hash(frames[0]["text"]) != row_hash:
            raise BrowserSafetyStop("assessment_rows_differ")
        if parse_assessment_text(frames[0]["text"]) != attempt.after:
            raise BrowserSafetyStop("score_source_receipt_mismatch")
    return (
        ledger.attempts[-1].after,
        row_hash,
        {
            "source_receipt": str(Path(path)),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "data_revision": revision,
            "assessment_report": report,
        },
    )


def read_outer_score(page):
    text = page.locator("body").inner_text(timeout=3000)
    matches = re.findall(r"\bScore:\s*([0-9]+(?:\.[0-9]+)?)(?![0-9.eE])", text)
    if len(matches) != 1 or not 0 <= float(matches[0]) <= 260:
        raise BrowserSafetyStop("missing_or_ambiguous_outer_score")
    return matches[0]


def exposed_button(button):
    return (
        button.is_visible()
        and button.is_enabled()
        and button.evaluate(
            "e=>{"
            + RENDER_VISIBILITY_JS
            + """
        const r=e.getBoundingClientRect(), h=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
        return e.tagName==='BUTTON' && styled(e) && h && (h===e || e.contains(h)) && exposed(r,h);
        }"""
        )
    )


def transfer_assessed_score(
    page,
    config,
    output,
    *,
    assessment_receipt,
    allow_score_transfer=False,
    timeout_seconds=15,
    before_dispatch=None,
    check_cancelled=None,
    settle_inventory=False,
    deadline=None,
):
    """One score dispatch, with optional raw paginated-list notice settling.

    Settling is read-only and shares fixed predispatch/acknowledgement deadlines,
    additionally capped by the supplied absolute monotonic owner deadline.
    A dispatched failure leaves the durable revision claim and is never retried.
    """
    from .browser_assessment_actions import _settled_assessment_report, _settling_options

    _settling_options(settle_inventory, deadline)
    for hook in (before_dispatch, check_cancelled):
        if hook is not None and not callable(hook):
            raise TypeError("Score dispatch hooks must be callable")
    if allow_score_transfer is not True:
        raise BrowserSafetyStop("score_transfer_not_enabled")
    if type(timeout_seconds) not in {int, float} or not 0.1 <= timeout_seconds <= 30:
        raise ValueError("Score transfer deadline must be bounded")
    assessed, expected_rows, provenance = load_assessed_revision(assessment_receipt)
    claim = Path(assessment_receipt).parent / f"score-transfer-revision-{provenance['data_revision']}.json"
    if claim.exists():
        raise BrowserSafetyStop("score_transfer_revision_already_reserved")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    config = assessment_config(config)
    original_frames = page.frames.copy()
    parent_deadline = deadline
    read_deadline = time.monotonic() + timeout_seconds if settle_inventory else None
    attempted, unexpected_dialog = False, False

    def dialog_handler(dialog):
        nonlocal unexpected_dialog
        unexpected_dialog = True
        dialog.dismiss()

    def check():
        page.wait_for_timeout(0)
        if unexpected_dialog:
            raise BrowserSafetyStop("unexpected_browser_dialog")
        if page.frames != original_frames or len(page.context.pages) != 1:
            raise BrowserSafetyStop("score_transfer_context_changed")
        if settle_inventory and check_cancelled is not None:
            check_cancelled()

    def read():
        check()
        if settle_inventory:
            report = _settled_assessment_report(
                page,
                config,
                deadline=min(read_deadline, parent_deadline)
                if parent_deadline is not None
                else read_deadline,
                check_cancelled=check,
                inspector=inspect_page,
                frames=tuple(original_frames),
            )
        else:
            report = inspect_page(page, config)
        if report["ignored_frame_urls"]:
            raise BrowserSafetyStop("unknown_visible_frame")
        simulation = next(f for f in report["frames"] if f["url"] == SIMULATION_URL)
        current = parse_assessment_text(simulation["text"])
        if (
            current.acknowledgement is not None
            or rows_hash(simulation["text"]) != expected_rows
            or any(
                getattr(current, k) != getattr(assessed, k)
                for k in ("funding", "collected", "data_quality_percent", "scavenger_found")
            )
        ):
            raise BrowserSafetyStop("assessed_project_changed")
        checkbox = page.get_by_role("checkbox", name="I am ready to submit project.", exact=True)
        if checkbox.count() != 1 or not checkbox.is_visible() or checkbox.is_checked():
            raise BrowserSafetyStop("submission_must_remain_unchecked")
        return report, read_outer_score(page)

    def bind():
        matches = []
        allowed = {r.url for r in config.frames if r.url != SIMULATION_URL}
        for frame in page.frames:
            if frame.url not in allowed:
                continue
            for button in frame.get_by_role(
                "button", name=re.compile(r"^Update Score$", re.IGNORECASE)
            ).all():
                if button.is_visible():
                    matches.append(button)
        if len(matches) != 1 or not exposed_button(matches[0]):
            raise BrowserSafetyStop("score_transfer_control_unavailable")
        return matches[0]

    def acknowledgement():
        # A generic success alert or stale notice is not a transfer receipt.
        from .browser_planet_chart import VISIBLE_TOOLTIP

        notices = [x for x in page.get_by_text("Score updated.", exact=True).all() if x.is_visible()]
        if len(notices) > 1:
            raise BrowserSafetyStop("ambiguous_score_acknowledgement")
        return len(notices) == 1 and notices[0].evaluate(VISIBLE_TOOLTIP)

    page.on("dialog", dialog_handler)
    try:
        before, score_before = read()
        if acknowledgement():
            raise BrowserSafetyStop("stale_score_acknowledgement")
        button = bind().element_handle(timeout=2000)
        save_probe(before, directory / "before")
        # A durable source-revision claim also prevents retry through a new
        # output directory or process after an uncertain transfer.
        persist_json(
            claim,
            {
                "source_sha256": provenance["source_sha256"],
                "data_revision": provenance["data_revision"],
                "output": str(directory),
                "max_clicks": 1,
                "automatic_retry": False,
            },
        )
        persist_json(
            directory / "reserved.json",
            {
                **provenance,
                "action": "Update Score",
                "max_clicks": 1,
                "score_before": score_before,
                "submission_enabled": False,
                "task_completed": False,
            },
        )
        if before_dispatch is not None:
            before_dispatch()
        current, score = read()
        latest = bind().element_handle(timeout=2000)
        if (
            comparable_screen(current) != comparable_screen(before)
            or score != score_before
            or acknowledgement()
            or not button.evaluate("(a,b)=>a.isConnected&&a===b", latest)
        ):
            raise BrowserSafetyStop("score_transfer_changed_after_reservation")
        if check_cancelled is not None:
            check_cancelled()
        if settle_inventory and time.monotonic() >= min(
            read_deadline, parent_deadline if parent_deadline is not None else read_deadline
        ):
            raise BrowserSafetyStop("score_transfer_read_deadline")
        attempted = True
        button.click(timeout=3000)
        deadline = time.monotonic() + timeout_seconds
        read_deadline = deadline
        while not acknowledgement():
            if check_cancelled is not None:
                check_cancelled()
            if unexpected_dialog or not config.allows(page.url):
                raise BrowserSafetyStop("score_transfer_context_changed")
            if time.monotonic() >= deadline:
                raise BrowserSafetyStop("score_transfer_acknowledgement_timeout")
            if settle_inventory and parent_deadline is not None and time.monotonic() >= parent_deadline:
                raise BrowserSafetyStop("score_transfer_read_deadline")
            page.wait_for_timeout(50)
        persist_json(directory / "acknowledgement.json", {"visible_text": "Score updated."})
        after, score_after = read()
        save_probe(after, directory / "after")
        persist_json(
            directory / "score-readback.json", {"score_before": score_before, "score_after": score_after}
        )
        if score_projection(after, feedback=True) != score_projection(before):
            raise BrowserSafetyStop("unexpected_score_transfer_side_effect")
        receipt = {
            **provenance,
            "score_before": score_before,
            "score_after": score_after,
            "score_transfer_verified": True,
            "score_computed_locally": False,
            "save_verified": False,
            "submitted": False,
            "task_completed": False,
            "browser_acceptance_passed": False,
            "automatic_retry": False,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "score_transfer_failed",
                "write_may_have_occurred": attempted,
                "automatic_retry": False,
                "submitted": False,
                "task_completed": False,
            },
        )
        raise
    finally:
        page.remove_listener("dialog", dialog_handler)


def reconcile_score_transfer(page, config, failed_output, output):
    """Read-only review of the one observed feedback-controls mismatch.

    No click, checkbox change, score recalculation, acknowledgement dismissal or
    retry. The original stopped report and source-revision claim stay unchanged.
    """
    failed, directory = Path(failed_output), Path(output)
    read_json = lambda path: json.loads(path.read_text())
    stopped = read_json(failed / "stopped.json")
    reservation = read_json(failed / "reserved.json")
    if stopped != {
        "reason": "unexpected_score_transfer_side_effect",
        "write_may_have_occurred": True,
        "automatic_retry": False,
        "submitted": False,
        "task_completed": False,
    } or read_json(failed / "acknowledgement.json") != {"visible_text": "Score updated."}:
        raise BrowserSafetyStop("unsupported_score_reconciliation")
    source = reservation["source_receipt"]
    assessed, expected_rows, provenance = load_assessed_revision(source)
    if any(reservation[k] != v for k, v in provenance.items()):
        raise BrowserSafetyStop("score_reconciliation_source_changed")
    claim = read_json(Path(source).parent / f"score-transfer-revision-{provenance['data_revision']}.json")
    if claim != {
        "source_sha256": provenance["source_sha256"],
        "data_revision": provenance["data_revision"],
        "output": str(failed),
        "max_clicks": 1,
        "automatic_retry": False,
    }:
        raise BrowserSafetyStop("score_reconciliation_claim_mismatch")
    captures, hashes = {}, {}
    for name in ("before", "after"):
        raw = (failed / name / "observation.json").read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if hashes[name] != read_json(failed / name / "manifest.json")["observation_sha256"]:
            raise BrowserSafetyStop("score_reconciliation_capture_hash_mismatch")
        captures[name] = json.loads(raw)
    if score_projection(captures["before"]) != score_projection(captures["after"], feedback=True):
        raise BrowserSafetyStop("score_reconciliation_has_other_changes")
    page.wait_for_timeout(0)
    if len(page.context.pages) != 1:
        raise BrowserSafetyStop("unexpected_popup")
    current = inspect_page(page, assessment_config(config))
    if current["ignored_frame_urls"] or comparable_screen(current) != comparable_screen(captures["after"]):
        raise BrowserSafetyStop("score_reconciliation_live_state_changed")
    check = page.get_by_role("checkbox", name="I am ready to submit project.", exact=True)
    if check.count() != 1 or check.is_checked():
        raise BrowserSafetyStop("submission_must_remain_unchecked")
    sim = next(f for f in current["frames"] if f["url"] == SIMULATION_URL)
    state = parse_assessment_text(sim["text"])
    if rows_hash(sim["text"]) != expected_rows or any(
        getattr(state, k) != getattr(assessed, k)
        for k in ("funding", "collected", "data_quality_percent", "scavenger_found")
    ):
        raise BrowserSafetyStop("assessed_project_changed")
    score_after = read_outer_score(page)
    if float(score_after) == float(reservation["score_before"]):
        raise BrowserSafetyStop("score_reconciliation_requires_changed_readback")
    directory.mkdir(parents=True, exist_ok=False)
    save_probe(current, directory / "current")
    result = {
        **provenance,
        "failed_run": str(failed),
        "source_capture_sha256": hashes,
        "score_before": reservation["score_before"],
        "score_after": score_after,
        "score_transfer_verified": True,
        "original_failure_preserved": True,
        "browser_actions_executed": 0,
        "submitted": False,
        "task_completed": False,
        "browser_acceptance_passed": False,
        "save_verified": False,
        "review": "known_score_feedback_controls_only_plus_original_visible_acknowledgement",
    }
    persist_json(directory / "reconciled.json", result)
    return result
