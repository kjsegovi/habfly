"""New assessment stages with immutable history, not retries of earlier data.

At most two simulation-only assessments per explicit changed-data revision.
An uncertain dispatched action blocks continuation. A separately reviewed,
hashed pre-dispatch disposition preserves the old reservation without treating
it as an assessment or authorizing the same revision again.
"""

import hashlib
import json
import time
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import (
    AssessmentActuator,
    _settled_assessment_report,
    _settling_options,
    assessment_config,
    persist_json,
    rows_hash,
)
from .browser_probe import inspect_page
from .browser_stellar import SIMULATION_URL
from .project_assessment import AssessmentError, AssessmentLedger, parse_assessment_text


def assessment_history(root):
    root = Path(root).resolve()
    history = []
    for path in root.glob("*/attempt-*-reserved.json"):
        raw = path.read_bytes()
        reserved = json.loads(raw)
        ledger = AssessmentLedger.model_validate(reserved["ledger"])
        attempt = ledger.pending
        if attempt is None or path.name != f"attempt-{attempt.sequence:03d}-reserved.json":
            raise AssessmentError("inconsistent_project_assessment_history")
        if reserved.get("data_revision") != attempt.data_revision:
            raise AssessmentError("inconsistent_project_assessment_revision")
        prefix = path.name.removesuffix("-reserved.json")
        confirmed = path.with_name(prefix + "-confirmed.json")
        sources = {str(path): hashlib.sha256(raw).hexdigest()}
        if confirmed.exists():
            payload = json.loads(confirmed.read_bytes())
            completed = AssessmentLedger.model_validate(payload["ledger"])
            if (
                completed.pending
                or len(completed.attempts) != len(ledger.attempts)
                or completed.attempts[:-1] != ledger.attempts[:-1]
                or completed.attempts[-1].model_copy(update={"after": None}) != attempt
                or payload.get("receipt") != completed.attempts[-1].model_dump(mode="json")
                or payload.get("project_rows_sha256") != reserved.get("project_rows_sha256")
            ):
                raise AssessmentError("inconsistent_project_assessment_confirmation")
            sources[str(confirmed)] = hashlib.sha256(confirmed.read_bytes()).hexdigest()
            outcome = "confirmed"
        else:
            disposition_path = path.with_name(prefix + "-undispatched.json")
            if not disposition_path.exists():
                raise AssessmentError("unresolved_project_assessment_reservation")
            disposition = json.loads(disposition_path.read_bytes())
            stopped_path = path.with_name(prefix + "-stopped.json")
            stopped = json.loads(stopped_path.read_bytes())
            if (
                disposition.get("disposition") != "guard_stopped_before_click_dispatch"
                or disposition.get("data_revision") != attempt.data_revision
                or disposition.get("kind") != attempt.kind
                or disposition.get("automatic_retry") is not False
                or disposition.get("assessment_confirmed") is not False
                or disposition.get("clicks_executed_by_reconciliation") != 0
                or stopped.get("reason") != "assessment_changed_after_reservation"
                or stopped.get("ledger") != reserved["ledger"]
                or (
                    stopped.get("click_dispatch_started") is not False
                    and not (
                        "click_dispatch_started" not in stopped
                        and disposition.get("legacy_guard_review") is True
                    )
                )
            ):
                raise AssessmentError("invalid_project_predispatch_disposition")
            hashes = disposition.get("source_sha256", {})
            required = {path.resolve(), stopped_path.resolve()}
            if not required <= {Path(p).resolve() for p in hashes}:
                raise AssessmentError("missing_project_disposition_sources")
            for name, digest in hashes.items():
                source = Path(name).resolve()
                if (
                    not source.is_relative_to(root)
                    or hashlib.sha256(source.read_bytes()).hexdigest() != digest
                ):
                    raise AssessmentError("project_disposition_source_changed")
            sources.update(hashes)
            sources[str(disposition_path)] = hashlib.sha256(disposition_path.read_bytes()).hexdigest()
            outcome = "not_dispatched_original_reservation_retained"
        history.append(
            {
                "revision": attempt.data_revision,
                "kind": attempt.kind,
                "outcome": outcome,
                "project_rows_sha256": reserved["project_rows_sha256"],
                "source_sha256": sources,
            }
        )
    identities = [(h["revision"], h["kind"]) for h in history]
    if len(set(identities)) != len(identities):
        raise AssessmentError("duplicate_project_assessment_revision")
    return sorted(history, key=lambda row: (row["revision"], row["kind"]))


def begin_assessment_stage(
    page,
    config,
    output,
    *,
    history_root,
    data_revision,
    collected,
    reason,
    max_attempts=2,
    settle_inventory=False,
    deadline=None,
    timeout_seconds=10,
    check_cancelled=None,
):
    """Open one explicit stage; optional settling is for live paginated sources.

    ``deadline`` is an absolute monotonic owner bound. Both initial reads share
    the same setup timeout; a later charge still has its original receipt budget.
    False/default preserves the legacy raw reads and actuator call arguments.
    """
    _settling_options(settle_inventory, deadline)
    if check_cancelled is not None and not callable(check_cancelled):
        raise TypeError("Assessment cancellation hook must be callable")
    if type(timeout_seconds) not in {int, float} or not 0.1 <= timeout_seconds <= 30:
        raise ValueError("Assessment deadline must be bounded")
    if not settle_inventory and (timeout_seconds != 10 or check_cancelled is not None):
        raise ValueError("Stage read hooks and timeout require explicit inventory settling")
    root, directory = Path(history_root).resolve(), Path(output).resolve()
    if (
        directory.parent != root
        or type(data_revision) is not int
        or data_revision < 1
        or type(collected) is not int
        or not 1 <= collected <= 500
        or type(max_attempts) is not int
        or max_attempts not in {1, 2}
        or not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > 1000
    ):
        raise ValueError("Explicit bounded assessment stage and owned output are required")
    history = assessment_history(root)
    if history and data_revision <= history[-1]["revision"]:
        raise AssessmentError("new_project_revision_required_not_retry")
    if settle_inventory:
        original_frames = tuple(page.frames)
        until = time.monotonic() + timeout_seconds
        if deadline is not None:
            until = min(until, deadline)
        before = _settled_assessment_report(
            page,
            assessment_config(config),
            deadline=until,
            check_cancelled=check_cancelled,
            inspector=inspect_page,
            frames=original_frames,
        )
    else:
        before = inspect_page(page, assessment_config(config))
    if before["ignored_frame_urls"] or len(page.context.pages) != 1:
        raise BrowserSafetyStop("unexpected_project_assessment_context")
    frames = [f for f in before["frames"] if f["url"] == SIMULATION_URL]
    if len(frames) != 1:
        raise BrowserSafetyStop("ambiguous_simulation_frame")
    snapshot, fingerprint = parse_assessment_text(frames[0]["text"]), rows_hash(frames[0]["text"])
    if snapshot.collected != collected or snapshot.acknowledgement is not None:
        raise AssessmentError("project_stage_collection_or_receipt_mismatch")
    if history and any(h["project_rows_sha256"] == fingerprint for h in history):
        raise AssessmentError("project_rows_unchanged_not_a_new_stage")
    actuator = AssessmentActuator(
        page,
        config,
        directory,
        budget=100 * max_attempts,
        max_attempts=max_attempts,
        **(
            {"settle_inventory": True, "deadline": deadline, "timeout_seconds": timeout_seconds}
            if settle_inventory
            else {}
        ),
    )
    if settle_inventory:
        actuator._settle_frames = original_frames
    _, current, current_rows = (
        actuator.read(deadline=until, check_cancelled=check_cancelled)
        if settle_inventory
        else actuator.read()
    )
    if current != snapshot or current_rows != fingerprint:
        raise AssessmentError("project_stage_changed_during_setup")
    persist_json(
        directory / "stage.json",
        {
            "data_revision": data_revision,
            "collected": collected,
            "reason": reason,
            "budget_simulation_dollars": 100 * max_attempts,
            "max_attempts": max_attempts,
            "history": history,
            "new_project_rows_sha256": fingerprint,
            "same_revision_retry_allowed": False,
            "real_money": False,
            "scope": "explicit_assessment_only_not_save_score_transfer_or_submission",
        },
    )
    return actuator
