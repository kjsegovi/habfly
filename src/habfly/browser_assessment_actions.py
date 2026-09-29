"""At-most-once visible assessment actions; simulation currency only.

No answer editing, automation unlocks, Save, score transfer or submission. The
caller supplies the data revision and must open the collected-list assessment
view first. Each reservation is fsynced before the native click. A timeout or
uncertain result leaves that reservation pending, including across processes.
"""

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_probe import BrowserProbeConfig, inspect_page, save_probe
from .browser_setup import RENDER_VISIBILITY_JS, rendered_control
from .browser_stellar import SIMULATION_URL
from .project_assessment import ACKNOWLEDGEMENTS, AssessmentError, AssessmentLedger, parse_assessment_text


def assessment_config(config):
    payload = config.model_dump()
    for frame in payload["frames"]:
        if frame["url"] == SIMULATION_URL:
            frame["required_text"] = ["FUNDING", "TOTAL COLLECTED"]
    return BrowserProbeConfig.model_validate(payload)


def rows_hash(text):
    normalized = " ".join(text.upper().split())
    rows = re.findall(r"\bOBSERVATIONS\b(.*?)\bTOTAL COLLECTED\s+[0-9]+\b", normalized)
    if len(rows) != 1:
        raise AssessmentError("ambiguous_visible_project_rows")
    return hashlib.sha256(rows[0].encode()).hexdigest()


def persist_json(path, value):
    # Exclusive immutable artifacts prevent accidental retries/overwrites.
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _settled_assessment_report(page, config, *, deadline, check_cancelled=None, inspector=None, frames=None):
    """Explicit paginated-only raw read; settling never extends an action budget."""
    from .browser_project_paginated_inventory_steps import read_settled_inventory

    original = tuple(page.frames) if frames is None else frames

    def check():
        if check_cancelled is not None:
            check_cancelled()
        if time.monotonic() >= deadline:
            raise AssessmentError("assessment_read_deadline")
        if len(page.context.pages) != 1 or tuple(page.frames) != original:
            raise BrowserSafetyStop("assessment_read_context_changed")
        if not config.allows(page.url):
            raise BrowserSafetyStop("navigation_outside_activity")

    def inspect(p, c):
        report = (inspector or inspect_page)(p, c)
        if report["ignored_frame_urls"]:
            raise BrowserSafetyStop("unknown_visible_frame")
        return report

    return read_settled_inventory(
        page, config, check=check, deadline=deadline, clock=time.monotonic, inspector=inspect
    )


def _settling_options(enabled, deadline):
    if type(enabled) is not bool:
        raise ValueError("Inventory settling must be an explicit boolean")
    if deadline is not None and (
        not enabled or type(deadline) not in {int, float} or not math.isfinite(deadline)
    ):
        raise ValueError("Inventory settling requires a finite absolute deadline")


def record_undispatched_assessment(source, observed_capture, *, accept_legacy_preclick_guard=False):
    """Offline disposition of a known guard stop before click dispatch.

    Does not remove the reservation, mutate a ledger, click, or permit retry.
    The exceptional legacy path is explicit and limited to the exact guard
    raised before button.click in v1. A funding readback alone never suffices.
    """
    source, capture = Path(source), Path(observed_capture)
    stopped_paths = sorted(source.glob("attempt-*-stopped.json"))
    if len(stopped_paths) != 1:
        raise AssessmentError("one_predispatch_stop_required")
    stopped_path = stopped_paths[0]
    stopped = json.loads(stopped_path.read_bytes())
    if stopped.get("reason") != "assessment_changed_after_reservation":
        raise AssessmentError("assessment_dispatch_not_proven_absent")
    legacy = "click_dispatch_started" not in stopped
    if legacy:
        if accept_legacy_preclick_guard is not True or stopped.get("writes_may_have_occurred") is not True:
            raise AssessmentError("legacy_predispatch_review_required")
    elif (
        stopped.get("click_dispatch_started") is not False
        or stopped.get("writes_may_have_occurred") is not False
    ):
        raise AssessmentError("assessment_dispatch_not_proven_absent")
    ledger = AssessmentLedger.model_validate(stopped["ledger"])
    attempt = ledger.pending
    if attempt is None:
        raise AssessmentError("pending_predispatch_reservation_required")
    prefix = f"attempt-{attempt.sequence:03d}"
    reserved_path = source / f"{prefix}-reserved.json"
    reserved = json.loads(reserved_path.read_bytes())
    if (
        stopped_path.name != f"{prefix}-stopped.json"
        or (source / f"{prefix}-confirmed.json").exists()
        or reserved.get("ledger") != stopped["ledger"]
        or reserved.get("data_revision") != attempt.data_revision
        or reserved.get("action") != {"kind": "CLICK", "visible_label": "ASSESS"}
        or reserved.get("simulation_dollars") != 100
        or reserved.get("real_money") is not False
    ):
        raise AssessmentError("predispatch_reservation_mismatch")
    paths = [stopped_path, reserved_path]
    for folder in (source / f"{prefix}-before", capture):
        raw = (folder / "observation.json").read_bytes()
        manifest = json.loads((folder / "manifest.json").read_bytes())
        if hashlib.sha256(raw).hexdigest() != manifest.get("observation_sha256"):
            raise AssessmentError("predispatch_capture_hash_mismatch")
        report = json.loads(raw)
        frames = [f for f in report["frames"] if f["url"] == SIMULATION_URL]
        if (
            report.get("ignored_frame_urls")
            or len(frames) != 1
            or parse_assessment_text(frames[0]["text"]) != attempt.before
            or rows_hash(frames[0]["text"]) != reserved.get("project_rows_sha256")
        ):
            raise AssessmentError("predispatch_readback_changed")
        paths.extend((folder / "observation.json", folder / "manifest.json"))
    receipt = {
        "disposition": "guard_stopped_before_click_dispatch",
        "kind": attempt.kind,
        "data_revision": attempt.data_revision,
        "clicks_executed_by_reconciliation": 0,
        "legacy_guard_review": legacy,
        "funding_unchanged": True,
        "original_pending_reservation_preserved": True,
        "unused_reserved_simulation_dollars": 100,
        "automatic_retry": False,
        "assessment_confirmed": False,
        "task_completed": False,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    persist_json(source / f"{prefix}-undispatched.json", receipt)
    return receipt


class AssessmentActuator:
    def __init__(
        self,
        page,
        config,
        output,
        *,
        budget=0,
        max_attempts=0,
        timeout_seconds=10,
        settle_inventory=False,
        deadline=None,
    ):
        _settling_options(settle_inventory, deadline)
        if not 0.1 <= timeout_seconds <= 30:
            raise ValueError("Assessment deadline must be bounded")
        self.page, self.config, self.output = page, assessment_config(config), Path(output)
        self.ledger = AssessmentLedger(budget=budget, max_attempts=max_attempts)
        self.timeout_seconds = timeout_seconds
        self.settle_inventory, self.read_deadline = settle_inventory, deadline
        self._settle_frames = None
        self.stopped = False
        self.frame = None
        self.output.mkdir(parents=True, exist_ok=False)
        persist_json(self.output / "ledger-initial.json", self.ledger.model_dump(mode="json"))

    def read(self, *, deadline=None, check_cancelled=None):
        if self.stopped:
            raise AssessmentError("assessment_actuator_stopped")
        if len(self.page.context.pages) != 1:
            raise BrowserSafetyStop("unexpected_popup")
        if self.settle_inventory:
            until = time.monotonic() + self.timeout_seconds if deadline is None else deadline
            if self.read_deadline is not None:
                until = min(until, self.read_deadline)
            if self._settle_frames is None:
                self._settle_frames = tuple(self.page.frames)

            def check():
                if self.stopped:
                    raise AssessmentError("assessment_actuator_stopped")
                if check_cancelled is not None:
                    check_cancelled()

            report = _settled_assessment_report(
                self.page,
                self.config,
                deadline=until,
                check_cancelled=check,
                frames=self._settle_frames,
            )
        else:
            report = inspect_page(self.page, self.config)
        if report["ignored_frame_urls"]:
            raise BrowserSafetyStop("unknown_visible_frame")
        frames = [f for f in self.page.frames if f.url == SIMULATION_URL]
        if len(frames) != 1 or self.frame is not None and frames[0] != self.frame:
            raise BrowserSafetyStop("simulation_frame_changed")
        self.frame = frames[0]
        visible = next(f for f in report["frames"] if f["url"] == SIMULATION_URL)
        return report, parse_assessment_text(visible["text"]), rows_hash(visible["text"])

    def button(self, label):
        controls = [
            e
            for e in self.frame.get_by_role(
                "button", name=re.compile("^" + re.escape(label) + "$", re.IGNORECASE)
            ).all()
            if e.is_visible()
        ]
        exposed = len(controls) == 1 and controls[0].evaluate(
            """e => {"""
            + RENDER_VISIBILITY_JS
            + """
            const r=e.getBoundingClientRect(), hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
            return e.tagName==='BUTTON' && styled(e) && hit && (hit===e || e.contains(hit)) &&
                styled(hit) && exposed(r,hit);
        }"""
        )
        if len(controls) != 1 or not controls[0].is_enabled() or not exposed:
            raise AssessmentError("ambiguous_or_unavailable_assessment_control")
        return controls[0]

    def assess(self, kind, *, data_revision, before_dispatch=None, check_cancelled=None):
        # Optional owner hooks never replace the adapter's own durable claim or
        # visible/native guards. The first can reserve the canonical journal;
        # the second is a non-emitting cancellation/source check at dispatch.
        for hook in (before_dispatch, check_cancelled):
            if hook is not None and not callable(hook):
                raise TypeError("Assessment dispatch hooks must be callable")
        deadline = time.monotonic() + self.timeout_seconds if self.settle_inventory else None
        read = (
            (lambda: self.read(deadline=deadline, check_cancelled=check_cancelled))
            if self.settle_inventory
            else self.read
        )
        before_report, before, before_rows = read()
        if kind not in {"data_quality", "scavenger_hunt"}:
            raise AssessmentError("unsupported_assessment_kind")
        if before.mode != kind:
            raise AssessmentError("open_requested_assessment_panel_first")
        button = self.button("ASSESS").element_handle(timeout=2000)
        # Recheck identity/values after control binding and before reservation.
        _, current, current_rows = read()
        if current != before or current_rows != before_rows:
            raise AssessmentError("assessment_observation_changed")
        attempt = self.ledger.reserve(kind, before, data_revision=data_revision)
        prefix = f"attempt-{attempt.sequence:03d}"
        click_started = False
        predispatch_checks = None
        try:
            save_probe(before_report, self.output / f"{prefix}-before")
            persist_json(
                self.output / f"{prefix}-reserved.json",
                {
                    "ledger": self.ledger.model_dump(mode="json"),
                    "project_rows_sha256": before_rows,
                    "data_revision": data_revision,
                    "action": {"kind": "CLICK", "visible_label": "ASSESS"},
                    "simulation_dollars": 100,
                    "real_money": False,
                },
            )
            # No UI action is allowed before the reservation above is durable.
            if before_dispatch is not None:
                before_dispatch()
            recheck_report, current, current_rows = read()
            live = self.button("ASSESS").element_handle(timeout=2000)
            predispatch_checks = {
                "snapshot_unchanged": current == before,
                "rows_unchanged": current_rows == before_rows,
                "native_control_unchanged": button.evaluate("(a,b)=>a.isConnected&&a===b", live),
            }
            if not all(predispatch_checks.values()):
                save_probe(recheck_report, self.output / f"{prefix}-predispatch")
                raise AssessmentError("assessment_changed_after_reservation")
            if check_cancelled is not None:
                check_cancelled()
            if self.settle_inventory and time.monotonic() >= min(deadline, self.read_deadline or deadline):
                raise AssessmentError("assessment_read_deadline")
            click_started = True
            button.click(timeout=3000)
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                if check_cancelled is not None:
                    check_cancelled()
                after_report, after, after_rows = read()
                if after_rows != before_rows:
                    # Preserve the actual visible discrepancy before stopping.
                    # This is evidence only: never relax the row guard or retry.
                    save_probe(after_report, self.output / f"{prefix}-changed-rows")
                    persist_json(
                        self.output / f"{prefix}-row-difference.json",
                        {
                            "before_sha256": before_rows,
                            "after_sha256": after_rows,
                            "click_dispatch_started": True,
                            "automatic_retry": False,
                        },
                    )
                    raise AssessmentError("project_rows_changed_during_assessment")
                if after.acknowledgement is not None:
                    break
                if time.monotonic() >= deadline:
                    raise AssessmentError("assessment_receipt_timeout")
                self.page.wait_for_timeout(100)
            # Save visible evidence before accepting the receipt.
            save_probe(after_report, self.output / f"{prefix}-after")
            receipt = self.ledger.confirm(attempt.sequence, after, data_revision=data_revision)
            persist_json(
                self.output / f"{prefix}-confirmed.json",
                {
                    "ledger": self.ledger.model_dump(mode="json"),
                    "receipt": receipt.model_dump(mode="json"),
                    "project_rows_sha256": after_rows,
                    "report": self.ledger.report(data_revision=data_revision),
                },
            )
            return self.ledger.report(data_revision=data_revision)
        except BaseException as exc:
            self.stopped = True
            reason = (
                str(exc)
                if isinstance(exc, (AssessmentError, BrowserSafetyStop))
                else "assessment_operation_uncertain"
            )
            persist_json(
                self.output / f"{prefix}-stopped.json",
                {
                    "reason": reason,
                    "ledger": self.ledger.model_dump(mode="json"),
                    "retry_allowed": False,
                    "writes_may_have_occurred": click_started,
                    "click_dispatch_started": click_started,
                    "predispatch_checks": predispatch_checks,
                },
            )
            raise

    def dismiss_receipt(self, *, before_dispatch=None, check_cancelled=None):
        """Only dismiss a receipt already confirmed by this exact actuator."""
        for hook in (before_dispatch, check_cancelled):
            if hook is not None and not callable(hook):
                raise TypeError("Assessment dispatch hooks must be callable")
        deadline = time.monotonic() + self.timeout_seconds if self.settle_inventory else None
        read = (
            (lambda: self.read(deadline=deadline, check_cancelled=check_cancelled))
            if self.settle_inventory
            else self.read
        )
        before_report, before, rows = read()
        if not self.ledger.attempts or self.ledger.pending:
            raise AssessmentError("no_confirmed_receipt_to_dismiss")
        last = self.ledger.attempts[-1]
        prefix = f"ack-{last.sequence:03d}"
        if (self.output / f"{prefix}-reserved.json").exists():
            raise AssessmentError("assessment_dismissal_already_attempted")
        if last.after != before:
            # Preserve transient mismatches too. A pre-click rejection is not
            # an uncertain dismissal, but its observed evidence must survive.
            if not (self.output / f"{prefix}-stale.json").exists():
                save_probe(before_report, self.output / f"{prefix}-stale-capture")
                persist_json(
                    self.output / f"{prefix}-stale.json",
                    {
                        "reason": "receipt_changed_before_dismiss",
                        "click_dispatch_started": False,
                        "expected": last.after.model_dump(mode="json"),
                        "observed": before.model_dump(mode="json"),
                        "automatic_retry": False,
                    },
                )
            raise AssessmentError("receipt_changed_before_dismiss")
        native = [
            e
            for e in self.frame.get_by_role("button", name=re.compile("^ok$", re.IGNORECASE)).all()
            if e.is_visible()
        ]
        if native:
            control = self.button("OK")
        else:
            custom = [
                e for e in self.frame.get_by_text(re.compile("^ok$", re.IGNORECASE)).all() if e.is_visible()
            ]
            if (
                len(custom) != 1
                or custom[0].evaluate("e=>e.tagName") != "BUTTON-POPUP"
                or not rendered_control(custom[0])
            ):
                raise AssessmentError("unavailable_assessment_acknowledgement")
            control = custom[0]
            text = " ".join(control.locator("..").inner_text().upper().split())
            if text != ACKNOWLEDGEMENTS[last.kind] + " OK":
                raise AssessmentError("unverified_assessment_acknowledgement")
        handle = control.element_handle(timeout=2000)
        dispatched = False
        try:
            persist_json(
                self.output / f"{prefix}-reserved.json",
                {
                    "kind": last.kind,
                    "data_revision": last.data_revision,
                    "max_acknowledgement_clicks": 1,
                    "assessment_clicks": 0,
                    "automatic_retry": False,
                },
            )
            if before_dispatch is not None:
                before_dispatch()
            _, current, current_rows = read()
            if (
                current != before
                or current_rows != rows
                or not handle.evaluate("(a,b)=>a.isConnected&&a===b", control.element_handle())
            ):
                raise AssessmentError("receipt_changed_before_dismiss_dispatch")
            if check_cancelled is not None:
                check_cancelled()
            if self.settle_inventory and time.monotonic() >= min(deadline, self.read_deadline or deadline):
                raise AssessmentError("assessment_read_deadline")
            dispatched = True
            handle.click(timeout=3000)
            after_report, after, new_rows = read()
            save_probe(after_report, self.output / f"{prefix}-after")
            expected = before.model_dump(exclude={"acknowledgement", "visible_text_sha256"})
            if (
                after.acknowledgement is not None
                or after.model_dump(exclude={"acknowledgement", "visible_text_sha256"}) != expected
                or rows != new_rows
            ):
                raise AssessmentError("assessment_dismissal_mismatch")
            persist_json(
                self.output / f"{prefix}-confirmed.json",
                {
                    "kind": last.kind,
                    "data_revision": last.data_revision,
                    "acknowledgement_dismissed": True,
                    "acknowledgement_clicks": 1,
                    "assessment_clicks": 0,
                    "project_rows_sha256": new_rows,
                    "automatic_retry": False,
                    "task_completed": False,
                },
            )
        except BaseException as exc:
            self.stopped = True
            persist_json(
                self.output / f"{prefix}-stopped.json",
                {
                    "reason": str(exc)
                    if isinstance(exc, (AssessmentError, BrowserSafetyStop))
                    else "assessment_dismissal_uncertain",
                    "click_dispatch_started": dispatched,
                    "automatic_retry": False,
                    "assessment_clicks": 0,
                    "task_completed": False,
                },
            )
            raise
