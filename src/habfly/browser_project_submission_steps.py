"""One-shot submission transport, deliberately without a success recognizer.

The real course's submission acknowledgement is not yet grounded. Even a returned
Submit click therefore ends unknown_pending, never in a canonical WriteReceipt.
One reservation covers readiness and Submit, and survives every later stop. The
caller owns browser lifetime; constructors and control commands perform no UI work.
"""

import fcntl
import hashlib
import json
import math
import os
import re
import time
from copy import deepcopy

from .browser import BrowserSafetyStop
from .browser_assessment_actions import assessment_config, persist_json, rows_hash
from .browser_no_planet_workflow import _Evidence
from .browser_numeric import comparable_screen, screen_identity
from .browser_probe import _visible_frame, inspect_page, public_url, save_probe
from .browser_project_submission_preflight import (
    _EXPOSED,
    READY,
    SUBMIT,
    _simulation,
    _sources,
    preflight_project_submission,
)
from .browser_score_transfer import read_outer_score
from .browser_stellar import SIMULATION_URL
from .browser_submission_controls import SUBMIT_EXPOSED, bind_readiness, outer_submit_box
from .browser_submission_controls import submit_outer_exposed as _outer_exposed
from .browser_submission_outcome_diagnostic import native_dialog_disposition, preserve_submission_outcome
from .contracts import RuntimeEvent
from .presentation_capture import evidence_screenshot
from .project_assessment import parse_assessment_text
from .project_progress import WriteReserved, _json

MODE = "canonical_project_submission_unknown_outcome"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("project_submission_" + reason)


def _reason(exc):
    code = str(exc)
    if isinstance(exc, BrowserSafetyStop) and code in {
        "authentication_required",
        "unexpected_modal",
        "navigation_outside_activity",
        "unknown_visible_frame",
    }:
        return "project_submission_" + code
    return (
        code
        if isinstance(exc, BrowserSafetyStop)
        and re.fullmatch(r"(?:project_submission_|project_submission_preflight_)[a-z0-9_]{1,120}", code)
        else "project_submission_operation_failed"
    )


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bytes(path, raw):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


class _Cancelled(Exception):
    pass


class BrowserProjectSubmissionSteps:
    """Four cooperative stages, at most one readiness and one Submit dispatch.

    ``advance`` is a manual single step even while paused; ``tick`` runs only
    when resumed. Pause takes effect between calls. Abort/callback failure is
    checked before each possible dispatch. Paused time counts toward the fixed
    wall deadline. No modal is accepted/dismissed and no browser is closed here.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        journal,
        scoring_dir,
        allow_submission=False,
        max_seconds=180,
        max_advances=4,
        emit=lambda _: None,
        cancelled=lambda: False,
        _clock=time.monotonic,
    ):
        _require(allow_submission is True, "explicit_opt_in_required")
        _require(
            type(max_seconds) in {int, float} and math.isfinite(max_seconds) and 1 <= max_seconds <= 600,
            "invalid_deadline",
        )
        _require(type(max_advances) is int and 1 <= max_advances <= 4, "invalid_advance_limit")
        _require(callable(emit) and callable(cancelled) and callable(_clock), "invalid_callback")
        self.book = _Evidence(run_history)
        self.history, self.journal = self.book.history, journal
        try:
            state, self.rows_sha, self.score, self.assessed = _sources(self.book, journal, scoring_dir)
        except BrowserSafetyStop:
            raise
        except Exception:  # noqa: BLE001 - source parsing must not expose private paths
            raise BrowserSafetyStop("project_submission_invalid_source_chain") from None
        self.scoring_dir = self.book.path(scoring_dir)
        self.journal_path = self.book.path(journal.path)
        self._journal_key = str(self.journal_path.relative_to(self.history))
        self._journal_before = self.book.read(self.journal_path)
        self._expected_journal = self._journal_before
        self.input_source_sha256 = deepcopy(self.book.hashes)
        # This single mutable file is checked byte-for-byte against its exact
        # original prefix + our sole reservation, not against an ignored hash.
        del self.book.hashes[self._journal_key]
        self.revision = state.revision
        self.output = self.book.path(output)
        _require(
            self.output != self.history
            and not self.output.exists()
            and not any(self.output.is_relative_to(path) for path in self.book.clean_directories),
            "invalid_output",
        )
        self.claim = self.book.path(
            self.history / "project-submission-claims" / f"revision-{self.revision}.json"
        )
        _require(not self.claim.exists(), "attempt_already_reserved")
        self.page, self.config = page, assessment_config(config)
        self._config = self.config.model_dump(mode="json")
        self._callback, self._cancelled, self._clock = emit, cancelled, _clock
        self.max_seconds, self.max_advances = max_seconds, max_advances
        self.allow_submission = allow_submission
        self._limits = (allow_submission, max_seconds, max_advances)
        self._started = self._clock()
        self.phase, self.status, self.failure = "preflight_pending", "paused", None
        self._busy = self._emitting = self._finalizing = self._abort_requested = False
        self._forward_failed = self.artifact_write_failed = False
        self.listener_cleanup_failed = False
        self._frames, self._dialogs, self._popups = None, [], []
        self._watching = False
        self._outcome_diagnostic_attempted = False
        self._outcome_diagnostic = None
        self._submission_feedback = None
        self._feedback_polls = 0
        self._proposal = self.report = self._before = self._ready_capture = None
        self._reservation_attempted = self._reservation_verified = False
        self.readiness_may_have_occurred = self.submit_may_have_occurred = False
        self.readiness_click_returned = self.submit_click_returned = False
        self.readiness_verified = False
        self.advances = self._sequence = 0
        self.artifacts = {}
        self.scope = {
            "schema_version": 1,
            "mode": MODE,
            "project_id": state.project_id,
            "attempt_id": state.attempt_id,
            "data_revision": self.revision,
            "project_rows_sha256": self.rows_sha,
            "allow_submission": True,
            "max_seconds": max_seconds,
            "max_advances": max_advances,
            "max_readiness_writes": 1,
            "max_submit_clicks": 1,
            "pause_counts_toward_deadline": True,
            "automatic_retry": False,
            "acknowledgement_recognizer_implemented": False,
            "canonical_success_receipt_implemented": False,
            "negative_feedback_recognizer_implemented": True,
            "feedback_wait_max_seconds": 3.0,
            "feedback_max_read_polls": 13,
            "diagnostic_screenshots": "authorized_post_auth_current_viewport_only",
            "submitted": False,
            "submission_verified": False,
            "task_completed": False,
            "project_completed": False,
            "browser_acceptance_passed": False,
            "scientific_verified": False,
            "training_label": False,
        }
        self.book.unchanged()
        self.output.mkdir(parents=True, exist_ok=False)
        persist_json(self.output / "scope.json", self.scope)
        self.book.read(self.output / "scope.json")
        self._stream = (self.output / "events.jsonl").open("x", encoding="utf-8")
        self._run_id = "submission-" + _sha(str(self.output.relative_to(self.history)).encode())[:16]

    @property
    def finished(self):
        return self.status in {"stopped", "aborted"}

    def _check(self):
        if self._abort_requested or self.finished or self._cancelled():
            raise _Cancelled
        _require(self._clock() - self._started < self.max_seconds, "time_limit")
        _require(
            (self.allow_submission, self.max_seconds, self.max_advances) == self._limits, "limits_changed"
        )
        _require(self.config.model_dump(mode="json") == self._config, "config_changed")
        _require(
            {self.book.path(p) for p in self.history.glob("project-progress-*.jsonl")} == {self.journal_path},
            "ambiguous_canonical_journal",
        )
        _require(self.journal_path.read_bytes() == self._expected_journal, "canonical_journal_changed")
        progress = self.journal.load()
        state = progress.reduce()
        _require(
            state.revision == self.revision
            and len(state.stars) == 30
            and all(s.task_completed for s in state.stars.values()),
            "canonical_tasks_changed",
        )
        _require(
            state.pending == ([self._proposal] if self._reservation_verified else []), "reservation_changed"
        )
        for kind in ("assessment_data_quality", "assessment_scavenger_hunt", "score_transfer"):
            receipt = state.receipt(kind)
            _require(
                receipt is not None and receipt.project_rows_sha256 == self.rows_sha, "assessments_changed"
            )
        self.book.unchanged()

    def _timeout(self, cap=3000):
        self._check()
        return max(1, min(cap, int((self.max_seconds - (self._clock() - self._started)) * 1000)))

    def _watch(self):
        if not self._watching:
            self._frames = self.page.frames.copy()
            self._feedback_visible_frames = [
                frame
                for frame in self._frames
                if frame is self.page.main_frame or _visible_frame(frame, self.page.main_frame)
            ]
            self._dialog_handler = lambda dialog: self._dialogs.append(native_dialog_disposition(dialog))
            self._popup_handler = lambda _: self._popups.append(True)
            self.page.on("dialog", self._dialog_handler)
            self.page.context.on("page", self._popup_handler)
            self._watching = True

    def _context(self):
        self._check()
        self.page.wait_for_timeout(0)
        _require(
            not self._dialogs
            and not self._popups
            and self.page.frames == self._frames
            and len(self.page.context.pages) == 1
            and self.config.allows(self.page.url),
            "context_changed",
        )

    def _controls(self, report, *, checked):
        """Native checked-state binder; preflight's unchecked-only API stays unchanged."""
        checkbox = self.page.get_by_role("checkbox", name=READY, exact=True)
        _require(
            checkbox.count() == 1
            and checkbox.is_visible()
            and checkbox.is_enabled()
            and checkbox.is_checked() is checked,
            "readiness_control_changed",
        )
        readiness = bind_readiness(
            self.page,
            checkbox,
            checked=checked,
            exposed=_EXPOSED,
            prefix="project_submission_",
            timeout=self._timeout(),
        )
        allowed = {rule.url for rule in self.config.frames if rule.url != SIMULATION_URL}
        buttons = [
            (frame, button)
            for frame in self.page.frames
            if frame.url in allowed
            for button in frame.get_by_role("button", name=SUBMIT, exact=True).all()
        ]
        _require(len(buttons) == 1, "submit_control_ambiguous")
        frame, button = buttons[0]
        _require(
            frame.parent_frame == self.page.main_frame
            and button.is_visible()
            and button.is_enabled() is True,
            "submit_control_unavailable",
        )
        button_box = button.evaluate(SUBMIT_EXPOSED)
        _require(
            isinstance(button_box, dict) and _outer_exposed(frame, [outer_submit_box(button_box)]),
            "submit_control_unexposed",
        )
        checkbox_label = f'- checkbox "{READY}"' + (" [checked]" if checked else "")
        outer = [
            c for c in report["outer_controls"] if c["role"] == "checkbox" and READY in c["accessibility"]
        ]
        captured = [
            (f, c)
            for f in report["frames"]
            for c in f["controls"]
            if c["role"] == "button" and SUBMIT in c["accessibility"]
        ]
        _require(
            len(outer) == len(captured) == 1
            and outer[0]["accessibility"] == checkbox.aria_snapshot() == checkbox_label
            and outer[0].get("enabled") is True
            and outer[0].get("protected") is True
            and not any(SUBMIT in c["accessibility"] for c in report["outer_controls"])
            and not any(READY in c["accessibility"] for f in report["frames"] for c in f["controls"]),
            "readiness_capture_mismatch",
        )
        rule = next(rule for rule in self.config.frames if rule.url == frame.url)
        siblings = [f for f in self.page.frames if f.url == frame.url]
        captured_frame, control = captured[0]
        _require(
            len(siblings) == rule.count
            and captured_frame["id"] == f"{rule.name}-{siblings.index(frame)}"
            and captured_frame["url"] == frame.url
            and control["accessibility"] == button.aria_snapshot() == f'- button "{SUBMIT}"'
            and control.get("enabled") is True
            and control.get("protected") is True,
            "submit_capture_mismatch",
        )
        return (
            readiness,
            button.element_handle(timeout=self._timeout()),
        )

    def _read(self, *, checked):
        self._context()
        report = inspect_page(self.page, self.config)
        text = _simulation(report)
        _require(
            parse_assessment_text(text) == self.assessed and rows_hash(text) == self.rows_sha, "rows_changed"
        )
        _require(float(read_outer_score(self.page)) == self.score, "score_changed")
        controls = self._controls(report, checked=checked)
        self._context()
        return report, controls

    def _save_capture(self, report, name):
        path = self.output / name
        save_probe(report, path)
        self.book.capture(path)
        self.artifacts[name] = str(path.relative_to(self.history))

    def _diagnostic(self, name, *, expected=None):
        """No modal/auth/unknown-frame bypass to obtain an image after failure.

        Text requirements alone may disappear after Submit; the exact authorized
        origin, frame inventory, authentication and modal guards remain in force.
        Bytes stay in memory until a second guarded read verifies capture safety.
        """
        self._context()
        diagnostic_config = self.config.model_copy(deep=True)
        for rule in diagnostic_config.frames:
            rule.required_text = []
        report = inspect_page(self.page, diagnostic_config)
        _require(report.get("ignored_frame_urls") == [], "diagnostic_unknown_frame")
        if expected is not None:
            _require(screen_identity(report) == screen_identity(expected), "diagnostic_screen_changed")
        text = self.page.locator("body").inner_text(timeout=self._timeout())
        accessibility = self.page.locator("body").aria_snapshot(timeout=self._timeout())
        _require(max(len(text), len(accessibility)) <= self.config.max_text_chars, "diagnostic_budget")
        png = evidence_screenshot(self.page, full_page=False, timeout=self._timeout(5000))
        _require(
            isinstance(png, bytes) and png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) <= 16_000_000,
            "invalid_diagnostic_image",
        )
        self._context()
        after = inspect_page(self.page, diagnostic_config)
        _require(
            after.get("ignored_frame_urls") == [] and screen_identity(report) == screen_identity(after),
            "diagnostic_screen_changed",
        )
        self._context()
        self._save_capture(after, name)
        path = self.output / name
        _bytes(path / "viewport.png", png)
        url = self.page.url
        persist_json(
            path / "visible-outer.json",
            {
                "text": text.replace(url, public_url(url)),
                "accessibility": accessibility.replace(url, public_url(url)),
                "authority": "diagnostic_only_not_submission_acknowledgement",
            },
        )
        persist_json(
            path / "diagnostic.json",
            {
                "viewport_sha256": _sha(self.book.read(path / "viewport.png")),
                "outer_sha256": _sha(self.book.read(path / "visible-outer.json")),
                "submission_verified": False,
                "acknowledgement_interpreted": False,
            },
        )
        self.book.read(path / "diagnostic.json")
        return after

    def _reserve(self):
        self._check()
        self.claim.parent.mkdir(exist_ok=True)
        _sync_directory(self.history)
        proposal = WriteReserved(
            action_id=f"submission-{self.revision}",
            write_kind="submission",
            revision=self.revision,
            project_rows_sha256=self.rows_sha,
            before_sha256=_sha(self.book.read(self.output / "preflight/after/observation.json")),
        )
        claim = {
            "mode": MODE,
            "output": str(self.output.relative_to(self.history)),
            "journal_before_sha256": _sha(self._journal_before),
            "proposal": proposal.model_dump(mode="json"),
            "max_readiness_writes": 1,
            "max_submit_clicks": 1,
            "automatic_retry": False,
        }
        persist_json(self.claim, claim)
        _sync_directory(self.claim.parent)
        self.book.read(self.claim)
        persist_json(self.output / "reserved.json", claim)
        self.book.read(self.output / "reserved.json")
        self._proposal = proposal
        self._check()
        with self.journal_path.open("r+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            _require(stream.read().encode() == self._journal_before, "canonical_journal_changed")
            progress = self.journal._read(stream).append(proposal)
            line = (_json(progress.records[-1].model_dump(mode="json")) + "\n").encode()
            self.book.unchanged()
            if self._cancelled() or self._abort_requested:
                raise _Cancelled
            self._reservation_attempted = True
            stream.seek(0, os.SEEK_END)
            stream.write(line.decode())
            stream.flush()
            os.fsync(stream.fileno())
            self._expected_journal = self._journal_before + line
            stream.seek(0)
            _require(stream.read().encode() == self._expected_journal, "reservation_readback_failed")
            self._reservation_verified = True
        self._check()

    def _feedback_context(self):
        from .browser_probe import _check_auth_and_modals, _url_identity

        self._context()
        visible = [
            frame
            for frame in self.page.frames
            if frame is self.page.main_frame or _visible_frame(frame, self.page.main_frame)
        ]
        _require(visible == self._feedback_visible_frames, "diagnostic_visible_frames_changed")
        expected = {_url_identity(rule.url): rule.count for rule in self.config.frames}
        actual = {}
        for frame in visible:
            if frame is not self.page.main_frame:
                try:
                    identity = _url_identity(frame.url)
                except ValueError:
                    _require(False, "diagnostic_unknown_frame")
                actual[identity] = actual.get(identity, 0) + 1
        _require(actual == expected, "diagnostic_unknown_frame")
        for frame in visible:
            _check_auth_and_modals(frame)
        self._context()
        _require(
            [
                frame
                for frame in self.page.frames
                if frame is self.page.main_frame or _visible_frame(frame, self.page.main_frame)
            ]
            == visible,
            "diagnostic_visible_frames_changed",
        )

    def _capture_outcome(self):
        """Bounded fourth-stage response reads; never another write or receipt.

        A short async opportunity uses the original absolute deadline. The
        final capture and exposure checks remain mandatory even after a shape
        appears; neither a poll nor raw diagnostic text proves a refusal.
        """
        from .browser_submission_feedback import (
            classify_submission_feedback,
            read_submission_feedback_exposure,
            submission_refusal_candidate,
        )

        before = self.book.json(self.output / "before/visible-outer.json")
        until = min(self._started + self.max_seconds, self._clock() + 3.0)
        diagnostic_config = self.config.model_copy(deep=True)
        for rule in diagnostic_config.frames:
            rule.required_text = []
        for index in range(13):
            self._context()
            report = inspect_page(self.page, diagnostic_config)
            _require(report.get("ignored_frame_urls") == [], "diagnostic_unknown_frame")
            body = self.page.locator("body")
            snapshot = {
                "text": body.inner_text(timeout=self._timeout()),
                "accessibility": body.aria_snapshot(timeout=self._timeout()),
            }
            _require(max(map(len, snapshot.values())) <= self.config.max_text_chars, "diagnostic_budget")
            self._context()
            self._feedback_polls += 1
            if submission_refusal_candidate(before, snapshot) or self._clock() >= until or index == 12:
                break
            self.page.wait_for_timeout(max(0, min(250, (until - self._clock()) * 1000)))
            self._check()
        self._diagnostic("post-submit")
        after = self.book.json(self.output / "post-submit/visible-outer.json")
        exposure = None
        if submission_refusal_candidate(before, after):
            exposure = read_submission_feedback_exposure(
                self.page, after, guard=self._feedback_context, timeout=self._timeout
            )
        feedback = classify_submission_feedback(
            before, after, submit_dispatched=self.submit_click_returned, exposure=exposure
        )
        self._check()
        feedback.update(
            source_sha256={
                str(path.relative_to(self.history)): _sha(self.book.read(path))
                for path in (
                    self.output / "before/visible-outer.json",
                    self.output / "post-submit/visible-outer.json",
                    self.output / "post-submit/viewport.png",
                    self.output / "submit-dispatch-reserved.json",
                )
            },
            read_polls=self._feedback_polls,
            wait_max_seconds=3.0,
        )
        path = self.output / "submission-feedback.json"
        persist_json(path, feedback)
        self.book.read(path)
        self.artifacts["submission_feedback"] = str(path.relative_to(self.history))
        self._submission_feedback = feedback
        self._check()
        # Even the known negative response leaves the canonical reservation
        # unmatched. It cannot authorize a retry or manufacture a positive ack.
        self._finish("project_submission_acknowledgement_not_grounded")

    def _prepare(self):
        self._watch()
        self._context()
        returned = preflight_project_submission(
            self.page,
            self.config,
            self.output / "preflight",
            run_history=self.history,
            journal=self.journal,
            scoring_dir=self.scoring_dir,
            timeout_seconds=min(60, self.max_seconds - (self._clock() - self._started)),
            cancelled=lambda: self._abort_requested or self._cancelled(),
            _clock=self._clock,
        )
        self._check()
        saved = self.book.json(self.output / "preflight/confirmed.json")
        _require(
            saved == returned
            and saved["journal_sha256"] == _sha(self._journal_before)
            and saved["project_rows_sha256"] == self.rows_sha
            and saved["eligible_for_readiness_selection"] is True
            and saved["eligible_for_grounded_dispatch"] is True,
            "preflight_not_eligible",
        )
        self._before = self.book.capture(self.output / "preflight/after")
        self.book.capture(self.output / "preflight/before")
        self.artifacts["preflight"] = str((self.output / "preflight").relative_to(self.history))
        self._diagnostic("before", expected=self._before)
        self._emit("state", {**self.state(), "boundary": "before_canonical_reservation"})
        self._check()
        self._reserve()
        self.phase = "selecting_readiness"
        self._emit("state", {**self.state(), "boundary": "canonical_reservation_verified"})

    def _dispatch(self, *, readiness):
        checked, name, index = not readiness, "readiness" if readiness else "submit", 0 if readiness else 1
        _require(
            self._reservation_verified
            and not (self.readiness_may_have_occurred if readiness else self.submit_may_have_occurred),
            "dispatch_already_attempted_or_unreserved",
        )
        _require(readiness or self.readiness_verified, "readiness_not_verified")
        expected = self._before if readiness else self._ready_capture
        before, handles = self._read(checked=checked)
        _require(screen_identity(before) == screen_identity(expected), "predispatch_screen_changed")
        self._save_capture(before, name + "-before")
        self._emit(
            "action_proposed",
            {
                "kind": "CLICK",
                "target": name,
                "value": READY if readiness else SUBMIT,
                "action_source": "explicit_reference_submission_not_learned",
                "canonical_action_id": self._proposal.action_id,
                "submission_verified": False,
            },
        )
        self._check()
        latest, current_handles = self._read(checked=checked)
        _require(
            screen_identity(before) == screen_identity(latest)
            and all(
                old.evaluate("(a,b)=>a.isConnected&&a===b", new) for old, new in zip(handles, current_handles)
            ),
            "native_binding_changed",
        )
        self._check()
        persist_json(
            self.output / (name + "-dispatch-reserved.json"),
            {"action_id": self._proposal.action_id, "max_dispatches": 1, "may_have_occurred": True},
        )
        _sync_directory(self.output)
        self.book.read(self.output / (name + "-dispatch-reserved.json"))
        # Conservative crash markers precede the native call. A thrown click
        # is uncertain and can never be retried through this or another output.
        if readiness:
            self.readiness_may_have_occurred = True
        else:
            self.submit_may_have_occurred = True
        self._context()
        final_handles = self._controls(latest, checked=checked)
        _require(
            all(old.evaluate("(a,b)=>a.isConnected&&a===b", new) for old, new in zip(handles, final_handles)),
            "native_binding_changed",
        )
        timeout = self._timeout()
        handles[index].click(timeout=timeout)
        if readiness:
            self.readiness_click_returned = True
        else:
            self.submit_click_returned = True
        self._check()
        if readiness:
            after, _ = self._read(checked=True)
            expected_after = deepcopy(before)
            for control in expected_after["outer_controls"]:
                if control["role"] == "checkbox" and control["accessibility"] == f'- checkbox "{READY}"':
                    control["accessibility"] += " [checked]"
            _require(comparable_screen(after) == comparable_screen(expected_after), "readiness_side_effect")
            self._save_capture(after, "readiness-after")
            self.readiness_verified, self._ready_capture = True, after
            self.phase = "submitting"
        else:
            self.phase = "capturing_outcome"
        self._emit(
            "action_result",
            {
                "target": name,
                "click_returned": True,
                "readiness_readback_verified": self.readiness_verified,
                "submission_verified": False,
                "task_completed": False,
                "project_completed": False,
            },
        )

    def _emit(self, kind, payload):
        envelope = RuntimeEvent(event=kind, sequence=self._sequence, run_id=self._run_id, payload=payload)
        raw = envelope.model_dump_json()
        self._stream.write(raw + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._sequence += 1
        if not self._forward_failed:
            self._emitting = True
            try:
                self._callback(json.loads(raw))
            except Exception:  # noqa: BLE001 - callbacks may contain credentials
                self._forward_failed = True
                raise BrowserSafetyStop("project_submission_event_forwarding_failed") from None
            finally:
                self._emitting = False

    def state(self):
        return {
            **self.scope,
            "component": "project_submission",
            "phase": self.phase,
            "status": self.status,
            "finished": self.finished,
            "failure_reason": self.failure,
            "advances": self.advances,
            "canonical_reservation_attempted": self._reservation_attempted,
            "canonical_reservation_verified": self._reservation_verified,
            "pending_canonical_action": (
                self._proposal.model_dump(mode="json") if self._reservation_attempted else None
            ),
            "reservation_intent": self._proposal.model_dump(mode="json") if self._proposal else None,
            "readiness_write_may_have_occurred": self.readiness_may_have_occurred,
            "submit_write_may_have_occurred": self.submit_may_have_occurred,
            "readiness_click_returned": self.readiness_click_returned,
            "submit_click_returned": self.submit_click_returned,
            "readiness_readback_verified": self.readiness_verified,
            "submission_outcome": "unknown" if self.submit_may_have_occurred else "not_dispatched",
            **(
                {"submission_feedback": deepcopy(self._submission_feedback)}
                if self._submission_feedback is not None
                else {}
            ),
            "event_forwarding_failed": self._forward_failed,
            "artifact_write_failed": self.artifact_write_failed,
            "listener_cleanup_failed": self.listener_cleanup_failed,
            "input_source_sha256": deepcopy(self.input_source_sha256),
            "source_sha256": deepcopy(self.book.hashes),
            "journal_before_sha256": _sha(self._journal_before),
            "journal_expected_sha256": _sha(self._expected_journal),
            "artifact_paths": deepcopy(self.artifacts),
            **(
                {"failure_outcome_diagnostic": deepcopy(self._outcome_diagnostic)}
                if self._outcome_diagnostic is not None
                else {}
            ),
        }

    def _preserve_failed_outcome(self, reason):
        if not self.submit_may_have_occurred or self._outcome_diagnostic_attempted:
            return
        self._outcome_diagnostic_attempted = True
        directory = self.output / "post-submit-failure"
        try:
            disposition = preserve_submission_outcome(
                self.page,
                type(self.config).model_validate(self._config),
                directory,
                deadline=self._started + self._limits[1],
                original_failure=reason,
                native_dialogs=self._dialogs,
                popup_observed=lambda: bool(self._popups),
                cancelled=lambda: self._abort_requested or self._cancelled(),
                _clock=self._clock,
            )
            self._outcome_diagnostic = {
                "directory": str(directory.relative_to(self.history)),
                **disposition,
            }
            for name in (*disposition["source_sha256"], "disposition.json"):
                self.book.read(directory / name)
        except Exception:  # noqa: BLE001 - original failure and reservation must survive diagnostics
            self._outcome_diagnostic = {
                "disposition": "diagnostic_preservation_failed",
                "consistency_unverified": True,
                "submission_verified": False,
                "automatic_retry": False,
            }

    def _finish(self, reason, *, aborted=False):
        if self.finished or self._finalizing:
            return
        self._finalizing = True
        self.failure, self.status = reason, "aborted" if aborted else "stopped"
        self.phase = "unknown_pending" if self._reservation_attempted else self.status
        diagnostic_interrupt = None
        try:
            if reason != "project_submission_acknowledgement_not_grounded":
                try:
                    self._preserve_failed_outcome(reason)
                except (KeyboardInterrupt, SystemExit) as exc:
                    # Diagnostic collection never outranks the original stop.
                    # Finish durable reporting/cleanup before returning control.
                    diagnostic_interrupt = exc
                    self._outcome_diagnostic = {
                        "disposition": "diagnostic_interrupted",
                        "consistency_unverified": True,
                        "submission_verified": False,
                        "automatic_retry": False,
                    }
            if self._watching:
                for remove, kind, handler in (
                    (self.page.remove_listener, "dialog", self._dialog_handler),
                    (self.page.context.remove_listener, "page", self._popup_handler),
                ):
                    try:
                        remove(kind, handler)
                    except Exception:  # noqa: BLE001 - cleanup cannot reopen a stopped write attempt
                        self.listener_cleanup_failed = True
                self._watching = False
            summary = self.state()
            try:
                self._emit("episode_summary", summary)
            except Exception:  # noqa: BLE001 - forwarding failure cannot authorize another dispatch
                self._forward_failed = True
            self.report = self.state()
            if self.report != summary:
                try:
                    self._emit("state", self.report)
                except Exception:  # noqa: BLE001 - corrective local evidence must not trigger more work
                    self.artifact_write_failed = True
                    self.report = self.state()
            self.report["events_sha256"] = _sha((self.output / "events.jsonl").read_bytes())
            try:
                persist_json(self.output / "report.json", self.report)
            except Exception:  # noqa: BLE001 - keep the terminal state when disk writes fail
                self.artifact_write_failed = True
                self.report = self.state()
        finally:
            self._stream.close()
            self._finalizing = False
        if diagnostic_interrupt is not None:
            raise diagnostic_interrupt

    def advance(self):
        if self.finished:
            return self.state()
        if self._busy or self._emitting:
            self._abort_requested = True
            raise BrowserSafetyStop("project_submission_reentrant_call")
        self._busy = True
        try:
            self._check()
            _require(self.advances < self.max_advances, "advance_limit")
            self.advances += 1
            if self.advances == 1:
                self._emit("hello", {"protocol_version": 1, **self.scope})
                self._check()
            if self.phase == "preflight_pending":
                self._prepare()
            elif self.phase == "selecting_readiness":
                self._dispatch(readiness=True)
            elif self.phase == "submitting":
                self._dispatch(readiness=False)
            elif self.phase == "capturing_outcome":
                self._capture_outcome()
            else:
                raise BrowserSafetyStop("project_submission_unknown_phase")
            if not self.finished:
                self._check()
                self._emit("state", self.state())
                self._check()
        except _Cancelled:
            self._finish("operator_aborted", aborted=True)
        except (KeyboardInterrupt, SystemExit):
            self._abort_requested = True
            self._finish("operator_aborted", aborted=True)
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize driver errors and preserve pending writes
            self._finish(_reason(exc))
        finally:
            self._busy = False
        return self.state()

    def tick(self):
        return self.advance() if self.status == "running" else self.state()

    def pause(self):
        if not self.finished:
            self.status = "paused"
        return self.state()

    def resume(self):
        if not self.finished:
            self.status = "running"
        return self.state()

    def abort(self):
        if not self.finished:
            self._abort_requested = True
            if not self._busy and not self._emitting:
                self._finish("operator_aborted", aborted=True)
        return self.state()

    def close(self):
        return self.abort()


SubmissionSteps = BrowserProjectSubmissionSteps
