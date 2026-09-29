"""Canonical, cooperative assessment and optional score transfer; no submission.

The owner must supply a strict complete inventory and explicitly expose each
assessment panel. Live paginated inventory requires a new validated full sweep
after each scoring boundary. This child never performs pagination/navigation.
Construction is offline. A real canonical reservation is flushed before every
charge/transfer; uncertain writes are never repaired, refunded, or retried.
"""

import fcntl
import hashlib
import json
import math
import os
import re
import time
from copy import deepcopy
from datetime import datetime

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json, rows_hash
from .browser_no_planet_workflow import _Evidence
from .browser_project_assessment import assessment_history, begin_assessment_stage
from .browser_score_transfer import load_assessed_revision, score_projection, transfer_assessed_score
from .browser_stellar import SIMULATION_URL
from .contracts import RuntimeEvent
from .project_assessment import AssessmentLedger, parse_assessment_text
from .project_inventory_source import load_inventory_source
from .project_progress import WriteReceipt, WriteReserved, _json

MODE = "canonical_project_assessment_and_score_transfer"
KINDS = ("data_quality", "scavenger_hunt")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("project_scoring_" + reason)


def _simulation(report):
    frames = [f for f in report["frames"] if f["url"] == SIMULATION_URL]
    _require(not report.get("ignored_frame_urls") and len(frames) == 1, "invalid_capture")
    return frames[0]["text"]


class _Cancelled(Exception):
    pass


class BrowserProjectAssessmentSteps:
    """One two-charge stage, not an unattended navigator or project completion.

    ``provide_panel(kind)`` is an offline scheduling handoff, not evidence that a
    panel is correct. The existing actuator validates that panel on the following
    advance. Each advance performs at most one native click. Pause time counts
    toward the fixed deadline. Browser ownership remains with the caller.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        journal,
        run_history,
        inventory_dir,
        assessment_history_root,
        allow_score_transfer=False,
        max_seconds=600,
        max_advances=6,
        assessment_timeout_seconds=10,
        score_timeout_seconds=15,
        emit=lambda _: None,
        cancelled=lambda: False,
        _clock=time.monotonic,
    ):
        _require(type(allow_score_transfer) is bool, "explicit_transfer_flag_required")
        _require(type(max_advances) is int and 1 <= max_advances <= 6, "invalid_advance_limit")
        for value, low, high in (
            (max_seconds, 1, 3600),
            (assessment_timeout_seconds, 0.1, 30),
            (score_timeout_seconds, 0.1, 30),
        ):
            _require(
                type(value) in {int, float} and math.isfinite(value) and low <= value <= high,
                "invalid_deadline",
            )
        _require(callable(emit) and callable(cancelled) and callable(_clock), "invalid_callback")
        self.book = _Evidence(run_history)
        self.history = self.book.history
        self.journal = journal
        self.journal_path = self.book.path(journal.path)
        _require(self.journal_path.parent == self.history, "journal_outside_attempt")
        self._canonical_identity()
        progress = journal.load()
        state = progress.reduce()
        _require(
            len(state.stars) == 30 and all(s.task_completed for s in state.stars.values()),
            "thirty_unique_verified_tasks_required",
        )
        _require(not state.pending and state.receipt("submission") is None, "pending_or_submitted_write")
        _require(
            not any(
                r.revision == state.revision
                and r.write_kind
                in {"assessment_data_quality", "assessment_scavenger_hunt", "score_transfer", "submission"}
                for r in state.reservations
            ),
            "revision_already_reserved",
        )
        self.revision = state.revision
        self._journal_sha = _sha(self.journal_path.read_bytes())
        self.inventory_dir = self.book.clean(inventory_dir)
        self.inventory_source = load_inventory_source(self.book, self.inventory_dir)
        inventory = self.inventory_source.receipt
        _require(
            {r["name"].casefold() for r in inventory["rows"]}
            == {s.name.casefold() for s in state.stars.values()},
            "inventory_names_mismatch",
        )
        self.rows_sha = rows_hash(_simulation(self.inventory_source.anchor))
        self.paginated = self.inventory_source.kind == "live_paginated"
        self._inventory_refreshes = {}
        self._accepted_inventory_refreshes = ()
        self._inventory_fingerprint = self._inventory_identity()
        self.assessment_root = self.book.path(assessment_history_root)
        _require(
            self.assessment_root.is_dir() and self.assessment_root != self.history,
            "owned_assessment_history_required",
        )
        self.prior_history = assessment_history(self.assessment_root)
        _require(
            not any(h["revision"] >= self.revision for h in self.prior_history),
            "assessment_revision_already_used",
        )
        for item in self.prior_history:
            for path, expected in item["source_sha256"].items():
                _require(_sha(self.book.read(path)) == expected, "prior_history_changed")
        self.output = self.book.path(output)
        _require(
            self.output != self.history
            and not self.output.is_relative_to(self.inventory_dir)
            and not self.output.is_relative_to(self.assessment_root),
            "invalid_output",
        )
        self.stage_dir = self.assessment_root / f"canonical-revision-{self.revision}"
        _require(not self.stage_dir.exists(), "assessment_stage_exists")
        claims = self.history / "project-scoring-claims"
        claim = self.book.path(claims / f"revision-{self.revision}.json")
        _require(not claim.exists(), "owner_already_reserved")
        self.page, self.config = page, config.model_copy(deep=True)
        self._config = self.config.model_dump(mode="json")
        self._callback, self._cancelled, self._clock = emit, cancelled, _clock
        self._started = _clock()
        self.max_seconds, self.max_advances = max_seconds, max_advances
        self.assessment_timeout_seconds, self.score_timeout_seconds = (
            assessment_timeout_seconds,
            score_timeout_seconds,
        )
        self.allow_score_transfer = allow_score_transfer
        self.phase, self.status, self.failure = "awaiting_data_quality_panel", "paused", None
        self.actuator, self.report = None, None
        self._assessment_read_deadline = None
        self.advances = self._sequence = 0
        self._busy = self._emitting = self._finalizing = self._abort_requested = self._forward_failed = False
        self._assessment_receipts, self._ack_receipts, self._score_receipt = [], [], None
        self._proposal = None
        self._ack_proposal = None
        self.scope = {
            "mode": MODE,
            "data_revision": self.revision,
            "project_rows_sha256": self.rows_sha,
            "allow_score_transfer": allow_score_transfer,
            "max_seconds": max_seconds,
            "max_advances": max_advances,
            "max_assessment_clicks": 2,
            "budget_simulation_dollars": 200,
            "max_score_transfer_clicks": int(allow_score_transfer),
            "assessment_timeout_seconds": assessment_timeout_seconds,
            "score_timeout_seconds": score_timeout_seconds,
            "real_money": False,
            "automatic_retry": False,
            "pagination_supported": self.paginated,
            "complete_visible_inventory_required": not self.paginated,
            "assessment_navigation_supported": False,
            "pause_counts_toward_deadline": True,
            "cancellation": "between_calls_and_before_native_dispatch",
            "submission_enabled": False,
            "submitted": False,
            "task_completed": False,
            "project_completed": False,
            "browser_acceptance_passed": False,
            "scientific_verified": False,
            "training_label": False,
            **self.inventory_source.binding_fields(),
        }
        self._fixed_options = (
            allow_score_transfer,
            max_seconds,
            max_advances,
            assessment_timeout_seconds,
            score_timeout_seconds,
        )
        self.book.unchanged()
        self.output.mkdir(parents=True, exist_ok=False)
        claims.mkdir(exist_ok=True)
        persist_json(
            claim,
            {
                **self.scope,
                "output": str(self.output.relative_to(self.history)),
                "journal_before_sha256": self._journal_sha,
            },
        )
        self.book.read(claim)
        persist_json(self.output / "scope.json", self.scope)
        self.book.read(self.output / "scope.json")
        self._stream = (self.output / "events.jsonl").open("x", encoding="utf-8")
        self._run_id = "project-scoring-" + _sha(str(self.output.relative_to(self.history)).encode())[:16]

    @property
    def finished(self):
        return self.status in {"completed", "stopped", "aborted"}

    def _canonical_identity(self):
        _require(
            {self.book.path(p) for p in self.history.glob("project-progress-*.jsonl")} == {self.journal_path},
            "ambiguous_canonical_journal",
        )

    def _inventory_identity(self):
        source = self.inventory_source
        return _sha(
            _json(
                {
                    "kind": source.kind,
                    "receipt": source.receipt,
                    "rows": source.rows,
                    "anchor": source.anchor,
                    "anchor_dir": str(source.anchor_dir),
                    "anchor_evidence": source.anchor_evidence,
                    "whole_collection_sha256": source.whole_collection_sha256,
                    "rows_sha256": self.rows_sha,
                    "paginated": self.paginated,
                }
            ).encode()
        )

    def _inventory_invariants(self):
        """Cached handoffs are not authority; immutable accepted records are."""
        _require(self._inventory_identity() == self._inventory_fingerprint, "inventory_cache_changed")
        accepted = dict(self._accepted_inventory_refreshes)
        _require(
            isinstance(self._inventory_refreshes, dict)
            and set(self._inventory_refreshes) == set(accepted)
            and len(accepted) == len(self._accepted_inventory_refreshes),
            "inventory_refresh_cache_changed",
        )
        for kind, expected in self._accepted_inventory_refreshes:
            raw = self.book.read(self.output / f"inventory-refresh-{kind}.json")
            _require(
                _sha(raw) == expected and json.loads(raw) == self._inventory_refreshes[kind],
                "inventory_refresh_cache_changed",
            )

    def _require_inventory_refresh(self, kind):
        self._inventory_invariants()
        _require(not self.paginated or kind in self._inventory_refreshes, "fresh_collection_sweep_required")

    def _check(self):
        if self._abort_requested or self.finished or self._cancelled():
            raise _Cancelled
        _require(self._clock() - self._started < self.max_seconds, "time_limit")
        _require(
            (
                self.allow_score_transfer,
                self.max_seconds,
                self.max_advances,
                self.assessment_timeout_seconds,
                self.score_timeout_seconds,
            )
            == self._fixed_options,
            "fixed_options_changed",
        )
        _require(self.config.model_dump(mode="json") == self._config, "config_changed")
        self._inventory_invariants()
        if self.actuator is not None:
            _require(
                self.actuator.timeout_seconds == self.assessment_timeout_seconds
                and self.actuator.settle_inventory is self.paginated
                and self.actuator.read_deadline == self._assessment_read_deadline
                and self.actuator.ledger.budget == 200
                and self.actuator.ledger.max_attempts == 2,
                "actuator_limits_changed",
            )
        self._canonical_identity()
        _require(_sha(self.journal_path.read_bytes()) == self._journal_sha, "canonical_journal_changed")
        state = self.journal.load().reduce()
        _require(
            state.revision == self.revision
            and len(state.stars) == 30
            and all(s.task_completed for s in state.stars.values()),
            "canonical_tasks_changed",
        )
        self.book.unchanged()

    def _append(self, item):
        self._check()
        # Compare-and-append under the same canonical lock: a concurrent writer
        # must not slip an Active or data event between source check and append.
        with self.journal_path.open("r+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            _require(_sha(stream.read().encode()) == self._journal_sha, "canonical_journal_changed")
            progress = self.journal._read(stream).append(item)
            self.book.unchanged()
            stream.seek(0, os.SEEK_END)
            stream.write(_json(progress.records[-1].model_dump(mode="json")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            self._journal_sha = _sha(stream.read().encode())

    def _emit(self, event, payload):
        item = RuntimeEvent(event=event, sequence=self._sequence, run_id=self._run_id, payload=payload)
        raw = item.model_dump_json()
        self._stream.write(raw + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._sequence += 1
        if not self._forward_failed:
            self._emitting = True
            try:
                self._callback(json.loads(raw))
            except Exception:  # noqa: BLE001 - callbacks may contain private session data
                self._forward_failed = True
                raise BrowserSafetyStop("project_scoring_event_forwarding_failed") from None
            finally:
                self._emitting = False

    def provide_panel(self, kind):
        """Schedule only; never click or trust the caller's panel claim."""
        _require(not self._busy and not self._emitting and not self.finished, "invalid_panel_handoff")
        self._check()
        _require(kind in KINDS and self.phase == f"awaiting_{kind}_panel", "unexpected_panel_handoff")
        self._require_inventory_refresh(kind)
        self.phase = "initializing_assessments" if kind == "data_quality" else "assessing_scavenger_hunt"
        self.status = "paused"
        persist_json(
            self.output / f"panel-{kind}.json",
            {
                "kind": kind,
                "verified": False,
                "meaning": "explicit_visible_panel_handoff_pending_adapter_validation",
            },
        )
        return self.state()

    def provide_inventory_refresh(self, kind, *, inventory_dir, inventory_sha256):
        """Accept a new full visible sweep, never fabricate unseen rows or navigate.

        Each assessment and score transfer has its own post-boundary sweep.
        This is observed consistency, not an atomic server-side snapshot.
        """
        _require(not self._busy and not self._emitting and not self.finished, "invalid_inventory_handoff")
        self._check()
        _require(
            isinstance(inventory_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", inventory_sha256),
            "invalid_inventory_hash",
        )
        phases = {**{k: f"awaiting_{k}_panel" for k in KINDS}, "score_transfer": "score_transfer_ready"}
        _require(
            self.paginated
            and kind in phases
            and self.phase == phases[kind]
            and kind not in self._inventory_refreshes,
            "unexpected_inventory_refresh",
        )
        directory = self.book.clean(inventory_dir)
        _require(
            directory != self.inventory_dir
            and all(
                str(directory.relative_to(self.history)) != old["inventory_path"]
                and inventory_sha256 != old["inventory_sha256"]
                for old in self._inventory_refreshes.values()
            ),
            "collection_sweep_reused",
        )
        refreshed = load_inventory_source(self.book, directory, expected_sha256=inventory_sha256)
        _require(
            refreshed.kind == "live_paginated"
            and refreshed.rows == self.inventory_source.rows
            and refreshed.whole_collection_sha256 == self.inventory_source.whole_collection_sha256
            and rows_hash(_simulation(refreshed.anchor)) == self.rows_sha,
            "refreshed_collection_changed",
        )
        first = self.book.path(self.history / refreshed.receipt["native_pages"][0]["path"]).parent
        captured = self.book.capture(first / "before")
        boundary_dir = (
            self.inventory_source.anchor_dir
            if kind == "data_quality"
            else self.stage_dir / ("ack-000-after" if kind == "scavenger_hunt" else "ack-001-after")
        )
        boundary = self.book.capture(boundary_dir)
        stamps = []
        for report in (boundary, captured):
            value = report.get("captured_at")
            _require(isinstance(value, str), "missing_refresh_timestamp")
            try:
                stamp = datetime.fromisoformat(value)
            except ValueError:
                raise BrowserSafetyStop("project_scoring_invalid_refresh_timestamp") from None
            _require(stamp.tzinfo is not None, "invalid_refresh_timestamp")
            stamps.append(stamp)
        _require(stamps[1] > stamps[0], "stale_collection_sweep")
        record = {
            "kind": kind,
            "inventory_path": str(directory.relative_to(self.history)),
            "inventory_sha256": inventory_sha256,
            **refreshed.binding_fields(),
            "after_capture_sha256": _sha(self.book.read(refreshed.anchor_dir / "observation.json")),
            "revision": self.revision,
            "browser_actions": 0,
            "freshness": {
                "boundary_capture": str(boundary_dir.relative_to(self.history)),
                "boundary_capture_sha256": _sha(self.book.read(boundary_dir / "observation.json")),
                "boundary_captured_at": boundary["captured_at"],
                "sweep_first_capture": str((first / "before").relative_to(self.history)),
                "sweep_first_capture_sha256": _sha(self.book.read(first / "before/observation.json")),
                "sweep_started_at": captured["captured_at"],
                "started_after_boundary_verified": True,
                "atomic_server_snapshot_verified": False,
            },
        }
        self._check()
        path = self.output / f"inventory-refresh-{kind}.json"
        persist_json(path, record)
        digest = _sha(self.book.read(path))
        self._inventory_refreshes[kind] = record
        self._accepted_inventory_refreshes += ((kind, digest),)
        self._inventory_invariants()
        return self.state()

    def pause(self):
        if not self.finished:
            self.status = "paused"
        return self.state()

    def resume(self):
        if not self.finished and not self.phase.startswith("awaiting_"):
            self.status = "running"
        return self.state()

    def _capture(self, folder, snapshot=None):
        report = self.book.capture(folder)
        text = _simulation(report)
        _require(rows_hash(text) == self.rows_sha, "visible_rows_changed")
        current = parse_assessment_text(text)
        _require(
            current.collected == 30 and (snapshot is None or current == snapshot), "capture_snapshot_mismatch"
        )
        return report, current

    def _reserve_assessment(self, kind, sequence):
        self._check()
        self._require_inventory_refresh(kind)
        prefix = f"attempt-{sequence:03d}"
        payload = self.book.json(self.stage_dir / f"{prefix}-reserved.json")
        ledger = AssessmentLedger.model_validate(payload["ledger"])
        attempt = ledger.pending
        _require(
            ledger.budget == 200
            and ledger.max_attempts == 2
            and attempt is not None
            and attempt.sequence == sequence
            and attempt.kind == kind
            and attempt.data_revision == self.revision
            and len(ledger.attempts) == sequence + 1
            and ledger == self.actuator.ledger
            and payload["data_revision"] == self.revision
            and payload["project_rows_sha256"] == self.rows_sha
            and payload["simulation_dollars"] == 100
            and payload["real_money"] is False
            and payload["action"] == {"kind": "CLICK", "visible_label": "ASSESS"},
            "invalid_local_reservation",
        )
        self._capture(self.stage_dir / f"{prefix}-before", attempt.before)
        if sequence:
            previous = AssessmentLedger.model_validate(self._assessment_receipts[-1]["ledger"])
            _require(ledger.attempts[:-1] == previous.attempts, "assessment_ledger_replaced")
        proposal = WriteReserved(
            action_id=f"assessment-{kind}-{self.revision}",
            write_kind="assessment_" + kind,
            revision=self.revision,
            project_rows_sha256=self.rows_sha,
            before_sha256=_sha(self.book.read(self.stage_dir / f"{prefix}-before/observation.json")),
        )
        self._append(proposal)
        self._proposal = proposal
        self._emit(
            "action_proposed",
            {
                "kind": "CLICK",
                "target": "assessment_" + kind,
                "value": "ASSESS",
                "simulation_dollars": 100,
                "action_source": "guarded_reference_workflow",
                "canonical_action_id": self._proposal.action_id,
            },
        )
        self._check()

    def _assess(self, kind):
        self._require_inventory_refresh(kind)
        sequence = KINDS.index(kind)
        self.actuator.assess(
            kind,
            data_revision=self.revision,
            before_dispatch=lambda: self._reserve_assessment(kind, sequence),
            check_cancelled=self._check,
        )
        self._check()
        path = self.stage_dir / f"attempt-{sequence:03d}-confirmed.json"
        payload = self.book.json(path)
        ledger = AssessmentLedger.model_validate(payload["ledger"])
        reserved = AssessmentLedger.model_validate(
            self.book.json(self.stage_dir / f"attempt-{sequence:03d}-reserved.json")["ledger"]
        )
        _require(
            not ledger.pending
            and ledger == self.actuator.ledger
            and ledger.attempts[-1].model_copy(update={"after": None}) == reserved.pending
            and ledger.attempts[:-1] == reserved.attempts[:-1]
            and ledger.budget == 200
            and ledger.max_attempts == 2
            and payload["receipt"] == ledger.attempts[-1].model_dump(mode="json")
            and payload["report"] == ledger.report(data_revision=self.revision)
            and payload["project_rows_sha256"] == self.rows_sha
            and self._proposal is not None,
            "invalid_assessment_confirmation",
        )
        self._capture(self.stage_dir / f"attempt-{sequence:03d}-after", ledger.attempts[-1].after)
        self._append(
            WriteReceipt(
                action_id=self._proposal.action_id,
                write_kind="assessment_" + kind,
                revision=self.revision,
                project_rows_sha256=self.rows_sha,
                source_sha256=_sha(self.book.read(path)),
                authority="course_feedback",
                acknowledgement=kind + "_updated",
                confirmed=True,
                assessment=ledger,
            )
        )
        self._assessment_receipts.append(payload)
        self._proposal = None
        self.phase = "acknowledging_" + kind
        self._emit(
            "action_result",
            {
                "target": "assessment_" + kind,
                "verified": True,
                "task_completed": False,
                "project_completed": False,
            },
        )

    def _acknowledge(self, kind):
        sequence = KINDS.index(kind)

        def proposed():
            self._check()
            reserved = self.book.json(self.stage_dir / f"ack-{sequence:03d}-reserved.json")
            _require(
                reserved
                == {
                    "kind": kind,
                    "data_revision": self.revision,
                    "max_acknowledgement_clicks": 1,
                    "assessment_clicks": 0,
                    "automatic_retry": False,
                },
                "invalid_acknowledgement_reservation",
            )
            self._ack_proposal = kind
            self._emit(
                "action_proposed",
                {
                    "kind": "CLICK",
                    "target": "acknowledgement_" + kind,
                    "value": "OK",
                    "action_source": "guarded_reference_workflow",
                },
            )
            self._check()

        self.actuator.dismiss_receipt(before_dispatch=proposed, check_cancelled=self._check)
        self._check()
        path = self.stage_dir / f"ack-{sequence:03d}-confirmed.json"
        payload = self.book.json(path)
        _require(
            payload
            == {
                "kind": kind,
                "data_revision": self.revision,
                "acknowledgement_dismissed": True,
                "acknowledgement_clicks": 1,
                "assessment_clicks": 0,
                "project_rows_sha256": self.rows_sha,
                "automatic_retry": False,
                "task_completed": False,
            },
            "invalid_acknowledgement_confirmation",
        )
        _, after = self._capture(self.stage_dir / f"ack-{sequence:03d}-after")
        before = self.actuator.ledger.attempts[-1].after
        excluded = {"acknowledgement", "visible_text_sha256"}
        _require(
            after.acknowledgement is None
            and after.model_dump(exclude=excluded) == before.model_dump(exclude=excluded),
            "acknowledgement_changed_data",
        )
        self._ack_receipts.append(payload)
        self._ack_proposal = None
        if kind == "data_quality":
            self.phase, self.status = "awaiting_scavenger_hunt_panel", "paused"
        else:
            self.phase = (
                "score_transfer_ready" if self.allow_score_transfer else "assessed_score_transfer_disabled"
            )
        self._emit(
            "action_result",
            {
                "target": "acknowledgement_" + kind,
                "verified": True,
                "task_completed": False,
                "project_completed": False,
            },
        )

    def _transfer(self):
        self._require_inventory_refresh("score_transfer")
        source = self.stage_dir / "attempt-001-confirmed.json"
        assessed, expected_rows, provenance = load_assessed_revision(source)
        _require(
            expected_rows == self.rows_sha
            and provenance["data_revision"] == self.revision
            and assessed.collected == 30
            and len(self._ack_receipts) == 2,
            "transfer_source_mismatch",
        )
        directory = self.output / "score-transfer"

        def proposed():
            self._check()
            self._require_inventory_refresh("score_transfer")
            reserved = self.book.json(directory / "reserved.json")
            _require(
                all(reserved.get(k) == v for k, v in provenance.items())
                and reserved.get("action") == "Update Score"
                and reserved.get("max_clicks") == 1
                and reserved.get("submission_enabled") is False
                and reserved.get("task_completed") is False,
                "invalid_score_reservation",
            )
            self._capture(directory / "before")
            claim = self.book.json(self.stage_dir / f"score-transfer-revision-{self.revision}.json")
            _require(
                claim
                == {
                    "source_sha256": provenance["source_sha256"],
                    "data_revision": self.revision,
                    "output": str(directory),
                    "max_clicks": 1,
                    "automatic_retry": False,
                },
                "invalid_score_claim",
            )
            proposal = WriteReserved(
                action_id=f"score-transfer-{self.revision}",
                write_kind="score_transfer",
                revision=self.revision,
                project_rows_sha256=self.rows_sha,
                before_sha256=_sha(self.book.read(directory / "before/observation.json")),
            )
            self._append(proposal)
            self._proposal = proposal
            self._emit(
                "action_proposed",
                {
                    "kind": "CLICK",
                    "target": "score_transfer",
                    "value": "Update Score",
                    "action_source": "guarded_reference_workflow",
                    "canonical_action_id": self._proposal.action_id,
                },
            )
            self._check()

        returned = transfer_assessed_score(
            self.page,
            self.config,
            directory,
            assessment_receipt=source,
            allow_score_transfer=self.allow_score_transfer,
            timeout_seconds=self.score_timeout_seconds,
            before_dispatch=proposed,
            check_cancelled=self._check,
            **(
                {
                    "settle_inventory": True,
                    "deadline": time.monotonic() + self.max_seconds - (self._clock() - self._started),
                }
                if self.paginated
                else {}
            ),
        )
        self._check()
        payload = self.book.json(directory / "confirmed.json")
        _require(
            payload == returned
            and all(payload.get(k) == v for k, v in provenance.items())
            and payload.get("score_transfer_verified") is True
            and all(
                payload.get(k) is False
                for k in (
                    "score_computed_locally",
                    "save_verified",
                    "submitted",
                    "task_completed",
                    "browser_acceptance_passed",
                    "automatic_retry",
                )
            )
            and self._proposal is not None,
            "invalid_score_confirmation",
        )
        _require(
            self.book.json(directory / "acknowledgement.json") == {"visible_text": "Score updated."}
            and self.book.json(directory / "score-readback.json")
            == {"score_before": payload["score_before"], "score_after": payload["score_after"]},
            "score_feedback_missing",
        )
        before, _ = self._capture(directory / "before")
        after, _ = self._capture(directory / "after")
        _require(score_projection(before) == score_projection(after, feedback=True), "score_side_effect")
        self._append(
            WriteReceipt(
                action_id=self._proposal.action_id,
                write_kind="score_transfer",
                revision=self.revision,
                project_rows_sha256=self.rows_sha,
                source_sha256=_sha(self.book.read(directory / "confirmed.json")),
                authority="course_feedback",
                acknowledgement="score_updated",
                confirmed=True,
                score=float(payload["score_after"]),
            )
        )
        self._score_receipt, self._proposal = payload, None
        self.phase = "score_transferred_not_submitted"
        self._emit(
            "action_result",
            {
                "target": "score_transfer",
                "verified": True,
                "score": float(payload["score_after"]),
                "task_completed": False,
                "project_completed": False,
            },
        )

    def state(self):
        return {
            **self.scope,
            "component": "project_scoring",
            "phase": self.phase,
            "status": self.status,
            "finished": self.finished,
            "failure_reason": self.failure,
            "advances": self.advances,
            "assessment_verified": {k: len(self._assessment_receipts) > i for i, k in enumerate(KINDS)},
            "acknowledgements_verified": len(self._ack_receipts),
            "score_transfer_verified": self._score_receipt is not None,
            "pending_canonical_action": self._proposal.model_dump(mode="json") if self._proposal else None,
            "pending_acknowledgement": self._ack_proposal,
            "assessment_outcome_uncertain": bool(self.actuator and self.actuator.ledger.pending),
            "event_forwarding_failed": self._forward_failed,
            "source_sha256": deepcopy(self.book.hashes),
            **({"inventory_refreshes": deepcopy(self._inventory_refreshes)} if self.paginated else {}),
            "journal_sha256": self._journal_sha,
            "artifact_paths": {
                "assessments": str(self.stage_dir.relative_to(self.history)),
                "score_transfer": str((self.output / "score-transfer").relative_to(self.history)),
            },
        }

    def _finish(self, reason=None, *, aborted=False):
        if self.finished or self._finalizing:
            return
        self._finalizing = True
        self.failure = reason
        self.status = "aborted" if aborted else "stopped" if reason else "completed"
        if reason:
            self.phase = self.status
        if self.actuator:
            self.actuator.stopped = True
        try:
            summary = self.state()
            try:
                self._emit("episode_summary", summary)
                if self._abort_requested:
                    self.status, self.phase, self.failure = "aborted", "aborted", "operator_aborted"
                self._inventory_invariants()
                self.book.unchanged()
                _require(
                    _sha(self.journal_path.read_bytes()) == self._journal_sha, "canonical_journal_changed"
                )
            except Exception:  # noqa: BLE001 - final failure cannot authorize another write
                self.status, self.phase, self.failure = (
                    "stopped",
                    "stopped",
                    self.failure or "project_scoring_terminal_validation_failed",
                )
            self.report = self.state()
            if self.report != summary:
                try:
                    self._emit("state", self.report)
                except Exception:  # noqa: BLE001 - correction stays in exact journal even if forwarding failed
                    self._forward_failed = True
                    self.report = self.state()
            self.report["events_sha256"] = _sha((self.output / "events.jsonl").read_bytes())
            persist_json(self.output / "report.json", self.report)
        finally:
            self._stream.close()
            self._finalizing = False

    def advance(self):
        if self.finished:
            return self.state()
        if self._busy or self._emitting:
            self._abort_requested = True
            raise BrowserSafetyStop("project_scoring_reentrant_call")
        self._busy = True
        try:
            self._check()
            if self.phase.startswith("awaiting_"):
                return self.state()
            _require(self.advances < self.max_advances, "advance_limit")
            self.advances += 1
            if self.phase == "initializing_assessments":
                if self.paginated:
                    self._assessment_read_deadline = (
                        time.monotonic() + self.max_seconds - (self._clock() - self._started)
                    )
                self.actuator = begin_assessment_stage(
                    self.page,
                    self.config,
                    self.stage_dir,
                    history_root=self.assessment_root,
                    data_revision=self.revision,
                    collected=30,
                    reason="Canonical thirty verified tasks with pinned complete visible inventory",
                    max_attempts=2,
                    **(
                        {
                            "settle_inventory": True,
                            "deadline": self._assessment_read_deadline,
                            "timeout_seconds": self.assessment_timeout_seconds,
                            "check_cancelled": self._check,
                        }
                        if self.paginated
                        else {}
                    ),
                )
                self.actuator.timeout_seconds = self.assessment_timeout_seconds
                _require(
                    self.book.json(self.stage_dir / "ledger-initial.json")
                    == AssessmentLedger(budget=200, max_attempts=2).model_dump(mode="json"),
                    "invalid_initial_assessment_ledger",
                )
                stage = self.book.json(self.stage_dir / "stage.json")
                _require(
                    stage["new_project_rows_sha256"] == self.rows_sha
                    and stage["history"] == self.prior_history,
                    "stage_source_mismatch",
                )
                self.phase = "assessing_data_quality"
            elif self.phase.startswith("assessing_"):
                self._assess(self.phase.removeprefix("assessing_"))
            elif self.phase.startswith("acknowledging_"):
                self._acknowledge(self.phase.removeprefix("acknowledging_"))
            elif self.phase == "score_transfer_ready":
                self._transfer()
            else:
                raise BrowserSafetyStop("project_scoring_unsupported_phase")
            self._check()
            if self.phase in {"assessed_score_transfer_disabled", "score_transferred_not_submitted"}:
                self._finish()
            else:
                self._emit("state", self.state())
                self._check()
        except _Cancelled:
            self._finish("operator_aborted", aborted=True)
        except (KeyboardInterrupt, SystemExit):
            self._abort_requested = True
            self._finish("operator_aborted", aborted=True)
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize driver failures and retain uncertainty
            reason = (
                str(exc)
                if isinstance(exc, BrowserSafetyStop) and str(exc).startswith("project_scoring_")
                else "project_scoring_operation_failed"
            )
            self._finish(reason)
        finally:
            self._busy = False
        return self.state()

    def abort(self):
        if not self.finished:
            self._abort_requested = True
            if not self._busy and not self._emitting:
                self._finish("operator_aborted", aborted=True)
        return self.state()

    def close(self):
        return self.abort()
