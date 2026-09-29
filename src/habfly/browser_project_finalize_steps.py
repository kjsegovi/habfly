"""Separate, opt-in finalization of an immutable thirty-star campaign handoff.

The caller retains Chromium. This owner never reopens a finished campaign or
reinterprets its historical journal hash as a current hash. Existing scoring
and submission components alone may append their canonical write records.
Submission still ends unknown/pending: there is no invented course receipt.
"""

import hashlib
import os
import re
import stat
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_no_planet_workflow import _Evidence
from .browser_project_campaign_steps import MODE as CAMPAIGN_MODE
from .browser_project_campaign_steps import _verified_owner
from .browser_project_scoring_coordinator import BrowserProjectScoringCoordinator
from .browser_project_submission_preflight import _sources as submission_sources
from .browser_project_submission_steps import BrowserProjectSubmissionSteps
from .contracts import RuntimeEvent
from .project_events import ProjectEventRelay
from .project_inventory_source import load_inventory_source
from .project_progress import ProjectJournal, ProjectProgress, TaskReceipt, WriteReceipt, WriteReserved, _json

MODE = "bounded_post_campaign_finalization"
SCORING_SECONDS, SUBMISSION_SECONDS, MAX_ADVANCES = 600, 180, 54


class _PinnedEvidence(_Evidence):
    """Cache only metadata of leaves already checked against their SHA-256.

    A new inode, size, nanosecond timestamp or ctime forces another exact hash
    check. Parent-directory inode/type checks prevent a swapped/symlinked path
    from bypassing that check. The mutable canonical journal is never cached.
    This avoids re-reading every historical star on every nested UI callback.
    """

    def __init__(self, history):
        super().__init__(history)
        self._files, self._parents = {}, {}

    @staticmethod
    def _stamp(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def read(self, path):
        before = self._stamp(self.path(path).stat(follow_symlinks=False))
        raw = super().read(path)
        path = self.path(path)
        after = self._stamp(path.stat(follow_symlinks=False))
        _require(before == after, "source_changed_during_read")
        self._files[str(path.relative_to(self.history))] = after
        for parent in path.parents:
            info = parent.stat(follow_symlinks=False)
            identity = (info.st_dev, info.st_ino, info.st_mode)
            _require(stat.S_ISDIR(info.st_mode), "source_parent_not_directory")
            _require(
                parent not in self._parents or self._parents[parent] == identity, "source_parent_changed"
            )
            self._parents[parent] = identity
            if parent == self.history:
                break
        return raw

    def unchanged(self):
        for parent, identity in self._parents.items():
            info = parent.stat(follow_symlinks=False)
            _require((info.st_dev, info.st_ino, info.st_mode) == identity, "source_parent_changed")
        for directory in self.clean_directories:
            _require(
                not any((directory / name).exists() for name in ("stopped.json", "invalidated.json")),
                "failed_source_directory",
            )
        for directory, tree in self.closed_trees.items():
            _require(self.tree(directory) == tree, "evidence_tree_changed")
        for name in tuple(self.hashes):
            path = self.history / name
            if self._stamp(path.stat(follow_symlinks=False)) != self._files.get(name):
                self.read(path)


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("project_finalize_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _reason(exc):
    value = str(exc)
    return (
        value
        if isinstance(exc, BrowserSafetyStop) and re.fullmatch(r"[a-z][a-z0-9_]{0,150}", value)
        else "project_finalize_operation_failed"
    )


def _relative(book, value):
    _require(
        isinstance(value, str) and value and not Path(value).is_absolute() and ".." not in Path(value).parts,
        "invalid_source_path",
    )
    return book.path(book.history / value)


def _adopt_sources(book, sources):
    _require(isinstance(sources, dict) and sources, "missing_source_hashes")
    for name, checksum in sources.items():
        _require(isinstance(checksum, str) and re.fullmatch(r"[a-f0-9]{64}", checksum), "invalid_source_hash")
        _require(_sha(book.read(_relative(book, name))) == checksum, "source_hash_changed")


def _latch_campaign(book, journal, runtime_dir):
    """Replay existing strict owner/import validators at their actual journal prefixes.

    Only the canonical, already-recorded prefix is supplied to each validator;
    no historical journal file or browser action is created. Large diagnostic
    outer/campaign event streams are not receipt authority and are not loaded.
    The final owner event stream and all declared campaign source leaves are
    still checked by the existing validators.
    """
    _require(isinstance(journal, ProjectJournal), "canonical_journal_required")
    directory = book.path(runtime_dir)
    path = book.path(journal.path)
    _require(directory == book.history and path.parent == directory, "runtime_attempt_mismatch")
    _require(set(book.history.glob("project-progress-*.jsonl")) == {path}, "ambiguous_canonical_journal")
    raw, progress = book.read(path), journal.load()
    _require(path.read_bytes() == raw, "concurrent_journal_change")
    state = progress.reduce()
    _require(
        len(state.stars) == 30
        and all(star.task_completed for star in state.stars.values())
        and not state.pending
        and not state.reservations
        and not state.receipts,
        "thirty_unscored_verified_tasks_required",
    )
    outer = book.json(directory / "report.json")
    campaign_dir = book.clean(directory / "campaign")
    campaign = book.json(campaign_dir / "report.json")
    scope = book.json(campaign_dir / "scope.json")
    for report in (outer, campaign):
        _require(
            report.get("status") == "handoff"
            and report.get("phase") == "awaiting_assessment"
            and report.get("finished") is True
            and report.get("target_workflows_verified") is True
            and report.get("target_stars") == 30
            and type(report.get("target_stars")) is int
            and report.get("failure_reason") is None
            and all(
                report.get(key) is False
                for key in (
                    "task_completed",
                    "project_completed",
                    "event_forwarding_failed",
                    "automatic_retry",
                )
            )
            and report.get("project_progress") == state.report(),
            "unverified_campaign_handoff",
        )
    _require(
        outer.get("mode") == "fresh_browser_campaign_project_runtime"
        and outer.get("project_campaign") is True
        and outer.get("browser_status") == "visible"
        and outer.get("campaign") == campaign
        and campaign.get("mode") == CAMPAIGN_MODE
        and campaign.get("verified_stars") == 30
        and campaign.get("owner_limits") == {"max_seconds": 1800, "max_advances": 512}
        and campaign.get("journal_sha256") == _sha(raw)
        and all(campaign.get(key) == value for key, value in scope.items()),
        "campaign_source_mismatch",
    )
    declared = campaign.get("source_sha256")
    _adopt_sources(book, declared)
    task_ends = [
        i + 1 for i, record in enumerate(progress.records) if isinstance(record.payload, TaskReceipt)
    ]
    _require(len(task_ends) == 30 and task_ends[-1] == len(progress.records), "unexpected_task_history")
    lines, names, final = raw.splitlines(keepends=True), [], None
    for ordinal, end in enumerate(task_ends, 1):
        prefix = ProjectProgress(**progress.header(), records=progress.records[:end])
        verified_path = campaign_dir / f"verified-{ordinal:03d}.json"
        _require(str(verified_path.relative_to(book.history)) in declared, "unbound_verified_star")
        verified = book.json(verified_path)
        context = book.json(campaign_dir / f"star-{ordinal:03d}.json")
        _require(
            type(context.get("ordinal")) is int
            and context["ordinal"] == ordinal
            and isinstance(context.get("star"), str)
            and context["star"].casefold() not in {name.casefold() for name in names},
            "invalid_star_sequence",
        )
        names.append(context["star"])
        # Existing _verified_owner consumes only .path and .load(). The supplied
        # historical state is an exact, hash-validated canonical prefix.
        historical = SimpleNamespace(path=path, load=lambda prefix=prefix: prefix)
        _state, _capture, sources = _verified_owner(
            book, historical, _relative(book, context["owner_output"]), names, context["star"]
        )
        _require(
            verified == {**context, **sources, "journal_sha256": _sha(b"".join(lines[: end + 1]))},
            "verified_star_source_mismatch",
        )
        final = verified
    _require(
        campaign.get("current_star") == context
        and outer.get("current_star") == context
        and {name.casefold() for name in names} == {s.name.casefold() for s in state.stars.values()},
        "final_star_context_changed",
    )
    inventory_dir = _relative(book, final["inventory_dir"])
    source = load_inventory_source(book, inventory_dir, expected_sha256=final["inventory_sha256"])
    _require(
        source.kind == "live_paginated"
        and len(source.rows) == 30
        and all(final.get(key) == value for key, value in source.binding_fields().items()),
        "complete_paginated_inventory_required",
    )
    book.unchanged()
    # Canonical journal alone becomes mutable, under explicit child guards.
    del book.hashes[str(path.relative_to(book.history))]
    return raw, progress, inventory_dir, source.binding_fields()


class BrowserProjectFinalizeSteps:
    """Caller-owned page; fixed separate budgets, one child advance per call.

    Constructor validates and exclusively claims this continuation, but creates
    no child and touches no page. start() begins the 600-second scoring budget;
    entering submission_initializing begins its separate 180-second budget.
    Paused wall time counts. close() aborts work, never closes Chromium.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        journal,
        campaign_runtime_dir,
        allow_score_transfer=False,
        allow_submission=False,
        emit=lambda _: None,
        cancelled=lambda: False,
        _clock=time.monotonic,
        _scoring_factory=None,
        _submission_factory=None,
    ):
        _require(
            type(allow_score_transfer) is bool and type(allow_submission) is bool, "explicit_flags_required"
        )
        _require(not allow_submission or allow_score_transfer, "submission_requires_score_transfer")
        _require(callable(emit) and callable(cancelled) and callable(_clock), "invalid_callback")
        self.book = _PinnedEvidence(run_history)
        self.history, self.journal = self.book.history, journal
        try:
            before, progress, inventory, binding = _latch_campaign(self.book, journal, campaign_runtime_dir)
        except BrowserSafetyStop:
            raise
        except Exception:  # noqa: BLE001 - source failures must not expose private paths
            raise BrowserSafetyStop("project_finalize_invalid_campaign_sources") from None
        self.output = self.book.path(output)
        _require(
            self.output != self.history
            and not self.output.exists()
            and not any(self.output.is_relative_to(p) for p in self.book.clean_directories),
            "invalid_output",
        )
        self.page, self.config = page, config.model_copy(deep=True)
        self._config = self.config.model_dump(mode="json")
        self._before, self._initial_progress, self._journal_sha = before, progress, _sha(before)
        self._cached_raw, self._cached_progress, self._cached_state = before, progress, progress.reduce()
        self._stars = progress.reduce().stars
        self.revision = progress.reduce().revision
        self.inventory_dir = inventory
        self._clock, self._callback, self._cancelled = _clock, emit, cancelled
        self._scoring_factory = _scoring_factory or BrowserProjectScoringCoordinator
        self._submission_factory = _submission_factory or BrowserProjectSubmissionSteps
        self.allow_score_transfer, self.allow_submission = allow_score_transfer, allow_submission
        self._flags = (allow_score_transfer, allow_submission)
        self._stage_started = None
        self._stage_seconds = SCORING_SECONDS
        self.status, self.phase, self.failure = "idle", "not_started", None
        self.scoring = self.submission = self.report = None
        self._component = None
        self._busy = self._emitting = self._closing = self._abort_requested = self._forward_failed = False
        self._cleanup_failed = False
        self._score_checkpoint_sha = None
        self._score_checkpoint_closed = False
        self.advances = self._sequence = 0
        self._completed_trees = {}
        self._relay = ProjectEventRelay(self._receive)
        self._scoring_forward = self._submission_forward = None
        self.handoff = {
            "historical_journal_sha256": _sha(before),
            "runtime_report_sha256": self.book.hashes["report.json"],
            "campaign_report_sha256": self.book.hashes["campaign/report.json"],
            "verified_thirtieth_sha256": self.book.hashes["campaign/verified-030.json"],
            "source_sha256": deepcopy(self.book.hashes),
            "inventory_dir": str(inventory.relative_to(self.history)),
            **binding,
        }
        self.handoff_sha = _sha(_json(self.handoff).encode())
        self.scope = {
            "schema_version": 1,
            "mode": MODE,
            "project_id": progress.project_id,
            "attempt_id": progress.attempt_id,
            "data_revision": self.revision,
            "allow_score_transfer": allow_score_transfer,
            "allow_submission": allow_submission,
            "scoring_max_seconds": SCORING_SECONDS,
            "scoring_max_advances": 46,
            "submission_max_seconds": SUBMISSION_SECONDS,
            "submission_max_advances": 4,
            "max_advances": MAX_ADVANCES,
            "pause_counts_toward_deadline": True,
            "cancellation": "between_bounded_calls_and_callback_abort",
            "browser_owned_by_caller": True,
            "campaign_handoff_sha256": self.handoff_sha,
            "campaign_journal_sha256": _sha(before),
            "task_completed": False,
            "project_completed": False,
            "submitted": False,
            "scientific_verified": False,
            "automatic_retry": False,
            "automatic_deadline_increase": False,
            "optimizer_updates": 0,
        }
        self._scope = deepcopy(self.scope)
        claims = self.history / "project-finalization-claims"
        self.claim = self.book.path(claims / f"revision-{self.revision}.json")
        _require(not self.claim.exists(), "continuation_already_claimed")
        _require(not cancelled(), "cancelled")
        self.book.unchanged()
        _require(journal.path.read_bytes() == before, "canonical_journal_changed")
        claims.mkdir(exist_ok=True)
        claim = {**self.scope, "output": str(self.output.relative_to(self.history))}
        # An exclusive claim survives every failure, including initialization.
        try:
            fd = os.open(self.claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise BrowserSafetyStop("project_finalize_continuation_already_claimed") from None
        with os.fdopen(fd, "w") as stream:
            stream.write(_json(claim) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(claims, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self.output.mkdir(parents=True, exist_ok=False)
        persist_json(self.output / "reserved.json", claim)
        persist_json(self.output / "scope.json", self.scope)
        persist_json(self.output / "campaign-handoff.json", self.handoff)
        with (self.output / "campaign-journal.jsonl").open("xb") as stream:
            stream.write(before)
            stream.flush()
            os.fsync(stream.fileno())
        for path in (
            self.claim,
            *(
                self.output / name
                for name in ("reserved.json", "scope.json", "campaign-handoff.json", "campaign-journal.jsonl")
            ),
        ):
            self.book.read(path)
        self._stream = (self.output / "events.jsonl").open("x")
        self._run_id = "project-finalize-" + self.handoff_sha[:16]

    @property
    def finished(self):
        return self.status in {"handoff", "stopped", "aborted"}

    def _expected_journal_sha(self):
        if self.submission is not None:
            return self.submission.state().get("journal_expected_sha256")
        if self.scoring is not None:
            return self.scoring.scoring.state().get("journal_sha256")
        return self._journal_sha

    def _current(self):
        raw = self.book.path(self.journal.path).read_bytes()
        if raw != self._cached_raw:
            progress = self.journal.load()
            _require(self.journal.path.read_bytes() == raw, "concurrent_journal_change")
            self._cached_progress, self._cached_state = progress, progress.reduce()
            self._cached_raw = raw
        return raw, self._cached_progress, self._cached_state

    def _check(self):
        _require(
            not self.finished and not self._abort_requested and not self._cancelled(), "cancelled_or_stopped"
        )
        _require(
            self.scope == self._scope and (self.allow_score_transfer, self.allow_submission) == self._flags,
            "options_changed",
        )
        _require(self.config.model_dump(mode="json") == self._config, "boundary_changed")
        expected_seconds = (
            SUBMISSION_SECONDS
            if self.phase.startswith("submission_") or self.phase == "unknown_pending"
            else SCORING_SECONDS
        )
        _require(self._stage_seconds == expected_seconds, "stage_budget_changed")
        if self.scoring is not None:
            child = self.scoring.state()
            _require(
                child.get("max_seconds") == SCORING_SECONDS
                and child.get("max_advances") == 46
                and child.get("allow_score_transfer") is self.allow_score_transfer
                and self.scoring.output == self.output / "scoring"
                and self.scoring.journal is self.journal,
                "scoring_identity_or_limits_changed",
            )
        if self.submission is not None:
            child = self.submission.state()
            _require(
                child.get("max_seconds") == SUBMISSION_SECONDS
                and child.get("max_advances") == 4
                and child.get("allow_submission") is True
                and self.submission.output == self.output / "submission"
                and self.submission.journal is self.journal,
                "submission_identity_or_limits_changed",
            )
        if self._stage_started is not None:
            _require(self._clock() - self._stage_started < self._stage_seconds, "stage_time_limit")
        _require(
            set(self.history.glob("project-progress-*.jsonl")) == {self.journal.path},
            "ambiguous_canonical_journal",
        )
        raw, progress, state = self._current()
        _require(
            raw.startswith(self._before) and _sha(raw) == self._expected_journal_sha(),
            "canonical_journal_changed",
        )
        _require(state.stars == self._stars and state.revision == self.revision, "canonical_tasks_changed")
        allowed = {"assessment_data_quality", "assessment_scavenger_hunt"}
        if self.allow_score_transfer:
            allowed.add("score_transfer")
        if self.allow_submission:
            allowed.add("submission")
        _require(
            all(
                isinstance(r.payload, (WriteReserved, WriteReceipt)) and r.payload.write_kind in allowed
                for r in progress.records[len(self._initial_progress.records) :]
            ),
            "unexpected_journal_write",
        )
        self.book.unchanged()
        for directory, paths in self._completed_trees.items():
            _require(self._tree(directory) == paths, "completed_tree_changed")

    def _tree(self, directory):
        return tuple(sorted(str(self.book.path(p).relative_to(self.history)) for p in directory.rglob("*")))

    def _pin_tree(self, directory):
        paths = self._tree(directory)
        _require(paths and len(paths) <= 4000, "invalid_completed_tree")
        for name in paths:
            path = self.history / name
            _require(
                path.name
                not in {
                    "stopped.json",
                    "invalidated.json",
                    "finalization_failed.json",
                    "event-forwarding-failed.json",
                }
                and not path.name.endswith("-stopped.json"),
                "failed_completed_tree",
            )
            if path.is_file():
                self.book.read(path)
        self._completed_trees[directory] = paths

    def _child_cancelled(self):
        self._check()
        return False

    def _emit(self, kind, payload):
        event = RuntimeEvent(event=kind, sequence=self._sequence, run_id=self._run_id, payload=payload)
        self._stream.write(event.model_dump_json() + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._sequence += 1
        if not self._forward_failed:
            self._emitting = True
            try:
                self._callback(event.model_dump(mode="json"))
            except Exception:  # noqa: BLE001 - callback exceptions can contain credentials
                self._forward_failed = True
                raise BrowserSafetyStop("project_finalize_event_forwarding_failed") from None
            finally:
                self._emitting = False

    def _receive(self, kind, payload):
        if self._closing:
            return
        self._check()
        value = next(
            (
                payload[k]
                for k in ("component_state", "component_summary", "component_hello")
                if isinstance(payload.get(k), dict)
            ),
            None,
        )
        if value is not None:
            self._component = deepcopy(value)
        self._emit(kind, {**payload, **self.state()} if kind == "state" else payload)
        self._check()

    def _from_scoring(self, event):
        if not self._closing:
            _require(self.phase in {"scoring_initializing", "scoring_active"}, "unscheduled_scoring_event")
            self._scoring_forward(event)

    def _from_submission(self, event):
        if not self._closing:
            _require(
                self.phase in {"submission_initializing", "submission_active"}, "unscheduled_submission_event"
            )
            self._submission_forward(event)

    def state(self):
        submission_state = self.submission.state() if self.submission else {}
        try:
            raw, _progress, current = self._current()
            progress, journal_sha = current.report(), _sha(raw)
        except Exception:  # noqa: BLE001 - malformed canonical data is unavailable, never stale success
            progress, journal_sha, current = None, None, None
        checkpoint = self._score_checkpoint(current, journal_sha)
        return {
            **deepcopy(self.scope),
            "status": self.status,
            "phase": self.phase,
            "finished": self.finished,
            "failure_reason": self.failure,
            "advances": self.advances,
            "finalization_component": deepcopy(self._component),
            "project_progress": progress,
            "canonical_progress_available": progress is not None,
            "journal_sha256": journal_sha,
            "score_checkpoint_completed": checkpoint,
            "reported_score": current.receipt("score_transfer").score if checkpoint else None,
            "scoring_dir": str((self.output / "scoring/scoring").relative_to(self.history))
            if self.scoring
            else None,
            "submission_dir": str((self.output / "submission").relative_to(self.history))
            if self.submission
            else None,
            "score_transfer_verified": bool(
                self.scoring
                and self.scoring.report
                and self.scoring.report.get("score_transfer_verified") is True
                and self.status not in {"stopped", "aborted"}
            ),
            "submission_outcome": submission_state.get("submission_outcome", "not_dispatched"),
            "submission_feedback": deepcopy(submission_state.get("submission_feedback")),
            "event_forwarding_failed": self._forward_failed,
            "cleanup_failed": self._cleanup_failed,
            "trace_path": str((self.output / "events.jsonl").relative_to(self.history)),
        }

    def _score_checkpoint(self, current, journal_sha):
        # A visible score alone, or a successful child, does not close this
        # checkpoint. Only the verified current revision and clean owner close do.
        if not (
            self._score_checkpoint_closed
            and self._score_checkpoint_sha is not None
            and journal_sha == self._score_checkpoint_sha
            and self.status == "handoff"
            and self.phase == "score_transferred_not_submitted"
            and self.failure is None
            and not self._abort_requested
            and not self._forward_failed
            and not self._cleanup_failed
            and self.allow_score_transfer is True
            and self.allow_submission is False
            and current is not None
            and current.revision == self.revision
            and len(current.stars) == 30
            and all(star.task_completed for star in current.stars.values())
            and not current.pending
            and all(
                current.receipt(k) is not None
                for k in ("assessment_data_quality", "assessment_scavenger_hunt", "score_transfer")
            )
            and current.receipt("submission") is None
        ):
            return False
        try:
            self.book.unchanged()
            return all(self._tree(path) == paths for path, paths in self._completed_trees.items())
        except Exception:  # noqa: BLE001 - unavailable proof cannot retain checkpoint success
            return False

    def start(self, *, paused=True):
        _require(self.status == "idle" and type(paused) is bool, "already_started_or_invalid_pause")
        self.status, self.phase = "paused" if paused else "running", "scoring_initializing"
        self._stage_started = self._clock()
        try:
            self._check()
            self._emit("hello", {"protocol_version": 1, **self.state()})
            self._check()
        except Exception as exc:  # noqa: BLE001 - the boundary preserves the durable claim
            self._stop(exc)
        return self.state()

    def _verify_scoring(self):
        child = self.scoring
        expected = (
            "score_transferred_not_submitted"
            if self.allow_score_transfer
            else "assessed_score_transfer_disabled"
        )
        report = self.book.json(child.output / "report.json")
        _require(
            child.report == report
            and report.get("finished") is True
            and report.get("phase") == expected
            and report.get("failure_reason") is None
            and report.get("event_forwarding_failed") is False
            and report.get("cleanup_failed") is False
            and report.get("journal_sha256") == _sha(self.journal.path.read_bytes())
            and report.get("scoring_dir") == str(child.scoring.output.relative_to(self.history)),
            "unverified_scoring_result",
        )
        if self.allow_score_transfer:
            submission_sources(self.book, self.journal, child.scoring.output)
            self.book.hashes.pop(str(self.journal.path.relative_to(self.history)), None)
        else:
            state = self.journal.load().reduce()
            _require(
                not state.pending
                and all(
                    state.receipt(k) is not None
                    for k in ("assessment_data_quality", "assessment_scavenger_hunt")
                )
                and state.receipt("score_transfer") is None,
                "unverified_assessments",
            )
        self._pin_tree(child.output)
        self._pin_tree(self.output / "assessment-history")
        self._check()
        if self.allow_score_transfer:
            self._score_checkpoint_sha = _sha(self.journal.path.read_bytes())
        self._relay.retire()
        if self.allow_submission:
            self.phase, self._component = "submission_initializing", None
            self._stage_started, self._stage_seconds = self._clock(), SUBMISSION_SECONDS
        else:
            self._finish(expected)

    def _verify_submission(self):
        child = self.submission
        report = self.book.json(child.output / "report.json")
        _require(
            child.report == report
            and report.get("submitted") is False
            and report.get("project_completed") is False
            and report.get("task_completed") is False
            and report.get("phase") == "unknown_pending"
            and report.get("failure_reason") == "project_submission_acknowledgement_not_grounded"
            and report.get("canonical_reservation_verified") is True
            and report.get("submit_click_returned") is True
            and report.get("submission_outcome") == "unknown"
            and report.get("event_forwarding_failed") is False,
            "submission_not_grounded_or_interrupted",
        )
        _adopt_sources(self.book, report["source_sha256"])
        state = self.journal.load().reduce()
        _require(
            len(state.pending) == 1
            and state.pending[0].write_kind == "submission"
            and state.receipt("submission") is None,
            "submission_pending_reservation_changed",
        )
        self._pin_tree(child.output)
        self._check()
        self._finish("unknown_pending", reason=report["failure_reason"])

    def advance(self):
        if self.finished:
            return self.state()
        if self._busy or self._emitting:
            self._abort_requested = True
            raise BrowserSafetyStop("project_finalize_reentrant_call")
        self._busy = True
        try:
            _require(self.status in {"running", "paused"}, "not_started")
            self._check()
            _require(self.advances < MAX_ADVANCES, "advance_limit")
            self.advances += 1
            if self.phase == "scoring_initializing":
                assessment_root = self.output / "assessment-history"
                assessment_root.mkdir(exist_ok=False)
                self._scoring_forward = self._relay.bind("project.finalization.scoring")
                self.scoring = self._scoring_factory(
                    self.page,
                    self.config,
                    self.output / "scoring",
                    run_history=self.history,
                    journal=self.journal,
                    inventory_dir=self.inventory_dir,
                    assessment_history_root=assessment_root,
                    allow_score_transfer=self.allow_score_transfer,
                    max_seconds=SCORING_SECONDS,
                    panel_max_seconds=60,
                    assessment_timeout_seconds=10,
                    score_timeout_seconds=15,
                    emit=self._from_scoring,
                    cancelled=self._child_cancelled,
                    _clock=self._clock,
                )
                _require(not self.scoring.finished, "scoring_constructor_failed")
                self.phase = "scoring_active"
            elif self.phase == "scoring_active":
                self.scoring.advance()
                if self.scoring.finished:
                    _require(self.scoring.failure is None, self.scoring.failure or "scoring_failed")
                    self.phase = "scoring_verifying"
            elif self.phase == "scoring_verifying":
                self._verify_scoring()
            elif self.phase == "submission_initializing":
                self._submission_forward = self._relay.bind("project.finalization.submission")
                self.submission = self._submission_factory(
                    self.page,
                    self.config,
                    self.output / "submission",
                    run_history=self.history,
                    journal=self.journal,
                    scoring_dir=self.scoring.scoring.output,
                    allow_submission=True,
                    max_seconds=SUBMISSION_SECONDS,
                    max_advances=4,
                    emit=self._from_submission,
                    cancelled=self._child_cancelled,
                    _clock=self._clock,
                )
                self.phase = "submission_active"
            elif self.phase == "submission_active":
                self.submission.advance()
                if self.submission.finished:
                    self.phase = "submission_verifying"
            elif self.phase == "submission_verifying":
                self._verify_submission()
            else:
                raise BrowserSafetyStop("project_finalize_unknown_phase")
            if not self.finished:
                self._check()
                self._emit("state", self.state())
                self._check()
        except (KeyboardInterrupt, SystemExit):
            self.abort()
            raise
        except Exception as exc:  # noqa: BLE001 - never retry an uncertain child operation
            self._stop(exc)
        finally:
            self._busy = False
        return self.state()

    def step(self):
        if self.finished:
            return self.state()
        _require(self.status == "paused", "pause_before_step")
        return self.advance()

    def tick(self):
        return self.advance() if self.status == "running" else self.state()

    advance_if_due = tick

    def pause(self):
        if not self.finished:
            _require(self.status in {"paused", "running"}, "not_started")
            self.status = "paused"
        return self.state()

    def resume(self):
        if not self.finished:
            _require(not self._busy and not self._emitting and self.status == "paused", "invalid_resume")
            self._check()
            self.status = "running"
        return self.state()

    def _finish(self, phase, reason=None):
        self._check()
        self.phase, self.failure = phase, reason
        self._emit("episode_summary", {**self.state(), "status": "handoff", "finished": True})
        self._check()
        self._relay.retire()
        self._stream.close()
        self.status = "handoff"
        self._score_checkpoint_closed = True
        try:
            self.report = {
                **self.state(),
                "source_sha256": deepcopy(self.book.hashes),
                "events_sha256": _sha((self.output / "events.jsonl").read_bytes()),
            }
            persist_json(self.output / "report.json", self.report)
        except Exception:
            self._score_checkpoint_closed = False
            self.status = "running"  # Let the original failure take the terminal stop path.
            raise

    def _stop(self, exc):
        if self.finished or self._closing:
            return
        self._closing = True
        self.failure = "operator_aborted" if self._abort_requested else _reason(exc)
        self.status, self.phase = ("aborted", "aborted") if self._abort_requested else ("stopped", "stopped")
        try:
            for child in (self.submission, self.scoring):
                if child is not None and not child.finished:
                    try:
                        child.abort()
                    except Exception:  # noqa: BLE001 - cleanup never authorizes another write
                        self._cleanup_failed = True
            self.report = {**self.state(), "source_sha256": deepcopy(self.book.hashes)}
            persist_json(self.output / "stopped.json", self.report)
            if not self._forward_failed:
                try:
                    self._emit("error", {"type": "ProjectFinalizationStop", "message": self.failure})
                    self._emit("episode_summary", self.report)
                except Exception:  # noqa: BLE001 - final callback failure cannot reopen work
                    self._forward_failed = True
                    self.report = {**self.state(), "source_sha256": deepcopy(self.book.hashes)}
                    # Preserve the already-durable original stop. A second
                    # callback failure is a new disposition, never an overwrite
                    # or another attempt at the completed/uncertain operation.
                    persist_json(
                        self.output / "event-forwarding-failed.json",
                        {
                            "original_stop_sha256": _sha((self.output / "stopped.json").read_bytes()),
                            "state": self.report,
                            "automatic_retry": False,
                        },
                    )
        finally:
            self._relay.retire()
            self._stream.close()
            self._closing = False

    def abort(self):
        if not self.finished:
            self._abort_requested = True
            if not self._busy and not self._emitting:
                self._stop(BrowserSafetyStop("operator_aborted"))
        return self.state()

    def close(self):
        return self.abort()
