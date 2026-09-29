"""Offline project progress, with explicit branches and a durable write journal.

This is an evidence reducer, not a browser driver or scientific classifier.
Adapters must verify the student-visible artifacts whose hashes they supply.
Predictions/readbacks do not themselves prove correctness or task completion.
Use one ``ProjectJournal`` under the owned browser-attempt root; never start a
new attempt identity to retry an uncertain action in the same browser attempt.
Journal reservations are independent of an action's artifact output directory.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .contracts import Contract, RuntimeEvent
from .project_assessment import AssessmentLedger

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identity = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,100}$")]
Provenance = Literal[
    "learned_prediction", "reference_prediction", "scientific_observation", "course_feedback", "tool_verified"
]
Stage = Literal["stellar_numeric", "stellar_color", "stellar_classification", "planet", "habitability"]
WriteKind = Literal[
    "star_update",
    "save_star",
    "assessment_data_quality",
    "assessment_scavenger_hunt",
    "score_transfer",
    "submission",
]
ACKNOWLEDGEMENTS = {
    "star_update": "star_updated",
    "save_star": "star_saved",
    "assessment_data_quality": "data_quality_updated",
    "assessment_scavenger_hunt": "scavenger_hunt_updated",
    "score_transfer": "score_updated",
    "submission": "project_submitted",
}


class ProgressError(ValueError):
    pass


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Evidence(Contract):
    """A decision's origin and readback are different from scientific support."""

    source_sha256: Digest
    provenance: Provenance
    value: str | None = Field(default=None, min_length=1, max_length=200)
    transport_verified: bool = Field(default=False, strict=True)
    scientific_verified: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def verification_is_not_prediction(self):
        if self.scientific_verified and self.provenance not in {"scientific_observation", "course_feedback"}:
            raise ValueError("Prediction or transport alone is not scientific verification")
        return self


class StageEvidence(Contract):
    decision: Evidence
    confirmation: Evidence | None = None
    outcome: (
        Literal[
            "candidate", "no_planet", "planet", "unresolved", "habitable", "not_habitable", "not_applicable"
        ]
        | None
    ) = None
    applicability_reason: Literal["no_planet", "non_terrestrial_planet"] | None = None
    branch_applicability_verified: bool = Field(default=False, strict=True)
    policy_label: str | None = Field(default=None, min_length=1, max_length=100)
    observation_limit_days: int | None = Field(default=None, ge=1, le=10000, strict=True)

    @model_validator(mode="after")
    def applicability_is_not_a_native_write(self):
        if self.branch_applicability_verified and self.outcome != "not_applicable":
            raise ValueError("Branch applicability verification requires an explicit not-applicable outcome")
        return self

    @property
    def transport_verified(self):
        return self.decision.transport_verified or bool(
            self.confirmation and self.confirmation.transport_verified
        )

    def report(self):
        return {
            "value": self.decision.value,
            "provenance": self.decision.provenance,
            "transport_verified": self.transport_verified,
            "scientific_verified": self.decision.scientific_verified
            or bool(self.confirmation and self.confirmation.scientific_verified),
            "outcome": self.outcome,
            "applicability_reason": self.applicability_reason,
            "branch_applicability_verified": self.branch_applicability_verified,
            "policy_label": self.policy_label,
            "observation_limit_days": self.observation_limit_days,
        }


class Collected(Contract):
    kind: Literal["collected"] = "collected"
    star_id: Identity
    name: str = Field(min_length=1, max_length=100)
    source_sha256: Digest

    @model_validator(mode="after")
    def canonical_name(self):
        if self.name != " ".join(self.name.split()) or not self.name:
            raise ValueError("Visible star name must be nonblank and whitespace normalized")
        return self


class StageRecorded(Contract):
    kind: Literal["stage"] = "stage"
    star_id: Identity
    stage: Stage
    evidence: StageEvidence

    @model_validator(mode="after")
    def stage_outcome(self):
        outcome = self.evidence.outcome
        if self.stage == "planet":
            if outcome not in {"candidate", "no_planet", "planet", "unresolved"}:
                raise ValueError("Explicit planet outcome required")
        elif self.stage == "habitability":
            if outcome not in {"habitable", "not_habitable", "not_applicable", "unresolved"}:
                raise ValueError("Explicit habitability outcome required")
        elif outcome is not None:
            raise ValueError("Stellar evidence cannot carry a planet or habitability outcome")
        if self.stage in {"stellar_color", "stellar_classification"} and self.evidence.decision.value is None:
            raise ValueError("A visible stellar selection value is required")
        reason = self.evidence.applicability_reason
        if (outcome == "not_applicable") != (reason is not None):
            raise ValueError("Not-applicable evidence requires an explicit applicability reason")
        return self


class Active(Contract):
    kind: Literal["active"] = "active"
    star_id: Identity | None = None
    stage: Stage | Literal["assessment", "score_transfer", "submission", "complete"]


class TaskReceipt(Contract):
    """An explicit workflow receipt, not a claim of perfect scientific answers."""

    kind: Literal["task_receipt"] = "task_receipt"
    star_id: Identity
    star_revision: int = Field(ge=0, strict=True)
    source_sha256: Digest
    authority: Literal["visible_workflow_readback", "course_feedback"]
    task_completed: Literal[True]

    @field_validator("task_completed", mode="before")
    @classmethod
    def explicit_boolean(cls, value):
        if value is not True:
            raise ValueError("Task receipt requires explicit true, not a truthy value")
        return value


class WriteReserved(Contract):
    kind: Literal["write_reserved"] = "write_reserved"
    action_id: Identity
    write_kind: WriteKind
    star_id: Identity | None = None
    revision: int = Field(ge=0, strict=True)
    before_sha256: Digest
    project_rows_sha256: Digest | None = None


class WriteReceipt(Contract):
    kind: Literal["write_receipt"] = "write_receipt"
    action_id: Identity
    write_kind: WriteKind
    star_id: Identity | None = None
    revision: int = Field(ge=0, strict=True)
    source_sha256: Digest
    project_rows_sha256: Digest | None = None
    authority: Literal["visible_readback", "course_feedback"]
    acknowledgement: str = Field(min_length=1, max_length=100)
    confirmed: Literal[True]
    assessment: AssessmentLedger | None = None
    score: float | None = Field(default=None, ge=0, le=260, strict=True)
    submitted: bool = Field(default=False, strict=True)

    @field_validator("confirmed", mode="before")
    @classmethod
    def explicit_boolean(cls, value):
        if value is not True:
            raise ValueError("Write receipt requires explicit true, not a truthy value")
        return value

    @model_validator(mode="after")
    def distinct_receipts(self):
        if self.acknowledgement != ACKNOWLEDGEMENTS[self.write_kind]:
            raise ValueError("Wrong write acknowledgement")
        assessment = self.write_kind.startswith("assessment_")
        if assessment != (self.assessment is not None):
            raise ValueError("Assessment receipt requires its validated spending ledger")
        if (self.write_kind == "score_transfer") != (self.score is not None):
            raise ValueError("Only score transfer carries a visible outer score")
        if (self.write_kind == "submission") != self.submitted:
            raise ValueError("Only an explicit submission receipt means submitted")
        if self.write_kind in {"score_transfer", "submission"} and self.authority != "course_feedback":
            raise ValueError("Score transfer and submission require authoritative course feedback")
        return self


Payload = Annotated[
    Collected | StageRecorded | Active | TaskReceipt | WriteReserved | WriteReceipt,
    Field(discriminator="kind"),
]


class ProgressRecord(Contract):
    schema_version: Literal[1] = 1
    sequence: int = Field(ge=0, strict=True)
    project_id: Identity
    attempt_id: Identity
    previous_sha256: Digest
    payload: Payload


class StarProgress(Contract):
    id: Identity
    name: str
    revision: int = 0
    stellar_numeric: StageEvidence | None = None
    stellar_color: StageEvidence | None = None
    stellar_classification: StageEvidence | None = None
    planet: StageEvidence | None = None
    habitability: StageEvidence | None = None
    completion: TaskReceipt | None = None

    def ready(self):
        parts = [self.stellar_numeric, self.stellar_color, self.stellar_classification, self.planet]
        if not all(part and part.transport_verified for part in parts):
            return False
        if self.habitability is None or not (
            self.habitability.transport_verified
            or (
                self.habitability.outcome == "not_applicable"
                and self.habitability.branch_applicability_verified
            )
        ):
            return False
        return bool(
            self.planet.outcome in {"planet", "no_planet"}
            and self.habitability.outcome in {"habitable", "not_habitable", "not_applicable"}
        )

    @property
    def task_completed(self):
        return bool(self.completion and self.completion.star_revision == self.revision and self.ready())

    def report(self, stage=None):
        return {
            "id": self.id,
            "name": self.name,
            "stage": stage,
            "revision": self.revision,
            "stellar": {
                label: part.report() if part else None
                for label, part in (
                    ("numeric", self.stellar_numeric),
                    ("color", self.stellar_color),
                    ("classification", self.stellar_classification),
                )
            },
            "planet": self.planet.report() if self.planet else {"outcome": "unresolved"},
            "habitability": self.habitability.report() if self.habitability else {"outcome": "unresolved"},
            "task_completed": self.task_completed,
        }


class ProgressState(Contract):
    project_id: Identity
    attempt_id: Identity
    revision: int = 0
    stars: dict[str, StarProgress] = Field(default_factory=dict)
    active: Active | None = None
    reservations: list[WriteReserved] = Field(default_factory=list)
    receipts: list[WriteReceipt] = Field(default_factory=list)
    evidence_owners: dict[str, str] = Field(default_factory=dict)

    @property
    def pending(self):
        confirmed = {receipt.action_id for receipt in self.receipts}
        return [item for item in self.reservations if item.action_id not in confirmed]

    def receipt(self, kind):
        return next(
            (
                item
                for item in reversed(self.receipts)
                if item.write_kind == kind and item.revision == self.revision
            ),
            None,
        )

    def report(self):
        verified = sum(star.task_completed for star in self.stars.values())
        assessment = {
            kind: self.receipt("assessment_" + kind) is not None
            for kind in ("data_quality", "scavenger_hunt")
        }
        transferred = self.receipt("score_transfer") is not None
        submitted = self.receipt("submission") is not None
        complete = (
            verified == 30 and all(assessment.values()) and transferred and submitted and not self.pending
        )
        current = self.stars.get(self.active.star_id) if self.active else None
        return {
            "schema_version": 1,
            "project_id": self.project_id,
            "attempt_id": self.attempt_id,
            "revision": self.revision,
            "target": 30,
            "collected": len(self.stars),
            "verified": verified,
            "unresolved": len(self.stars) - verified,
            "active_star": current.report(self.active.stage) if current else None,
            "active_stage": self.active.stage if self.active else None,
            "uncertain_actions": [
                {
                    "id": item.action_id,
                    "kind": item.write_kind,
                    "star_id": item.star_id,
                    "revision": item.revision,
                }
                for item in self.pending
            ],
            "assessment": assessment,
            "score_transfer_verified": transferred,
            "submitted": submitted,
            "project_completed": complete,
            "ladder": {"one_star": verified >= 1, "three_star": verified >= 3, "thirty_star": complete},
        }

    def runtime_event(self, *, seq: int):
        """Additive v1 state event; no existing observation or command changes."""
        return RuntimeEvent(event="state", sequence=seq, payload={"project_progress": self.report()})


def _owned_evidence(state, star_id, digest):
    owner = state.evidence_owners.get(digest)
    if owner is not None and owner != star_id:
        raise ProgressError("cross_star_evidence_reuse")
    state.evidence_owners[digest] = star_id


def _star(state, star_id):
    if star_id not in state.stars:
        raise ProgressError("unknown_star")
    return state.stars[star_id]


def _branch(star, evidence):
    planet = star.planet.outcome if star.planet else "unresolved"
    outcome, reason = evidence.outcome, evidence.applicability_reason
    if evidence.branch_applicability_verified:
        if not star.planet or not star.planet.transport_verified:
            raise ProgressError("applicability_requires_verified_planet_selection")
        if not star.planet.report()["scientific_verified"] and evidence.report()["scientific_verified"]:
            raise ProgressError("assumed_planet_branch_is_not_scientifically_verified")
    if outcome == "unresolved":
        return
    if planet == "no_planet":
        if outcome != "not_applicable" or reason != "no_planet":
            raise ProgressError("no_planet_requires_explicit_not_applicable")
    elif planet == "planet":
        if outcome == "not_applicable" and (
            reason != "non_terrestrial_planet" or star.planet.decision.value not in {"gas_giant", "ice_giant"}
        ):
            raise ProgressError("not_applicable_requires_non_terrestrial_class")
        if outcome in {"habitable", "not_habitable"} and star.planet.decision.value != "terrestrial":
            raise ProgressError("habitability_requires_explicit_terrestrial_class")
    else:
        raise ProgressError("planet_outcome_unresolved")


def _reserve(state, item):
    if item.revision != state.revision:
        raise ProgressError("stale_project_revision")
    if any(old.action_id == item.action_id for old in state.reservations):
        raise ProgressError("action_already_reserved")
    key = (item.write_kind, item.star_id, item.revision)
    if any((old.write_kind, old.star_id, old.revision) == key for old in state.reservations):
        raise ProgressError("logical_write_already_reserved")
    star_write = item.write_kind in {"star_update", "save_star"}
    if star_write:
        _star(state, item.star_id)
    elif item.star_id is not None:
        raise ProgressError("project_write_cannot_claim_one_star")
    if not star_write and item.project_rows_sha256 is None:
        raise ProgressError("project_rows_evidence_required")
    if item.write_kind in {"score_transfer", "submission"}:
        for kind in ("assessment_data_quality", "assessment_scavenger_hunt"):
            receipt = state.receipt(kind)
            if receipt is None or receipt.project_rows_sha256 != item.project_rows_sha256:
                raise ProgressError("both_current_assessments_required")
    if item.write_kind == "submission":
        transferred = state.receipt("score_transfer")
        if transferred is None or transferred.project_rows_sha256 != item.project_rows_sha256:
            raise ProgressError("current_score_transfer_required")
        if sum(star.task_completed for star in state.stars.values()) != 30:
            raise ProgressError("thirty_unique_completed_stars_required")
    state.reservations.append(item.model_copy(deep=True))


def _confirm(state, item):
    pending = state.pending
    if len(pending) != 1 or pending[0].action_id != item.action_id:
        raise ProgressError("unknown_pending_write")
    reserved = pending[0]
    if (
        any(
            getattr(reserved, name) != getattr(item, name)
            for name in ("write_kind", "star_id", "revision", "project_rows_sha256")
        )
        or item.revision != state.revision
    ):
        raise ProgressError("write_receipt_identity_mismatch")
    if item.source_sha256 == reserved.before_sha256:
        raise ProgressError("unchanged_receipt_evidence")
    if any(receipt.source_sha256 == item.source_sha256 for receipt in state.receipts):
        raise ProgressError("duplicate_write_receipt")
    if item.assessment is not None:
        ledger = item.assessment
        if ledger.pending or not ledger.attempts:
            raise ProgressError("assessment_unconfirmed")
        attempt = ledger.attempts[-1]
        if (
            "assessment_" + attempt.kind != item.write_kind
            or attempt.data_revision != item.revision
            or attempt.after.collected != len(state.stars)
        ):
            raise ProgressError("assessment_receipt_project_mismatch")
    if item.star_id is not None:
        _owned_evidence(state, item.star_id, item.source_sha256)
    state.receipts.append(item.model_copy(deep=True))


def _apply(state, item):
    if state.receipt("submission") is not None and not isinstance(item, Active):
        raise ProgressError("submitted_attempt_is_closed")
    if state.pending and not isinstance(item, (Active, WriteReceipt)):
        raise ProgressError("write_outcome_uncertain_no_retry")
    if isinstance(item, Collected):
        if len(state.stars) >= 30:
            raise ProgressError("collection_limit_reached")
        if item.star_id in state.stars or any(
            star.name.casefold() == item.name.casefold() for star in state.stars.values()
        ):
            raise ProgressError("duplicate_collected_star")
        state.stars[item.star_id] = StarProgress(id=item.star_id, name=item.name)
        state.revision += 1
    elif isinstance(item, StageRecorded):
        star = _star(state, item.star_id)
        if getattr(star, item.stage) == item.evidence:
            raise ProgressError("duplicate_stage_record")
        if item.stage == "habitability":
            _branch(star, item.evidence)
        for evidence in (item.evidence.decision, item.evidence.confirmation):
            if evidence:
                _owned_evidence(state, item.star_id, evidence.source_sha256)
        # New inputs or class choices invalidate downstream analysis; no old
        # branch can become complete merely because its controls still exist.
        if item.stage.startswith("stellar_"):
            star.planet, star.habitability = None, None
        elif item.stage == "planet":
            star.habitability = None
        setattr(star, item.stage, item.evidence.model_copy(deep=True))
        star.revision += 1
        star.completion = None
        state.revision += 1
    elif isinstance(item, Active):
        if item.star_id is not None:
            _star(state, item.star_id)
        state.active = item.model_copy(deep=True)
    elif isinstance(item, TaskReceipt):
        star = _star(state, item.star_id)
        if item.star_revision != star.revision or not star.ready():
            raise ProgressError("task_requires_current_complete_explicit_branch")
        if star.completion is not None:
            raise ProgressError("task_receipt_already_recorded")
        _owned_evidence(state, item.star_id, item.source_sha256)
        star.completion = item.model_copy(deep=True)
    elif isinstance(item, WriteReserved):
        _reserve(state, item)
    elif isinstance(item, WriteReceipt):
        _confirm(state, item)


class ProjectProgress(Contract):
    schema_version: Literal[1] = 1
    project_id: Identity
    attempt_id: Identity
    records: list[ProgressRecord] = Field(default_factory=list, max_length=10000)

    @model_validator(mode="after")
    def validate_history(self):
        self.reduce()
        return self

    def header(self):
        return {"schema_version": 1, "project_id": self.project_id, "attempt_id": self.attempt_id}

    def reduce(self):
        state = ProgressState(project_id=self.project_id, attempt_id=self.attempt_id)
        previous = _hash(self.header())
        for sequence, record in enumerate(self.records):
            record = ProgressRecord.model_validate(record.model_dump(mode="json"))
            if (
                record.sequence != sequence
                or record.previous_sha256 != previous
                or record.project_id != self.project_id
                or record.attempt_id != self.attempt_id
            ):
                raise ProgressError("inconsistent_project_history")
            _apply(state, record.payload)
            previous = _hash(record.model_dump(mode="json"))
        return state

    def append(self, payload: Payload):
        """Pure, non-mutating append. Persist the returned record BEFORE a click."""
        previous = _hash(self.records[-1].model_dump(mode="json") if self.records else self.header())
        record = ProgressRecord(
            sequence=len(self.records),
            project_id=self.project_id,
            attempt_id=self.attempt_id,
            previous_sha256=previous,
            payload=payload,
        )
        # Keep operational guard errors as ProgressError for callers, rather
        # than wrapping them inside a Pydantic history-validation exception.
        record = ProgressRecord.model_validate(record.model_dump(mode="json"))
        _apply(self.reduce(), record.payload)
        return ProjectProgress(
            project_id=self.project_id,
            attempt_id=self.attempt_id,
            records=[*(old.model_copy(deep=True) for old in self.records), record.model_copy(deep=True)],
        )


class ProjectJournal:
    """Single canonical, lock-serialized JSONL log per owned browser attempt.

    A partial final write fails closed. Reconciliation may append a verified
    receipt without another click; there is intentionally no unreserve/retry.
    Files are 0600, immutable-header and append-only. No credentials are needed.
    """

    def __init__(self, root, *, project_id, attempt_id):
        self.identity = ProjectProgress(project_id=project_id, attempt_id=attempt_id)
        self.path = Path(root) / f"project-progress-{_hash(self.identity.header())}.jsonl"

    def create(self):
        # Deliberately do not create an arbitrary new output tree implicitly.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_json(self.identity.header()) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return self

    def _read(self, stream):
        stream.seek(0)
        raw = stream.read()
        if not raw.endswith("\n"):
            raise ProgressError("incomplete_journal_no_automatic_repair")
        try:
            lines = [json.loads(line) for line in raw.splitlines()]
            if not lines or lines[0] != self.identity.header():
                raise ProgressError("journal_identity_mismatch")
            return ProjectProgress(**lines[0], records=lines[1:])
        except (ValueError, TypeError) as exc:
            raise ProgressError("invalid_journal_no_automatic_repair") from exc

    def load(self):
        with self.path.open("r", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            return self._read(stream)

    def append(self, payload: Payload):
        with self.path.open("r+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            progress = self._read(stream).append(payload)
            stream.seek(0, os.SEEK_END)
            stream.write(_json(progress.records[-1].model_dump(mode="json")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            return progress.reduce()


def progress_from_runtime_event(event):
    """Old v1 streams remain readable, but cannot establish project completion."""
    parsed = RuntimeEvent.model_validate(event)
    value = parsed.payload.get("project_progress")
    if value is None:
        return {"status": "unsupported_legacy_project_progress", "project_completed": False}
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ProgressError("unsupported_project_progress_payload")
    # Runtime summaries are display data, never authoritative reducer input.
    return {"status": "display_only_requires_journal_for_verification", "project_progress": value}
