"""Visible assessment evidence and an at-most-once simulation-spending ledger.

This module does not click the browser or choose/revise scientific answers. The
future project actuator must reserve before clicking and persist the ledger at
that point. An uncertain click stays pending: a timeout is NOT permission to
spend again. Assessment, saving, score transfer and submission are distinct.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import Field, model_validator

from .contracts import Contract

AssessmentKind = Literal["data_quality", "scavenger_hunt"]
ACKNOWLEDGEMENTS = {
    "data_quality": "DATA QUALITY UPDATED",
    "scavenger_hunt": "SCAVENGER HUNT UPDATED",
}


class AssessmentError(ValueError):
    pass


class QualityScores(Contract):
    stars: float = Field(ge=0, le=100)
    planets: float = Field(ge=0, le=100)
    habitability: float = Field(ge=0, le=100)
    overall: float = Field(ge=0, le=100)


class AssessmentSnapshot(Contract):
    schema_version: Literal[1] = 1
    mode: Literal["data_quality", "scavenger_hunt", "automation"]
    funding: int = Field(ge=0, strict=True)
    cost: int | None = Field(default=None, ge=0, strict=True)
    collected: int = Field(ge=0, strict=True)
    data_quality_percent: float = Field(ge=0, le=100)
    scavenger_found: int = Field(ge=0, le=8, strict=True)
    quality: QualityScores | None = None
    acknowledgement: AssessmentKind | None = None
    visible_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def mode_fields(self):
        if (self.mode == "data_quality") != (self.quality is not None):
            raise ValueError("Quality details must match the assessment mode")
        if self.quality and self.quality.overall != self.data_quality_percent:
            raise ValueError("Inconsistent quality readback")
        if self.mode != "automation" and self.cost is None:
            raise ValueError("Assessment cost is required")
        if self.acknowledgement is not None and self.acknowledgement != self.mode:
            raise ValueError("Wrong assessment acknowledgement")
        return self


def _one(pattern, text, field):
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise AssessmentError(f"missing_or_ambiguous_{field}")
    return matches[0]


def parse_assessment_text(visible_text: str) -> AssessmentSnapshot:
    """Parse only rendered simulation text, never source HTML or app state.

    Zero is a real score; absent/duplicate values are errors. Scavenger category
    labels alone do not prove which category is checked, so only its visible
    count is returned. No inferred analysis count or completion flag is emitted.
    """
    if not isinstance(visible_text, str) or len(visible_text) > 100000:
        raise AssessmentError("invalid_assessment_text")
    text = " ".join(visible_text.upper().split())
    funding = int(_one(r"\bFUNDING\s*\$([0-9]+)\b", text, "funding"))
    quality = float(_one(r"\bDATA QUALITY\s+([0-9]+(?:\.[0-9]+)?)%", text, "quality"))
    found = int(_one(r"\bSCAVENGER HUNT\s+([0-9]+)/8\b", text, "scavenger_count"))
    collected = int(_one(r"\bTOTAL COLLECTED\s+([0-9]+)\b", text, "collected"))
    scores = re.findall(
        r"\bSTARS\s+([0-9]+(?:\.[0-9]+)?)%\s+"
        r"PLANETS\s+([0-9]+(?:\.[0-9]+)?)%\s+"
        r"HABITABILITY\s+([0-9]+(?:\.[0-9]+)?)%\s+"
        r"OVERALL\s+([0-9]+(?:\.[0-9]+)?)%",
        text,
    )
    automation = "ASSESSMENT FIELD" in text
    scavenger = all(
        name in text
        for name in (
            "MAIN SEQUENCE",
            "WHITE DWARF",
            "RED GIANT",
            "SUPERGIANT",
            "GAS GIANT",
            "ICE GIANT",
            "TERRESTRIAL",
            "HABITABLE WORLD",
        )
    )
    # The eight categories can also occur in rows. A quality panel takes
    # precedence, but a simultaneously visible automation panel is ambiguous.
    if automation and scores or len(scores) > 1:
        raise AssessmentError("ambiguous_assessment_mode")
    if scores:
        mode = "data_quality"
        details = QualityScores(
            **dict(zip(("stars", "planets", "habitability", "overall"), map(float, scores[0])))
        )
        if details.overall != quality:
            raise AssessmentError("inconsistent_quality_readback")
    elif automation:
        mode, details = "automation", None
    elif scavenger:
        mode, details = "scavenger_hunt", None
    else:
        raise AssessmentError("unknown_assessment_mode")
    cost = None if mode == "automation" else int(_one(r"\bCOST\s*\$([0-9]+)\b", text, "cost"))
    acknowledgements = [kind for kind, label in ACKNOWLEDGEMENTS.items() if label in text]
    if len(acknowledgements) > 1:
        raise AssessmentError("ambiguous_assessment_acknowledgement")
    acknowledgement = acknowledgements[0] if acknowledgements else None
    if acknowledgement and acknowledgement != mode:
        raise AssessmentError("wrong_assessment_acknowledgement")
    return AssessmentSnapshot(
        mode=mode,
        funding=funding,
        cost=cost,
        collected=collected,
        data_quality_percent=quality,
        scavenger_found=found,
        quality=details,
        acknowledgement=acknowledgement,
        visible_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


class AssessmentAttempt(Contract):
    sequence: int = Field(ge=0, strict=True)
    kind: AssessmentKind
    data_revision: int = Field(ge=0, strict=True)
    before: AssessmentSnapshot
    after: AssessmentSnapshot | None = None


class AssessmentLedger(Contract):
    """Serializable guard; no refunds or automatic retry of uncertain attempts.

    Budget is in *simulation dollars*, not real currency. A revision must change
    whenever project data or the collection changes. Mode switches do not change
    that revision. Save/score/submission must have their own verified receipts.
    """

    budget: int = Field(default=0, ge=0, le=50000, strict=True)
    max_attempts: int = Field(default=0, ge=0, le=20, strict=True)
    attempts: list[AssessmentAttempt] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_history(self):
        if len(self.attempts) > self.max_attempts or self.reserved > self.budget:
            raise ValueError("Assessment history exceeds budget")
        seen = set()
        for index, attempt in enumerate(self.attempts):
            identity = (attempt.kind, attempt.data_revision)
            if attempt.sequence != index or identity in seen:
                raise ValueError("Duplicate or unordered assessment history")
            seen.add(identity)
            if index and attempt.data_revision < self.attempts[index - 1].data_revision:
                raise ValueError("Unordered project revisions")
            if attempt.before.mode != attempt.kind or attempt.before.cost != 100:
                raise ValueError("Unsupported assessment history")
            if attempt.before.acknowledgement is not None or attempt.before.funding < 100:
                raise ValueError("Invalid assessment starting state")
            if attempt.after is None and index != len(self.attempts) - 1:
                raise ValueError("Unresolved assessment before a later attempt")
            if attempt.after is not None:
                self._verify_receipt(attempt, attempt.after)
        return self

    @property
    def reserved(self):
        return sum(attempt.before.cost or 0 for attempt in self.attempts)

    @property
    def pending(self):
        return self.attempts[-1] if self.attempts and self.attempts[-1].after is None else None

    def reserve(self, kind: AssessmentKind, before: AssessmentSnapshot, *, data_revision: int):
        if self.pending:
            raise AssessmentError("assessment_outcome_uncertain")
        if kind not in ACKNOWLEDGEMENTS or before.mode != kind:
            raise AssessmentError("assessment_mode_not_authorized")
        if before.acknowledgement is not None:
            raise AssessmentError("stale_assessment_modal")
        if before.cost != 100:
            raise AssessmentError("assessment_cost_changed")
        if type(data_revision) is not int or data_revision < 0:
            raise AssessmentError("invalid_data_revision")
        if self.attempts and data_revision < self.attempts[-1].data_revision:
            raise AssessmentError("stale_project_revision")
        if any(a.kind == kind and a.data_revision == data_revision for a in self.attempts):
            raise AssessmentError("already_assessed_revision")
        if len(self.attempts) >= self.max_attempts or self.reserved + before.cost > self.budget:
            raise AssessmentError("assessment_budget_exhausted")
        if before.funding < before.cost:
            raise AssessmentError("insufficient_simulation_funding")
        attempt = AssessmentAttempt(
            sequence=len(self.attempts),
            kind=kind,
            data_revision=data_revision,
            before=before.model_copy(deep=True),
        )
        self.attempts.append(attempt)
        return attempt.model_copy(deep=True)

    @staticmethod
    def _verify_receipt(attempt, after):
        if after.mode != attempt.kind or after.acknowledgement != attempt.kind:
            raise AssessmentError("assessment_acknowledgement_missing")
        if after.funding != attempt.before.funding - attempt.before.cost:
            raise AssessmentError("assessment_funding_mismatch")
        if after.collected != attempt.before.collected or after.cost != attempt.before.cost:
            raise AssessmentError("assessment_project_changed")
        if after.visible_text_sha256 == attempt.before.visible_text_sha256:
            raise AssessmentError("unchanged_assessment_evidence")

    def confirm(self, sequence: int, after: AssessmentSnapshot, *, data_revision: int):
        attempt = self.pending
        if attempt is None or type(sequence) is not int or sequence != attempt.sequence:
            raise AssessmentError("unknown_pending_assessment")
        if type(data_revision) is not int or data_revision != attempt.data_revision:
            raise AssessmentError("assessment_project_changed")
        self._verify_receipt(attempt, after)
        attempt.after = after.model_copy(deep=True)
        return attempt.model_copy(deep=True)

    def report(self, *, data_revision: int):
        if type(data_revision) is not int or data_revision < 0:
            raise AssessmentError("invalid_data_revision")
        receipts = {
            kind: next(
                (
                    a.after
                    for a in reversed(self.attempts)
                    if a.kind == kind and a.data_revision == data_revision and a.after is not None
                ),
                None,
            )
            for kind in ACKNOWLEDGEMENTS
        }
        return {
            "schema_version": 1,
            "scope": "assessment_evidence_only",
            "reserved_simulation_dollars": self.reserved,
            "outcome_uncertain": self.pending is not None,
            "data_quality_assessed": receipts["data_quality"] is not None,
            "scavenger_hunt_assessed": receipts["scavenger_hunt"] is not None,
            "quality": receipts["data_quality"].quality.model_dump() if receipts["data_quality"] else None,
            "scavenger_found": receipts["scavenger_hunt"].scavenger_found
            if receipts["scavenger_hunt"]
            else None,
            # Do not infer any of these from assessment feedback or 100% quality.
            "save_verified": False,
            "score_transfer_verified": False,
            "submitted": False,
            "task_completed": False,
            "browser_acceptance_passed": False,
        }
