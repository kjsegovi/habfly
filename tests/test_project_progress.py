"""Pure offline branch/replay tests; no browser, credentials, or training."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from pydantic import ValidationError

from habfly.project_assessment import AssessmentLedger, AssessmentSnapshot, QualityScores
from habfly.project_progress import (
    Active,
    Collected,
    Evidence,
    ProgressError,
    ProjectJournal,
    ProjectProgress,
    StageEvidence,
    StageRecorded,
    TaskReceipt,
    WriteReceipt,
    WriteReserved,
    progress_from_runtime_event,
)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fresh():
    return ProjectProgress(project_id="habworlds", attempt_id="preview-001")


def collected(progress, name="ALPHA", *, star_id=None):
    return progress.append(Collected(star_id=star_id or name, name=name, source_sha256=digest(name)))


def stage(
    progress,
    star_id,
    label,
    *,
    value=None,
    outcome=None,
    reason=None,
    provenance="learned_prediction",
    transport=True,
    scientific=False,
    suffix="",
):
    if value is None and label in {"stellar_color", "stellar_classification"}:
        value = "IR" if label == "stellar_color" else "main_sequence"
    return progress.append(
        StageRecorded(
            star_id=star_id,
            stage=label,
            evidence=StageEvidence(
                decision=Evidence(
                    source_sha256=digest((star_id, label, suffix)),
                    provenance=provenance,
                    value=value,
                    transport_verified=transport,
                    scientific_verified=scientific,
                ),
                outcome=outcome,
                applicability_reason=reason,
            ),
        )
    )


def complete_star(
    progress, name="ALPHA", *, planet="no_planet", habitability="not_applicable", planet_class=None
):
    progress = collected(progress, name)
    for label in ("stellar_numeric", "stellar_color", "stellar_classification"):
        progress = stage(progress, name, label)
    progress = stage(
        progress, name, "planet", value=planet_class, outcome=planet, provenance="reference_prediction"
    )
    reason = None
    if habitability == "not_applicable":
        reason = "no_planet" if planet == "no_planet" else "non_terrestrial_planet"
    progress = stage(
        progress, name, "habitability", outcome=habitability, reason=reason, provenance="reference_prediction"
    )
    return confirm_task(progress, name)


def confirm_task(progress, star_id):
    return progress.append(
        TaskReceipt(
            star_id=star_id,
            star_revision=progress.reduce().stars[star_id].revision,
            source_sha256=digest((star_id, "task")),
            authority="visible_workflow_readback",
            task_completed=True,
        )
    )


def reserve(progress, kind, *, action_id=None, star_id=None, rows="rows"):
    revision = progress.reduce().revision
    return progress.append(
        WriteReserved(
            action_id=action_id or f"{kind}-{revision}",
            write_kind=kind,
            star_id=star_id,
            revision=revision,
            before_sha256=digest((kind, revision, "before")),
            project_rows_sha256=None if star_id else digest(rows),
        )
    )


def assessment_ledger(progress, kind):
    revision, count = progress.reduce().revision, len(progress.reduce().stars)
    details = (
        QualityScores(stars=70, planets=20, habitability=5, overall=31) if kind == "data_quality" else None
    )
    before = AssessmentSnapshot(
        mode=kind,
        funding=50000,
        cost=100,
        collected=count,
        data_quality_percent=31,
        scavenger_found=0,
        quality=details,
        visible_text_sha256=digest("assessment-before"),
    )
    after = before.model_copy(
        update={
            "funding": 49900,
            "acknowledgement": kind,
            "visible_text_sha256": digest("assessment-after"),
        }
    )
    ledger = AssessmentLedger(budget=100, max_attempts=1)
    ledger.reserve(kind, before, data_revision=revision)
    ledger.confirm(0, after, data_revision=revision)
    return ledger


def receipt_for(progress):
    pending = progress.reduce().pending[0]
    kind = pending.write_kind
    names = {
        "star_update": "star_updated",
        "save_star": "star_saved",
        "assessment_data_quality": "data_quality_updated",
        "assessment_scavenger_hunt": "scavenger_hunt_updated",
        "score_transfer": "score_updated",
        "submission": "project_submitted",
    }
    return WriteReceipt(
        action_id=pending.action_id,
        write_kind=kind,
        star_id=pending.star_id,
        revision=pending.revision,
        project_rows_sha256=pending.project_rows_sha256,
        source_sha256=digest((pending.action_id, "receipt")),
        authority="course_feedback",
        acknowledgement=names[kind],
        confirmed=True,
        assessment=assessment_ledger(progress, kind.removeprefix("assessment_"))
        if kind.startswith("assessment_")
        else None,
        score=43.4 if kind == "score_transfer" else None,
        submitted=kind == "submission",
    )


def write(progress, kind, **kwargs):
    progress = reserve(progress, kind, **kwargs)
    return progress.append(receipt_for(progress))


def test_zero_stars_no_fields_and_termination_never_mean_completion():
    progress = fresh()
    report = progress.reduce().report()
    assert report["collected"] == report["verified"] == 0
    assert not report["project_completed"] and not any(report["ladder"].values())
    progress = collected(progress)
    assert progress.reduce().report()["unresolved"] == 1
    with pytest.raises(ProgressError, match="explicit_branch"):
        confirm_task(progress, "ALPHA")
    legacy = {
        "version": 1,
        "event": "episode_summary",
        "sequence": 1,
        "payload": {"terminated": True, "submitted": True, "task_completed": True},
    }
    assert progress_from_runtime_event(legacy) == {
        "status": "unsupported_legacy_project_progress",
        "project_completed": False,
    }


@pytest.mark.parametrize(
    "planet,habitability,classification",
    [
        ("no_planet", "not_applicable", None),
        ("planet", "habitable", "terrestrial"),
        ("planet", "not_habitable", "terrestrial"),
        ("planet", "not_applicable", "gas_giant"),
        ("planet", "not_applicable", "ice_giant"),
    ],
)
def test_valid_branches_do_not_require_every_star_to_have_a_habitable_planet(
    planet, habitability, classification
):
    progress = complete_star(fresh(), planet=planet, habitability=habitability, planet_class=classification)
    state = progress.reduce()
    assert state.report()["verified"] == 1 and state.report()["ladder"]["one_star"]
    assert not state.report()["project_completed"]
    assert state.stars["ALPHA"].planet.report()["scientific_verified"] is False
    # Workflow verified is not a claim that a reference prediction is correct.
    assert state.stars["ALPHA"].planet.report()["provenance"] == "reference_prediction"


@pytest.mark.parametrize("outcome", ["candidate", "unresolved"])
def test_candidate_or_unresolved_planet_cannot_complete(outcome):
    progress = collected(fresh())
    for label in ("stellar_numeric", "stellar_color", "stellar_classification"):
        progress = stage(progress, "ALPHA", label)
    progress = stage(progress, "ALPHA", "planet", outcome=outcome)
    progress = stage(progress, "ALPHA", "habitability", outcome="unresolved")
    with pytest.raises(ProgressError, match="explicit_branch"):
        confirm_task(progress, "ALPHA")


@pytest.mark.parametrize(
    "planet,classification,outcome,reason",
    [
        ("no_planet", None, "habitable", None),
        ("no_planet", None, "not_habitable", None),
        ("no_planet", None, "not_applicable", "non_terrestrial_planet"),
        ("planet", "terrestrial", "not_applicable", "no_planet"),
        ("planet", "terrestrial", "not_applicable", "non_terrestrial_planet"),
        ("planet", "gas_giant", "habitable", None),
        ("planet", "ice_giant", "not_habitable", None),
    ],
)
def test_contradictory_or_unjustified_not_applicable_branch_rejected(planet, classification, outcome, reason):
    progress = collected(fresh())
    progress = stage(progress, "ALPHA", "planet", value=classification, outcome=planet)
    with pytest.raises(ProgressError):
        stage(progress, "ALPHA", "habitability", outcome=outcome, reason=reason)


def test_explicit_readback_and_workflow_receipt_required():
    progress = complete_star(fresh())
    progress = stage(progress, "ALPHA", "stellar_color", transport=False, suffix="new")
    assert progress.reduce().stars["ALPHA"].planet is None
    assert progress.reduce().stars["ALPHA"].habitability is None
    assert progress.reduce().report()["verified"] == 0
    progress = stage(progress, "ALPHA", "planet", outcome="no_planet")
    progress = stage(progress, "ALPHA", "habitability", outcome="not_applicable", reason="no_planet")
    with pytest.raises(ProgressError, match="explicit_branch"):
        confirm_task(progress, "ALPHA")


def test_provenance_transport_and_scientific_confirmation_are_distinct():
    with pytest.raises(ValidationError, match="not scientific"):
        Evidence(source_sha256=digest(1), provenance="learned_prediction", scientific_verified=True)
    evidence = StageEvidence(
        decision=Evidence(source_sha256=digest(1), provenance="learned_prediction", value="main_sequence"),
        confirmation=Evidence(
            source_sha256=digest(2),
            provenance="course_feedback",
            scientific_verified=True,
            transport_verified=True,
        ),
    )
    report = evidence.report()
    assert report["provenance"] == "learned_prediction"
    assert report["transport_verified"] and report["scientific_verified"]


@pytest.mark.parametrize("invalid", ["", " ALPHA", "ALPHA ", "A  B"])
def test_ambiguous_star_names_rejected(invalid):
    with pytest.raises(ValidationError):
        collected(fresh(), invalid, star_id="stable")


def test_duplicate_star_identity_and_casefold_name_cannot_inflate_ladder():
    progress = complete_star(fresh())
    for name, identity in [("OTHER", "ALPHA"), ("ALPHA", "different"), ("alpha", "different")]:
        with pytest.raises(ProgressError, match="duplicate_collected"):
            collected(progress, name, star_id=identity)


def test_cross_star_stage_and_task_receipts_rejected():
    progress = complete_star(fresh())
    progress = collected(progress, "BETA")
    alpha = progress.reduce().stars["ALPHA"]
    with pytest.raises(ProgressError, match="cross_star"):
        progress.append(
            StageRecorded(star_id="BETA", stage="stellar_numeric", evidence=alpha.stellar_numeric)
        )
    progress = complete_star(fresh(), "BETA")
    progress = complete_star(progress, "ALPHA")
    # Even a current explicit receipt cannot be moved to a different star.
    progress = stage(
        progress, "BETA", "habitability", outcome="not_applicable", reason="no_planet", suffix="new"
    )
    moved = alpha.completion.model_copy(update={"star_id": "BETA", "star_revision": 6})
    with pytest.raises(ProgressError, match="cross_star"):
        progress.append(moved)


def test_uncertain_reservation_survives_roundtrip_and_blocks_new_outputs_ids_and_revisions():
    progress = reserve(complete_star(fresh()), "save_star", star_id="ALPHA")
    restored = ProjectProgress.model_validate_json(progress.model_dump_json())
    assert len(restored.reduce().report()["uncertain_actions"]) == 1
    for change in (
        lambda: reserve(restored, "save_star", star_id="ALPHA", action_id="different-output"),
        lambda: reserve(restored, "assessment_data_quality"),
        lambda: collected(restored, "BETA"),
        lambda: stage(restored, "ALPHA", "stellar_numeric", suffix="new-revision"),
    ):
        with pytest.raises(ProgressError, match="outcome_uncertain"):
            change()
    # Observing which screen is active is read-only and remains allowed.
    observed = restored.append(Active(star_id="ALPHA", stage="stellar_numeric"))
    assert observed.reduce().report()["uncertain_actions"]
    confirmed = observed.append(receipt_for(observed))
    assert not confirmed.reduce().pending
    with pytest.raises(ProgressError, match="logical_write_already_reserved"):
        reserve(confirmed, "save_star", star_id="ALPHA", action_id="new-output")


@pytest.mark.parametrize(
    "change",
    [
        {"action_id": "unknown"},
        {"star_id": "BETA"},
        {"revision": 0},
        {"project_rows_sha256": digest("other")},
        {"write_kind": "star_update", "acknowledgement": "star_updated"},
    ],
)
def test_wrong_pending_receipt_cannot_resolve_uncertain_write(change):
    progress = reserve(complete_star(fresh()), "save_star", star_id="ALPHA")
    changed = receipt_for(progress).model_copy(update=change)
    with pytest.raises(ProgressError):
        progress.append(changed)
    assert progress.reduce().pending


def test_old_before_capture_and_duplicate_receipt_are_not_new_confirmation():
    progress = reserve(complete_star(fresh()), "save_star", star_id="ALPHA")
    receipt = receipt_for(progress)
    old = receipt.model_copy(update={"source_sha256": progress.reduce().pending[0].before_sha256})
    with pytest.raises(ProgressError, match="unchanged_receipt"):
        progress.append(old)
    progress = progress.append(receipt)
    with pytest.raises(ProgressError, match="unknown_pending"):
        progress.append(receipt)


def test_assessment_binding_and_current_revision_required_for_score_transfer():
    progress = complete_star(fresh())
    with pytest.raises(ProgressError, match="both_current"):
        reserve(progress, "score_transfer")
    progress = write(progress, "assessment_data_quality")
    with pytest.raises(ProgressError, match="both_current"):
        reserve(progress, "score_transfer")
    progress = write(progress, "assessment_scavenger_hunt", rows="different-rows")
    with pytest.raises(ProgressError, match="both_current"):
        reserve(progress, "score_transfer")
    progress = collected(progress, "BETA")
    assert not any(progress.reduce().report()["assessment"].values())


def test_assessment_wrong_collection_and_pending_ledger_rejected():
    progress = reserve(complete_star(fresh()), "assessment_data_quality")
    receipt = receipt_for(progress)
    wrong = receipt.model_copy(deep=True)
    wrong.assessment.attempts[0].before.collected = 2
    wrong.assessment.attempts[0].after.collected = 2
    with pytest.raises(ProgressError, match="project_mismatch"):
        progress.append(wrong)
    pending = receipt.model_copy(deep=True)
    pending.assessment.attempts[0].after = None
    with pytest.raises(ProgressError, match="unconfirmed"):
        progress.append(pending)


def test_ladder_requires_explicit_receipts_and_thirty_star_submission_not_perfect_score():
    progress = fresh()
    for index in range(30):
        progress = complete_star(progress, f"STAR-{index:02}")
        report = progress.reduce().report()
        assert report["verified"] == index + 1
        assert report["ladder"]["one_star"] is True
        assert report["ladder"]["three_star"] == (index >= 2)
        assert not report["ladder"]["thirty_star"]
    with pytest.raises(ProgressError, match="collection_limit"):
        collected(progress, "EXTRA")
    with pytest.raises(ProgressError, match="both_current"):
        reserve(progress, "submission")
    for kind in ("assessment_data_quality", "assessment_scavenger_hunt"):
        progress = write(progress, kind)
    with pytest.raises(ProgressError, match="score_transfer_required"):
        reserve(progress, "submission")
    progress = write(progress, "score_transfer")
    assert progress.reduce().report()["score_transfer_verified"]
    assert not progress.reduce().report()["submitted"]
    progress = reserve(progress, "submission")
    assert not progress.reduce().report()["project_completed"]
    progress = progress.append(receipt_for(progress))
    report = progress.reduce().report()
    assert report["project_completed"] and report["ladder"]["thirty_star"]
    assert progress.reduce().receipt("score_transfer").score == 43.4  # Not perfect; still completed.
    progress = progress.append(Active(stage="complete"))
    assert progress.reduce().report()["active_stage"] == "complete"
    assert progress.reduce().report()["project_completed"]
    with pytest.raises(ProgressError, match="closed"):
        stage(progress, "STAR-00", "stellar_numeric", suffix="after-submission")


def test_three_stars_with_all_assessments_and_score_still_cannot_submit():
    progress = fresh()
    for name in ("ALPHA", "BETA", "GAMMA"):
        progress = complete_star(progress, name)
    for kind in ("assessment_data_quality", "assessment_scavenger_hunt", "score_transfer"):
        progress = write(progress, kind)
    with pytest.raises(ProgressError, match="thirty_unique"):
        reserve(progress, "submission")


def test_typed_receipts_require_the_right_authority_and_do_not_conflate_operations():
    progress = reserve(complete_star(fresh()), "save_star", star_id="ALPHA")
    source = receipt_for(progress).model_dump(mode="json")
    for changes in (
        {"submitted": True},
        {"score": 100},
        {"acknowledgement": "project_submitted"},
        {"confirmed": False},
        {"confirmed": 1},
        {"revision": True},
    ):
        with pytest.raises(ValidationError):
            WriteReceipt.model_validate({**source, **changes})
    with pytest.raises(ValidationError, match="authoritative"):
        WriteReceipt.model_validate(
            {
                **source,
                "write_kind": "submission",
                "submitted": True,
                "authority": "visible_readback",
                "acknowledgement": "project_submitted",
            }
        )


def test_deterministic_replay_and_mutable_aliases_do_not_change_prior_log():
    first = complete_star(fresh())
    before = first.model_dump_json()
    second = collected(first, "BETA")
    second.records[0].payload.name = "MODIFIED"
    assert first.model_dump_json() == before
    restored = ProjectProgress.model_validate_json(before)
    assert restored.reduce().report() == first.reduce().report()
    assert restored.reduce().runtime_event(seq=7) == first.reduce().runtime_event(seq=7)
    raw = json.loads(before)
    for position, field, value in (
        (0, "attempt_id", "other"),
        (1, "sequence", 99),
        (2, "previous_sha256", digest("tampered")),
    ):
        bad = deepcopy(raw)
        bad["records"][position][field] = value
        with pytest.raises(ValueError, match="inconsistent_project_history"):
            ProjectProgress.model_validate(bad)


def test_runtime_state_addition_preserves_protocol_one_and_is_display_only():
    progress = complete_star(fresh()).append(Active(star_id="ALPHA", stage="habitability"))
    event = progress.reduce().runtime_event(seq=12).model_dump(mode="json")
    assert event["version"] == 1 and event["event"] == "state" and event["sequence"] == 12
    current = event["payload"]["project_progress"]["active_star"]
    assert current["planet"]["outcome"] == "no_planet"
    assert current["habitability"]["outcome"] == "not_applicable"
    assert current["task_completed"] is True
    assert progress_from_runtime_event(event)["status"].startswith("display_only")
    event["payload"]["project_progress"]["schema_version"] = 2
    with pytest.raises(ProgressError, match="unsupported"):
        progress_from_runtime_event(event)


def test_durable_canonical_journal_no_overwrite_and_reloaded_reservation(tmp_path):
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="preview-001").create()
    for record in complete_star(fresh()).records:
        journal.append(record.payload)
    progress = journal.load()
    proposal = reserve(progress, "save_star", star_id="ALPHA").records[-1].payload
    journal.append(proposal)
    independent = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="preview-001")
    assert independent.path == journal.path
    with pytest.raises(FileExistsError):
        independent.create()
    before = journal.path.read_bytes()
    with pytest.raises(ProgressError, match="outcome_uncertain"):
        independent.append(proposal.model_copy(update={"action_id": "new-artifact-output"}))
    assert journal.path.read_bytes() == before
    independent.append(receipt_for(independent.load()))
    assert not journal.load().reduce().pending
    assert journal.path.stat().st_mode & 0o777 == 0o600


def test_truncated_or_modified_journal_fails_closed_without_repair(tmp_path):
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="preview-001").create()
    original = journal.path.read_bytes()
    journal.path.write_bytes(original + b'{"schema_version":1')
    with pytest.raises(ProgressError, match="incomplete_journal"):
        journal.load()
    unchanged = journal.path.read_bytes()
    with pytest.raises(ProgressError):
        journal.append(Collected(star_id="ALPHA", name="ALPHA", source_sha256=digest(1)))
    assert journal.path.read_bytes() == unchanged
    journal.path.write_bytes(original.replace(b"preview-001", b"preview-002"))
    with pytest.raises(ProgressError, match="invalid_journal"):
        journal.load()


def test_approved_capped_observation_assumption_is_labelled_not_scientific_proof():
    progress = complete_star(fresh())
    decision = StageEvidence(
        decision=Evidence(
            source_sha256=digest("5000-day-no-dip"),
            provenance="reference_prediction",
            transport_verified=True,
        ),
        outcome="no_planet",
        policy_label="bounded_no_dip_assumption",
        observation_limit_days=5000,
    )
    progress = progress.append(StageRecorded(star_id="ALPHA", stage="planet", evidence=decision))
    progress = stage(progress, "ALPHA", "habitability", outcome="not_applicable", reason="no_planet")
    progress = confirm_task(progress, "ALPHA")
    progress = progress.append(Active(star_id="ALPHA", stage="habitability"))
    report = progress.reduce().report()
    assert report["verified"] == 1
    branch = report["active_star"]["planet"]
    assert branch["policy_label"] == "bounded_no_dip_assumption"
    assert branch["observation_limit_days"] == 5000
    assert branch["transport_verified"] and not branch["scientific_verified"]


@pytest.mark.parametrize("label", ["stellar_color", "stellar_classification"])
def test_missing_stellar_choice_does_not_become_completed_by_a_transport_flag(label):
    with pytest.raises(ValidationError, match="selection value"):
        StageRecorded(
            star_id="ALPHA",
            stage=label,
            evidence=StageEvidence(
                decision=Evidence(
                    source_sha256=digest(1), provenance="reference_prediction", transport_verified=True
                )
            ),
        )


def test_unknown_planet_class_is_not_terrestrial_by_default():
    progress = stage(collected(fresh()), "ALPHA", "planet", outcome="planet")
    with pytest.raises(ProgressError, match="explicit_terrestrial"):
        stage(progress, "ALPHA", "habitability", outcome="not_habitable")


@pytest.mark.parametrize("value", [False, 0, 1, "true", "yes"])
def test_task_receipt_needs_literal_boolean_not_truthiness(value):
    with pytest.raises(ValidationError):
        TaskReceipt(
            star_id="ALPHA",
            star_revision=1,
            source_sha256=digest(1),
            authority="visible_workflow_readback",
            task_completed=value,
        )


def test_concurrent_journal_writers_cannot_both_reserve_one_operation(tmp_path):
    journal = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="preview-001").create()
    for record in complete_star(fresh()).records:
        journal.append(record.payload)
    reservation = reserve(journal.load(), "save_star", star_id="ALPHA").records[-1].payload

    def attempt(action_id):
        other = ProjectJournal(tmp_path, project_id="habworlds", attempt_id="preview-001")
        try:
            other.append(reservation.model_copy(update={"action_id": action_id}))
            return "reserved"
        except ProgressError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, ["worker-one", "worker-two"]))
    assert sorted(outcomes) == ["blocked", "reserved"]
    assert len(journal.load().reduce().pending) == 1


def test_explicit_no_planet_applicability_needs_no_fake_habitability_transport():
    progress = complete_star(fresh())
    evidence = StageEvidence(
        decision=Evidence(source_sha256=digest("explicit-na-branch"), provenance="reference_prediction"),
        outcome="not_applicable",
        applicability_reason="no_planet",
        branch_applicability_verified=True,
    )
    progress = progress.append(StageRecorded(star_id="ALPHA", stage="habitability", evidence=evidence))
    progress = confirm_task(progress, "ALPHA")
    star = progress.reduce().stars["ALPHA"]
    assert star.task_completed
    assert not star.habitability.transport_verified
    report = star.habitability.report()
    assert report["branch_applicability_verified"]
    assert not report["transport_verified"] and not report["scientific_verified"]
    assert (
        ProjectProgress.model_validate_json(progress.model_dump_json()).reduce().stars["ALPHA"].task_completed
    )


def test_explicit_nonterrestrial_applicability_needs_verified_class_and_no_fake_transport():
    progress = complete_star(
        fresh(), planet="planet", habitability="not_applicable", planet_class="ice_giant"
    )
    evidence = StageEvidence(
        decision=Evidence(source_sha256=digest("ice-giant-na"), provenance="reference_prediction"),
        outcome="not_applicable",
        applicability_reason="non_terrestrial_planet",
        branch_applicability_verified=True,
    )
    progress = progress.append(StageRecorded(star_id="ALPHA", stage="habitability", evidence=evidence))
    assert confirm_task(progress, "ALPHA").reduce().stars["ALPHA"].task_completed


@pytest.mark.parametrize("bad", ["unverified_no", "scientific_claim", "wrong_reason", "cross_star"])
def test_applicability_flag_cannot_substitute_missing_or_wrong_planet_evidence(bad):
    progress = complete_star(fresh())
    if bad == "unverified_no":
        progress = stage(progress, "ALPHA", "planet", outcome="no_planet", transport=False, suffix="new")
    elif bad == "cross_star":
        progress = complete_star(progress, "BETA")
    decision = Evidence(
        source_sha256=digest(("BETA", "planet", "")) if bad == "cross_star" else digest("applicability"),
        provenance="course_feedback" if bad == "scientific_claim" else "reference_prediction",
        scientific_verified=bad == "scientific_claim",
    )
    evidence = StageEvidence(
        decision=decision,
        outcome="not_applicable",
        branch_applicability_verified=True,
        applicability_reason="non_terrestrial_planet" if bad == "wrong_reason" else "no_planet",
    )
    with pytest.raises(ProgressError):
        progress.append(StageRecorded(star_id="ALPHA", stage="habitability", evidence=evidence))


def test_applicability_flag_is_strict_and_only_for_an_explicit_not_applicable_branch():
    base = {"decision": {"source_sha256": digest(1), "provenance": "reference_prediction"}}
    for value in ("yes", 1):
        with pytest.raises(ValidationError):
            StageEvidence(**base, outcome="not_applicable", branch_applicability_verified=value)
    with pytest.raises(ValidationError):
        StageEvidence(**base, outcome="habitable", branch_applicability_verified=True)
