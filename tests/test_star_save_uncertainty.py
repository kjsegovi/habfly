"""Owned failed-Save diagnostics only; no browser, retries or canonical promotion."""
# ruff: noqa: F401,F811

import hashlib
import json
from pathlib import Path

import pytest
from test_browser_star_session import advance_to, complete, create, rig

import habfly.browser_star_session as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_save import MODE
from habfly.project_progress import ProjectJournal

FAILURE = "no_planet_save_reserved_phase_timeout"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failed_records(session, *, reserved=True, attempted=True, dispatch=True, ack=True):
    directory = session.output / "save"
    directory.mkdir()
    source = session.output / "window/choice/confirmed.json"
    key = hashlib.sha256(session.star.casefold().encode()).hexdigest()
    claim = session.history / "no-planet-save-reservations" / f"{key}.json"
    if reserved:
        intent = {
            "schema_version": 1,
            "mode": MODE,
            "star": session.star,
            "kind": "CLICK",
            "visible_label": "Save",
            "max_save_clicks": 1,
            "choice_path": str(source.relative_to(session.history)),
            "choice_sha256": sha(source),
            "output": str(directory.relative_to(session.history)),
            "task_completed": False,
            "automatic_retry": False,
        }
        write(claim, intent)
        write(directory / "reserved.json", intent)
    write(
        directory / "stopped.json",
        {
            "mode": MODE,
            "reason": FAILURE,
            "reservation_created": reserved,
            "save_may_have_occurred": attempted,
            "automatic_retry": False,
            "task_completed": False,
        },
    )
    if dispatch:
        write(directory / "dispatch.json", {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1})
    if ack:
        write(
            directory / "acknowledgement.json",
            {
                "visible_text": "Data saved",
                "source": "fully_exposed_footer_text",
                "notice_was_already_present": False,
            },
        )
    return directory, claim


def install_failure(rig, monkeypatch, *, records=None, mutate=lambda *_: None):
    session = create(rig)
    calls = []

    def fail(*args, **kwargs):
        assert session.artifact_paths["save_dir"] == "run/save"
        assert args[2] == session.output / "save" and kwargs["settle_reserved_notice"] is True
        calls.append("save")
        if records is not None:
            directory, claim = failed_records(session, **records)
            mutate(directory, claim)
        raise BrowserSafetyStop(FAILURE)

    monkeypatch.setattr(module, "save_no_planet_work", fail)
    return session, calls


def assert_failed(session):
    state = session.state()
    assert session.phase == "stopped" and session.failure == FAILURE
    assert state["artifact_paths"]["save_dir"] == "run/save"
    assert state["task_completed"] is state["project_completed"] is False
    assert state["save_outcome"]["final_readback_verified"] is False
    assert state["save_outcome"]["canonical_receipt"] is False
    assert state["save_outcome"]["automatic_retry"] is False
    assert state["save_outcome"]["task_completed"] is state["save_outcome"]["project_completed"] is False
    assert session.report == json.loads((session.output / "report.json").read_bytes())
    return state["save_outcome"]


def test_late_final_readback_reports_uncertain_save_but_not_canonical_success(rig, monkeypatch):
    journal = ProjectJournal(rig.root, project_id="fixture", attempt_id="attempt").create()
    before = journal.path.read_bytes()
    session, calls = install_failure(rig, monkeypatch, records={})
    complete(session)
    outcome = assert_failed(session)
    assert session.state()["save_outcome_uncertain"] is True
    assert outcome["evidence_status"] == "recorded_stop"
    assert outcome["reservation_retained"] is outcome["dispatch_recorded"] is True
    assert outcome["acknowledgement_recorded"] is True
    assert len(outcome["source_sha256"]) == 6
    for name, digest in outcome["source_sha256"].items():
        assert sha(rig.root / name) == digest
    saved = {name: (rig.root / name).read_bytes() for name in outcome["source_sha256"]}
    for _ in range(3):
        session.advance()
        session.abort()
        session.close()
    assert calls == ["save"] and not any(call[0] == "verify" for call in rig.calls)
    assert saved == {name: (rig.root / name).read_bytes() for name in saved}
    assert journal.path.read_bytes() == before and journal.load().reduce().report()["uncertain_actions"] == []


@pytest.mark.parametrize("reserved", [False, True])
def test_explicit_predispatch_stop_is_distinct_from_uncertain_write(rig, monkeypatch, reserved):
    session, calls = install_failure(
        rig,
        monkeypatch,
        records={
            "reserved": reserved,
            "attempted": False,
            "dispatch": False,
            "ack": False,
        },
    )
    complete(session)
    outcome = assert_failed(session)
    assert session.state()["save_outcome_uncertain"] is False
    assert outcome["reservation_retained"] is reserved
    assert outcome["dispatch_recorded"] is outcome["acknowledgement_recorded"] is False
    assert calls == ["save"]


@pytest.mark.parametrize("dispatch", [False, True])
def test_attempt_without_ack_is_uncertain_not_confirmed_click(rig, monkeypatch, dispatch):
    session, _ = install_failure(rig, monkeypatch, records={"dispatch": dispatch, "ack": False})
    complete(session)
    outcome = assert_failed(session)
    assert outcome["save_outcome_uncertain"] is True
    assert outcome["dispatch_recorded"] is dispatch and outcome["acknowledgement_recorded"] is False
    assert "save_click_delivered" not in outcome


def test_fake_save_without_records_stays_unknown_and_keeps_intended_path(rig, monkeypatch):
    session, calls = install_failure(rig, monkeypatch)
    complete(session)
    outcome = assert_failed(session)
    assert session.state()["save_outcome_uncertain"] is None
    assert outcome["evidence_status"] == "unavailable_or_invalid"
    assert outcome["source_sha256"] == {} and outcome["reservation_retained"] is None
    assert calls == ["save"]


@pytest.mark.parametrize(
    "change",
    [
        "wrong_star",
        "wrong_output",
        "changed_claim",
        "changed_choice",
        "missing_choice",
        "bool_count",
        "bool_attempt",
        "bool_ack",
        "bad_dispatch",
        "bad_json",
        "duplicate_json",
        "nonfinite",
        "oversized",
        "invalidated",
        "confirmed",
        "symlink",
        "directory_symlink",
        "contradictory",
    ],
)
def test_untrusted_or_mixed_failure_evidence_remains_unknown(rig, monkeypatch, change):
    def mutate(directory, claim):
        path = directory / "reserved.json"
        if change in {"wrong_star", "wrong_output", "bool_count"}:
            value = json.loads(path.read_bytes())
            key, data = {
                "wrong_star": ("star", "Other"),
                "wrong_output": ("output", "other/save"),
                "bool_count": ("max_save_clicks", True),
            }[change]
            value[key] = data
            write(path, value)
            write(claim, value)
        elif change == "changed_claim":
            write(claim, {"star": "Other"})
        elif change == "changed_choice":
            write(directory.parent / "window/choice/confirmed.json", {"changed": True})
        elif change == "missing_choice":
            (directory.parent / "window/choice/confirmed.json").unlink()
        elif change in {"bool_attempt", "contradictory"}:
            path = directory / "stopped.json"
            value = json.loads(path.read_bytes())
            value["save_may_have_occurred"] = 1 if change == "bool_attempt" else False
            write(path, value)
        elif change == "bool_ack":
            path = directory / "acknowledgement.json"
            value = json.loads(path.read_bytes())
            value["notice_was_already_present"] = 0
            write(path, value)
        elif change == "bad_dispatch":
            write(directory / "dispatch.json", {"kind": "CLICK", "visible_label": "Save", "max_clicks": True})
        elif change in {"bad_json", "duplicate_json", "nonfinite", "oversized"}:
            (directory / "stopped.json").write_text(
                {
                    "bad_json": "private driver secret",
                    "duplicate_json": '{"mode":1,"mode":2}',
                    "nonfinite": '{"value":NaN}',
                    "oversized": "x" * 64_001,
                }[change]
            )
        elif change in {"invalidated", "confirmed"}:
            write(directory / f"{change}.json", {})
        elif change == "symlink":
            path = directory / "stopped.json"
            copied = directory.parent / "other-stopped.json"
            path.rename(copied)
            path.symlink_to(copied)
        else:
            moved = directory.with_name("other-save")
            directory.rename(moved)
            directory.symlink_to(moved, target_is_directory=True)

    session, _ = install_failure(rig, monkeypatch, records={}, mutate=mutate)
    complete(session)
    outcome = assert_failed(session)
    assert outcome["evidence_status"] == "unavailable_or_invalid"
    assert outcome["save_outcome_uncertain"] is None and outcome["source_sha256"] == {}
    assert "private driver secret" not in json.dumps(session.report)


def test_diagnostic_read_failure_cannot_replace_original_save_failure(rig, monkeypatch):
    session, _ = install_failure(rig, monkeypatch, records={})
    original = Path.read_bytes

    def unavailable(path):
        if path == session.output / "save/stopped.json":
            raise OSError("secret driver text must stay private")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", unavailable)
    complete(session)
    assert assert_failed(session)["save_outcome_uncertain"] is None
    assert "secret driver" not in json.dumps(session.report)


def test_even_unexpected_diagnostic_exception_preserves_original_stop(rig, monkeypatch):
    session, _ = install_failure(rig, monkeypatch)
    monkeypatch.setattr(session, "_record_failed_save", lambda: (_ for _ in ()).throw(OSError("private")))
    complete(session)
    assert session.failure == FAILURE and session.phase == "stopped"
    assert session.report["artifact_paths"]["save_dir"] == "run/save"
    assert not session.report["task_completed"]


def test_success_and_pre_save_stops_keep_legacy_diagnostic_omission(rig):
    session = create(rig)
    assert "save_outcome_uncertain" not in session.state()
    assert "save_outcome" not in session.state()
    complete(session)
    assert session.phase == "verified_no_planet"
    assert "save_outcome_uncertain" not in session.report and "save_outcome" not in session.report
