"""Injected failure records only: no browser, transport, or scientific claims."""
# ruff: noqa: F811

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_project_steps import create, rig, sha, write  # noqa: F401

from habfly.project_wire import compact_project_event


@pytest.fixture(params=["planet", "habitability"])
def failed_save(rig, request):
    terrestrial = request.param == "habitability"
    owner = create(rig)
    owner.start()
    parent = owner.output / ("terrestrial" if terrestrial else "positive-finalization")
    directory = parent / "save"
    child = SimpleNamespace(output=parent, finished=True, save_receipt=None, book=SimpleNamespace(hashes={}))
    setattr(owner, "terrestrial_component" if terrestrial else "positive_component", child)
    name = "terrestrial" if terrestrial else "ice_giant"
    class_path = owner.output / "planet-class/confirmed.json"
    write(class_path, {"star": "Alpha", "value": name, "readback_verified": True})
    owner._planet_class_request = {"name": name}
    owner._planet_class_source = {
        "directory": str(class_path.parent.relative_to(rig.root)),
        "sha256": sha(class_path),
    }
    intent = {
        "kind": "CLICK",
        "visible_label": "Save",
        "star": "ALPHA",
        "max_save_clicks": 1,
        "task_completed": False,
        "automatic_retry": False,
    }
    sources = {}
    if terrestrial:
        for stage in ("phase", "choice"):
            path = parent / stage / "confirmed.json"
            write(path, {"star": "Alpha", "readback_verified": True, "choice": "not_habitable"})
            sources[str(path.relative_to(rig.root))] = sha(path)
        child.book.hashes.update(sources)
        intent.update(
            schema_version=1,
            mode="terrestrial_habitability_save",
            choice="not_habitable",
            output=str(directory.relative_to(rig.root)),
            phase_dir=str((parent / "phase").relative_to(rig.root)),
            choice_dir=str((parent / "choice").relative_to(rig.root)),
            source_sha256=sources,
        )
        canonical = deepcopy(intent)
    else:
        canonical = {
            "schema_version": 1,
            "mode": "cooperative_positive_planet_finalization",
            "star": "Alpha",
            "planet_class": name,
            "planet_class_sha256": sha(class_path),
            "output": str(parent.relative_to(rig.root)),
            "save_output": str(directory.relative_to(rig.root)),
            "native_intent": deepcopy(intent),
            "maximum_save_dispatches": 1,
            "automatic_retry": False,
            "task_completed": False,
        }
    key = hashlib.sha256(owner.star.casefold().encode()).hexdigest() + ".json"
    claim = (
        rig.root
        / ("habitability-save-reservations" if terrestrial else "positive-finalization-reservations")
        / key
    )
    write(claim, canonical)
    write(directory / "reserved.json", intent)
    write(
        directory / "stopped.json",
        {
            **({"mode": "terrestrial_habitability_save"} if terrestrial else {}),
            "reason": "injected_reserved_phase_timeout",
            "reservation_created": True,
            "save_may_have_occurred": True,
            "automatic_retry": False,
            "task_completed": False,
        },
    )
    write(directory / "dispatch.json", {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1})
    write(
        directory / "acknowledgement.json",
        {
            "visible_text": "Data saved",
            "source": "fully_exposed_footer_text",
            "notice_was_already_present": False,
        },
    )
    return SimpleNamespace(
        owner=owner,
        rig=rig,
        child=child,
        directory=directory,
        claim=claim,
        intent=intent,
        canonical=canonical,
        terrestrial=terrestrial,
        class_path=class_path,
    )


def stop(item):
    before = item.rig.journal.path.read_bytes()
    item.owner._stop("original_causal_failure")
    state = item.owner.state()
    assert item.owner.failure == "original_causal_failure"
    assert state["task_completed"] is state["project_completed"] is False
    assert item.rig.journal.path.read_bytes() == before
    assert state["save_outcome"]["final_readback_verified"] is False
    assert state["save_outcome"]["canonical_receipt"] is False
    assert state["save_outcome"]["automatic_retry"] is False
    return state


def test_real_diagnostic_contract_reaches_terminal_report_and_compact_wire(failed_save):
    item = failed_save
    before = {p: p.read_bytes() for p in (item.claim, *item.directory.glob("*.json"), item.class_path)}
    state = stop(item)
    summary = state["save_outcome"]
    assert state["save_outcome_uncertain"] is True
    assert summary["evidence_status"] == "recorded_stop"
    assert (
        summary["reservation_retained"]
        is summary["dispatch_recorded"]
        is summary["acknowledgement_recorded"]
        is True
    )
    assert summary["save_surface"] == ("habitability" if item.terrestrial else "planet")
    assert summary["source_sha256"]
    for name, digest in summary["source_sha256"].items():
        assert sha(item.rig.root / name) == digest
    assert item.owner.report["save_outcome"] == summary
    wire = compact_project_event("state", state)
    assert wire["save_outcome"] == summary and wire["save_outcome_uncertain"] is True
    assert wire["project_wire"]["evidence_receipt"] is False
    for path, raw in before.items():
        assert path.read_bytes() == raw
    state["save_outcome"]["task_completed"] = True
    assert item.owner.state()["save_outcome"]["task_completed"] is False


@pytest.mark.parametrize("durable_marker", [False, True])
def test_budget_rejection_preserves_preclick_marker_uncertainty(failed_save, durable_marker):
    item = failed_save
    (item.directory / "acknowledgement.json").unlink()
    if not durable_marker:
        (item.directory / "dispatch.json").unlink()
    path = item.directory / "stopped.json"
    value = json.loads(path.read_bytes())
    value.update(reason="reserved_save_insufficient_readback_budget", save_may_have_occurred=durable_marker)
    write(path, value)
    write(
        item.directory / "dispatch-budget-rejected.json",
        {
            "estimate_not_guarantee": True,
            "deadline_extended": False,
            "save_click_dispatched": False,
            "automatic_retry": False,
            "task_completed": False,
        },
    )
    state = stop(item)
    assert state["save_outcome_uncertain"] is durable_marker
    assert state["save_outcome"]["dispatch_recorded"] is durable_marker
    assert state["save_outcome"]["acknowledgement_recorded"] is False
    assert state["save_outcome"]["reservation_retained"] is True


@pytest.mark.parametrize("contradiction", ["acknowledgement", "missing_marker", "false_uncertainty"])
def test_budget_rejection_with_contradictory_evidence_stays_unknown(failed_save, contradiction):
    item = failed_save
    if contradiction != "acknowledgement":
        (item.directory / "acknowledgement.json").unlink()
    if contradiction == "missing_marker":
        (item.directory / "dispatch.json").unlink()
    if contradiction == "false_uncertainty":
        path = item.directory / "stopped.json"
        value = json.loads(path.read_bytes())
        value["save_may_have_occurred"] = False
        write(path, value)
    write(
        item.directory / "dispatch-budget-rejected.json",
        {
            "estimate_not_guarantee": True,
            "deadline_extended": False,
            "save_click_dispatched": False,
            "automatic_retry": False,
            "task_completed": False,
        },
    )
    state = stop(item)
    assert state["save_outcome_uncertain"] is None
    assert state["save_outcome"]["evidence_status"] == "unavailable_or_invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        "other_star",
        "other_output",
        "class_changed",
        "class_pin_changed",
        "boolean_count",
        "duplicate_json",
        "nonfinite",
        "oversized",
        "symlink",
        "directory",
        "missing_stopped",
        "false_with_dispatch",
        "ack_without_dispatch",
        "changed_ack",
        "conflicting_confirmation",
    ],
)
def test_invalid_failure_proof_is_unknown_and_cannot_mask_original_stop(failed_save, mutation):
    item = failed_save
    if mutation in {"other_star", "other_output", "boolean_count"}:
        value = deepcopy(item.canonical)
        key = (
            "star"
            if mutation == "other_star"
            else "output"
            if mutation == "other_output"
            else ("max_save_clicks" if item.terrestrial else "maximum_save_dispatches")
        )
        value[key] = True if mutation == "boolean_count" else "wrong"
        write(item.claim, value)
    elif mutation == "class_changed":
        write(item.class_path, {"star": "OTHER", "value": "gas_giant"})
    elif mutation == "class_pin_changed":
        item.owner._planet_class_source["sha256"] = "0" * 64
    elif mutation in {"duplicate_json", "nonfinite", "oversized"}:
        raw = (
            '{"task_completed":false,"task_completed":false}'
            if mutation == "duplicate_json"
            else ('{"value":NaN}' if mutation == "nonfinite" else " " * 64_001)
        )
        (item.directory / "stopped.json").write_text(raw)
    elif mutation in {"symlink", "directory"}:
        path = item.directory / "dispatch.json"
        raw = path.read_bytes()
        path.unlink()
        if mutation == "symlink":
            target = item.rig.root / "other-dispatch.json"
            target.write_bytes(raw)
            path.symlink_to(target)
        else:
            path.mkdir()
    elif mutation in {"missing_stopped", "ack_without_dispatch"}:
        (item.directory / ("stopped.json" if mutation == "missing_stopped" else "dispatch.json")).unlink()
    elif mutation == "false_with_dispatch":
        path = item.directory / "stopped.json"
        value = json.loads(path.read_bytes())
        value["save_may_have_occurred"] = False
        write(path, value)
    elif mutation == "changed_ack":
        write(
            item.directory / "acknowledgement.json",
            {"visible_text": "Data saved", "notice_was_already_present": True},
        )
    else:
        write(item.directory / "confirmed.json", {"task_completed": True})
    state = stop(item)
    assert state["save_outcome_uncertain"] is None
    assert state["save_outcome"]["evidence_status"] == "unavailable_or_invalid"
    assert state["save_outcome"]["source_sha256"] == {}


@pytest.mark.parametrize("source", ["phase", "choice"])
def test_terrestrial_requires_current_owned_phase_and_choice_pins(failed_save, source):
    if not failed_save.terrestrial:
        pytest.skip("Terrestrial-only source contract")
    path = failed_save.directory.parent / source / "confirmed.json"
    failed_save.child.book.hashes[str(path.relative_to(failed_save.rig.root))] = "0" * 64
    assert stop(failed_save)["save_outcome_uncertain"] is None


def test_confirmed_save_followed_by_workflow_failure_is_not_uncertain_save(failed_save):
    item = failed_save
    (item.directory / "stopped.json").unlink()
    receipt = {
        **item.intent,
        "save_click_delivered": True,
        "data_saved_notice_observed": True,
        "answers_unchanged": True,
    }
    write(item.directory / "confirmed.json", receipt)
    item.child.save_receipt = deepcopy(receipt)
    item.owner._stop("workflow_readback_failed")
    assert "save_outcome" not in item.owner.report
    assert "save_outcome_uncertain" not in item.owner.report
    assert not item.owner.report["task_completed"]


def test_no_save_files_means_no_invented_failed_save(rig):
    owner = create(rig)
    owner.start()
    owner.positive_component = SimpleNamespace(output=owner.output / "positive-finalization", finished=True)
    owner._stop("before_save")
    assert "save_outcome" not in owner.report


def test_diagnostic_failure_does_not_mask_original_stop(rig, monkeypatch):
    owner = create(rig)
    owner.start()

    def broken_reader():
        raise OSError("untrusted filesystem exception text")

    monkeypatch.setattr(owner, "_record_failed_continuation_save", broken_reader)
    owner._stop("original_causal_failure")
    assert owner.report["failure_reason"] == "original_causal_failure"
    assert owner.report["task_completed"] is False
    assert "untrusted filesystem" not in json.dumps(owner.report)
