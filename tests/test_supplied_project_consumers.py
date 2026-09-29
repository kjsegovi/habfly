"""Offline supplied-mode dispatch; no native/model/transfer evaluation runs.

The upstream numerical validator seam is explicit in the reused importer
fixtures. Real readbacks, exact mode dispatch, inventory, journal plans and
completed-owner proof are not mocked. These are not course acceptance tests.
"""

from types import SimpleNamespace

import pytest
from test_project_evidence import inventory, read, sha, write
from test_project_positive_evidence import offline as positive_offline  # noqa: F401
from test_project_terrestrial_evidence import offline as terrestrial_offline  # noqa: F401
from test_supplied_browser_workflows import _convert

import habfly.browser_project_campaign_steps as campaign
import habfly.browser_project_inventory_steps as collection
import habfly.browser_project_next_star_steps as transition
from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_workflow import _Evidence
from habfly.contracts import RuntimeEvent
from habfly.project_progress import ProgressError
from habfly.supplied_browser_modes import NON_MAIN_CLASSES, POSITIVE_SUPPLIED_MODE, TERRESTRIAL_SUPPLIED_MODE


@pytest.fixture(params=[False, True], ids=["positive-supplied", "terrestrial-supplied"])
def branch(request):
    terrestrial = request.param
    fixture = request.getfixturevalue("terrestrial_offline" if terrestrial else "positive_offline")

    def make(actual_class="white_dwarf", extra_collected=False):
        data = _convert(fixture, terrestrial_branch=terrestrial, actual_class=actual_class)
        journal, history, _, workflow, _, calls = data
        receipt = read(workflow / "confirmed.json")
        star = receipt["star"]
        names = [star, "Beta"] if extra_collected else [star]
        inv = inventory(history, names, "owner", classes={star: actual_class})
        mode = TERRESTRIAL_SUPPLIED_MODE if terrestrial else POSITIVE_SUPPLIED_MODE
        phase, helper = transition.WORKFLOWS[mode]
        importer = (
            helper.import_verified_terrestrial if terrestrial else helper.import_verified_positive_planet
        )
        result = importer(journal, history, inv, workflow)
        owner = history / "completed-owner"
        report = {
            "mode": "bounded_single_star_project_runtime",
            "status": "completed",
            "phase": phase,
            "star": star,
            "finished": True,
            "task_completed": True,
            "project_completed": False,
            "failure_reason": None,
            "event_forwarding_failed": False,
            "project_progress": result["progress"],
        }
        write(owner / "report.json", report)
        event = RuntimeEvent(
            event="episode_summary",
            sequence=0,
            run_id="offline-supplied-owner",
            payload={**report, "completed": True},
        )
        (owner / "events.jsonl").write_text(event.model_dump_json() + "\n")
        write(owner / "workflow-import.json", result)
        return SimpleNamespace(
            journal=journal,
            history=history,
            workflow=workflow,
            mode=mode,
            names=names,
            star=star,
            owner=owner,
            helper=helper,
            calls=calls,
            inventory=inv,
        )

    return make


@pytest.mark.parametrize("actual_class", NON_MAIN_CLASSES)
def test_new_modes_pin_inventory_and_revalidate_exact_import_before_next_star(branch, actual_class):
    item = branch(actual_class)
    before = item.journal.path.read_bytes()
    raw = (item.workflow / "confirmed.json").read_bytes()
    receipt, capture = collection._workflow(
        _Evidence(item.history), item.workflow, sha(item.workflow / "confirmed.json"), item.star
    )
    assert receipt["mode"] == item.mode and receipt["classification"] == actual_class
    assert collection.project_view(capture)["section"] == collection.MODES[item.mode][-1]
    state, previous, _ = transition._completed_owner(
        _Evidence(item.history), item.journal, item.owner, item.names
    )
    assert previous.task_completed and state.report()["verified"] == 1
    assert item.calls[-1]["supplied_inputs"] is True
    assert item.journal.path.read_bytes() == before
    assert (item.workflow / "confirmed.json").read_bytes() == raw
    assert not state.report()["project_completed"] and not state.report()["submitted"]


def test_campaign_and_finalizer_shared_verifier_reuses_same_strict_mode(branch):
    item = branch()
    before = item.journal.path.read_bytes()
    state, _, sources = campaign._verified_owner(
        _Evidence(item.history), item.journal, item.owner, item.names, item.star
    )
    assert campaign.WORKFLOWS is transition.WORKFLOWS
    assert sources["workflow_sha256"] == sha(item.workflow / "confirmed.json")
    assert item.calls[-1]["supplied_inputs"] is True
    assert all(star.task_completed for star in state.stars.values())
    assert item.journal.path.read_bytes() == before and not state.report()["project_completed"]


def test_readiness_still_requires_every_collected_star_verified(branch):
    item = branch(extra_collected=True)
    before = item.journal.path.read_bytes()
    with pytest.raises(BrowserSafetyStop, match="incomplete_or_unexpected_collection"):
        campaign._verified_owner(_Evidence(item.history), item.journal, item.owner, item.names, item.star)
    assert item.journal.path.read_bytes() == before


@pytest.mark.parametrize("consumer", ["next", "campaign"])
def test_strict_supplied_provenance_refusal_is_not_aliased_or_swallowed(branch, monkeypatch, consumer):
    item = branch()
    before = item.journal.path.read_bytes()
    calls = []

    def reject(book, **directories):
        calls.append(directories)
        raise BrowserSafetyStop("required_supplied_transfer_gate_changed")

    monkeypatch.setattr(item.helper, "_load_sources", reject)
    with pytest.raises(BrowserSafetyStop, match="required_supplied_transfer_gate_changed"):
        if consumer == "next":
            transition._completed_owner(_Evidence(item.history), item.journal, item.owner, item.names)
        else:
            campaign._verified_owner(_Evidence(item.history), item.journal, item.owner, item.names, item.star)
    assert len(calls) == 1 and calls[0]["supplied_inputs"] is True
    assert item.journal.path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["source", "mode", "phase", "capture"])
def test_changed_source_or_mixed_legacy_mode_stops_before_transition(branch, mutation):
    item = branch()
    before = item.journal.path.read_bytes()
    receipt = read(item.workflow / "confirmed.json")
    if mutation == "source":
        path = item.history / next(iter(receipt["source_sha256"]))
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation == "capture":
        path = item.workflow / "stellar-readback/verified/observation.json"
        value = read(path)
        value["frames"][0]["text"] += " altered public data"
        write(path, value)
    elif mutation == "mode":
        receipt["mode"] = item.helper.MODE
        write(item.workflow / "confirmed.json", receipt)
    else:
        report = read(item.owner / "report.json")
        report["phase"] = "verified_no_planet"
        write(item.owner / "report.json", report)
        event = RuntimeEvent(
            event="episode_summary",
            sequence=0,
            run_id="offline-supplied-owner",
            payload={**report, "completed": True},
        )
        (item.owner / "events.jsonl").write_text(event.model_dump_json() + "\n")
    with pytest.raises((BrowserSafetyStop, ProgressError)):
        transition._completed_owner(_Evidence(item.history), item.journal, item.owner, item.names)
    assert item.journal.path.read_bytes() == before


def test_unknown_mode_never_enters_inventory_navigation(branch):
    item = branch()
    receipt = read(item.workflow / "confirmed.json")
    receipt["mode"] += "_unrecognized"
    write(item.workflow / "confirmed.json", receipt)
    with pytest.raises(BrowserSafetyStop, match="unsupported_completed_workflow"):
        collection._workflow(
            _Evidence(item.history), item.workflow, sha(item.workflow / "confirmed.json"), item.star
        )
