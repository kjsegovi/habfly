"""Offline admission/scheduling regression; no browser or model is started.

The upstream scientific source is an explicitly synthetic seam, as in the
existing coordinator rig. Native source rebuilding is covered separately by
the gated autosave inventory Chromium test.
"""
# ruff: noqa: F811

import json

import pytest
from test_browser_project_inventory_steps import rig, sha  # noqa: F401

import habfly.browser_project_inventory_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_autosave import ACKNOWLEDGEMENT_SOURCE, autosave_flags, autosave_workflow_flags


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def autosave_source(rig, mode="no_planet_visible_workflow_readback"):
    directory = (
        rig.directory
        if mode == "no_planet_visible_workflow_readback"
        else rig.source(mode=mode, name="other-workflow")
    )
    workflow = read(directory / "confirmed.json")
    branch = (
        "no_planet"
        if mode.startswith("no_planet")
        else "positive"
        if mode.startswith("positive")
        else "terrestrial"
    )
    source_path = rig.root / "readback/confirmed.json"
    upstream = next(iter(workflow["source_sha256"]))
    source = {
        "schema_version": 1,
        **autosave_flags(),
        "branch": branch,
        "star": workflow["star"],
        "output": "readback",
        "source_sha256": dict(workflow["source_sha256"]),
        "before_sha256": "a" * 64,
        "after_sha256": "b" * 64,
    }
    if branch == "no_planet":
        source.update(choice_path=upstream, choice_sha256=workflow["source_sha256"][upstream])
    write(source_path, source)
    workflow.update(
        **autosave_workflow_flags(),
        save_acknowledgement_verified=False,
        source_save_click_delivered=False,
        source_save_resumed_same_intent=False,
        source_save_continuation_stopped_predispatch=False,
        save_acknowledgement_source=ACKNOWLEDGEMENT_SOURCE,
    )
    workflow["source_sha256"]["readback/confirmed.json"] = sha(source_path)
    write(directory / "confirmed.json", workflow)
    return directory, source_path


def create(rig, directory, **kwargs):
    return rig.create(workflow_dir=directory, workflow_sha256=sha(directory / "confirmed.json"), **kwargs)


def repin(directory, source):
    value = read(directory / "confirmed.json")
    value["source_sha256"]["readback/confirmed.json"] = sha(source)
    write(directory / "confirmed.json", value)


@pytest.mark.parametrize("mode", list(module.MODES))
def test_every_autosave_branch_enters_real_coordinator_and_bounded_navigation(rig, mode):
    directory, _ = autosave_source(rig, mode)
    owner = create(rig, directory)
    assert rig.flags.calls == []
    for expected in ("to_stellar", "verify_inventory", "inventory_verified"):
        owner.advance()
        assert owner.phase == expected
    assert owner.finished and owner.status == "completed"
    assert owner.navigation_attempts == owner.navigation_clicks == 2
    assert owner.advances == 3
    assert owner.report["task_completed"] is owner.report["project_completed"] is False
    assert owner.source["persistence_verified"] is False
    assert owner.source["save_acknowledgement_verified"] is False


@pytest.mark.parametrize("mode", list(module.MODES))
def test_legacy_explicit_acknowledgement_admission_is_unchanged(rig, mode):
    directory = rig.directory if mode.startswith("no_planet") else rig.source(mode=mode, name="old")
    owner = create(rig, directory)
    assert owner.source == read(directory / "confirmed.json")
    assert owner.source["save_acknowledgement_verified"] is True
    assert not rig.flags.calls
    owner.close()


@pytest.mark.parametrize(
    "key,value",
    [
        ("save_acknowledgement_verified", True),
        ("save_acknowledgement_verified", 0),
        ("source_save_click_delivered", True),
        ("source_save_click_delivered", 0),
        ("source_save_resumed_same_intent", True),
        ("source_save_continuation_stopped_predispatch", True),
        ("save_acknowledgement_source", "Data saved"),
        ("save_authority", "persistence"),
        ("autosave_readback_mode", "unknown"),
        ("persistence_verified", True),
        ("persistence_verified", 0),
        ("save_strategy", "explicit"),
        ("save_strategy", None),
    ],
)
def test_mixed_or_malformed_workflow_authority_never_navigates(rig, key, value):
    directory, _ = autosave_source(rig)
    receipt = read(directory / "confirmed.json")
    receipt[key] = value
    write(directory / "confirmed.json", receipt)
    with pytest.raises(BrowserSafetyStop):
        create(rig, directory)
    assert not rig.flags.calls and not (rig.root / "run").exists()


def test_false_ack_without_new_metadata_is_not_a_legacy_success(rig):
    receipt = read(rig.directory / "confirmed.json")
    receipt["save_acknowledgement_verified"] = False
    write(rig.directory / "confirmed.json", receipt)
    with pytest.raises(BrowserSafetyStop, match="unsupported_completed_workflow"):
        rig.create()
    assert not rig.flags.calls


@pytest.mark.parametrize(
    "key,value",
    [
        ("branch", "positive"),
        ("star", "Other"),
        ("output", "elsewhere"),
        ("save_click_delivered", True),
        ("save_click_delivered", 0),
        ("save_acknowledgement_verified", True),
        ("visible_readback_verified", 1),
        ("persistence_verified", True),
        ("browser_actions", False),
        ("task_completed", True),
        ("schema_version", True),
        ("mode", "unknown"),
        ("source_sha256", {"workflow/upstream.json": "f" * 64}),
    ],
)
def test_readback_receipt_must_match_exact_branch_source_and_typed_authority(rig, key, value):
    directory, source = autosave_source(rig)
    receipt = read(source)
    receipt[key] = value
    write(source, receipt)
    repin(directory, source)
    with pytest.raises(BrowserSafetyStop):
        create(rig, directory)
    assert not rig.flags.calls and not (rig.root / "run").exists()


@pytest.mark.parametrize("change", ["missing", "multiple", "stopped", "unhashed_mutation", "symlink"])
def test_readback_evidence_cannot_be_missing_ambiguous_failed_or_rebound(rig, change):
    directory, source = autosave_source(rig)
    receipt = read(directory / "confirmed.json")
    if change == "missing":
        del receipt["source_sha256"]["readback/confirmed.json"]
        write(directory / "confirmed.json", receipt)
    elif change == "multiple":
        duplicate = rig.root / "second/confirmed.json"
        write(duplicate, {**read(source), "output": "second"})
        receipt["source_sha256"]["second/confirmed.json"] = sha(duplicate)
        write(directory / "confirmed.json", receipt)
    elif change == "stopped":
        write(source.parent / "stopped.json", {"task_completed": False})
    elif change == "unhashed_mutation":
        write(source, {**read(source), "star": "Other"})
    else:
        actual = rig.root / "actual.json"
        source.rename(actual)
        source.symlink_to(actual)
    with pytest.raises(BrowserSafetyStop):
        create(rig, directory)
    assert not rig.flags.calls


def test_callback_source_mutation_stops_before_any_navigation(rig):
    directory, source = autosave_source(rig)

    def callback(kind, payload):
        if kind == "state" and payload.get("scheduled_call") == "to_list":
            write(source, {**read(source), "star": "Other"})

    owner = create(rig, directory, emit=callback)
    owner.advance()
    assert owner.finished and owner.status == "stopped"
    assert owner.navigation_attempts == 0
    assert not any(call in {"list", "stellar", "verify"} for call in rig.flags.calls)
