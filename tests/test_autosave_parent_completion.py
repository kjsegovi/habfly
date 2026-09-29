"""Full parent handoffs using explicitly injected transport fixture children.

Unlike standalone importer tests these drive the real parent completion gates.
No browser, model or scientific result is supplied by these synthetic children.
"""
# ruff: noqa: F401

from copy import deepcopy

import pytest
from test_browser_project_steps import rig, sha, write
from test_browser_project_steps_positive import finalizing, positive_rig
from test_browser_project_steps_terrestrial import completed_child, terrestrial_rig

import habfly.browser_project_steps as module
from habfly.browser_autosave import autosave_workflow_flags


def autosave_constructor(monkeypatch):
    original = module.BrowserProjectSteps

    def create(*args, **kwargs):
        kwargs["model_options"] = {**kwargs["model_options"], "save_strategy": "autosave"}
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "BrowserProjectSteps", create)


def adapt_fixture(item, branch, monkeypatch, *, change=None):
    """Represent a readback-only child before the parent consumes its output."""
    flags = item.flags
    children = item.finalizers if branch == "positive" else item.children

    def construct():
        child = children[-1]
        original_state = child.state

        def state():
            return {
                **original_state(),
                **autosave_workflow_flags(),
                "save_acknowledgement_verified": False,
                "visible_readback_verified": child.finished and not child.aborted,
                "save_dispatch_attempts_recorded": 0,
            }

        monkeypatch.setattr(child, "state", state)

    def completed(child):
        path = child.output / "workflow/confirmed.json"
        import json

        workflow = json.loads(path.read_bytes())
        workflow.update(
            **autosave_workflow_flags(),
            save_acknowledgement_verified=False,
            source_save_click_delivered=False,
        )
        if change == "workflow_strategy":
            workflow.pop("save_strategy")
        elif change == "workflow_ack":
            workflow["save_acknowledgement_verified"] = True
        write(path, workflow)
        child.report.update(
            **autosave_workflow_flags(),
            save_acknowledgement_verified=False,
            save_dispatch_attempts_recorded=0,
            workflow_sha256=sha(path),
        )
        if branch == "positive":
            child.report.update(save_may_have_occurred=False, save_click_delivery_confirmed=False)
        if change in {"ack", "persistence", "strategy", "dispatch", "dispatch_bool", "readback"}:
            key, value = {
                "ack": ("save_acknowledgement_verified", True),
                "persistence": ("persistence_verified", True),
                "strategy": ("save_strategy", "explicit"),
                "dispatch": ("save_dispatch_attempts_recorded", 1),
                "dispatch_bool": ("save_dispatch_attempts_recorded", False),
                "readback": ("visible_readback_verified", False),
            }[change]
            child.report[key] = value
            old = child.state
            monkeypatch.setattr(child, "state", lambda: {**old(), key: value})
        write(child.output / "report.json", child.report)

    flags.constructor_hook = construct
    flags.final_hook = completed


@pytest.mark.parametrize("branch", ["positive", "terrestrial"])
@pytest.mark.parametrize(
    "change",
    [
        None,
        "ack",
        "persistence",
        "strategy",
        "dispatch",
        "dispatch_bool",
        "readback",
        "workflow_strategy",
        "workflow_ack",
    ],
)
def test_autosave_child_completion_reaches_inventory_only_with_matching_authority(
    request, monkeypatch, branch, change
):
    item = request.getfixturevalue("positive_rig" if branch == "positive" else "terrestrial_rig")
    autosave_constructor(monkeypatch)
    adapt_fixture(item, branch, monkeypatch, change=change)
    if branch == "positive":
        owner = finalizing(item)
        child = item.finalizers[-1]
        for _ in range(3):
            owner.step()
    else:
        owner = completed_child(item)
        child = item.children[-1]
    try:
        assert child.options["save_strategy"] == "autosave"
        assert owner.state()["task_completed"] is False
        assert item.rig.journal.load().reduce().report()["verified"] == 0
        if change:
            assert owner.finished and owner.phase == "stopped", owner.state()
            assert owner._workflow is None
        else:
            assert not owner.finished and owner.phase == "awaiting_inventory", owner.state()
            assert owner._workflow is not None
            assert child.report["save_acknowledgement_verified"] is False
            assert child.report["save_dispatch_attempts_recorded"] == 0
        # Completion of this injected child cannot itself increment the ledger.
        before = deepcopy(owner.state()["project_progress"])
        owner.close()
        assert owner.state()["project_progress"] == before
    finally:
        owner.close()
