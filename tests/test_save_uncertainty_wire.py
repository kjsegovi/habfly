"""Noncanonical Save uncertainty stays visible without promoting completion."""
# ruff: noqa: F811

from copy import deepcopy

import pytest
from test_browser_project_campaign_steps import rig as campaign_rig  # noqa: F401
from test_browser_project_runtime import create as create_bridge
from test_browser_project_runtime import rig as bridge_rig  # noqa: F401
from test_browser_project_steps import create as create_owner
from test_browser_project_steps import rig as owner_rig  # noqa: F401

from habfly.project_wire import compact_project_event


def diagnostic(value):
    return {
        "save_outcome_uncertain": value,
        "save_outcome": {
            "scope": "noncanonical_failed_save_diagnostic",
            "save_may_have_occurred": value,
            "task_completed": False,
            "project_completed": False,
            "automatic_retry": False,
        },
    }


@pytest.mark.parametrize("uncertain", [True, False, None])
def test_owner_forwards_nullable_save_diagnostic_without_touching_journal(owner_rig, uncertain):
    owner = create_owner(owner_rig)
    before = owner_rig.journal.path.read_bytes()
    source = diagnostic(uncertain)
    owner._child_state = deepcopy(source)
    owner.status, owner.phase = "stopped", "stopped"
    owner.failure = "no_planet_save_reserved_phase_timeout"
    state = owner.state()
    assert state["save_outcome_uncertain"] is uncertain
    assert state["save_outcome"] == source["save_outcome"]
    assert not state["task_completed"] and not state["project_completed"]
    assert owner_rig.journal.path.read_bytes() == before
    state["save_outcome"]["task_completed"] = True
    assert not owner._child_state["save_outcome"]["task_completed"]


@pytest.mark.parametrize("uncertain", [True, False, None])
def test_bridge_and_compact_wire_keep_failed_save_separate_from_empty_canonical_uncertainty(
    bridge_rig, uncertain
):
    bridge = create_bridge(bridge_rig)
    source = diagnostic(uncertain)
    bridge._project_state = {**source, "phase": "stopped", "project_progress": {"uncertain_actions": []}}
    bridge.status, bridge.phase = "stopped", "stopped"
    bridge.failure = "no_planet_save_reserved_phase_timeout"
    state = bridge.state()
    assert state["save_outcome_uncertain"] is uncertain
    assert state["save_outcome"] == source["save_outcome"]
    assert not state["task_completed"] and not state["project_completed"]
    wire = compact_project_event("state", state)
    assert wire["save_outcome_uncertain"] is uncertain
    assert wire["save_outcome"] == source["save_outcome"]
    assert not wire["task_completed"] and not wire["project_completed"]
    assert wire["project_wire"]["evidence_receipt"] is False
    bridge.close()


def test_old_wire_without_diagnostic_does_not_invent_known_save_result():
    source = {"status": "stopped", "task_completed": False, "project_completed": False}
    wire = compact_project_event("state", source)
    assert "save_outcome" not in wire and "save_outcome_uncertain" not in wire


@pytest.mark.parametrize("uncertain", [True, False, None])
def test_campaign_diagnostic_cannot_create_a_star_receipt(campaign_rig, uncertain):
    campaign = campaign_rig.c
    original = campaign_rig.journal.path.read_bytes()
    campaign._owner_state = diagnostic(uncertain)
    campaign.status, campaign.phase = "stopped", "stopped"
    campaign.failure = "no_planet_save_reserved_phase_timeout"
    state = campaign.state()
    assert state["save_outcome_uncertain"] is uncertain
    assert state["save_outcome"] == campaign._owner_state["save_outcome"]
    assert not state["target_workflows_verified"] and state["verified_stars"] == 0
    assert campaign_rig.journal.path.read_bytes() == original
