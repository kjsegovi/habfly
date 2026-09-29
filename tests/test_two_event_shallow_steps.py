"""Two-event scheduler tests: injected native seams, actual reference validation."""

from copy import deepcopy

import pytest
import test_browser_shallow_transit_steps as fixtures
from test_shallow_two_hint_stop import SAVED, SAVED_PNG_SHA, SAVED_REPORT_SHA, axes, representative_png

from habfly.browser import BrowserSafetyStop
from habfly.browser_shallow_transit_probe import (
    EXACT_TWO_HINT_POLICY,
    FIRST_THREE_TWO_ROW_HINT_POLICY,
    candidate_columns,
    iterate_shallow_feature,
)
from habfly.planet_tooltip_reference import MODE, TWO_MODE

rig = fixtures.rig


@pytest.mark.parametrize("rig", [2], indirect=True)
def test_explicit_two_event_owner_confirms_both_restores_and_labels_single_spacing(rig):
    owner = rig.make(allow_two_events=True)
    assert not rig.state.calls and not rig.state.native
    assert owner.scope["max_native_actions"] == 38
    assert owner.scope["measurement_mode"] == TWO_MODE
    state = fixtures.drive(owner)
    assert state["phase"] == "measurements_ready", state
    assert state["native_action_attempts"] == state["native_actions_confirmed"] == 26
    assert state["probes"] == state["restorations"] == 2
    assert state["window_evidence"]["progress_path"] == "sensor/restore-02/progress/report.json"
    assert set(state["tooltip_reference"]) == {"expected_star", "diagnostics", "mode"}
    assert state["tooltip_reference"]["mode"] == TWO_MODE
    measured = state["measurements"]
    assert measured["period_days"]["value"] == "1000"
    assert measured["period_days"]["estimate_kind"] == "single_spacing"
    assert measured["confirmed_feature_count"] == 2 and measured["observed_interval_count"] == 1
    assert measured["consistency_redundancy"] == 0
    assert measured["consecutive_events_assumed"] is True
    assert measured["recurrence_confirmed"] is measured["observed_recurrence_compatible"] is False
    assert measured["single_spacing_compatible"] is True
    assert measured["answer_authorized"] is state["task_completed"] is state["project_completed"] is False
    assert not (owner.output / "probe-03").exists()
    assert len([p for p in owner.output.glob("restore-*")]) == 2
    original = deepcopy(state)
    assert owner.advance() == original
    owner.close()


@pytest.mark.parametrize("rig", [3, 4], indirect=True)
def test_optin_with_three_or_more_uses_unmodified_three_event_path(rig):
    owner = rig.make(allow_two_events=True)
    state = fixtures.drive(owner)
    assert state["phase"] == "measurements_ready", state
    assert state["measurement_mode"] == MODE
    assert state["overview_hint_policy"] == FIRST_THREE_TWO_ROW_HINT_POLICY
    assert state["max_native_actions"] == 64
    assert state["probes"] == state["restorations"] == 3
    assert set(state["tooltip_reference"]) == {"expected_star", "diagnostics"}
    assert "consecutive_events_assumed" not in state["measurements"]
    owner.close()


def test_failed_third_probe_never_becomes_two_event_handoff(rig):
    rig.state.fail_probe = 2
    owner = rig.make(allow_two_events=True)
    state = fixtures.drive(owner)
    assert state["phase"] == "stopped"
    # The failed probe's stopped.json also invalidates the owned evidence tree.
    assert state["failure_reason"] == "raster_planet_failed_evidence"
    assert state["measurement_mode"] == MODE
    assert state["measurements"] is state["tooltip_reference"] is None
    assert len(state["completed_probes"]) == 2
    assert not state["task_completed"]
    with pytest.raises(BrowserSafetyStop, match="owner_already_claimed"):
        rig.make("another", allow_two_events=True)
    owner.close()


@pytest.mark.parametrize("rig", [0, 1], indirect=True)
def test_insufficient_groups_fail_before_owner_reservation_or_browser(rig):
    with pytest.raises(BrowserSafetyStop, match="exact_two_overview_hints_required"):
        rig.make(allow_two_events=True)
    assert not rig.state.calls and not rig.state.native
    assert not (rig.root / "sensor").exists()
    assert not (rig.root / "shallow-transit-owner-claims").exists()


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_strict_optin_never_coerces(rig, value):
    with pytest.raises(BrowserSafetyStop, match="invalid_two_event_opt_in"):
        rig.make(allow_two_events=value)
    assert not rig.state.calls and not rig.state.native


@pytest.mark.parametrize("rig", [2], indirect=True)
@pytest.mark.parametrize(
    "field,value",
    [
        ("hint_policy", FIRST_THREE_TWO_ROW_HINT_POLICY),
        ("probe_count", 3),
        ("measurement_mode", MODE),
        ("action_limit", 64),
    ],
)
def test_permission_and_recipe_cannot_change_after_reservation(rig, field, value):
    owner = rig.make(allow_two_events=True)
    setattr(owner, field, value)
    state = owner.advance()
    assert state["phase"] == "stopped"
    assert not rig.state.calls and not rig.state.native
    assert state["measurements"] is None
    owner.close()


def test_exact_two_policy_accepts_saved_shape_but_never_truncates():
    assert candidate_columns(representative_png(), axes()[1], hint_policy=EXACT_TWO_HINT_POLICY) == [
        [102, 103],
        [194, 195],
    ]
    for count in (0, 1, 3, 4):
        raw = fixtures.png(
            baseline=20, features=tuple((70 + i * 40, 70 + i * 40, 22) for i in range(count)), size=(280, 196)
        )
        with pytest.raises(BrowserSafetyStop, match="exact_two_overview_hints_required"):
            candidate_columns(raw, axes()[1], hint_policy=EXACT_TWO_HINT_POLICY)


@pytest.mark.parametrize("index", [-1, 2, 3, True, None, "0"])
def test_two_event_probe_cannot_address_third_or_coerce_index(tmp_path, index):
    probe = iterate_shallow_feature(
        None,
        None,
        tmp_path / "output",
        run_history=tmp_path,
        source_dir=tmp_path / "missing",
        source_report_sha256="0" * 64,
        candidate_index=index,
        overview_hint_policy=EXACT_TWO_HINT_POLICY,
    )
    with pytest.raises(BrowserSafetyStop, match="invalid_candidate_index"):
        next(probe)
    assert not (tmp_path / "output").exists()


def test_optional_saved_bisperon_stays_immutable_and_is_only_two_search_hints():
    if not SAVED.is_dir():
        pytest.skip("Optional public crop not distributed")
    assert fixtures.sha(SAVED / "chart.png") == SAVED_PNG_SHA
    assert fixtures.sha(SAVED / "report.json") == SAVED_REPORT_SHA
    report = fixtures.load(SAVED / "report.json")
    assert candidate_columns(
        (SAVED / "chart.png").read_bytes(), report["flux_axis_labels"], hint_policy=EXACT_TWO_HINT_POLICY
    ) == [[102, 103], [194, 195]]
    assert fixtures.sha(SAVED / "chart.png") == SAVED_PNG_SHA
