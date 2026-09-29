"""Native bridge safety with intercepted pages and a fixture-only expert policy."""
# ruff: noqa: F811

import json
from types import SimpleNamespace

import pytest
from test_browser_habitability_numeric import habitat_page  # noqa: F401
from test_browser_numeric import chromium, config, page  # noqa: F401

from habfly.browser_habitability_policy import run_habitability_temperature
from habfly.environments.habitability_calculations import habitability_expert
from habfly.habitability_knowledge import HabitabilityCalculator


@pytest.fixture
def fixture_model(monkeypatch, tmp_path):
    import habfly.browser_habitability_policy as module

    class FixturePolicy:
        def eval(self):
            return self

        def act(self, observation, state):
            return habitability_expert(observation, HabitabilityCalculator().pack), None, {}

    pilot = tmp_path / "pilot"
    (pilot / "training").mkdir(parents=True)
    (pilot / "training/checkpoint.pt").write_bytes(b"fixture-only-not-trained")
    evaluation = tmp_path / "final"
    evaluation.mkdir()
    (evaluation / "report.json").write_text("{}")
    monkeypatch.setattr(module, "require_frozen_final_gate", lambda *_: None)
    monkeypatch.setattr(
        module,
        "load_validated_pilot",
        lambda *_: (
            FixturePolicy(),
            {"graph_hash": "fixture", "knowledge_pack_hash": "fixture"},
            None,
        ),
    )
    return {"pilot": pilot, "final_evaluation": evaluation, "graph": SimpleNamespace()}


def test_native_bridge_copies_equilibrium_once_and_does_not_select_gases_or_strength(
    habitat_page, fixture_model, tmp_path
):
    page, frame = habitat_page
    report = run_habitability_temperature(
        page, config(), tmp_path / "run", supplied_greenhouse_increment=30, **fixture_model
    )
    assert report["equilibrium_transport_verified"] and report["steps"] == 28
    assert report["checkpoint_unchanged"] and report["optimizer_updates"] == 0
    assert not report["task_completed"] and not report["course_acceptance_passed"]
    assert frame.locator("#greenhouse").input_value() == frame.locator("#phase").input_value() == ""
    assert frame.locator("#surface").inner_text() == frame.locator("#temperature").input_value()
    assert report["local_proposals"]["surface_temp"] - report["local_proposals"]["equilibrium_temp"] == 30
    events = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["event"] == "neural_activity"]) == 28
    assert all(e["version"] == 1 for e in events)
    assert not page.get_by_role("checkbox").is_checked()


def test_changed_page_during_local_choices_cannot_authorize_copy(habitat_page, fixture_model, tmp_path):
    page, frame = habitat_page

    def change(row):
        if row["steps"] == 2:
            frame.locator("#phase").select_option(label="Gas")

    report = run_habitability_temperature(
        page, config(), tmp_path / "run", supplied_greenhouse_increment=0, notify=change, **fixture_model
    )
    assert not report["equilibrium_transport_verified"]
    assert report["outcome"] == "stale_habitability_observation"
    assert frame.locator("#temperature").input_value() == "0"
    assert not (tmp_path / "run/native-copy/reserved.json").exists()


def test_native_uncertain_copy_is_not_retried_or_reported_complete(habitat_page, fixture_model, tmp_path):
    page, frame = habitat_page
    frame.locator("#temperature").evaluate(
        "e=>e.onblur=()=>document.querySelector('#surface').textContent='999'"
    )
    report = run_habitability_temperature(
        page, config(), tmp_path / "run", supplied_greenhouse_increment=0, **fixture_model
    )
    assert not report["equilibrium_transport_verified"] and not report["task_completed"]
    assert (tmp_path / "run/native-copy/reserved.json").exists()
    assert (tmp_path / "run/native-copy/stopped.json").exists()
    assert not (tmp_path / "run/native-copy/confirmed.json").exists()
