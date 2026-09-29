"""Outer lifecycle transport tests with injected drivers; no browser or model launch."""

# Imported pytest fixtures intentionally also name test parameters.
# ruff: noqa: F811

import pytest
from test_browser_project_runtime import (
    create,
    ready,
    rig,  # noqa: F401
)
from test_browser_project_runtime import rig as bridge_rig  # noqa: F401
from test_runtime_browser_project import integrated, start  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.runtime import RunOptions


def planet_ready(rig, **kwargs):
    class PlanetSteps(rig.steps):
        def __init__(self, *args, **options):
            self.constructor_options = options.copy()
            self.choices = []
            self.dispatched = []
            super().__init__(*args, **options)

        def step(self):
            if self.phase == "ready":
                self.phase, self.status = "awaiting_planet_class", "paused"
                return
            sequence = {
                "selecting_planet_class": "positive_initializing",
                "positive_initializing": "positive_active",
                "positive_active": "inventory_initializing",
                "inventory_initializing": "inventory_active",
                "inventory_active": "inventory_import",
            }
            self.dispatched.append(self.phase)
            self.phase = sequence[self.phase]

        def provide_planet_class(self, *, name, rationale):
            if name not in {"ice_giant", "gas_giant", "terrestrial"} or not rationale:
                raise BrowserSafetyStop("project_steps_invalid_planet_class")
            self.choices.append((name, rationale))
            self.phase = "selecting_planet_class"

    bridge = ready(rig, steps=PlanetSteps, reference_planet_continuation=True, **kwargs)
    bridge.step()
    assert bridge.phase == "awaiting_planet_class" and bridge.status == "paused"
    return bridge


def provide(bridge, **changes):
    return bridge.command(
        {
            "command": "step",
            "payload": {
                "planet_class": {
                    "name": "ice_giant",
                    "rationale": "Explicit fixture reference, not learned classification.",
                    **changes,
                }
            },
        }
    )


def test_choice_is_offline_and_subsequent_phases_are_separate_steps(rig):
    bridge = planet_ready(rig, positive_save_settle_seconds=30)
    assert bridge.project.constructor_options["reference_planet_continuation"] is True
    assert bridge.project.constructor_options["positive_save_settle_seconds"] == 30
    assert bridge.project.constructor_options["automatic_inventory"] is True
    calls = list(rig.calls)
    bridge.step()
    bridge.tick()
    assert rig.calls == calls and not bridge.project.choices
    with pytest.raises(BrowserSafetyStop, match="explicit_handoff_required"):
        bridge.resume()
    provide(bridge)
    assert bridge.phase == "selecting_planet_class" and bridge.status == "paused"
    assert not bridge.project.dispatched
    bridge.advance_if_due()
    assert not bridge.project.dispatched
    for phase in (
        "positive_initializing",
        "positive_active",
        "inventory_initializing",
        "inventory_active",
        "inventory_import",
    ):
        count = len(bridge.project.dispatched)
        bridge.step()
        assert bridge.phase == phase and len(bridge.project.dispatched) == count + 1
        assert not bridge.state()["task_completed"] and not bridge.state()["project_completed"]
    bridge.close()


def test_resume_waits_for_explicit_decision(rig):
    bridge = planet_ready(rig)
    provide(bridge)
    bridge.resume()
    assert bridge.status == bridge.project.status == "running"
    bridge.tick()
    assert bridge.phase == "positive_initializing"
    bridge.pause()
    count = len(bridge.project.dispatched)
    bridge.tick()
    assert len(bridge.project.dispatched) == count
    bridge.abort()
    bridge.step()
    assert len(bridge.project.dispatched) == count
    bridge.close()


def test_modified_initial_evidence_blocks_reference_choice(rig):
    bridge = planet_ready(rig)
    (bridge.output / "initial-star/stellar/observation.json").write_text("{}")
    provide(bridge)
    assert bridge.status == "stopped" and not bridge.project.choices
    bridge.close()


@pytest.mark.parametrize("changes", [{"name": "invented"}, {"rationale": ""}])
def test_child_rejection_stops_without_dispatch(rig, changes):
    bridge = planet_ready(rig)
    provide(bridge, **changes)
    assert bridge.status == "stopped" and not bridge.project.dispatched
    bridge.close()


def test_legacy_default_cannot_accept_reference_planet_handoff(rig):
    bridge = ready(rig)
    assert bridge.scope["reference_planet_continuation"] is False
    assert bridge.scope["positive_save_settle_seconds"] == 20
    with pytest.raises(BrowserSafetyStop, match="planet_class_handoff_not_ready"):
        provide(bridge)
    assert bridge.phase == "ready"
    bridge.close()


@pytest.mark.parametrize(
    "name,value",
    [
        ("reference_planet_continuation", 1),
        ("reference_planet_continuation", "true"),
        ("positive_save_settle_seconds", True),
        ("positive_save_settle_seconds", "30"),
        ("positive_save_settle_seconds", 31),
        ("positive_save_settle_seconds", 0),
        ("positive_save_settle_seconds", float("nan")),
        ("positive_save_settle_seconds", float("inf")),
    ],
)
def test_invalid_constructor_options_have_no_browser_or_output(rig, name, value):
    with pytest.raises(BrowserSafetyStop):
        create(rig, **{name: value})
    assert not rig.calls and not (rig.root / "run").exists()


@pytest.mark.parametrize(
    "name,value",
    [
        ("project_reference_planet_continuation", 1),
        ("project_reference_planet_continuation", "true"),
        ("browser_positive_save_settle_seconds", True),
        ("browser_positive_save_settle_seconds", "30"),
        ("browser_positive_save_settle_seconds", 31),
        ("browser_positive_save_settle_seconds", 0),
        ("browser_positive_save_settle_seconds", float("nan")),
    ],
)
def test_run_options_are_strict(name, value):
    with pytest.raises(ValueError):
        RunOptions(**{name: value})


def test_v1_runtime_forwards_fixed_options_without_launch(integrated):
    bridge = start(
        integrated, project_reference_planet_continuation=True, browser_positive_save_settle_seconds=30
    )
    assert bridge.reference_planet_continuation is True
    assert bridge.positive_save_settle_seconds == 30
    assert bridge.status == "paused" and not integrated.calls
