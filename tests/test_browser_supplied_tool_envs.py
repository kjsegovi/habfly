"""Synthetic contract checks, not learned rollouts or native acceptance."""

import pytest
from test_browser_habitability_policy import mapping
from test_browser_planet_policy import FakeSession

from habfly.browser import BrowserSafetyStop
from habfly.browser_supplied_tool_envs import (
    NON_MAIN_CLASSES,
    SuppliedPlanetBrowserToolEnv,
    SuppliedTemperatureBrowserToolEnv,
)
from habfly.contracts import Action
from habfly.environments.habitability_calculations import habitability_expert
from habfly.environments.planet_calculations import planet_expert
from habfly.habitability_supplied_inputs import SCOPE as TEMPERATURE_SCOPE
from habfly.planet_supplied_inputs import SCOPE as PLANET_SCOPE


@pytest.mark.parametrize("actual_class", NON_MAIN_CLASSES)
@pytest.mark.parametrize("task", ["planet", "temperature"])
def test_same_visible_controls_keep_actual_class_and_no_private_grades(task, actual_class, monkeypatch):
    session = FakeSession()
    env = (
        SuppliedPlanetBrowserToolEnv(session, supplied_star_class=actual_class)
        if task == "planet"
        else SuppliedTemperatureBrowserToolEnv(
            mapping(), supplied_star_class=actual_class, supplied_greenhouse_increment=30
        )
    )
    expert = planet_expert if task == "planet" else habitability_expert
    calls, execute = [], env.adapter.execute

    def captured(*args, **kwargs):
        calls.append(kwargs["star_class"])
        return execute(*args, **kwargs)

    monkeypatch.setattr(env.adapter, "execute", captured)
    with pytest.raises(RuntimeError, match="no scripted expert"):
        env.expert_action(env.observe())
    assert env.grades() == (set(), {}) and "expected" not in env.case
    for _ in range(128):
        obs = env.observe()
        assert obs.values["star_class"] == actual_class
        assert obs.progress["scope"] == (PLANET_SCOPE if task == "planet" else TEMPERATURE_SCOPE)
        result = env.step(expert(obs, env.pack))  # Explicit fixture driver, never used by browser owner.
        assert result.reward == result.cumulative_reward == 0
        assert not result.observation.progress["task_completed"]
        assert not result.observation.progress["browser_acceptance_passed"]
        if result.terminated:
            break
    assert result.terminated and result.failure_reason is None
    assert calls and set(calls) == {actual_class}
    if task == "planet":
        assert env.transport_verified and len(session.attempted) == 4
    else:
        assert env.proposals_complete and env.answers["surface_temp"] - env.answers["equilibrium_temp"] == 30
        assert not session.attempted  # No native temperature-copy primitive exists in this environment.
    env.reset(seed=env.case["seed"])
    assert not env.results and not env.answers and not env.bindings
    assert not (env.transport_verified if task == "planet" else env.proposals_complete)
    assert env.observe().values["star_class"] == actual_class


@pytest.mark.parametrize("actual_class", [None, True, "", "giant", "main_sequence"])
@pytest.mark.parametrize("task", ["planet", "temperature"])
def test_unknown_or_main_class_cannot_enter_new_browser_path(task, actual_class):
    with pytest.raises(BrowserSafetyStop, match="explicit_non_main_class"):
        if task == "planet":
            SuppliedPlanetBrowserToolEnv(FakeSession(), supplied_star_class=actual_class)
        else:
            SuppliedTemperatureBrowserToolEnv(
                mapping(), supplied_star_class=actual_class, supplied_greenhouse_increment=0
            )


@pytest.mark.parametrize("value", ["", "0", "nan", "-1"])
def test_missing_or_invalid_supplied_mass_stops_without_repair(value):
    session = FakeSession()
    session.mapping["observation"]["values"]["stellar_inputs"]["stellar_mass"]["display_text"] = value
    with pytest.raises(BrowserSafetyStop):
        SuppliedPlanetBrowserToolEnv(session, supplied_star_class="white_dwarf")
    assert not session.attempted


@pytest.mark.parametrize("task", ["planet", "temperature"])
def test_incomplete_check_does_not_select_or_repair_inputs(task):
    session = FakeSession()
    env = (
        SuppliedPlanetBrowserToolEnv(session, supplied_star_class="white_dwarf")
        if task == "planet"
        else SuppliedTemperatureBrowserToolEnv(
            mapping(), supplied_star_class="white_dwarf", supplied_greenhouse_increment=0
        )
    )
    control = next(c for c in env.observe().controls if c.id.endswith(":check"))
    result = env.step(Action(kind="CLICK", target=control.id))
    assert result.terminated and result.failure_reason
    assert not env.answers and not env.results and not env.bindings and not session.attempted
