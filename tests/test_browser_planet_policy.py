"""Public readbacks only; fixture expert checks transport, not learned success."""

from copy import deepcopy

import pytest

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_policy import PlanetBrowserToolEnv, require_final_gate, visible_measurements
from habfly.contracts import Action
from habfly.environments.planet_calculations import FIELDS, QUANTITY_UNITS, SCOPE, planet_expert


class FakeSession:
    def __init__(self):
        self.attempted, self.verified, self.reads = set(), {}, 0
        self.mapping = {
            "observation": {
                "values": {
                    "has_planet": "Yes",
                    "browser_field_map": {
                        k: {"current_value": str(v), "unit": QUANTITY_UNITS[k]}
                        for k, v in {
                            "period_days": "108",
                            "line_shift": "0.000000629",
                            "brightness_drop": "0.008",
                            **dict.fromkeys(FIELDS, ""),
                        }.items()
                    },
                    "stellar_inputs": {
                        "stellar_mass": {"display_text": "2.368", "unit": "Msun"},
                        "stellar_radius": {"display_text": "1.961", "unit": "Rsun"},
                    },
                }
            }
        }

    def current(self):
        self.reads += 1

    def copy(self, destination, text, unit, *, source):
        assert destination in FIELDS and source == "checkpoint"
        assert unit == QUANTITY_UNITS[destination] and destination not in self.attempted
        self.attempted.add(destination)
        self.verified[destination] = {"value": text, "unit": unit}


def test_shared_reference_actions_copy_four_fields_without_grading_or_claiming_success():
    session = FakeSession()
    env = PlanetBrowserToolEnv(session, supplied_star_class="main_sequence")
    assert "expected" not in env.case
    with pytest.raises(RuntimeError, match="no scripted expert"):
        env.expert_action(env.observe())
    for _ in range(62):
        result = env.step(planet_expert(env.observe(), env.pack))
        assert result.reward == result.cumulative_reward == 0
        assert not result.observation.progress["task_completed"]
    assert result.terminated and not result.failure_reason and env.transport_verified
    assert session.reads == 1 and set(session.verified) == set(FIELDS)
    assert {k: float(v["value"]) for k, v in session.verified.items()} == env.answers
    assert env.grades() == (set(), {})
    assert not result.observation.progress["browser_acceptance_passed"]


@pytest.mark.parametrize(
    "kind", ["missing", "zero", "unit", "nonfinite", "huge_drop", "prepopulated", "no_planet"]
)
def test_missing_or_invalid_readbacks_never_create_answer_values(kind):
    mapping = FakeSession().mapping
    values = mapping["observation"]["values"]
    fields = values["browser_field_map"]
    if kind == "no_planet":
        values["has_planet"] = "No"
    elif kind == "unit":
        fields["line_shift"]["unit"] = "m"
    elif kind == "prepopulated":
        fields["planet_mass"]["current_value"] = "12"
    else:
        key = "brightness_drop" if kind == "huge_drop" else "line_shift"
        fields[key]["current_value"] = {"missing": "", "zero": "0", "nonfinite": "nan", "huge_drop": "101"}[
            kind
        ]
    with pytest.raises(BrowserSafetyStop):
        visible_measurements(mapping)


def test_wrong_unit_or_incomplete_check_stops_without_repair():
    for label, value in (("Unit for planet_mass", "au"), ("Check derived planet analysis", None)):
        session = FakeSession()
        env = PlanetBrowserToolEnv(session, supplied_star_class="main_sequence")
        control = next(c for c in env.observe().controls if c.label == label)
        action = Action(kind="SELECT" if value else "CLICK", target=control.id, value=value)
        result = env.step(action)
        assert result.terminated and result.failure_reason
        assert not session.attempted and not env.transport_verified
    with pytest.raises(BrowserSafetyStop, match="supplied_main_sequence"):
        PlanetBrowserToolEnv(FakeSession(), supplied_star_class="giant")


def test_browser_promotion_requires_the_exact_frozen_checkpoint_result():
    report = {
        "scope": SCOPE,
        "checkpoint_sha256": "frozen",
        "checkpoint_unchanged": True,
        "optimizer_updates": 0,
        "final_test_episodes": 100,
        "learning_gate_passed": True,
        "closed_loop": {
            "test": {
                "episodes": 100,
                "completed": 100,
                "invalid_actions": 0,
                "tool_errors": 0,
                "infrastructure_failures": 0,
                "api_failures": 0,
            }
        },
    }
    require_final_gate(report, "frozen")
    with pytest.raises(ValueError, match="matching frozen"):
        require_final_gate(report, "different")
    changed = deepcopy(report)
    changed["closed_loop"]["test"]["completed"] = 89
    with pytest.raises(ValueError, match="matching frozen"):
        require_final_gate(changed, "frozen")
