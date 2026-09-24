"""Explicit identities and budgets for the separately promoted local workflows."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from habfly.environments.lifetime import SCOPE as LIFETIME_SCOPE
from habfly.environments.lifetime import LifetimeEnv, lifetime_cases
from habfly.environments.lifetime import required_fields as lifetime_required_fields
from habfly.environments.luminosity import MAX_STEPS, SCOPE, LuminosityEnv, luminosity_cases
from habfly.environments.mass import SCOPE as MASS_SCOPE
from habfly.environments.mass import MassEnv, mass_cases, required_fields
from habfly.environments.radius import SCOPE as RADIUS_SCOPE
from habfly.environments.radius import RadiusEnv, radius_cases
from habfly.environments.radius import required_fields as radius_required_fields
from habfly.environments.temperature import SCOPE as TEMPERATURE_SCOPE
from habfly.environments.temperature import TemperatureEnv, temperature_cases


@dataclass(frozen=True)
class Workflow:
    task: str
    scope: str
    required: tuple[str, ...]
    steps: int
    environment: type
    cases: Callable
    parent: Path
    recognition_seed: int
    max_steps: int = MAX_STEPS

    def required_for(self, star_class):
        if self.task == "lifetime":
            return lifetime_required_fields(star_class)
        if self.task == "radius":
            return radius_required_fields(star_class)
        return required_fields(star_class) if self.task == "mass" else list(self.required)

    def expected_steps(self, case):
        if self.task in {"mass", "radius", "lifetime"} and case["star_class"] != "main_sequence":
            return 31
        return self.steps

    @property
    def applicability_metrics(self):
        if self.task == "lifetime":
            return (
                "mass_applicability_correct",
                "radius_applicability_correct",
                "lifetime_applicability_correct",
            )
        if self.task == "radius":
            return ("mass_applicability_correct", "radius_applicability_correct")
        return ("mass_applicability_correct",) if self.task == "mass" else ()


def workflow_spec(task):
    if task == "luminosity":
        return Workflow(
            task,
            SCOPE,
            ("distance", "luminosity"),
            22,
            LuminosityEnv,
            luminosity_cases,
            Path("experiments/distance-aliases-002/training/checkpoint.pt"),
            3600000,
        )
    if task == "temperature":
        return Workflow(
            task,
            TEMPERATURE_SCOPE,
            ("distance", "luminosity", "temperature"),
            31,
            TemperatureEnv,
            temperature_cases,
            Path("experiments/luminosity-002/training/checkpoint.pt"),
            4600000,
        )
    if task == "mass":
        return Workflow(
            task,
            MASS_SCOPE,
            ("distance", "luminosity", "temperature", "mass"),
            40,
            MassEnv,
            mass_cases,
            Path("experiments/temperature-source-003/training/checkpoint.pt"),
            6600000,
        )
    if task == "radius":
        return Workflow(
            task,
            RADIUS_SCOPE,
            ("distance", "luminosity", "temperature", "mass", "radius"),
            51,
            RadiusEnv,
            radius_cases,
            Path("experiments/mass-003/training/checkpoint.pt"),
            7600000,
        )
    if task == "lifetime":
        return Workflow(
            task,
            LIFETIME_SCOPE,
            ("distance", "luminosity", "temperature", "mass", "radius", "lifetime"),
            60,
            LifetimeEnv,
            lifetime_cases,
            Path("experiments/radius-001/training/checkpoint.pt"),
            8600000,
        )
    raise ValueError("Unknown chained stellar workflow")
