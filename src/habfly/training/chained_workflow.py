"""Explicit identities and budgets for the separately promoted local workflows."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from habfly.environments.luminosity import MAX_STEPS, SCOPE, LuminosityEnv, luminosity_cases
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
    raise ValueError("Unknown chained stellar workflow")
