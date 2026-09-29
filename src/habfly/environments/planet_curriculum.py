"""Explicitly bounded prerequisite-to-full derived-field curriculum.

Stages change the visible assignment, not which actions the engine chooses.
Every stage keeps the complete six-operation catalog and distractor measurements.
"""

import hashlib
from dataclasses import dataclass

from .planet_calculations import FIELDS, SCOPE, TEMPLATES, PlanetCalculationEnv, planet_cases
from .planet_period import SCOPE as PERIOD_SCOPE
from .planet_period import TEMPLATES as PERIOD_TEMPLATES
from .planet_period import period_cases


@dataclass(frozen=True)
class Stage:
    scope: str
    required: tuple[str, ...]
    steps: int
    parent_scope: str
    templates: dict[str, str]


def templates(fields):
    text = ", ".join(fields)
    return {
        "train": f"Derive {text} for the current star with the local tools, including necessary intermediate calculations. Copy outputs and choose units. Ignore reference-star measurements.",
        "calibration": f"For the current star, reconstruct these properties: {text}. Read the public calculation cards, reuse required results and enter answers with units. Disregard reference-star data.",
        "development": f"Complete {text} for the current star's planet from the supplied observations. Use explicit tools and intermediate results, not the reference star's measurements, then copy outputs with units.",
        "test": f"Calculate only {text} for this current-star system under the stated assumptions. Bind supplied observations or previous results; transfer outputs and units. Do not use reference-star readings.",
        "gate": f"Use the visible calculation tools to fill {text} for the current star, reusing necessary intermediate outputs, copying exactly and choosing units. Ignore the reference star.",
    }


ORBIT = "independent_physics_planet_orbital_radius_v1"
MASS = "independent_physics_planet_orbit_mass_v1"
RADIUS = "independent_physics_planet_orbit_mass_radius_v1"
STAGES = {
    "period": Stage(PERIOD_SCOPE, ("period_years",), 10, SCOPE, PERIOD_TEMPLATES),
    "orbit": Stage(ORBIT, FIELDS[:1], 18, PERIOD_SCOPE, templates(FIELDS[:1])),
    "mass": Stage(MASS, FIELDS[:2], 38, ORBIT, templates(FIELDS[:2])),
    "radius": Stage(RADIUS, FIELDS[:3], 50, MASS, templates(FIELDS[:3])),
    "full": Stage(SCOPE, FIELDS, 62, RADIUS, TEMPLATES),
}


def stage_cases(stage, split, count):
    spec = STAGES[stage]
    if stage == "period":
        return period_cases(split, count)
    rows = planet_cases(split, count)
    if stage == "full":
        return rows
    for case in rows:
        case.update(
            scope=spec.scope,
            required=list(spec.required),
            expected={k: case["expected"][k] for k in spec.required},
            case_id=hashlib.sha256((spec.scope + case["case_id"]).encode()).hexdigest(),
        )
    return rows


class PlanetCurriculumEnv(PlanetCalculationEnv):
    def __init__(self, adapter, cases, *, stage, max_steps=128):
        self.stage = STAGES[stage]
        self.scope, self.required_fields = self.stage.scope, self.stage.required
        super().__init__(adapter, cases, max_steps=max_steps)

    def task_instruction(self):
        return self.stage.templates[self.case["split"]]


def stage_environment(stage):
    if stage not in STAGES:
        raise ValueError("Unknown planet curriculum stage")
    return lambda adapter, cases: PlanetCurriculumEnv(adapter, cases, stage=stage)
