"""Independent-physics tool-use curriculum, not a HabWorlds grading oracle.

Inputs are supplied measurements, not inferred chart data. The first scope has
four derived answers and six explicit operations; raw measurement copying,
planet existence/classification, atmospheres and submission are separate gates.
Cases are inverse-generated from chosen physical quantities, independently of
the forward knowledge engine. Course-specific validation remains explicitly
pending in the pack and is never promoted by simulator completion.
"""

import hashlib
import json
import math
import random
from typing import ClassVar

from habfly.contracts import Action, Observation
from habfly.model.tool_state import visible_sources
from habfly.planet_knowledge import PlanetCalculator

from .local_stellar import LocalStellarEnv

SCOPE = "independent_physics_planet_derived_four_field_v1"
QUANTITY_UNITS = {
    "period_days": "day",
    "line_shift": "nm",
    "brightness_drop": "%",
    "stellar_mass": "Msun",
    "stellar_radius": "Rsun",
    "period_years": "yr",
    "radial_velocity": "m/s",
    "orbital_radius": "au",
    "planet_radius": "REarth",
    "planet_mass": "MEarth",
    "planet_density": "g/cm3",
}
FIELDS = ("orbital_radius", "planet_mass", "planet_radius", "planet_density")
SEEDS = {
    "train": 10000000,
    "calibration": 11000000,
    "development": 12000000,
    "test": 13000000,
    "gate": 14000000,
}
TEMPLATES = {
    "train": "Analyze the planet around the current star, not the reference star. Use supplied measurements and the explicit physics assumptions to calculate orbital radius, planet mass, planet radius and density. Copy derived results and units.",
    "calibration": "Ignore reference-star readings. For the current star system, use the local tool and its stated assumptions to obtain the four required derived planet quantities and select their units.",
    "development": "Reconstruct the current star's planet from its supplied observations. Disregard the reference star; bind inputs and intermediate results, then fill the derived properties with units.",
    "test": "Use only measurements associated with the current star to derive this planet's orbital radius, mass, radius and density under the supplied assumptions. Transfer tool outputs and choose units; ignore the reference star.",
    "gate": "Complete the current star's derived planet fields using the visible physics cards and explicit assumptions. Do not use the reference star's values.",
}


def planet_cases(split, count, *, offset=0):
    """Inverse fixtures, not calls to the forward tool or private course state.

    Constants match the documented candidate conventions (365-day year, solar
    radius 109 Earth radii, RV coefficient 11.177, Earth mass/radius conversions).
    Independent forward goldens and SI-coefficient tests validate those
    conventions separately. No real-browser case or final-test tuning is used.
    """
    if (
        split not in SEEDS
        or type(count) is not int
        or not 1 <= count <= 100
        or type(offset) is not int
        or not 0 <= offset <= 100000 - count
    ):
        raise ValueError("Unknown or unbounded planet split")
    cases = []
    for index in range(count):
        seed = SEEDS[split] + offset + index
        rng = random.Random(seed)
        stellar_mass = rng.uniform(0.5, 2.5)
        stellar_radius = rng.uniform(0.5, 2.5)
        orbital_radius = rng.uniform(0.1, 5)
        planet_mass = rng.uniform(0.5, 300)
        planet_radius = rng.uniform(0.5, 12)
        inputs = {
            "stellar_mass": stellar_mass,
            "stellar_radius": stellar_radius,
            "period_days": 365 * math.sqrt(orbital_radius**3 / stellar_mass),
            "line_shift": planet_mass / (11.177 * math.sqrt(orbital_radius * stellar_mass)) * 656.3 / 3e8,
            "brightness_drop": 100 * (planet_radius / (109 * stellar_radius)) ** 2,
        }
        expected = {
            "orbital_radius": orbital_radius,
            "planet_mass": planet_mass,
            "planet_radius": planet_radius,
            "planet_density": planet_mass * 5.97e27 / (4 * math.pi / 3 * (planet_radius * 6.37e8) ** 3),
        }
        rows = []
        for quantity, value in inputs.items():
            for source, factor in (("current star", 1), ("reference star", 1.31)):
                rows.append(
                    {
                        "kind": quantity,
                        "value": value * factor,
                        "unit": QUANTITY_UNITS[quantity],
                        "source": source,
                    }
                )
        rng.shuffle(rows)
        cases.append(
            {
                "seed": seed,
                "split": split,
                "scope": SCOPE,
                "star_class": "main_sequence",
                "case_id": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
                "measurements": {f"m{i}": row for i, row in enumerate(rows)},
                "required": list(FIELDS),
                "expected": expected,
                "expected_provenance": "inverse_generated_physics_not_course_grading",
            }
        )
    return cases


class PlanetCalculationEnv(LocalStellarEnv):
    task_name = "planet_calculations"
    scope = SCOPE
    required_fields = FIELDS
    check_label = "Check derived planet analysis"
    quantity_units: ClassVar[dict] = QUANTITY_UNITS

    def __init__(self, adapter, cases, *, max_steps=128):
        if not isinstance(adapter, PlanetCalculator) or not 1 <= max_steps <= 128:
            raise ValueError("Planet task needs its explicit calculator and bounded episode")
        if not cases or any(
            c.get("scope") != self.scope
            or c.get("required") != list(self.required_fields)
            or c.get("star_class") != "main_sequence"
            or set(c.get("expected", {})) != set(self.required_fields)
            for c in cases
        ):
            raise ValueError("Incompatible planet case contract")
        if len({c["seed"] for c in cases}) != len(cases):
            raise ValueError("Duplicate planet seeds")
        adapter.verify()
        super().__init__(adapter, cases, max_steps=max_steps)

    def task_instruction(self):
        return TEMPLATES[self.case["split"]]

    def reference_information(self):
        return {
            "assumptions": self.pack.assumptions,
            "source_conflicts": self.pack.source_conflicts,
            "provenance": self.pack.provenance,
            "validation_scope": self.pack.validation_scope,
        }

    def relevant_operations(self):
        return [operation.id for operation in self.pack.operations]

    def observe(self):
        observation = super().observe()
        observation.values["physics_assumptions"] = list(self.pack.assumptions)
        observation.progress.update(
            scope=self.scope,
            course_acceptance_passed=False,
            supplied_measurements=True,
            classification_learned=False,
        )
        return observation

    def calculate(self, bound):
        return self.adapter.execute(self.operation, bound, assumptions=self.pack.assumptions)

    def bind_input(self):
        self.bindings[self.parameter] = self.source
        source = visible_sources(self.observe().model_dump(mode="json"))[self.source]
        spec = self.pack.operation(self.operation).inputs[self.parameter]
        self.metrics["input_attempts"] += 1
        self.metrics["input_correct"] += int(
            source.get("kind") == spec.quantity
            and source.get("unit") == spec.unit
            and source.get("source") == "current star"
        )

    def grades(self):
        numeric = {
            k
            for k in self.required_fields
            if k in self.answers
            and math.isclose(self.answers[k], self.case["expected"][k], rel_tol=1e-6, abs_tol=1e-12)
        }
        units = {k for k in self.required_fields if self.units.get(k) == QUANTITY_UNITS[k]}
        return numeric & units, {
            "numeric_fields_correct": len(numeric),
            "required_fields": len(self.required_fields),
            "units_correct": len(units),
        }

    def expert_action(self, observation):
        return planet_expert(observation, self.pack)


def planet_expert(observation, pack):
    """Follow visible dependencies and provenance; never inspect grading data."""
    obs = Observation.model_validate(observation)
    state, values = obs.calculation, obs.values
    sources = visible_sources(obs.model_dump(mode="json"))
    valid = {
        ref: row
        for ref, row in sources.items()
        if row.get("valid", True) and row.get("source") == "current star"
    }

    def act(key, value=None):
        target = next(c.id for c in obs.controls if c.id.split(":", 1)[1] == key)
        return Action(
            kind="SELECT" if value is not None else "CLICK",
            target=target,
            value=value,
            observation_revision=obs.revision,
        )

    def existing(quantity):
        return next(
            (
                ref
                for ref, row in valid.items()
                if row["kind"] == quantity and row["unit"] == QUANTITY_UNITS[quantity]
            ),
            None,
        )

    def missing_operation(quantity, seen):
        if quantity in seen:
            raise ValueError("Cyclic public tool dependencies")
        operation = pack.operation(quantity)
        if operation is None:
            raise ValueError("Missing visible measurement")
        for spec in operation.inputs.values():
            if existing(spec.quantity) is None:
                return missing_operation(spec.quantity, seen | {quantity})
        return operation

    for field in values["required_fields"]:
        ref = existing(field)
        if ref is None:
            operation = missing_operation(field, set())
            if state["operation"] != operation.id:
                return act("operation", operation.id)
            for parameter, spec in operation.inputs.items():
                source = existing(spec.quantity)
                if state["bindings"].get(parameter) != source:
                    if state["parameter"] != parameter:
                        return act("parameter", parameter)
                    if state["source"] != source:
                        return act("source", source)
                    return act("bind")
            return act("execute")
        result = valid[ref]
        if values["answers"].get(field) != result["value"]:
            if state["selected_result"] != ref:
                return act("result", ref)
            if state["destination"] != field:
                return act("destination", field)
            return act("copy")
        if values["units"].get(field) != QUANTITY_UNITS[field]:
            return act(f"unit_{field}", QUANTITY_UNITS[field])
    return act("check")
