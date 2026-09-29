"""Supplied-input temperature practice, not a course habitability oracle.

Inverse-generated fixtures test tool selection, binding, chaining, copying and
units. They do not infer gases, greenhouse strength, water phase or habitability.
Expected temperatures never appear in observations or the reference expert.
"""

import hashlib
import json
import math
import random
from typing import ClassVar

from habfly.contracts import Action, Observation
from habfly.habitability_knowledge import HabitabilityCalculator
from habfly.model.tool_state import visible_sources

from .local_stellar import LocalStellarEnv

SCOPE = "independent_physics_supplied_habitability_temperatures_v1"
FIELDS = ("equilibrium_temp", "surface_temp")
QUANTITY_UNITS = {
    "stellar_luminosity": "Lsun",
    "orbital_radius": "au",
    "albedo": "fraction",
    "greenhouse_increment": "K",
    "equilibrium_temp": "K",
    "surface_temp": "K",
}
SEEDS = {
    "train": 15000000,
    "calibration": 16000000,
    "development": 17000000,
    "test": 18000000,
    "gate": 19000000,
}
TEMPLATES = {
    "train": "Use the current star's supplied terrestrial-planet inputs to calculate equilibrium and surface temperatures. Ignore reference-star readings. Copy results and choose units; do not infer habitability.",
    "calibration": "Derive both temperatures for this star's planet from the provided luminosity, orbit, albedo and warming increment. Reference-star data are distractors; select tools, bind inputs and transfer units.",
    "development": "Reconstruct the two required temperatures under the stated assumptions for the current star. Select its measurements rather than the reference star, reuse calculated results, and copy exact values with units.",
    "test": "Complete equilibrium and surface temperature fields for the current star's terrestrial planet using the explicit physical assumptions and supplied warming increment. Do not use reference-star measurements or infer water phase.",
    "gate": "Follow the visible reference cards to fill the current star's two temperature fields. Use supplied inputs and intermediate results, then choose units. No gas or habitability classification is requested.",
}


def habitability_cases(split, count, *, offset=0):
    if (
        split not in SEEDS
        or type(count) is not int
        or not 1 <= count <= 100
        or type(offset) is not int
        or not 0 <= offset <= 100000 - count
    ):
        raise ValueError("Unknown or unbounded habitability split")
    cases = []
    for index in range(count):
        seed = SEEDS[split] + offset + index
        rng = random.Random(seed)
        equilibrium = rng.uniform(150, 900)
        albedo, orbit = rng.uniform(0, 0.85), rng.uniform(0.1, 4)
        increment = (0, 10, 30, 100)[index % 4]
        # Inverse energy balance, independent of the forward expression evaluator.
        emission = 4 * math.pi * (orbit * 149597870700) ** 2 * 5.670374419e-8 * equilibrium**4
        luminosity = 4 * emission / ((1 - albedo) * 3.827e26)
        inputs = {
            "stellar_luminosity": luminosity,
            "orbital_radius": orbit,
            "albedo": albedo,
            "greenhouse_increment": increment,
        }
        rows = []
        for quantity, value in inputs.items():
            other = (10 if value == 0 else 0) if quantity == "greenhouse_increment" else value * 1.1
            for source, payload in (("current star", value), ("reference star", other)):
                rows.append(
                    {"kind": quantity, "value": payload, "unit": QUANTITY_UNITS[quantity], "source": source}
                )
        rng.shuffle(rows)
        cases.append(
            {
                "seed": seed,
                "split": split,
                "scope": SCOPE,
                "star_class": "main_sequence",
                "planet_class": "terrestrial",
                "required": list(FIELDS),
                "case_id": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
                "measurements": {f"m{i}": row for i, row in enumerate(rows)},
                "expected": {"equilibrium_temp": equilibrium, "surface_temp": equilibrium + increment},
                "expected_provenance": "inverse_generated_energy_balance_not_course_grading",
            }
        )
    return cases


class HabitabilityCalculationEnv(LocalStellarEnv):
    task_name = "habitability_calculations"
    scope = SCOPE
    check_label = "Check supplied temperature calculations"
    quantity_units: ClassVar[dict] = QUANTITY_UNITS

    def __init__(self, adapter, cases, *, max_steps=128):
        if (
            not isinstance(adapter, HabitabilityCalculator)
            or type(max_steps) is not int
            or not 1 <= max_steps <= 128
        ):
            raise ValueError("Explicit habitability calculator and bounded steps are required")
        if not cases or any(
            c.get("scope") != SCOPE
            or c.get("required") != list(FIELDS)
            or c.get("planet_class") != "terrestrial"
            or set(c.get("expected", {})) != set(FIELDS)
            for c in cases
        ):
            raise ValueError("Incompatible supplied-temperature case")
        if len({c["seed"] for c in cases}) != len(cases):
            raise ValueError("Duplicate habitability seeds")
        adapter.verify()
        super().__init__(adapter, cases, max_steps=max_steps)

    def task_instruction(self):
        return TEMPLATES[self.case["split"]]

    def reference_information(self):
        return {
            "assumptions": self.pack.assumptions,
            "provenance": self.pack.provenance,
            "validation_scope": self.pack.validation_scope,
        }

    def observe(self):
        observation = super().observe()
        for control in observation.controls:
            if control.label.startswith("Unit for "):
                control.options = list(dict.fromkeys(control.options))
        observation.values.update(planet_class="terrestrial", physics_assumptions=list(self.pack.assumptions))
        observation.progress.update(
            scope=SCOPE,
            course_acceptance_passed=False,
            supplied_measurements=True,
            greenhouse_increment_supplied=True,
            gas_identification_learned=False,
            water_phase_learned=False,
            habitability_decision_learned=False,
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
            for k in FIELDS
            if k in self.answers
            and math.isclose(self.answers[k], self.case["expected"][k], rel_tol=1e-6, abs_tol=1e-12)
        }
        units = {k for k in FIELDS if self.units.get(k) == "K"}
        return numeric & units, {
            "numeric_fields_correct": len(numeric),
            "required_fields": len(FIELDS),
            "units_correct": len(units),
        }

    def expert_action(self, observation):
        return habitability_expert(observation, self.pack)


def habitability_expert(observation, pack):
    """Read only public cards, source identity, dependencies and exact results."""
    obs = Observation.model_validate(observation)
    state, values = obs.calculation, obs.values
    sources = {
        ref: row
        for ref, row in visible_sources(obs.model_dump(mode="json")).items()
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
                for ref, row in sources.items()
                if row["kind"] == quantity and row["unit"] == QUANTITY_UNITS[quantity]
            ),
            None,
        )

    def missing(quantity, seen):
        if quantity in seen:
            raise ValueError("Cyclic public tool dependencies")
        op = pack.operation(quantity)
        if op is None:
            raise ValueError("Missing supplied measurement")
        for spec in op.inputs.values():
            if existing(spec.quantity) is None:
                return missing(spec.quantity, seen | {quantity})
        return op

    for field in values["required_fields"]:
        ref = existing(field)
        if ref is None:
            op = missing(field, set())
            if state["operation"] != op.id:
                return act("operation", op.id)
            for parameter, spec in op.inputs.items():
                source = existing(spec.quantity)
                if state["bindings"].get(parameter) != source:
                    if state["parameter"] != parameter:
                        return act("parameter", parameter)
                    if state["source"] != source:
                        return act("source", source)
                    return act("bind")
            return act("execute")
        if values["answers"].get(field) != sources[ref]["value"]:
            if state["selected_result"] != ref:
                return act("result", ref)
            if state["destination"] != field:
                return act("destination", field)
            return act("copy")
        if values["units"].get(field) != "K":
            return act(f"unit_{field}", "K")
    return act("check")
