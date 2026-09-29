"""Single explicit prerequisite, using the same six-option planet tool."""

import hashlib
import math

from .planet_calculations import PlanetCalculationEnv, planet_cases

SCOPE = "independent_physics_planet_period_conversion_v1"
TEMPLATES = {
    "train": "Use the current star's measured orbital period in days. Select a calculation to convert it to years, then copy the result and choose its unit. Ignore reference-star measurements.",
    "calibration": "Calculate only period_years for the current star. Its observations give the period in days. Use the local tool, copy its output and select the unit; disregard reference-star readings.",
    "development": "Convert the current star's orbital-period observation into years with the local calculator. Transfer that result and its unit to the required answer; do not use the reference star.",
    "test": "For the current star, use the observed day-based period to calculate period_years. Copy the tool's result and its unit, ignoring reference-star data.",
    "gate": "Read the current star's period in days and complete its period in years using explicit local-tool operations, exact copying and unit selection.",
}


def period_cases(split, count):
    rows = planet_cases(split, count)
    for case in rows:
        stellar_mass = next(
            m["value"]
            for m in case["measurements"].values()
            if m["kind"] == "stellar_mass" and m["source"] == "current star"
        )
        # Inverse physical generator's chosen orbit, not a call to the forward
        # calculator or access to course grading. Inputs stay byte-for-byte the
        # same as the corresponding full-task split; this introduces no new data.
        expected = math.sqrt(case["expected"]["orbital_radius"] ** 3 / stellar_mass)
        case.update(
            scope=SCOPE,
            required=["period_years"],
            expected={"period_years": expected},
            case_id=hashlib.sha256((SCOPE + case["case_id"]).encode()).hexdigest(),
        )
    return rows


class PlanetPeriodEnv(PlanetCalculationEnv):
    scope = SCOPE
    required_fields = ("period_years",)

    def task_instruction(self):
        return TEMPLATES[self.case["split"]]
