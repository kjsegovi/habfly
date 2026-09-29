"""Independent planet calculation candidate; not yet a course training oracle.

The original spreadsheet is preserved verbatim in provenance. The separately
named physical candidate uses stellar MASS in the radial-velocity relation.
Course-specific planet classes and habitability rules remain pending.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from .contracts import Contract
from .knowledge import CalculationResult, KnowledgeOperation, LocalCalculator, finite_number

DEFAULT_PLANET_PACK = Path(__file__).parent / "packs" / "planet_physical_candidate.json"


class PlanetKnowledgePack(Contract):
    schema_version: Literal[1] = 1
    id: Literal["planet_physical_candidate_v1"]
    validation_scope: Literal["independent_physics_only_course_pending"]
    operations: list[KnowledgeOperation]
    assumptions: list[str]
    source_conflicts: list[dict[str, str]]
    pending: list[str]
    provenance: dict[str, str]

    @model_validator(mode="after")
    def dependencies(self):
        expected = {
            "period_years": {"period_days": "day"},
            "radial_velocity": {"line_shift": "nm"},
            "orbital_radius": {"period_years": "yr", "stellar_mass": "Msun"},
            "planet_radius": {"brightness_drop": "%", "stellar_radius": "Rsun"},
            "planet_mass": {"radial_velocity": "m/s", "orbital_radius": "au", "stellar_mass": "Msun"},
            "planet_density": {"planet_mass": "MEarth", "planet_radius": "REarth"},
        }
        outputs = {
            "period_years": "yr",
            "radial_velocity": "m/s",
            "orbital_radius": "au",
            "planet_radius": "REarth",
            "planet_mass": "MEarth",
            "planet_density": "g/cm3",
        }
        if len(self.operations) != 6 or {o.id for o in self.operations} != set(expected):
            raise ValueError("Planet candidate requires exactly its six declared operations")
        for op in self.operations:
            if {k: v.unit for k, v in op.inputs.items()} != expected[op.id]:
                raise ValueError("Planet input unit/dependency contract changed")
            if (
                any(name != spec.quantity for name, spec in op.inputs.items())
                or op.output_unit != outputs[op.id]
                or op.applicable_classes != ["main_sequence"]
            ):
                raise ValueError("Planet output, quantity or applicability contract changed")
        if not self.source_conflicts or not self.assumptions or not self.pending:
            raise ValueError("Unresolved source conflicts and assumptions must remain visible")
        return self

    @property
    def checksum(self):
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def operation(self, operation_id):
        return next((o for o in self.operations if o.id == operation_id), None)


def load_planet_pack(path=None):
    return PlanetKnowledgePack.model_validate_json(Path(path or DEFAULT_PLANET_PACK).read_text())


class PlanetCalculator(LocalCalculator):
    def __init__(self, pack=None):
        super().__init__(pack or load_planet_pack())

    def execute(self, operation_id, bindings, *, assumptions=None):
        if (
            not isinstance(assumptions, (list, tuple, set, frozenset))
            or not all(isinstance(item, str) for item in assumptions)
            or len(assumptions) != len(set(assumptions))
            or set(assumptions) != set(self.pack.assumptions)
        ):
            return CalculationResult(
                ok=False, operation_id=operation_id, error="planet_assumptions_not_explicit"
            )
        if operation_id == "planet_radius" and isinstance(bindings, dict):
            binding = bindings.get("brightness_drop")
            depth = binding.get("value") if isinstance(binding, dict) else None
            if finite_number(depth) and depth > 100:
                return CalculationResult(
                    ok=False, operation_id=operation_id, error="brightness_drop_exceeds_100_percent"
                )
        return self._execute(operation_id, bindings, star_class="main_sequence")

    def reference_answers(self, *args, **kwargs):
        raise ValueError("Planet course validation is pending; no training oracle is enabled")

    def verify(self):
        # Fixed independent geometry/Kepler/unit cases, not generated references.
        cases = [
            ("period_years", {"period_days": {"value": 365, "unit": "day"}}, 1.0, "yr"),
            ("radial_velocity", {"line_shift": {"value": 0.000002187666666666667, "unit": "nm"}}, 1.0, "m/s"),
            (
                "orbital_radius",
                {"period_years": {"value": 8, "unit": "yr"}, "stellar_mass": {"value": 1, "unit": "Msun"}},
                4.0,
                "au",
            ),
            (
                "planet_radius",
                {
                    "brightness_drop": {"value": 0.008416799932665601, "unit": "%"},
                    "stellar_radius": {"value": 1, "unit": "Rsun"},
                },
                1.0,
                "REarth",
            ),
            (
                "planet_mass",
                {
                    "radial_velocity": {"value": 1, "unit": "m/s"},
                    "orbital_radius": {"value": 4, "unit": "au"},
                    "stellar_mass": {"value": 1, "unit": "Msun"},
                },
                22.354,
                "MEarth",
            ),
            (
                "planet_density",
                {
                    "planet_mass": {"value": 1, "unit": "MEarth"},
                    "planet_radius": {"value": 1, "unit": "REarth"},
                },
                5.514008418404722,
                "g/cm3",
            ),
        ]
        for name, bindings, expected, unit in cases:
            result = self.execute(name, bindings, assumptions=self.pack.assumptions)
            if (
                not result.ok
                or result.unit != unit
                or not math.isclose(result.value, expected, rel_tol=1e-12)
            ):
                raise ValueError("Planet independent golden case failed")
        # Derive the RV coefficient independently in SI, allowing the source's
        # rounded constants. This checks mass vs radius dimensional dependence.
        coefficient = math.sqrt(149597870700 * 1.98847e30 / 6.67430e-11) / 5.9722e24
        source = self.pack.operation("planet_mass").constants["rv_mass_factor"]
        if not math.isclose(source, coefficient, rel_tol=0.001):
            raise ValueError("Planet mass coefficient fails independent dimensional check")
        return {
            "valid": True,
            "golden_cases": len(cases),
            "independent_rv_coefficient": coefficient,
            "pack_hash": self.pack.checksum,
            "scope": self.pack.validation_scope,
            "course_validation_passed": False,
            "training_oracle_enabled": False,
        }
