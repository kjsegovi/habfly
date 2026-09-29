"""Explicit energy-balance tools; no hidden gas/phase/habitability oracle."""

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from .contracts import Contract
from .knowledge import CalculationResult, KnowledgeOperation, LocalCalculator, finite_number

DEFAULT_HABITABILITY_PACK = Path(__file__).parent / "packs/habitability_physical_candidate.json"


class GreenhouseBand(Contract):
    name: Literal["none", "weak", "moderate", "strong"]
    minimum: str
    maximum: str
    increment_kelvin: int


class HabitabilityKnowledgePack(Contract):
    schema_version: Literal[1] = 1
    id: Literal["habitability_physical_candidate_v1"]
    validation_scope: Literal["independent_energy_balance_course_pending"]
    operations: list[KnowledgeOperation]
    assumptions: list[str]
    greenhouse_bands: list[GreenhouseBand]
    pending: list[str]
    provenance: dict[str, str]

    @model_validator(mode="after")
    def contract(self):
        inputs = {
            "equilibrium_temp": {"stellar_luminosity": "Lsun", "orbital_radius": "au", "albedo": "fraction"},
            "surface_temp": {"equilibrium_temp": "K", "greenhouse_increment": "K"},
        }
        if len(self.operations) != 2 or {o.id for o in self.operations} != set(inputs):
            raise ValueError("Exactly two declared temperature operations are required")
        for op in self.operations:
            if (
                {k: v.unit for k, v in op.inputs.items()} != inputs[op.id]
                or any(k != v.quantity for k, v in op.inputs.items())
                or op.output_unit != "K"
            ):
                raise ValueError("Habitability dependency/unit contract changed")
        bands = [(b.name, b.minimum, b.maximum, b.increment_kelvin) for b in self.greenhouse_bands]
        if bands != [
            ("none", "0", "0.49", 0),
            ("weak", "0.5", "39.99", 10),
            ("moderate", "40", "59.99", 30),
            ("strong", "60", "100", 100),
        ]:
            raise ValueError("Published course greenhouse bands must be preserved, including gaps")
        if (
            set(self.assumptions)
            != {
                "terrestrial_planet",
                "blackbody_emission",
                "full_surface_heat_redistribution",
                "negligible_internal_heating",
            }
            or len(self.assumptions) != 4
            or not self.pending
        ):
            raise ValueError("Physical assumptions and unresolved rules must remain explicit")
        return self

    @property
    def checksum(self):
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def operation(self, name):
        return next((op for op in self.operations if op.id == name), None)


def load_habitability_pack(path=None):
    return HabitabilityKnowledgePack.model_validate_json(Path(path or DEFAULT_HABITABILITY_PACK).read_text())


class HabitabilityCalculator(LocalCalculator):
    def __init__(self, pack=None):
        super().__init__(pack or load_habitability_pack())

    def execute(self, operation_id, bindings, *, assumptions=None):
        fail = lambda error: CalculationResult(ok=False, operation_id=operation_id, error=error)
        if (
            not isinstance(assumptions, (list, tuple, set, frozenset))
            or not all(isinstance(v, str) for v in assumptions)
            or len(assumptions) != len(set(assumptions))
            or set(assumptions) != set(self.pack.assumptions)
        ):
            return fail("habitability_assumptions_not_explicit")
        if isinstance(bindings, dict):
            for key, allowed in (
                ("albedo", lambda v: 0 <= v < 1),
                ("greenhouse_increment", lambda v: v in {0, 10, 30, 100}),
            ):
                item = bindings.get(key)
                value = item.get("value") if isinstance(item, dict) else None
                if finite_number(value) and not allowed(value):
                    return fail(f"invalid_{key}_domain")
        return self._execute(operation_id, bindings, star_class="main_sequence")

    def greenhouse_reference(self, absorption_text):
        """Look up a supplied percent; don't select any browser control or gas.

        Gaps in the published decimal ranges are errors, not rounded/repaired.
        """
        try:
            if not isinstance(absorption_text, str) or len(absorption_text) > 32:
                raise ValueError
            value = Decimal(absorption_text)
            if not value.is_finite() or not 0 <= value <= 100:
                raise ValueError
        except (ValueError, InvalidOperation):
            return {"ok": False, "error": "invalid_absorption_percent"}
        for band in self.pack.greenhouse_bands:
            if Decimal(band.minimum) <= value <= Decimal(band.maximum):
                return {
                    "ok": True,
                    "name": band.name,
                    "increment_kelvin": band.increment_kelvin,
                    "source": "course_greenhouse_reference",
                    "absorption_percent": absorption_text,
                }
        return {"ok": False, "error": "unresolved_published_band_gap"}

    def reference_answers(self, *args, **kwargs):
        raise ValueError("Habitability course validation is pending; no training oracle is enabled")

    def verify(self):
        # Independently published solar effective temperatures (~278.3 K for
        # zero albedo; ~254.6 K for Bond albedo 0.3), plus additive examples.
        for albedo, expected in [(0, 278.3), (0.3, 254.6)]:
            result = self.execute(
                "equilibrium_temp",
                {
                    "stellar_luminosity": {"value": 1, "unit": "Lsun"},
                    "orbital_radius": {"value": 1, "unit": "au"},
                    "albedo": {"value": albedo, "unit": "fraction"},
                },
                assumptions=self.pack.assumptions,
            )
            if not result.ok or not math.isclose(result.value, expected, rel_tol=0.001):
                raise ValueError("Independent energy-balance case failed")
        for increment in (0, 10, 30, 100):
            result = self.execute(
                "surface_temp",
                {
                    "equilibrium_temp": {"value": 250, "unit": "K"},
                    "greenhouse_increment": {"value": increment, "unit": "K"},
                },
                assumptions=self.pack.assumptions,
            )
            if not result.ok or result.value != 250 + increment:
                raise ValueError("Course increment golden case failed")
        return {
            "valid": True,
            "golden_cases": 6,
            "pack_hash": self.pack.checksum,
            "scope": self.pack.validation_scope,
            "course_validation_passed": False,
            "training_oracle_enabled": False,
        }
