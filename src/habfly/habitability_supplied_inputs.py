"""Explicit actual-class temperature arithmetic; offline and additive only.

No old pack or model encoding is changed. The authoritative environment/tool
retains the supplied stellar class; a separately hash-pinned inference view
omits that one field without choosing or repairing any action or binding.
"""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from .contracts import Contract, Observation
from .environments.habitability_calculations import FIELDS, QUANTITY_UNITS, HabitabilityCalculationEnv
from .environments.habitability_calculations import SCOPE as LEGACY_SCOPE
from .environments.local_stellar import LocalStellarEnv
from .habitability_knowledge import (
    DEFAULT_HABITABILITY_PACK,
    GreenhouseBand,
    HabitabilityCalculator,
    load_habitability_pack,
)
from .knowledge import CalculationResult, KnowledgeOperation, LocalCalculator, finite_number

CLASSES = ("main_sequence", "white_dwarf", "red_giant", "supergiant")
SCOPE = "independent_energy_balance_supplied_class_inputs_v1"
ADAPTER = "temperature_class_omitted_inference_view_v1"
PACK_PATH = Path(__file__).parent / "packs/habitability_supplied_inputs_v1.json"
BASE_FILE_SHA256 = "806f8be3def21ae37b11432fc3964cc2f231b13206752d5bdb82b3a238fa2b47"
BASE_PACK_HASH = "95fec7129d126ed7b5866916538f7df16833fecdad8e4606d74f63015abe23f5"
DECLARATION_SHA256 = "a0e14aeb96bf35df2a6773dfbdc2feeb3d8b92b376e4e3f5fbbf50d10c48461e"


def checksum(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _base():
    if hashlib.sha256(DEFAULT_HABITABILITY_PACK.read_bytes()).hexdigest() != BASE_FILE_SHA256:
        raise ValueError("supplied_temperature_base_source_changed")
    pack = load_habitability_pack()
    if pack.checksum != BASE_PACK_HASH:
        raise ValueError("supplied_temperature_base_pack_changed")
    return pack


class SuppliedTemperatureOperation(KnowledgeOperation):
    applicable_classes: list[Literal["main_sequence", "white_dwarf", "red_giant", "supergiant"]]


class SuppliedTemperaturePack(Contract):
    schema_version: Literal[1]
    id: Literal["temperature_explicit_supplied_class_v1"]
    scope: Literal["independent_energy_balance_supplied_class_inputs_v1"]
    validation_scope: Literal["independent_supplied_class_energy_balance_only_course_pending"]
    declaration: dict
    operations: list[SuppliedTemperatureOperation]
    assumptions: list[str]
    greenhouse_bands: list[GreenhouseBand]
    pending: list[str]
    provenance: dict[str, str]

    @model_validator(mode="before")
    @classmethod
    def exact_declared_extension(cls, value):
        # Validate before Pydantic can coerce numeric/bool source identities.
        if type(value) is not dict or checksum(value) != checksum(_pack_value()):
            raise ValueError("supplied_temperature_math_or_provenance_changed")
        return value

    @property
    def checksum(self):
        return checksum(self.model_dump(mode="json"))

    def operation(self, identifier):
        return next((operation for operation in self.operations if operation.id == identifier), None)


def _pack_value():
    original = _base().model_dump(mode="json")
    raw = PACK_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DECLARATION_SHA256:
        raise ValueError("supplied_temperature_declaration_changed")
    return {
        "schema_version": 1,
        "id": "temperature_explicit_supplied_class_v1",
        "scope": SCOPE,
        "validation_scope": "independent_supplied_class_energy_balance_only_course_pending",
        "declaration": json.loads(raw),
        "operations": [{**op, "applicable_classes": list(CLASSES)} for op in original["operations"]],
        **{key: original[key] for key in ("assumptions", "greenhouse_bands", "pending", "provenance")},
    }


def load_supplied_temperature_pack():
    return SuppliedTemperaturePack.model_validate(_pack_value())


class SuppliedTemperatureCalculator(LocalCalculator):
    def __init__(self):
        super().__init__(load_supplied_temperature_pack())
        self.pack_hash = self.pack.checksum

    def execute(self, operation_id, bindings, *, star_class, assumptions=None):
        def fail(reason):
            return CalculationResult(ok=False, operation_id=operation_id, error=reason)

        if self.pack.checksum != self.pack_hash:
            return fail("supplied_temperature_pack_changed")
        if type(star_class) is not str or star_class not in CLASSES:
            return fail("unsupported_supplied_class")
        if (
            type(assumptions) is not list
            or any(type(item) is not str for item in assumptions)
            or assumptions != self.pack.assumptions
        ):
            return fail("temperature_assumptions_not_explicit")
        if isinstance(bindings, dict):
            for key, allowed in (
                ("albedo", lambda value: 0 <= value < 1),
                ("greenhouse_increment", lambda value: value in {0, 10, 30, 100}),
            ):
                binding = bindings.get(key)
                value = binding.get("value") if isinstance(binding, dict) else None
                if finite_number(value) and not allowed(value):
                    return fail(f"invalid_{key}_domain")
        return self._execute(operation_id, bindings, star_class=star_class)

    def greenhouse_reference(self, absorption_text):
        if self.pack.checksum != self.pack_hash:
            return {"ok": False, "error": "supplied_temperature_pack_changed"}
        # Pure exact published lookup; never selects a gas/menu or repairs gaps.
        return HabitabilityCalculator.greenhouse_reference(self, absorption_text)

    def reference_answers(self, *args, **kwargs):
        raise ValueError("supplied_temperature_no_training_oracle")

    def verify(self):
        # Keep the existing independently sourced approximate energy-balance
        # goldens plus exact additive goldens; no new class physics is asserted.
        for actual_class in CLASSES:
            for albedo, expected in ((0, 278.3), (0.3, 254.6)):
                result = self.execute(
                    "equilibrium_temp",
                    {
                        "stellar_luminosity": {"value": 1, "unit": "Lsun"},
                        "orbital_radius": {"value": 1, "unit": "au"},
                        "albedo": {"value": albedo, "unit": "fraction"},
                    },
                    star_class=actual_class,
                    assumptions=self.pack.assumptions,
                )
                if (
                    not result.ok
                    or result.unit != "K"
                    or not math.isclose(result.value, expected, rel_tol=0.001)
                ):
                    raise ValueError("supplied_temperature_independent_golden_failed")
            for increment in (0, 10, 30, 100):
                result = self.execute(
                    "surface_temp",
                    {
                        "equilibrium_temp": {"value": 250, "unit": "K"},
                        "greenhouse_increment": {"value": increment, "unit": "K"},
                    },
                    star_class=actual_class,
                    assumptions=self.pack.assumptions,
                )
                if not result.ok or result.unit != "K" or result.value != 250 + increment:
                    raise ValueError("supplied_temperature_additive_golden_failed")
        return {
            "valid": True,
            "operation_golden_cases": 24,
            "pack_hash": self.pack.checksum,
            "scope": SCOPE,
            "course_acceptance_passed": False,
            "scientific_verified": False,
            "native_browser_enabled": False,
            "training_oracle_enabled": False,
        }


def supplied_case(original, actual_class):
    """Pair a synthetic numeric fixture with a label, not a population claim."""
    if (
        actual_class not in CLASSES
        or original.get("scope") != LEGACY_SCOPE
        or original.get("split") != "development"
        or original.get("planet_class") != "terrestrial"
    ):
        raise ValueError("supplied_temperature_requires_recorded_development_fixture")
    value = deepcopy(original)
    value.update(
        scope=SCOPE,
        star_class=actual_class,
        numeric_case_id=original["case_id"],
        case_id=checksum({"scope": SCOPE, "class": actual_class, "numeric_case_id": original["case_id"]}),
        supplied_input_authority="synthetic_inverse_generated_fixture_not_native_measurement",
    )
    return value


class SuppliedTemperatureEnv(HabitabilityCalculationEnv):
    scope = SCOPE

    def __init__(self, adapter, cases, *, max_steps=128):
        if (
            type(adapter) is not SuppliedTemperatureCalculator
            or type(max_steps) is not int
            or not 1 <= max_steps <= 128
        ):
            raise ValueError("supplied_temperature_environment_contract")
        if not cases or len({case["seed"] for case in cases}) != len(cases):
            raise ValueError("supplied_temperature_duplicate_or_missing_cases")
        for case in cases:
            if (
                case.get("scope") != SCOPE
                or case.get("star_class") not in CLASSES
                or case.get("planet_class") != "terrestrial"
                or case.get("required") != list(FIELDS)
                or set(case.get("expected", {})) != set(FIELDS)
                or case.get("split") != "development"
                or case.get("supplied_input_authority")
                != "synthetic_inverse_generated_fixture_not_native_measurement"
            ):
                raise ValueError("supplied_temperature_case_scope")
            for kind in ("stellar_luminosity", "orbital_radius", "albedo", "greenhouse_increment"):
                rows = [
                    row
                    for row in case["measurements"].values()
                    if row.get("kind") == kind and row.get("source") == "current star"
                ]
                if (
                    len(rows) != 1
                    or rows[0].get("unit") != QUANTITY_UNITS[kind]
                    or not finite_number(rows[0].get("value"))
                ):
                    raise ValueError("supplied_temperature_explicit_input_required")
                value = rows[0]["value"]
                allowed = (
                    0 <= value < 1
                    if kind == "albedo"
                    else value in {0, 10, 30, 100}
                    if kind == "greenhouse_increment"
                    else value > 0
                )
                if not allowed:
                    raise ValueError("supplied_temperature_explicit_input_domain")
        adapter.verify()
        LocalStellarEnv.__init__(self, adapter, deepcopy(cases), max_steps=max_steps)

    def calculate(self, bound):
        return self.adapter.execute(
            self.operation, bound, star_class=self.case["star_class"], assumptions=self.pack.assumptions
        )

    def observe(self):
        observation = super().observe()
        observation.progress.update(scope=SCOPE, native_browser_enabled=False, scientific_verified=False)
        return observation


def adapter_manifest():
    value = {
        "version": 1,
        "id": ADAPTER,
        "scope": SCOPE,
        "parent_encoding": "structured_habitability_tool_v1",
        "control_encoding": "semantic_habitability_tool_v1",
        "transformation": "Omit only values.star_class from a deep-copied model input; retain actual class in authoritative observation and calculation applicability.",
        "class_feature_mode": "explicitly_omitted_not_unknown_class_support",
        "class_relabelled_as_main_sequence": False,
        "policy_choices_repaired": False,
        "calibration_verified": False,
        "native_browser_enabled": False,
    }
    return {**value, "sha256": checksum(value)}


def inference_view(observation, *, expected_adapter_sha256, expected_pack_hash):
    if expected_adapter_sha256 != adapter_manifest()["sha256"]:
        raise ValueError("supplied_temperature_adapter_hash_mismatch")
    original = Observation.model_validate(observation).model_dump(mode="json")
    if (
        expected_pack_hash != load_supplied_temperature_pack().checksum
        or original["calculation"].get("pack_hash") != expected_pack_hash
    ):
        raise ValueError("supplied_temperature_inference_pack_mismatch")
    if (
        original["values"].get("star_class") not in CLASSES
        or original["progress"].get("scope") != SCOPE
        or original["values"].get("planet_class") != "terrestrial"
    ):
        raise ValueError("supplied_temperature_inference_scope_mismatch")
    view = deepcopy(original)
    del view["values"]["star_class"]
    return Observation.model_validate(view)
