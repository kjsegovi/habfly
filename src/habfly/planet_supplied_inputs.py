"""Additive offline supplied-input scope; no browser authority or stellar relations.

The v1 pack, model encoders and checkpoints remain unchanged. Actual class is
retained by the tool/environment; only an explicitly declared inference view
omits it. Neither operation, source, result nor destination is auto-selected.
"""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from .contracts import Contract, Observation
from .environments.local_stellar import LocalStellarEnv
from .environments.planet_calculations import FIELDS, PlanetCalculationEnv
from .environments.planet_calculations import SCOPE as LEGACY_SCOPE
from .knowledge import CalculationResult, KnowledgeOperation, LocalCalculator, finite_number
from .planet_knowledge import DEFAULT_PLANET_PACK, PlanetCalculator, load_planet_pack

CLASSES = ("main_sequence", "white_dwarf", "red_giant", "supergiant")
SCOPE = "independent_physics_supplied_stellar_inputs_v1"
ADAPTER = "planet_class_omitted_inference_view_v1"
PACK_PATH = Path(__file__).parent / "packs/planet_supplied_inputs_v1.json"
BASE_FILE_SHA256 = "a102f70249a7af610906c6806aa4faa762db94b7c45b3835cf40f895943fc036"
BASE_PACK_HASH = "f5acfb5e1b6ce0a4bcdc49176d20e939b0a2939635f259e4b9a82fdad66299da"
DECLARATION_SHA256 = "9687da7ed6ff793943ef46976dbbd541df916c1572c04746ab7a4856ad9ba17e"


def checksum(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _base():
    if hashlib.sha256(DEFAULT_PLANET_PACK.read_bytes()).hexdigest() != BASE_FILE_SHA256:
        raise ValueError("supplied_planet_base_source_changed")
    pack = load_planet_pack()
    if pack.checksum != BASE_PACK_HASH:
        raise ValueError("supplied_planet_base_pack_changed")
    return pack


class SuppliedOperation(KnowledgeOperation):
    applicable_classes: list[Literal["main_sequence", "white_dwarf", "red_giant", "supergiant"]]


class SuppliedPlanetPack(Contract):
    schema_version: Literal[1] = 1
    id: Literal["planet_explicit_supplied_inputs_v1"]
    scope: Literal["independent_physics_supplied_stellar_inputs_v1"]
    validation_scope: Literal["independent_supplied_input_arithmetic_only_course_pending"]
    declaration: dict
    operations: list[SuppliedOperation]
    assumptions: list[str]
    source_conflicts: list[dict[str, str]]
    provenance: dict[str, str]
    pending: list[str]

    @model_validator(mode="before")
    @classmethod
    def unchanged_math_and_sources(cls, value):
        # Check before coercion: True/1 and integer/float aliases are not source
        # identity. This scope is an exact declared extension, not an editable DSL.
        if type(value) is not dict or checksum(value) != checksum(_pack_value()):
            raise ValueError("supplied_planet_math_or_provenance_changed")
        return value

    @property
    def checksum(self):
        return checksum(self.model_dump(mode="json"))

    def operation(self, identifier):
        return next((op for op in self.operations if op.id == identifier), None)


def _pack_value():
    original = _base()
    raw = PACK_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DECLARATION_SHA256:
        raise ValueError("supplied_planet_declaration_changed")
    declaration = json.loads(raw)
    operations = []
    for op in original.operations:
        operations.append({**op.model_dump(mode="json"), "applicable_classes": list(CLASSES)})
    return {
        "schema_version": 1,
        "id": "planet_explicit_supplied_inputs_v1",
        "scope": SCOPE,
        "validation_scope": "independent_supplied_input_arithmetic_only_course_pending",
        "declaration": declaration,
        "operations": operations,
        **{
            key: getattr(original, key)
            for key in ("assumptions", "source_conflicts", "provenance", "pending")
        },
    }


def load_supplied_planet_pack():
    return SuppliedPlanetPack.model_validate(_pack_value())


class SuppliedPlanetCalculator(LocalCalculator):
    def __init__(self):
        super().__init__(load_supplied_planet_pack())
        self.pack_hash = self.pack.checksum

    def execute(self, operation_id, bindings, *, star_class, assumptions=None):
        if self.pack.checksum != self.pack_hash:
            return CalculationResult(
                ok=False, operation_id=operation_id, error="supplied_planet_pack_changed"
            )
        if type(star_class) is not str or star_class not in CLASSES:
            return CalculationResult(ok=False, operation_id=operation_id, error="unsupported_supplied_class")
        if (
            type(assumptions) is not list
            or any(type(item) is not str for item in assumptions)
            or assumptions != self.pack.assumptions
        ):
            return CalculationResult(
                ok=False, operation_id=operation_id, error="planet_assumptions_not_explicit"
            )
        if operation_id == "planet_radius" and isinstance(bindings, dict):
            row = bindings.get("brightness_drop")
            depth = row.get("value") if isinstance(row, dict) else None
            if finite_number(depth) and depth > 100:
                return CalculationResult(
                    ok=False, operation_id=operation_id, error="brightness_drop_exceeds_100_percent"
                )
        return self._execute(operation_id, bindings, star_class=star_class)

    def reference_answers(self, *args, **kwargs):
        raise ValueError("supplied_planet_no_training_oracle")

    def verify(self):
        # Fixed independent SI/geometry/unit goldens, not the calculator's own outputs.
        cases = [
            ("period_years", {"period_days": (365, "day")}, 1, "yr"),
            ("radial_velocity", {"line_shift": (0.000002187666666666667, "nm")}, 1, "m/s"),
            ("orbital_radius", {"period_years": (8, "yr"), "stellar_mass": (1, "Msun")}, 4, "au"),
            (
                "planet_radius",
                {"brightness_drop": (0.008416799932665601, "%"), "stellar_radius": (1, "Rsun")},
                1,
                "REarth",
            ),
            (
                "planet_mass",
                {"radial_velocity": (1, "m/s"), "orbital_radius": (4, "au"), "stellar_mass": (1, "Msun")},
                22.354,
                "MEarth",
            ),
            (
                "planet_density",
                {"planet_mass": (1, "MEarth"), "planet_radius": (1, "REarth")},
                5.514008418404722,
                "g/cm3",
            ),
        ]
        for actual_class in CLASSES:
            for op, bindings, expected, unit in cases:
                result = self.execute(
                    op,
                    {k: {"value": v, "unit": u} for k, (v, u) in bindings.items()},
                    star_class=actual_class,
                    assumptions=self.pack.assumptions,
                )
                if (
                    not result.ok
                    or result.unit != unit
                    or not math.isclose(result.value, expected, rel_tol=1e-12)
                ):
                    raise ValueError("supplied_planet_independent_golden_failed")
        # Retain the original independent RV coefficient verification as well.
        PlanetCalculator().verify()
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
    """Explicitly relabel a synthetic numeric fixture, not a physical star claim."""
    if (
        actual_class not in CLASSES
        or original.get("scope") != LEGACY_SCOPE
        or original.get("split") != "development"
    ):
        raise ValueError("supplied_planet_requires_recorded_development_fixture")
    case = deepcopy(original)
    case.update(
        scope=SCOPE,
        star_class=actual_class,
        numeric_case_id=original["case_id"],
        case_id=checksum({"scope": SCOPE, "class": actual_class, "numeric_case_id": original["case_id"]}),
        supplied_input_authority="synthetic_inverse_generated_fixture_not_native_measurement",
    )
    return case


class SuppliedPlanetEnv(PlanetCalculationEnv):
    scope = SCOPE

    def __init__(self, adapter, cases, *, max_steps=128):
        if (
            type(adapter) is not SuppliedPlanetCalculator
            or type(max_steps) is not int
            or not 1 <= max_steps <= 128
        ):
            raise ValueError("supplied_planet_environment_contract")
        if not cases or len({c["seed"] for c in cases}) != len(cases):
            raise ValueError("supplied_planet_duplicate_or_missing_cases")
        for case in cases:
            if (
                case.get("scope") != SCOPE
                or case.get("star_class") not in CLASSES
                or case.get("required") != list(FIELDS)
                or set(case.get("expected", {})) != set(FIELDS)
                or case.get("split") != "development"
                or case.get("supplied_input_authority")
                != "synthetic_inverse_generated_fixture_not_native_measurement"
            ):
                raise ValueError("supplied_planet_case_scope")
            for kind, unit in (("stellar_mass", "Msun"), ("stellar_radius", "Rsun")):
                rows = [
                    row
                    for row in case["measurements"].values()
                    if row.get("kind") == kind and row.get("source") == "current star"
                ]
                if (
                    len(rows) != 1
                    or rows[0].get("unit") != unit
                    or not finite_number(rows[0].get("value"))
                    or rows[0]["value"] <= 0
                ):
                    raise ValueError("supplied_planet_explicit_stellar_input_required")
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
        "parent_encoding": "structured_planet_tool_v1",
        "control_encoding": "semantic_planet_tool_v1",
        "transformation": "Omit only values.star_class from a deep-copied model input; preserve original observation and actual class for tool applicability and logs.",
        "class_relabelled_as_main_sequence": False,
        "class_feature_mode": "explicitly_omitted_not_unknown_class_support",
        "policy_choices_repaired": False,
        "calibration_verified": False,
        "native_browser_enabled": False,
    }
    return {**value, "sha256": checksum(value)}


def inference_view(observation, *, expected_adapter_sha256, expected_pack_hash):
    if expected_adapter_sha256 != adapter_manifest()["sha256"]:
        raise ValueError("supplied_planet_adapter_hash_mismatch")
    original = Observation.model_validate(observation).model_dump(mode="json")
    if (
        expected_pack_hash != load_supplied_planet_pack().checksum
        or original["calculation"].get("pack_hash") != expected_pack_hash
    ):
        raise ValueError("supplied_planet_inference_pack_mismatch")
    if original["values"].get("star_class") not in CLASSES or original["progress"].get("scope") != SCOPE:
        raise ValueError("supplied_planet_inference_scope_mismatch")
    view = deepcopy(original)
    del view["values"]["star_class"]
    return Observation.model_validate(view)
