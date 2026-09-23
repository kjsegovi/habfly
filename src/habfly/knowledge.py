"""Versioned stellar reference cards and a bounded, deterministic local tool.

No model-authored expressions, downloads, credentials, or spreadsheet access.
The reference-answer helper is for private simulator grading, not policy use.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .content import calculate
from .contracts import Contract

DEFAULT_PACK = Path(__file__).parent / "packs" / "stellar_knowledge.json"
STAR_CLASSES = ("main_sequence", "white_dwarf", "giant")


class KnowledgeInput(Contract):
    quantity: str
    unit: str
    positive: bool = True


class KnowledgeOperation(Contract):
    id: str
    description: str
    inputs: dict[str, KnowledgeInput]
    expression: str
    constants: dict[str, float] = Field(default_factory=dict)
    output_unit: str
    applicable_classes: list[Literal["main_sequence", "white_dwarf", "giant"]] = Field(
        default_factory=lambda: list(STAR_CLASSES)
    )
    sources: list[dict[str, str]]

    @model_validator(mode="after")
    def grammar(self):
        if not self.id.isidentifier() or len(self.id) > 18 or not self.inputs or not self.sources:
            raise ValueError("Operation requires a short identifier, inputs, and provenance")
        if set(self.inputs) & set(self.constants):
            raise ValueError("Input and constant names must not overlap")
        if not self.applicable_classes or not all(math.isfinite(v) for v in self.constants.values()):
            raise ValueError("Invalid applicability or constants")
        try:
            tree = ast.parse(self.expression, mode="eval")
        except (SyntaxError, RecursionError) as exc:
            raise ValueError("Invalid knowledge expression") from exc
        nodes = list(ast.walk(tree))
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.UAdd,
            ast.USub,
        )
        if len(nodes) > 64 or any(not isinstance(n, allowed) for n in nodes):
            raise ValueError("Unsupported knowledge expression")
        names = {n.id for n in nodes if isinstance(n, ast.Name)}
        if not names <= set(self.inputs) | set(self.constants):
            raise ValueError("Undeclared expression name")
        if any(
            isinstance(n, ast.Constant) and (type(n.value) not in {int, float} or not math.isfinite(n.value))
            for n in nodes
        ):
            raise ValueError("Invalid expression constant")
        return self


class KnowledgePack(Contract):
    version: Literal[1] = 1
    id: str
    title: str
    description: str
    operations: list[KnowledgeOperation]
    references: dict[str, str]
    pending: list[str]
    provenance: dict[str, str]

    @model_validator(mode="after")
    def dependencies(self):
        ids = [op.id for op in self.operations]
        if (
            set(ids) != {"distance", "luminosity", "temperature", "mass", "radius", "lifetime"}
            or len(ids) != 6
        ):
            raise ValueError("Stellar v1 requires exactly the six stellar operations")
        for op in self.operations:
            if op.id in {"mass", "radius", "lifetime"} and op.applicable_classes != ["main_sequence"]:
                raise ValueError("Derived mass, radius, and lifetime require supplied main-sequence class")
            if any(
                spec.quantity not in set(ids) | {"flux", "parallax", "wavelength"}
                for spec in op.inputs.values()
            ):
                raise ValueError("Unknown input quantity")
        graph = {op.id: {v.quantity for v in op.inputs.values()} & set(ids) for op in self.operations}
        visited, active = set(), set()

        def visit(node):
            if node in active:
                raise ValueError("Calculation dependency cycle")
            if node not in visited:
                active.add(node)
                for child in graph[node]:
                    visit(child)
                active.remove(node)
                visited.add(node)

        for node in graph:
            visit(node)
        return self

    @property
    def checksum(self):
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def operation(self, operation_id):
        return next((op for op in self.operations if op.id == operation_id), None)

    def content_identity(self):
        return {
            "task": "stellar",
            "calculation_backend": "local",
            "calculation_mode": "local_tool_assisted",
            "knowledge_pack_id": self.id,
            "knowledge_pack_hash": self.checksum,
            "version": self.version,
        }


def load_knowledge_pack(path=None):
    return KnowledgePack.model_validate_json(Path(path or DEFAULT_PACK).read_text())


class CalculationResult(Contract):
    ok: bool
    operation_id: str
    value: float | None = None
    unit: str | None = None
    error: str | None = None


class LocalCalculator:
    backend = "local"

    def __init__(self, pack: KnowledgePack):
        self.config = self.pack = pack
        self.generation = 0

    def invalidate(self):
        self.generation += 1

    def reset(self):
        self.invalidate()

    def close(self):
        pass

    def execute(self, operation_id, bindings, star_class):
        def fail(reason):
            return CalculationResult(ok=False, operation_id=operation_id, error=reason)

        op = self.pack.operation(operation_id)
        if op is None:
            return fail("unknown_operation")
        if star_class not in op.applicable_classes:
            return fail("operation_not_applicable")
        if not isinstance(bindings, dict) or set(bindings) != set(op.inputs):
            return fail("missing_or_extra_inputs")
        values = dict(op.constants)
        for name, spec in op.inputs.items():
            binding = bindings[name]
            if not isinstance(binding, dict):
                return fail("invalid_binding")
            value = binding.get("value")
            if type(value) not in {int, float} or not math.isfinite(value):
                return fail("input_not_finite_number")
            if binding.get("unit") != spec.unit:
                return fail("incompatible_unit")
            if spec.positive and value <= 0:
                return fail("input_must_be_positive")
            # Semantic correctness belongs to the learner: a same-unit distractor
            # is evaluated as bound, not swapped for the correct measurement.
            values[name] = value
        try:
            value = calculate(op.expression, values)
        except ValueError:
            return fail("invalid_calculation_domain")
        if not math.isfinite(value) or value <= 0:
            return fail("invalid_calculation_result")
        return CalculationResult(ok=True, operation_id=operation_id, value=value, unit=op.output_unit)

    def reference_answers(self, measurements, star_class):
        """Private grading only. Policy execution calls execute with its own bindings."""
        from .environments.stellar_common import UNITS

        known = {k: {"value": v, "unit": UNITS[k]} for k, v in measurements.items()}
        answers = {}

        def resolve(operation_id):
            if operation_id in known:
                return known[operation_id]
            op = self.pack.operation(operation_id)
            bindings = {name: resolve(spec.quantity) for name, spec in op.inputs.items()}
            result = self.execute(operation_id, bindings, star_class)
            if not result.ok:
                raise ValueError(f"Invalid private reference: {result.error}")
            known[operation_id] = {"value": result.value, "unit": result.unit}
            answers[operation_id] = result.value
            return known[operation_id]

        for op in self.pack.operations:
            if star_class in op.applicable_classes:
                resolve(op.id)
        return answers

    def verify(self):
        from .environments.stellar_common import UNITS

        # Fixed, independent reference values, not recomputed with execute().
        cases = [
            ("distance", {"parallax": {"value": 0.032, "unit": "arcsec"}}, 101.875),
            (
                "luminosity",
                {"flux": {"value": 5.15e-13, "unit": "W/m2"}, "distance": {"value": 101.875, "unit": "ly"}},
                0.015708042022455068,
            ),
            ("temperature", {"wavelength": {"value": 212, "unit": "nm"}}, 13668.719339622641),
            ("mass", {"luminosity": {"value": 0.015708042022455068, "unit": "Lsun"}}, 0.3052153006226406),
            (
                "radius",
                {
                    "luminosity": {"value": 0.015708042022455068, "unit": "Lsun"},
                    "temperature": {"value": 13668.719339622641, "unit": "K"},
                },
                0.02256635219411464,
            ),
            ("lifetime", {"mass": {"value": 0.3052153006226406, "unit": "Msun"}}, 194305121024.2034),
            ("mass", {"luminosity": {"value": 1, "unit": "Lsun"}}, 1),
            (
                "radius",
                {"luminosity": {"value": 1, "unit": "Lsun"}, "temperature": {"value": 5800, "unit": "K"}},
                1,
            ),
            ("lifetime", {"mass": {"value": 1, "unit": "Msun"}}, 10000000000),
        ]
        for name, inputs, expected in cases:
            result = self.execute(name, inputs, "main_sequence")
            if (
                not result.ok
                or result.unit != UNITS[name]
                or not math.isclose(result.value, expected, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError(f"Knowledge golden case failed: {name}")
        return {
            "verified": True,
            "cases": len(cases),
            "knowledge_pack_hash": self.pack.checksum,
            "calculation_mode": "local_tool_assisted",
            "reference_kind": "fixed_spreadsheet_derived_goldens",
        }
