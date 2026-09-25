"""Versioned visible color reference. Its oracle is for private labels, not inference.

The public guard can reject uncovered/ambiguous measurements but never returns
a color to the model. No changes to the already-promoted arithmetic pack.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .contracts import Contract

COLOR_LABELS = ("UV", "Violet", "Blue", "Cyan", "Green", "Yellow", "Orange", "Red", "IR")
COLOR_CONTROL = "Peak wavelength color"
COLOR_CONTROLS = ("Measurement or result", COLOR_CONTROL, "Check color selection")


class ColorReferenceError(ValueError):
    pass


class ColorBand(Contract):
    label: str
    minimum: float = Field(ge=0)
    maximum: float | None
    include_minimum: bool
    include_maximum: bool

    @model_validator(mode="after")
    def interval(self):
        if self.maximum is not None and self.maximum <= self.minimum:
            raise ValueError("Invalid color interval")
        return self

    def contains(self, value):
        lower = value >= self.minimum if self.include_minimum else value > self.minimum
        upper = self.maximum is None or (
            value <= self.maximum if self.include_maximum else value < self.maximum
        )
        return lower and upper


class ColorReference(Contract):
    id: Literal["habworlds-peak-wavelength-color"]
    version: Literal[1]
    quantity: Literal["wavelength"]
    unit: Literal["nm"]
    description: str
    bands: list[ColorBand]
    ambiguity_policy: Literal["abstain_when_zero_or_multiple_bands_match"]
    pending: list[str]
    sources: list[dict[str, str]]

    @model_validator(mode="after")
    def labels(self):
        if tuple(b.label for b in self.bands) != COLOR_LABELS or not self.sources:
            raise ValueError("Color reference requires all nine bands and provenance")
        return self

    @property
    def checksum(self):
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()

    def _matches(self, value, unit):
        if unit != self.unit:
            raise ColorReferenceError("incompatible_wavelength_unit")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ColorReferenceError("missing_or_nonnumeric_wavelength")
        if not math.isfinite(value) or value <= 0:
            raise ColorReferenceError("invalid_wavelength_domain")
        return [b.label for b in self.bands if b.contains(value)]

    def guard(self, value, unit):
        """Coverage check only. Does not return, choose or repair a color."""
        matches = self._matches(value, unit)
        if len(matches) != 1:
            raise ColorReferenceError("ambiguous_color_boundary" if matches else "uncovered_color_gap")

    def private_label(self, value, unit="nm"):
        """Only expert/case generation and grading may consume this label."""
        self.guard(value, unit)
        return self._matches(value, unit)[0]

    def card(self):
        return {
            "id": self.id,
            "description": self.description,
            "inputs": {"wavelength": {"quantity": self.quantity, "unit": self.unit}},
            "bands": [b.model_dump() for b in self.bands],
            "ambiguity_policy": self.ambiguity_policy,
            "pending": self.pending,
            "sources": self.sources,
        }


def load_color_reference(path=None):
    path = Path(path) if path else Path(__file__).parent / "packs/stellar_color.json"
    return ColorReference.model_validate_json(path.read_text())


def validate_color_reference(reference):
    # Independently transcribed interior and unique endpoint cases. No generator
    # or band-loop is used to construct these expected answers.
    golden = (
        (212, "UV"),
        (380, "Violet"),
        (410, "Violet"),
        (460, "Blue"),
        (480, "Cyan"),
        (494, "Cyan"),
        (495, "Green"),
        (530, "Green"),
        (580, "Yellow"),
        (605, "Orange"),
        (680, "Red"),
        (744, "Red"),
        (1000, "IR"),
    )
    for value, expected in golden:
        if reference.private_label(value) != expected:
            raise ValueError("Color reference golden case mismatch")
    for value in (450, 475, 570, 590, 620, 494.5):
        try:
            reference.guard(value, "nm")
        except ColorReferenceError:
            continue
        raise ValueError("Color reference silently resolves an ambiguous case")
    return {"golden_cases": len(golden), "ambiguous_cases": 6, "reference_hash": reference.checksum}


def color_features(observation):
    """Raw selected wavelength and visible occupancy, never interval membership.

    Magnitude reaches sensory neurons, not a deterministic answer lookup. This
    head must learn the band decision; old numeric-policy encodings are unchanged.
    """
    values, state = observation.get("values", {}), observation.get("calculation", {})
    row = values.get("measurements", {}).get(state.get("source"), {})
    value = row.get("value")
    valid = (
        row.get("kind") == "wavelength"
        and row.get("unit") == "nm"
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )
    return [
        float(valid),
        math.log(value) - math.log(550.0) if valid else 0.0,
        float(bool(values.get("answers", {}).get("color"))),
    ]
