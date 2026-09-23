"""Backend-independent stellar cases and private numeric grading."""

import hashlib
import json
import math
import random

QUANTITIES = ("distance", "luminosity", "temperature", "mass", "radius", "lifetime")
UNITS = {
    "flux": "W/m2",
    "parallax": "arcsec",
    "wavelength": "nm",
    "distance": "ly",
    "luminosity": "Lsun",
    "temperature": "K",
    "mass": "Msun",
    "radius": "Rsun",
    "lifetime": "yr",
}
TEMPLATES = {
    "train": "Use the spreadsheet to analyze the current star. Copy its stellar results and select units.",
    "calibration": "Enter the current star's measurements in the sheet, then complete its stellar fields.",
    "development": "Reconstruct the supplied star using the spreadsheet. Ignore reference measurements.",
    "test": "Determine this star's properties with the permitted spreadsheet and fill the appropriate fields.",
    "gate": "Complete a stellar analysis using only the current star measurements and the sheet.",
}
LOCAL_TEMPLATES = {
    "train": "Analyze the current star with the local calculation reference. Choose operations and inputs, then copy results and units.",
    "calibration": "Use the permitted calculation tool to reconstruct the current star. Bind its measurements and complete the fields.",
    "development": "Determine the supplied star's properties with local reference cards. Ignore reference-star measurements.",
    "test": "Complete a stellar analysis: select suitable calculations, supply current-star inputs, and transfer the appropriate quantities and units.",
    "gate": "Solve the current star's stellar fields using only the visible local tool and measurements.",
}


def make_case(seed, split):
    if split not in TEMPLATES:
        raise ValueError("Unknown stellar split")
    rng = random.Random(seed)
    star_class = rng.choice(["main_sequence", "main_sequence", "white_dwarf", "giant"])
    inputs = {
        "flux": float(f"{10 ** rng.uniform(-14, -8):.8g}"),
        "parallax": float(f"{rng.uniform(0.005, 0.2):.8g}"),
        "wavelength": float(f"{rng.uniform(120, 1600):.8g}"),
    }
    measurements = []
    for kind, value in inputs.items():
        measurements.extend(
            [
                {"kind": kind, "value": value, "unit": UNITS[kind], "source": "current star"},
                {"kind": kind, "value": value * 1.7, "unit": UNITS[kind], "source": "reference star"},
            ]
        )
    rng.shuffle(measurements)
    return {
        "seed": seed,
        "split": split,
        "case_id": hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "star_class": star_class,
        "inputs": inputs,
        "measurements": {f"m{i}": value for i, value in enumerate(measurements)},
        "required": list(QUANTITIES if star_class == "main_sequence" else QUANTITIES[:3]),
    }


def grade_fields(case, answers, units):
    numeric = {
        k
        for k in case["required"]
        if k in answers and math.isclose(answers[k], case["expected"][k], rel_tol=1e-6, abs_tol=1e-9)
    }
    unit_correct = {k for k in case["required"] if units.get(k) == UNITS[k]}
    return numeric & unit_correct, {
        "numeric_fields_correct": len(numeric),
        "required_fields": len(case["required"]),
        "units_correct": len(unit_correct),
    }
