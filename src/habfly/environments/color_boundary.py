"""Reference-derived training curriculum, independent of development failures.

Sampling metadata and expert labels are private case data, never observations.
This module is not used by policy inference.
"""

import random
from collections import Counter
from copy import deepcopy

from habfly.color_reference import COLOR_LABELS
from habfly.environments.color import SAMPLE_RANGES

BOUNDARY_SEED = 10200000
BOUNDARY_TEMPLATE = (
    "For the current star, choose the band containing its peak wavelength; ignore other sources."
)
SAMPLING_RECIPE = "reference_edges_two_samples_each_side_1_to_10_percent_capped_span_v1"


def new_boundary_cases(reference, count=32):
    if count not in (2, 32):
        raise ValueError("Boundary budget is 32 pilot cases or two engineering smoke cases")
    cases = []
    for index in range(count):
        seed = BOUNDARY_SEED + index
        rng = random.Random(seed)
        transition, side = index // 4, index % 2
        left, right = reference.bands[transition : transition + 2]
        span = min(
            50.0,
            left.maximum - left.minimum,
            right.maximum - right.minimum if right.maximum is not None else 50.0,
        )
        # On a gap, use each printed edge independently; never interpolate it.
        edge = left.maximum if side == 0 else right.minimum
        offset = span * rng.uniform(0.01, 0.10)
        value = edge - offset if side == 0 else edge + offset
        label = reference.private_label(value)
        if label != (left.label if side == 0 else right.label) or not 100 <= value <= 1800:
            raise ValueError("Boundary sample does not match the supplied reference")
        other_range = SAMPLE_RANGES[(COLOR_LABELS.index(label) + 4) % len(COLOR_LABELS)]
        rows = [
            ("wavelength", "nm", "current star", value),
            ("wavelength", "nm", "reference star", rng.uniform(*other_range)),
            ("temperature", "K", "current star", rng.uniform(2500, 25000)),
            ("wavelength", "K", "current star", rng.uniform(100, 1800)),
        ]
        rng.shuffle(rows)
        measurements, expected_source = {}, None
        for kind, unit, source, number in rows:
            ident = f"m-{rng.getrandbits(80):x}"
            measurements[ident] = {"kind": kind, "unit": unit, "source": source, "value": number}
            if (kind, unit, source) == ("wavelength", "nm", "current star"):
                expected_source = ident
        cases.append(
            {
                "seed": seed,
                "case_id": f"color-{seed}",
                "split": "train",
                "instruction": BOUNDARY_TEMPLATE,
                "measurements": measurements,
                "expected_source": expected_source,
                "expected_color": label,
                "reference_hash": reference.checksum,
                "curriculum_sampling": {"transition": transition, "side": side, "replicate": index % 4 // 2},
            }
        )
    return cases


def boundary_curriculum(reference, original):
    """Half retained cases, half new cases; balance the combined color counts."""
    if len(original) not in (4, 64) or len({c["case_id"] for c in original}) != len(original):
        raise ValueError("Boundary curriculum requires a unique four- or 64-case original training set")
    new = new_boundary_cases(reference, len(original) // 2)
    counts = Counter(c["expected_color"] for c in new)
    available = deepcopy(original)
    random.Random(0).shuffle(available)
    retained = []
    for _ in range(len(original) // 2):
        case = min(
            available, key=lambda c: (counts[c["expected_color"]], COLOR_LABELS.index(c["expected_color"]))
        )
        available.remove(case)
        retained.append(case)
        counts[case["expected_color"]] += 1
    mixed = retained + new
    random.Random(0).shuffle(mixed)
    if len(original) == 64 and (
        set(counts) != set(COLOR_LABELS) or max(counts.values()) - min(counts.values()) > 1
    ):
        raise ValueError("Original cases cannot support a balanced boundary curriculum")
    return mixed
