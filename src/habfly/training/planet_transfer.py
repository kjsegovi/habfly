"""Explicit semantic-column transfer, not an expert workflow or action repair."""

from habfly.model import planet_tool_state as planet
from habfly.model import tool_state as stellar

from .planet_calculations import new_policy

# These remain stellar properties in both tasks. Do not alias unrelated
# quantities merely because their calculations occupy similar sequence steps.
QUANTITY_ALIASES = {"mass": "stellar_mass", "radius": "stellar_radius"}


def transfer_stellar_policy(parent):
    if (
        parent.observation_encoding != "structured_tool_v6"
        or parent.control_encoding != "semantic_tool_v1"
        or parent.selection_mode != "measurement_result_v3"
        or parent.hidden_size != 16
    ):
        raise ValueError("Planet transfer requires the explicit stellar v6 tool contract")
    child = new_policy(parent.graph)
    weights = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    mappings = {}

    def remap(name, width, pairs):
        old = weights[name]
        new = old.new_zeros((parent.hidden_size, width))
        for source, target in pairs:
            new[:, target] = old[:, source]
        weights[name] = new
        mappings[name] = [{"old_column": a, "new_column": b} for a, b in pairs]

    quantities = [
        (stellar.QUANTITIES.index(old), planet.QUANTITIES.index(new)) for old, new in QUANTITY_ALIASES.items()
    ]
    # Ten public state flags per quantity, then four operation-status flags,
    # displayed required-field membership, then the supplied class indicators.
    old_base, new_base = len(stellar.QUANTITIES) * 10, len(planet.QUANTITIES) * 10
    pairs = [(a * 10 + i, b * 10 + i) for a, b in quantities for i in range(10)]
    pairs += [(old_base + i, new_base + i) for i in range(4)]
    pairs += [(old_base + 4 + a, new_base + 4 + b) for a, b in quantities]
    pairs += [
        (old_base + 4 + len(stellar.QUANTITIES) + i, new_base + 4 + len(planet.QUANTITIES) + i)
        for i in range(len(stellar.STAR_CLASSES))
    ]
    remap("tool_state_projection.weight", planet.STATE_WIDTH, pairs)

    labels = {label: label for label in stellar.CONTROL_LABELS if label in planet.CONTROL_LABELS}
    labels["Check stellar analysis"] = "Check derived planet analysis"
    labels.update({f"Unit for {old}": f"Unit for {new}" for old, new in QUANTITY_ALIASES.items()})
    pairs = [
        (stellar.CONTROL_LABELS.index(old), planet.CONTROL_LABELS.index(new)) for old, new in labels.items()
    ]
    remap("control_projection.weight", len(planet.CONTROL_LABELS), pairs)

    units = [
        (len(stellar.QUANTITIES) + i, len(planet.QUANTITIES) + planet.UNITS.index(unit))
        for i, unit in enumerate(stellar.YEAR_UNITS)
        if unit in planet.UNITS
    ]
    remap("option_projection.weight", planet.OPTION_WIDTH, quantities + units)
    child.load_state_dict(weights, strict=True)
    child.calibration = {"status": "uncalibrated", "reason": "explicit_stellar_to_planet_transfer"}
    child.action_temperature = child.target_temperature = 1.0
    return child, {
        "version": 1,
        "source_encoding": parent.observation_encoding,
        "target_encoding": child.observation_encoding,
        "quantity_aliases": QUANTITY_ALIASES,
        "control_aliases": labels,
        "projection_columns": mappings,
        "new_columns": "zero_initialized_then_learned",
        "other_parameters": "exact_copy",
        "biological_topology": "unchanged",
        "formula_or_correct_action_rules_added": False,
    }
