"""Explicit planet-to-temperature workflow transfer, without action rules."""

from habfly.model import habitability_tool_state as temperature
from habfly.model import planet_tool_state as planet

from .habitability_calculations import new_policy


def transfer_planet_policy(parent):
    if (
        parent.observation_encoding != "structured_planet_tool_v1"
        or parent.control_encoding != "semantic_planet_tool_v1"
        or parent.selection_mode != "measurement_result_v3"
        or parent.hidden_size != 16
    ):
        raise ValueError("Temperature transfer requires the explicit planet v1 tool contract")
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

    # Identical physical quantities only. No relabeling density/radius/mass as
    # temperatures or albedo merely to borrow their positions in the workflow.
    shared = [q for q in planet.QUANTITIES if q in temperature.QUANTITIES]
    quantities = [(planet.QUANTITIES.index(q), temperature.QUANTITIES.index(q)) for q in shared]
    old_base, new_base = len(planet.QUANTITIES) * 10, len(temperature.QUANTITIES) * 10
    pairs = [(a * 10 + i, b * 10 + i) for a, b in quantities for i in range(10)]
    pairs += [(old_base + i, new_base + i) for i in range(4)]
    pairs += [(old_base + 4 + a, new_base + 4 + b) for a, b in quantities]
    pairs += [
        (old_base + 4 + len(planet.QUANTITIES) + i, new_base + 4 + len(temperature.QUANTITIES) + i)
        for i in range(len(planet.STAR_CLASSES))
    ]
    remap("tool_state_projection.weight", temperature.STATE_WIDTH, pairs)
    labels = {label: label for label in planet.CONTROL_LABELS if label in temperature.CONTROL_LABELS}
    labels["Check derived planet analysis"] = "Check supplied temperature calculations"
    remap(
        "control_projection.weight",
        len(temperature.CONTROL_LABELS),
        [(planet.CONTROL_LABELS.index(a), temperature.CONTROL_LABELS.index(b)) for a, b in labels.items()],
    )
    units = [
        (len(planet.QUANTITIES) + i, len(temperature.QUANTITIES) + temperature.UNITS.index(unit))
        for i, unit in enumerate(planet.UNITS)
        if unit in temperature.UNITS
    ]
    remap("option_projection.weight", temperature.OPTION_WIDTH, quantities + units)
    child.load_state_dict(weights, strict=True)
    child.calibration = {"status": "uncalibrated", "reason": "explicit_planet_to_temperature_transfer"}
    child.action_temperature = child.target_temperature = 1.0
    return child, {
        "version": 1,
        "source_encoding": parent.observation_encoding,
        "target_encoding": child.observation_encoding,
        "shared_quantities": shared,
        "control_aliases": labels,
        "projection_columns": mappings,
        "new_columns": "zero_initialized_then_learned",
        "other_parameters": "exact_copy",
        "biological_topology": "unchanged",
        "formula_or_correct_action_rules_added": False,
    }
