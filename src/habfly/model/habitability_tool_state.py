"""Versioned public temperature-tool features, never grading data or magnitudes."""

from .tool_state import STAR_CLASSES, state_features, visible_sources

QUANTITIES = (
    "stellar_luminosity",
    "orbital_radius",
    "albedo",
    "greenhouse_increment",
    "equilibrium_temp",
    "surface_temp",
)
UNITS = ("Lsun", "au", "fraction", "K")
STATE_WIDTH = len(QUANTITIES) * 11 + 4 + len(STAR_CLASSES)
OPTION_WIDTH = len(QUANTITIES) + len(UNITS)
CONTROL_LABELS = (
    "Calculation",
    "Required input",
    "Measurement or result",
    "Bind selected input",
    "Execute calculation",
    "Calculated result",
    "Answer destination",
    "Copy selected result",
    "Check supplied temperature calculations",
) + tuple(f"Unit for {q}" for q in QUANTITIES)


def habitability_state_features(observation):
    values = observation.get("values") or {}
    return (
        state_features(observation, quantities=QUANTITIES)
        + [q in values.get("required_fields", []) for q in QUANTITIES]
        + [values.get("star_class") == cls for cls in STAR_CLASSES]
    )


def habitability_option_features(observation, option):
    metadata = visible_sources(observation).get(option, {})
    return [metadata.get("kind", option) == q for q in QUANTITIES] + [
        metadata.get("unit", option) == unit for unit in UNITS
    ]


def habitability_control_features(control):
    return [control.get("label") == label for label in CONTROL_LABELS]
