"""Versioned public planet-tool features; no numeric answers or expert stages.

Separate widths and encoding names intentionally prevent silently loading a
stellar workflow checkpoint as a planet policy. All features enter the same
sensory/connectome path; there are no new per-neuron learned embeddings.
"""

from .tool_state import STAR_CLASSES, state_features, visible_sources

QUANTITIES = (
    "period_days",
    "line_shift",
    "brightness_drop",
    "stellar_mass",
    "stellar_radius",
    "period_years",
    "radial_velocity",
    "orbital_radius",
    "planet_radius",
    "planet_mass",
    "planet_density",
)
UNITS = ("day", "nm", "%", "Msun", "Rsun", "yr", "m/s", "au", "REarth", "MEarth", "g/cm3")
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
    "Check derived planet analysis",
) + tuple(f"Unit for {q}" for q in QUANTITIES)


def planet_state_features(observation):
    values = observation.get("values") or {}
    return (
        state_features(observation, quantities=QUANTITIES)
        + [q in values.get("required_fields", []) for q in QUANTITIES]
        + [values.get("star_class") == cls for cls in STAR_CLASSES]
    )


def planet_option_features(observation, option):
    metadata = visible_sources(observation).get(option, {})
    return [metadata.get("kind", option) == q for q in QUANTITIES] + [
        metadata.get("unit", option) == unit for unit in UNITS
    ]


def planet_control_features(control):
    return [control.get("label") == label for label in CONTROL_LABELS]
