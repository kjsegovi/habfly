"""Public tool-state encoding; no expert stages, grading answers or next-action rules.

Field identities supplement the character encoder at sensory neurons. Numeric
payloads stay in observations but do not determine the identity of a selection.
"""

QUANTITIES = (
    "parallax",
    "flux",
    "wavelength",
    "distance",
    "luminosity",
    "temperature",
    "mass",
    "radius",
    "lifetime",
)
UNITS = ("arcsec", "W/m2", "nm", "ly", "Lsun", "K", "Msun", "Rsun", "Gyr")
STATE_WIDTH = len(QUANTITIES) * 10 + 4
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
    "Check stellar analysis",
) + tuple(f"Unit for {q}" for q in QUANTITIES)


def control_features(control):
    return [control.get("label") == label for label in CONTROL_LABELS]


def visible_sources(observation):
    """Resolve result provenance only from its recorded, visible input lineage.

    Mixed, missing and cyclic lineage is not assumed to belong to the requested
    star. Wrong-reference calculations retain their wrong-reference identity.
    """
    measurements = (observation.get("values") or {}).get("measurements", {})
    results = (observation.get("calculation") or {}).get("results", {})

    def source(ref, seen):
        if ref in measurements:
            return measurements[ref].get("source", "unknown source")
        if ref in seen or ref not in results:
            return "unknown source"
        lineage = {source(parent, seen | {ref}) for parent in results[ref].get("bindings", {}).values()}
        return next(iter(lineage)) if len(lineage) == 1 else "mixed or unknown sources"

    return {**measurements, **{ref: {**row, "source": source(ref, set())} for ref, row in results.items()}}


def state_features(observation):
    state, values = observation.get("calculation") or {}, observation.get("values") or {}
    sources = visible_sources(observation)
    results = {k: r for k, r in state.get("results", {}).items() if r.get("valid")}
    source = sources.get(state.get("source"), {})
    selected = results.get(state.get("selected_result"), {})
    pending = results.get(state.get("pending_result"), {})
    vectors = []
    for quantity in QUANTITIES:
        vectors.extend(
            [
                state.get("operation") == quantity,
                state.get("parameter") == quantity,
                source.get("kind") == quantity,
                quantity in state.get("bindings", {}),
                any(r.get("kind") == quantity for r in results.values()),
                pending.get("kind") == quantity,
                selected.get("kind") == quantity,
                state.get("destination") == quantity,
                quantity in values.get("answers", {}),
                quantity in values.get("units", {}),
            ]
        )
    last = state.get("last_operation") or {}
    return vectors + [
        bool(state.get("tool_error")),
        last.get("kind") == "calculate",
        last.get("kind") == "copy",
        bool(state),
    ]


def option_features(observation, option):
    metadata = visible_sources(observation).get(option, {})
    kind = metadata.get("kind", option)
    unit = metadata.get("unit", option)
    return [kind == q for q in QUANTITIES] + [unit == u for u in UNITS]
