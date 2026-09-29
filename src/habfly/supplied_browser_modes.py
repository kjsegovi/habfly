"""Explicit opt-in identities; no legacy or supplied-input mode is an alias."""

from .browser import BrowserSafetyStop

DERIVED_SUPPLIED_MODE = "four_derived_planet_supplied_inputs_transport_v1"
TEMPERATURE_SUPPLIED_MODE = "supplied_class_equilibrium_copy_v1"
POSITIVE_SUPPLIED_MODE = "positive_planet_supplied_inputs_workflow_readback_v1"
TERRESTRIAL_SUPPLIED_MODE = "terrestrial_supplied_inputs_workflow_readback_v1"
NON_MAIN_CLASSES = ("white_dwarf", "red_giant", "supergiant")


def source_options(supplied_inputs):
    """Omit the new keyword entirely for legacy calls and serialized payloads."""
    if type(supplied_inputs) is not bool:
        raise BrowserSafetyStop("supplied_browser_invalid_opt_in")
    return {"supplied_inputs": True} if supplied_inputs else {}


def positive_mode(supplied_inputs):
    source_options(supplied_inputs)
    return POSITIVE_SUPPLIED_MODE if supplied_inputs else "positive_planet_visible_workflow_readback"


def terrestrial_mode(supplied_inputs):
    source_options(supplied_inputs)
    return TERRESTRIAL_SUPPLIED_MODE if supplied_inputs else "terrestrial_visible_workflow_readback"
