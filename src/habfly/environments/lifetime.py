"""Six supplied-class stellar calculations, with explicit mass-to-lifetime reuse."""

from .luminosity import LuminosityEnv
from .radius import radius_cases
from .radius import required_fields as radius_required_fields

SCOPE = "local_tool_assisted_distance_luminosity_temperature_mass_radius_lifetime"
TEMPLATES = {
    "train": (
        "Use the current star's measurements, not those of the reference star. Find distance, luminosity and temperature; also find mass, radius and lifetime when the supplied class is main_sequence. Copy the required results and select their units.",
        "Ignore the reference star. Analyze the current star: calculate distance, luminosity, temperature and, for main_sequence only, mass, radius and lifetime. Fill the required answers and units.",
    ),
    "calibration": (
        "For the current star, obtain distance, luminosity and temperature. Include mass, radius and lifetime only for main_sequence stars. Ignore reference-star readings; transfer the required values and units.",
        "Use readings from the current star. Determine its distance, luminosity, temperature and main-sequence mass, radius and lifetime where applicable; copy the answers and units, not reference-star values.",
    ),
    "development": (
        "Analyze the current star, not the reference star. Determine distance, luminosity and temperature, adding mass, radius and lifetime only if its supplied class is main_sequence. Copy each required result and its unit.",
        "Disregard the reference star's measurements. For the current star, calculate the required fields: distance, luminosity, temperature, and mass, radius and lifetime for main_sequence only. Select the units.",
    ),
    "test": (
        "Calculate distance, luminosity and temperature for the current star using its readings. Ignore the reference star. If the supplied class is main_sequence, also calculate mass, radius and lifetime; copy required results and units.",
        "Use the current star's parallax, flux and wavelength, not those belonging to the reference star. Complete distance, luminosity and temperature, plus mass, radius and lifetime only for main_sequence stars, with their units.",
        "The current star is requested. Disregard reference-star readings; find distance, luminosity and temperature, and mass, radius and lifetime when applicable to its supplied main_sequence class. Transfer each required result and unit.",
        "Ignore measurements for the reference star; analyze the current star. Copy distance, luminosity, temperature and, only for main_sequence, mass, radius and lifetime from the calculation tool. Select each unit.",
    ),
    "manual": (
        "Use measurements of the current star, not the reference star. Complete distance, luminosity and temperature; include mass, radius and lifetime only for main_sequence stars. Copy the required tool results and units.",
        "For the current star, calculate distance, luminosity and temperature, and mass, radius and lifetime if its supplied class is main_sequence. Ignore the reference star and fill the required answers and units.",
    ),
}


def required_fields(star_class):
    fields = radius_required_fields(star_class)
    return [*fields, "lifetime"] if star_class == "main_sequence" else fields


def lifetime_cases(split, count, calculator, *, offset=0):
    rows = radius_cases(split, count, calculator, offset=1000000 + offset)
    for index, case in enumerate(rows):
        case["required"] = required_fields(case["star_class"])
        answers = calculator.reference_answers(case["inputs"], case["star_class"])
        case["expected"] = {key: answers[key] for key in case["required"]}
        case["instruction"] = TEMPLATES[split][(index // 4) % len(TEMPLATES[split])]
    return rows


class LifetimeEnv(LuminosityEnv):
    def observe(self):
        observation = super().observe()
        observation.progress["diagnostic"] = "distance_luminosity_temperature_mass_radius_lifetime_v1"
        return observation
