"""Five supplied-class stellar calculations, without classification learning."""

from .luminosity import LuminosityEnv
from .mass import mass_cases
from .mass import required_fields as mass_required_fields

SCOPE = "local_tool_assisted_distance_luminosity_temperature_mass_radius"
TEMPLATES = {
    "train": (
        "Use the current star's measurements, not those of the reference star. Find distance, luminosity and temperature; also find mass and radius when the supplied class is main_sequence. Copy the required results and select their units.",
        "Ignore the reference star. Analyze the current star: calculate distance, luminosity, temperature and, for main_sequence only, mass and radius. Fill the required answers and units.",
    ),
    "calibration": (
        "For the current star, obtain distance, luminosity and temperature. Include mass and radius only for main_sequence stars. Ignore reference-star readings; transfer the required values and units.",
        "Use readings from the current star. Determine its distance, luminosity, temperature and main-sequence mass and radius where applicable; copy the answers and units, not reference-star values.",
    ),
    "development": (
        "Analyze the current star, not the reference star. Determine distance, luminosity and temperature, adding mass and radius only if its supplied class is main_sequence. Copy each required result and its unit.",
        "Disregard the reference star's measurements. For the current star, calculate the required fields: distance, luminosity, temperature, and mass and radius for main_sequence only. Select the units.",
    ),
    "test": (
        "Calculate distance, luminosity and temperature for the current star using its readings. Ignore the reference star. If the supplied class is main_sequence, also calculate mass and radius; copy required results and units.",
        "Use the current star's parallax, flux and wavelength, not those belonging to the reference star. Complete distance, luminosity and temperature, plus mass and radius only for main_sequence stars, with their units.",
        "The current star is requested. Disregard reference-star readings; find distance, luminosity and temperature, and mass and radius when applicable to its supplied main_sequence class. Transfer each required result and unit.",
        "Ignore measurements for the reference star; analyze the current star. Copy distance, luminosity, temperature and, only for main_sequence, mass and radius from the calculation tool. Select each unit.",
    ),
    "manual": (
        "Use measurements of the current star, not the reference star. Complete distance, luminosity and temperature; include mass and radius only for main_sequence stars. Copy the required tool results and units.",
        "For the current star, calculate distance, luminosity and temperature, and mass and radius if its supplied class is main_sequence. Ignore the reference star and fill the required answers and units.",
    ),
}


def required_fields(star_class):
    fields = mass_required_fields(star_class)
    return [*fields, "radius"] if star_class == "main_sequence" else fields


def radius_cases(split, count, calculator, *, offset=0):
    rows = mass_cases(split, count, calculator, offset=1000000 + offset)
    for index, case in enumerate(rows):
        case["required"] = required_fields(case["star_class"])
        answers = calculator.reference_answers(case["inputs"], case["star_class"])
        case["expected"] = {key: answers[key] for key in case["required"]}
        # Every complete instruction template covers all three supplied classes.
        case["instruction"] = TEMPLATES[split][(index // 4) % len(TEMPLATES[split])]
    return rows


class RadiusEnv(LuminosityEnv):
    def observe(self):
        observation = super().observe()
        observation.progress["diagnostic"] = "distance_luminosity_temperature_mass_radius_v1"
        return observation
