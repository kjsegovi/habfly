"""Supplied-class distance/luminosity/temperature, plus main-sequence mass.

The visible required fields reflect the assignment, not a hidden expert stage.
All six operations remain selectable; the calculator rejects inapplicable mass.
"""

from .luminosity import LuminosityEnv
from .stellar_common import make_case

SCOPE = "local_tool_assisted_distance_luminosity_temperature_mass"
BASE_FIELDS = ("distance", "luminosity", "temperature")
CLASSES = ("main_sequence", "white_dwarf", "main_sequence", "giant")
SEEDS = {"train": 6100000, "calibration": 6200000, "development": 6300000, "test": 6400000, "manual": 6500000}
TEMPLATES = {
    "train": (
        "Use the current star's measurements, not those of the reference star. Find distance, luminosity and temperature; also find mass when the supplied class is main_sequence. Copy the required results and select their units.",
        "Ignore the reference star. Analyze the current star: calculate distance, luminosity, temperature and, for main_sequence only, mass. Fill the required answers and units.",
    ),
    "calibration": (
        "For the current star, obtain distance, luminosity and temperature. Include mass only for main_sequence stars. Ignore reference-star readings; transfer the required values and units.",
        "Use readings from the current star. Determine its distance, luminosity, temperature and main-sequence mass where applicable; copy the answers and units, not reference-star values.",
    ),
    "development": (
        "Analyze the current star, not the reference star. Determine distance, luminosity and temperature, adding mass only if its supplied class is main_sequence. Copy each required result and its unit.",
        "Disregard the reference star's measurements. For the current star, calculate the required fields: distance, luminosity, temperature, and mass for main_sequence only. Select the units.",
    ),
    "test": (
        "Calculate distance, luminosity and temperature for the current star using its readings. Ignore the reference star. If the supplied class is main_sequence, also calculate mass; copy required results and units.",
        "Use the current star's parallax, flux and wavelength, not those belonging to the reference star. Complete distance, luminosity and temperature, plus mass only for main_sequence stars, with their units.",
        "The current star is requested. Disregard reference-star readings; find distance, luminosity and temperature, and mass when applicable to its supplied main_sequence class. Transfer each required result and unit.",
        "Ignore measurements for the reference star; analyze the current star. Copy distance, luminosity, temperature and, only for main_sequence, mass from the calculation tool. Select each unit.",
    ),
    "manual": (
        "Use measurements of the current star, not the reference star. Complete distance, luminosity and temperature; include mass only for main_sequence stars. Copy the required tool results and units.",
        "For the current star, calculate distance, luminosity and temperature, and mass if its supplied class is main_sequence. Ignore the reference star and fill the required answers and units.",
    ),
}


def required_fields(star_class):
    if star_class not in CLASSES:
        raise ValueError("Unknown supplied stellar class")
    return [*BASE_FIELDS, "mass"] if star_class == "main_sequence" else list(BASE_FIELDS)


def mass_cases(split, count, calculator, *, offset=0):
    rows = []
    for index in range(count):
        case = make_case(SEEDS[split] + offset + index, "test" if split == "manual" else split)
        # Each template sees both branches; numeric cases do not determine class.
        case["star_class"] = CLASSES[index % len(CLASSES)]
        case["required"] = required_fields(case["star_class"])
        answers = calculator.reference_answers(case["inputs"], case["star_class"])
        case["expected"] = {key: answers[key] for key in case["required"]}
        case["instruction"] = TEMPLATES[split][(index // len(CLASSES)) % len(TEMPLATES[split])]
        rows.append(case)
    return rows


class MassEnv(LuminosityEnv):
    def observe(self):
        observation = super().observe()
        observation.progress["diagnostic"] = "distance_luminosity_temperature_mass_v1"
        return observation
