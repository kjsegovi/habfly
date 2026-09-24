"""Distance, luminosity and temperature on the existing visible calculator UI."""

from .luminosity import LuminosityEnv
from .stellar_common import make_case

TEMPLATES = {
    "train": (
        "Find the current star's distance, luminosity and temperature. Use its parallax, flux and peak wavelength. Copy each result and select its unit.",
        "Use the current star's measurements. Calculate distance, luminosity using that distance, and temperature using peak wavelength. Fill all three answers and units.",
    ),
    "calibration": (
        "For the current star, obtain distance, luminosity and temperature with the local tool. Transfer the three values and their units.",
    ),
    "development": (
        "Analyze the current star: determine distance, luminosity and temperature, then copy each result with its unit. Ignore the reference star's measurements.",
    ),
    "test": (
        "Calculate the current star's distance, luminosity and temperature, not those of the reference star. Enter the tool results and select their units.",
    ),
    "manual": (
        "Complete the current star's distance, luminosity and temperature fields using its measurements and calculated results. Select all three units.",
    ),
}
SEEDS = {"train": 4100000, "calibration": 4200000, "development": 4300000, "test": 4400000, "manual": 4500000}
SCOPE = "local_tool_assisted_distance_luminosity_temperature"


def temperature_cases(split, count, calculator, *, offset=0):
    rows = []
    for index in range(count):
        case = make_case(SEEDS[split] + offset + index, "test" if split == "manual" else split)
        case["required"] = ["distance", "luminosity", "temperature"]
        answers = calculator.reference_answers(case["inputs"], case["star_class"])
        case["expected"] = {key: answers[key] for key in case["required"]}
        case["instruction"] = TEMPLATES[split][index % len(TEMPLATES[split])]
        rows.append(case)
    return rows


class TemperatureEnv(LuminosityEnv):
    def observe(self):
        observation = super().observe()
        observation.progress["diagnostic"] = "distance_luminosity_temperature_v1"
        return observation
