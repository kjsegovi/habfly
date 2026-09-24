"""Two explicitly requested outputs, using the unchanged public calculator UI."""

from .local_stellar import LocalStellarEnv
from .stellar_common import make_case

TEMPLATES = {
    "train": (
        "Find the current star's distance and luminosity. Use its parallax and flux; copy the results and select their units.",
        "Use the current star's measurements. Calculate distance, then luminosity using that distance. Fill both answers and units.",
    ),
    "calibration": (
        "For the current star, obtain distance and luminosity with the local tool. Transfer both values and units.",
    ),
    "development": (
        "Analyze the current star: determine distance and luminosity, then copy each result with its unit. Ignore the reference star.",
    ),
    "test": (
        "Calculate distance and luminosity for the current star, not the reference star. Enter the tool results and select their units.",
    ),
    "manual": (
        "Complete the current star's distance and luminosity fields using its measurements and calculated results. Select both units.",
    ),
}
SEEDS = {"train": 3100000, "calibration": 3200000, "development": 3300000, "test": 3400000, "manual": 3500000}
MAX_STEPS = 64
SCOPE = "local_tool_assisted_distance_luminosity"


def luminosity_cases(split, count, calculator, *, offset=0):
    rows = []
    for index in range(count):
        case = make_case(SEEDS[split] + offset + index, "test" if split == "manual" else split)
        case["required"] = ["distance", "luminosity"]
        answers = calculator.reference_answers(case["inputs"], case["star_class"])
        case["expected"] = {key: answers[key] for key in case["required"]}
        case["instruction"] = TEMPLATES[split][index % len(TEMPLATES[split])]
        rows.append(case)
    return rows


class LuminosityEnv(LocalStellarEnv):
    def observe(self):
        observation = super().observe()
        observation.instruction = self.case["instruction"]
        observation.progress["diagnostic"] = "distance_luminosity_v1"
        return observation
