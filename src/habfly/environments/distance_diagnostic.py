"""A single distance goal on the unchanged local stellar control surface."""

from .local_stellar import LocalStellarEnv
from .stellar_common import make_case

TEMPLATES = {
    "train": "Find the current star's distance from its parallax using the local tool. Copy the distance and select its unit.",
    "calibration": "Use the calculation reference to obtain distance for this star, then transfer its value and unit.",
    "development": "Complete only the distance field using the current star's parallax. Ignore reference-star measurements.",
    "test": "Determine how far away the supplied star is. Bind its parallax, calculate, and fill the distance and unit.",
}


def distance_case(seed, split, calculator):
    case = make_case(seed, split)
    case["required"] = ["distance"]
    case["expected"] = {
        "distance": calculator.reference_answers(case["inputs"], case["star_class"])["distance"]
    }
    return case


class DistanceDiagnosticEnv(LocalStellarEnv):
    """No stage hints, forced next actions, automatic binding or error repair."""

    def observe(self):
        observation = super().observe()
        observation.instruction = self.case.get("instruction", TEMPLATES[self.case["split"]])
        observation.progress["diagnostic"] = "distance_v1"
        return observation
