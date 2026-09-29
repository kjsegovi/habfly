"""New explicit-class tool bridges; construction alone grants no browser authority.

Separate owners must validate the supplied-input receipt and frozen transfer gate
before constructing these bridges. Existing native copy primitives still own
every write. No hidden stellar mass/radius/lifetime or private grading answers
are supplied, and no policy choice is repaired.
"""

import random
from copy import deepcopy

from .browser import BrowserSafetyStop
from .browser_habitability_policy import HabitabilityBrowserToolEnv, visible_temperature_inputs
from .browser_planet_policy import PlanetBrowserToolEnv, visible_measurements
from .environments.habitability_calculations import FIELDS as TEMPERATURE_FIELDS
from .environments.local_stellar import LocalStellarEnv
from .environments.planet_calculations import FIELDS as PLANET_FIELDS
from .habitability_supplied_inputs import SCOPE as TEMPERATURE_SCOPE
from .habitability_supplied_inputs import SuppliedTemperatureCalculator
from .planet_supplied_inputs import SCOPE as PLANET_SCOPE
from .planet_supplied_inputs import SuppliedPlanetCalculator

NON_MAIN_CLASSES = ("white_dwarf", "red_giant", "supergiant")


def _case(rows, actual_class, seed, required):
    # The old main-sequence path is not silently migrated to the new adapter.
    if type(actual_class) is not str or actual_class not in NON_MAIN_CLASSES or type(seed) is not int:
        raise BrowserSafetyStop("supplied_tool_explicit_non_main_class_required")
    rows = deepcopy(rows)
    random.Random(seed).shuffle(rows)
    return {
        "seed": seed,
        "split": "browser",
        "star_class": actual_class,
        "required": list(required),
        "measurements": {f"m{i}": row for i, row in enumerate(rows)},
    }


class SuppliedPlanetBrowserToolEnv(PlanetBrowserToolEnv):
    """Same visible controls and native transport; actual-class calculation tool."""

    def __init__(self, session, *, supplied_star_class, seed=15000000):
        self.session, self.transport_verified = session, False
        self.starting_measurements = visible_measurements(session.mapping)
        case = _case(self.starting_measurements, supplied_star_class, seed, PLANET_FIELDS)
        LocalStellarEnv.__init__(self, SuppliedPlanetCalculator(), [case], max_steps=128)
        self.reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        self.transport_verified = False
        return super().reset(seed=seed, options=options)

    def calculate(self, bound):
        return self.adapter.execute(
            self.operation, bound, star_class=self.case["star_class"], assumptions=self.pack.assumptions
        )

    def observe(self):
        observation = super().observe()
        observation.progress.update(scope=PLANET_SCOPE, scientific_verified=False)
        return observation


class SuppliedTemperatureBrowserToolEnv(HabitabilityBrowserToolEnv):
    """Two local proposals, with no native copy until the owner checks completion."""

    def __init__(self, mapping, *, supplied_star_class, supplied_greenhouse_increment, seed=20000000):
        rows = visible_temperature_inputs(
            mapping, supplied_greenhouse_increment=supplied_greenhouse_increment
        )
        case = _case(rows, supplied_star_class, seed, TEMPERATURE_FIELDS)
        case["planet_class"] = "terrestrial"
        self.proposals_complete = False
        LocalStellarEnv.__init__(self, SuppliedTemperatureCalculator(), [case], max_steps=128)
        self.reset(seed=seed)

    def calculate(self, bound):
        return self.adapter.execute(
            self.operation, bound, star_class=self.case["star_class"], assumptions=self.pack.assumptions
        )

    def observe(self):
        observation = super().observe()
        observation.progress.update(scope=TEMPERATURE_SCOPE, scientific_verified=False)
        return observation
