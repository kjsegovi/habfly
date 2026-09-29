"""The three raw copies require same-star evidence and an already-selected Yes."""
# ruff: noqa: F811

import json

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_evidence import evidence
from test_browser_planet_numeric import planet_page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_evidence import copy_measured_inputs


@pytest.mark.parametrize("same_star", [True, False])
def test_only_same_star_raw_copies_can_precede_the_learned_derived_stage(planet_page, tmp_path, same_star):
    page, frame = planet_page
    spectrum, transit = evidence()
    spectrum["events"][0]["payload"]["chart"]["star"] = "JYREMIS" if same_star else "OTHER"
    transit["star"] = "JYREMIS" if same_star else "OTHER"
    for name, data in (("spectrum", spectrum), ("transit", transit)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))

    def run():
        return copy_measured_inputs(
            page,
            config(),
            tmp_path / "run",
            spectrum_path=tmp_path / "spectrum.json",
            transit_path=tmp_path / "transit.json",
        )

    if not same_star:
        with pytest.raises(BrowserSafetyStop, match="star_changed"):
            run()
        assert frame.locator("#line_shift").input_value() == ""
        return
    report = run()
    assert report["raw_measurement_transport_verified"] and not report["task_completed"]
    assert len(report["verified_fields"]) == 3
    assert frame.locator("#line_shift").input_value() == "0.001"
    assert frame.locator("#brightness_drop").input_value() == "0.01"
    assert frame.locator("#period_days").input_value() == "20"
    assert frame.locator("#planet_mass").input_value() == ""
