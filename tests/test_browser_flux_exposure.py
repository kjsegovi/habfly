import pytest

from habfly.browser import BrowserSafetyStop
from habfly.browser_flux_exposure import exposed_flux_sample
from habfly.planet_charts import FluxSample


class Exposure:
    def __init__(self, brightness=95, *, missing=False, stuck=False, drift=False):
        self.offset, self.brightness = 0, brightness
        self.missing, self.stuck, self.drift = missing, stuck, drift
        self.pans, self.events = [], []

    def emit(self, *event):
        self.events.append(event)

    def flux_axis_labels(self):
        return [
            {"value": str(v + self.offset), "center_y": 31 + (100 - v) * 60}
            for v in (98.4, 98.8, 99.2, 99.6, 100)
        ]

    def probe_hover(self, x, y):
        assert (x, y) == (0.2, 80 / 195)
        if self.missing or not 98.4 + self.offset <= self.brightness <= 100 + self.offset:
            return None
        return FluxSample(day=2625, brightness_percent=str(self.brightness))

    def pan(self, start, end):
        self.pans.append((start, end))
        if not self.stuck:
            self.offset += -1.5 if end[1] < start[1] else 1.6 if self.drift else 1.5


def test_missing_tooltip_is_not_a_value_and_deep_transit_restores_original_view():
    session = Exposure()
    sample = exposed_flux_sample(session, 0.2)
    assert sample.day == 2625 and sample.brightness_percent == "95"
    assert session.offset == 0 and len(session.pans) == 6
    assert session.events[-1][1]["chart_visibility_recovery"]["view_restored"]
    assert session.events[-1][1]["chart_visibility_recovery"]["answer_writes"] == 0


def test_already_visible_sample_performs_no_recovery():
    session = Exposure(brightness=100)
    assert exposed_flux_sample(session, 0.2).brightness_percent == "100"
    assert not session.pans and not session.events


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"stuck": True}, "pan_not_verified"),
        ({"missing": True}, "bounded_flux_exposure_not_reached"),
        ({"drift": True}, "restore_not_verified"),
    ],
)
def test_unverified_pans_missing_data_and_failed_restore_never_return_a_sample(kwargs, reason):
    session = Exposure(**kwargs)
    with pytest.raises(BrowserSafetyStop, match=reason):
        exposed_flux_sample(session, 0.2, max_pans=4)
    assert not session.events


def test_zero_search_budget_and_unsupported_origin_stop():
    with pytest.raises(BrowserSafetyStop, match="bounded_flux_exposure_not_reached"):
        exposed_flux_sample(Exposure(), 0.2, max_pans=0)
    session = Exposure(missing=True)
    session.flux_axis_labels = lambda: [{"value": str(v), "center_y": 20 + 100 - v} for v in (80, 90, 100)]
    with pytest.raises(BrowserSafetyStop, match="unsupported_flux_exposure_search_origin"):
        exposed_flux_sample(session, 0.2)
    assert not session.pans
