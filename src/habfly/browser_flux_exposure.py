"""Explicit read-only vertical search when a deep transit is below the viewport.

Nothing is read from clipped tooltip text. Every candidate is an ordinary pan
and a fully exposed hover. The original vertical view must be restored before
returning a sample. No time-axis zoom, answer write or inferred flux value.
"""

from .browser import BrowserSafetyStop


def exposed_flux_sample(session, x, *, max_pans=64):
    if type(max_pans) is not int or not 0 <= max_pans <= 64:
        raise ValueError("Choose at most 64 explicit vertical search pans")
    from .browser_transit_sampling import baseline_position

    if not hasattr(session, "probe_hover"):
        return session.hover(x, 80 / 195)  # Existing synthetic session contract.
    sample = session.probe_hover(x, 80 / 195)
    if sample is not None:
        return sample
    original = session.flux_axis_labels()
    original_baseline = baseline_position(original)
    # Search only from the previously verified, visible 100% band. Missing
    # readings in an arbitrary view do not justify an inferred direction.
    if not 30 <= original_baseline <= 120:
        raise BrowserSafetyStop("unsupported_flux_exposure_search_origin")
    labels, pans = original, 0
    start, end = (0.5, 0.7), (0.5, 0.15)
    while sample is None and pans < max_pans:
        lower = min(float(row["value"]) for row in labels)
        if lower <= 0:
            break
        session.pan(start, end)
        pans += 1
        newer = session.flux_axis_labels()
        if len(newer) < 3 or min(float(row["value"]) for row in newer) >= lower:
            raise BrowserSafetyStop("flux_exposure_pan_not_verified")
        labels = newer
        sample = session.probe_hover(x, 80 / 195)
    if sample is None:
        raise BrowserSafetyStop("bounded_flux_exposure_not_reached")
    # Reverse exactly the ordinary drag sequence, then check rendered axes.
    for _ in range(pans):
        session.pan(end, start)
    restored = session.flux_axis_labels()
    if abs(baseline_position(restored) - original_baseline) > 1.5:
        raise BrowserSafetyStop("flux_exposure_restore_not_verified")
    session.emit(
        "state",
        {
            "chart_visibility_recovery": {
                "source": "ordinary_vertical_pan_and_visible_hover",
                "search_pans": pans,
                "restore_pans": pans,
                "view_restored": True,
                "answer_writes": 0,
            }
        },
    )
    return sample
