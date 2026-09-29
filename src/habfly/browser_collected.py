"""Name-scoped navigation of a blank collected stellar row, with no answer writes."""

import math
import re
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import read_class_choices
from .browser_numeric import screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_setup import RENDER_VISIBILITY_JS
from .browser_stellar import SIMULATION_URL, map_stellar_capture


def row_icon_exposed(icon):
    # Native SVG eye icons hit their own painted child, not the root SVG.
    # Inspect only exposure/geometry, never path data or event handlers.
    return icon.evaluate(
        "e=>{"
        + RENDER_VISIBILITY_JS
        + """
        const r=e.getBoundingClientRect(), h=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
        return styled(e)&&h&&(h===e||e.contains(h))&&styled(h)&&exposed(r,h);
    }"""
    )


def blank_stellar_row(frame, star):
    if not isinstance(star, str) or not re.fullmatch(r"[A-Za-z][A-Za-z -]{0,79}", star):
        raise BrowserSafetyStop("invalid_collected_star_name")
    names = [
        e
        for e in frame.get_by_text(re.compile(r"^" + re.escape(star) + r"$", re.IGNORECASE)).all()
        if e.is_visible()
    ]
    if len(names) != 1:
        raise BrowserSafetyStop("ambiguous_collected_star")
    name = names[0]
    data, row = name.locator("../.."), name.locator("../../..")
    tokens = data.inner_text().split()
    if len(tokens) != 12 or tokens[0].casefold() != star.casefold() or tokens[4:] != ["-"] * 8:
        raise BrowserSafetyStop("collected_star_not_blank")
    try:
        measurements = dict(zip(("parallax", "wavelength", "flux"), map(float, tokens[1:4])))
    except ValueError:
        raise BrowserSafetyStop("invalid_collected_measurements") from None
    if any(not math.isfinite(v) or v <= 0 for v in measurements.values()):
        raise BrowserSafetyStop("invalid_collected_measurements")
    icons = [e for e in row.get_by_role("img").all() if e.is_visible()]
    eye = []
    nb, rb = name.bounding_box(), row.bounding_box()
    for icon in icons:
        b = icon.bounding_box()
        if (
            b
            and nb
            and rb
            and icon.aria_snapshot() == "- img"
            and b["width"] == 22
            and b["height"] == 25
            and rb["x"] <= b["x"] < b["x"] + b["width"] <= nb["x"]
            and rb["y"] <= b["y"] < b["y"] + b["height"] <= rb["y"] + rb["height"]
            and row_icon_exposed(icon)
        ):
            eye.append(icon)
    if len(eye) != 1:
        raise BrowserSafetyStop("unavailable_collected_view_control")
    return eye[0], measurements


def open_blank_collected_star(page, config, output, *, star):
    """One ordinary row-eye click; never reset, delete, select class or repair answers."""
    boundary = config.model_copy(deep=True)
    for rule in boundary.frames:
        if rule.url == SIMULATION_URL:
            rule.required_text = ["FUNDING", "TOTAL COLLECTED", "ANALYZED DATA"]
    before = inspect_page(page, boundary)
    if before["ignored_frame_urls"] or len(page.context.pages) != 1:
        raise BrowserSafetyStop("unexpected_collected_context")
    frames = page.frames.copy()
    sim = [f for f in frames if f.url == SIMULATION_URL]
    if len(sim) != 1:
        raise BrowserSafetyStop("ambiguous_simulation_frame")
    frame = sim[0]
    control, measurements = blank_stellar_row(frame, star)
    handle = control.element_handle(timeout=2000)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    save_probe(before, directory / "before")
    clicked = False
    outside = lambda r: (r["outer_controls"], [f for f in r["frames"] if f["url"] != SIMULATION_URL])
    try:
        current, again = blank_stellar_row(frame, star)
        if (
            again != measurements
            or page.frames != frames
            or screen_identity(inspect_page(page, boundary)) != screen_identity(before)
            or not handle.evaluate("(a,b)=>a.isConnected&&a===b", current.element_handle())
        ):
            raise BrowserSafetyStop("collected_row_changed")
        persist_json(directory / "reserved.json", {"star": star, "max_clicks": 1, "answer_writes": 0})
        clicked = True
        handle.click(timeout=3000)
        for rule in boundary.frames:
            if rule.url == SIMULATION_URL:
                rule.required_text = []  # The strict stellar mapper validates detail labels.
        after = inspect_page(page, boundary)
        save_probe(after, directory / "stellar")
        mapping = map_stellar_capture(after, capture_sha256=screen_identity(after))
        values = mapping["observation"]["values"]
        choices, _ = read_class_choices(frame)
        if (
            page.frames != frames
            or len(page.context.pages) != 1
            or outside(before) != outside(after)
            or mapping["star_name"].casefold() != star.casefold()
            or any(f["current_value"] != "" for f in values["browser_field_map"].values())
            or values["color"]["selected"] is not None
        ):
            raise BrowserSafetyStop("collected_detail_mismatch")
        for key, expected in measurements.items():
            if values["measurements"][f"browser_{key}"]["value"] != expected:
                raise BrowserSafetyStop("collected_measurement_mismatch")
        receipt = {
            "star": mapping["star_name"],
            "action_source": "deterministic_navigation",
            "fresh_blank_numeric_answers_verified": True,
            "class_selection_verified": False,
            "painted_stellar_class": choices["selected"],
            "answer_writes": 0,
            "navigation_clicks": 1,
            "task_completed": False,
            "automatic_retry": False,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "collected_navigation_failed",
                "click_may_have_occurred": clicked,
                "automatic_retry": False,
            },
        )
        raise
