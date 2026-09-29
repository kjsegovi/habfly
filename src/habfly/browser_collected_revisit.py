"""Reopen a visible, named collected stellar row without modifying answers.

The current displayed row is the source, not a hidden catalog or grading key.
This only handles an already visible stellar-list page; it does not paginate,
change list tabs, delete, save, assess, or infer scientific classifications.
"""

import re
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import read_class_choices
from .browser_collected import row_icon_exposed
from .browser_numeric import screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_stellar import COLOR_OPTIONS, NUMBER, SIMULATION_URL, map_stellar_capture

CLASS_LABELS = {
    "main sequence": "main_sequence",
    "red giant": "red_giant",
    "supergiant": "supergiant",
    "white dwarf": "white_dwarf",
}


def collected_status_projection(report):
    """Ignore only the observed transient list footer, never a row or score.

    The raw snapshots remain evidence. Exact text/AX footer placement is needed;
    the same phrase in a row, modal, or another frame is not normalized.
    """
    value = deepcopy(report)
    for frame in value.get("frames", []):
        if frame.get("url") != SIMULATION_URL:
            continue
        raw = frame["text"]
        if raw.count("Data saved") == 1:
            frame["text"] = re.sub(
                r"\nData saved(?=\n\s*\nVIEWING\n[0-9]+-[0-9]+ OF [0-9]+\n\s*\nTOTAL COLLECTED\n[0-9]+\nSave$)",
                "",
                raw,
            )
        ax = frame["accessibility"]
        if ax.count("Data saved") == 1:
            frame["accessibility"] = re.sub(
                r'\n- text: Data saved(?=\n- button(?: \[disabled\])?\n- button(?: \[disabled\])?\n- text: viewing [0-9]+-[0-9]+ of [0-9]+ total collected [0-9]+\n- button "Save"(?: \[disabled\])?$)',
                "",
                ax,
            )
        for control in frame["controls"]:
            if control["role"] == "button" and control["accessibility"] in {
                '- button "Save"',
                '- button "Save" [disabled]',
            }:
                control["accessibility"], control["enabled"] = '- button "Save"', True
        frame["accessibility"] = frame["accessibility"].replace(
            '- button "Save" [disabled]', '- button "Save"'
        )
    return value


def _number(value, *, positive=False):
    if value == "-":
        return ""
    if not re.fullmatch(NUMBER, value):
        raise BrowserSafetyStop("invalid_collected_number")
    number = Decimal(value)
    if not number.is_finite() or number < 0 or positive and number <= 0:
        raise BrowserSafetyStop("invalid_collected_number")
    return value


def parse_stellar_row(text, star):
    if not isinstance(star, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", star):
        raise BrowserSafetyStop("invalid_collected_star_name")
    number = rf"(?:{NUMBER}|-)"
    pattern = (
        re.escape(star)
        + rf"\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})"
        + rf"\s+({number})\s+({number})\s+({'|'.join(COLOR_OPTIONS)}|-)\s+({number})"
        + rf"\s+({'|'.join(CLASS_LABELS)}|-)\s+({number})\s+({number})"
        + rf"\s+({number})(?:\s+(ka|Ma|Ga|Ta))?"
    )
    match = re.fullmatch(pattern, " ".join(text.split()), re.IGNORECASE)
    if not match:
        raise BrowserSafetyStop("unsupported_collected_stellar_row")
    p, w, f, distance, luminosity, color, temperature, kind, mass, radius, lifetime, prefix = match.groups()
    classification = CLASS_LABELS.get(kind.casefold())
    fields = dict(zip(("distance", "luminosity", "temperature"), (distance, luminosity, temperature)))
    if classification == "main_sequence":
        fields.update(mass=mass, radius=radius, lifetime=lifetime)
        if (lifetime != "-") != (prefix is not None):
            raise BrowserSafetyStop("ambiguous_collected_lifetime_unit")
    elif (mass, radius, lifetime, prefix) != ("-", "-", "-", None):
        raise BrowserSafetyStop("inapplicable_collected_stellar_answers")
    return {
        "star": star,
        "measurements": {
            k: _number(v, positive=True) for k, v in zip(("parallax", "wavelength", "flux"), (p, w, f))
        },
        "fields": {k: _number(v) for k, v in fields.items()},
        "color": next((c for c in COLOR_OPTIONS if c.casefold() == color.casefold()), None),
        "classification": classification,
        "lifetime_prefix": prefix,
    }


def visible_stellar_row(frame, star):
    # Parse the explicit unit/header row before treating positional values as
    # stellar data. Planet/list layouts are not interchangeable.
    text = " ".join(frame.locator("body").inner_text().split()).casefold()
    headers = (
        'parallax (")',
        "peak λ (nm)",
        "flux (w/m2)",
        "distance (ly)",
        "luminosity (ls)",
        "peak λ color",
        "temp. (k)",
        "classification",
        "mass (ms)",
        "radius (rs)",
        "lifetime (years)",
    )
    if any(text.count(label) != 1 for label in headers):
        raise BrowserSafetyStop("unsupported_collected_stellar_headers")
    # Validate before using the name in a locator or expression.
    if not isinstance(star, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", star):
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
    evidence = parse_stellar_row(data.inner_text(), star)
    nb, rb = name.bounding_box(), row.bounding_box()
    eyes = []
    for icon in row.get_by_role("img").all():
        if not icon.is_visible():
            continue
        box = icon.bounding_box()
        if (
            box
            and nb
            and rb
            and icon.aria_snapshot() == "- img"
            and box["width"] == 22
            and box["height"] == 25
            and rb["x"] <= box["x"] < box["x"] + box["width"] <= nb["x"]
            and rb["y"] <= box["y"] < box["y"] + box["height"] <= rb["y"] + rb["height"]
            and row_icon_exposed(icon)
        ):
            eyes.append(icon)
    if len(eyes) != 1:
        raise BrowserSafetyStop("unavailable_collected_view_control")
    return eyes[0], evidence


def _same_number(actual, expected):
    # Display formatting may differ, but never round or tolerate changed values.
    if not actual or not expected:
        return actual == expected
    return bool(re.fullmatch(NUMBER, actual)) and Decimal(actual) == Decimal(expected)


def verify_detail_row(mapping, choices, evidence):
    values = mapping["observation"]["values"]
    fields = values["browser_field_map"]
    if (
        mapping["star_name"].casefold() != evidence["star"].casefold()
        or set(fields) != set(evidence["fields"])
        or values["color"]["selected"] != evidence["color"]
        or (evidence["classification"] is not None and choices["selected"] != evidence["classification"])
        or any(not _same_number(fields[k]["current_value"], v) for k, v in evidence["fields"].items())
        or any(
            not _same_number(values["measurements"][f"browser_{k}"]["display_text"], v)
            for k, v in evidence["measurements"].items()
        )
    ):
        raise BrowserSafetyStop("collected_detail_values_changed")
    if evidence["classification"] == "main_sequence" and (
        values["lifetime_prefix"]["selected"] != evidence["lifetime_prefix"]
    ):
        raise BrowserSafetyStop("collected_detail_lifetime_unit_changed")


def reopen_collected_stellar(page, config, output, *, star):
    """One name-bound eye click with row/detail equality; no answer writes."""
    directory = Path(output)
    boundary = config.model_copy(deep=True)
    for rule in boundary.frames:
        if rule.url == SIMULATION_URL:
            rule.required_text = ["FUNDING", "TOTAL COLLECTED", "ANALYZED DATA"]
    before = inspect_page(page, boundary)
    frames = page.frames.copy()
    candidates = [f for f in frames if f.url == SIMULATION_URL]
    if before["ignored_frame_urls"] or len(page.context.pages) != 1 or len(candidates) != 1:
        raise BrowserSafetyStop("unexpected_collected_context")
    frame = candidates[0]
    control, evidence = visible_stellar_row(frame, star)
    handle = control.element_handle(timeout=2000)
    directory.mkdir(parents=True, exist_ok=False)
    save_probe(before, directory / "before")
    attempted, dialog_seen = False, []

    def dialog_handler(dialog):
        dialog_seen.append(True)
        dialog.dismiss()

    page.on("dialog", dialog_handler)
    outside = lambda report: (
        report["outer_controls"],
        [f for f in report["frames"] if f["url"] != SIMULATION_URL],
    )
    try:
        current, again = visible_stellar_row(frame, star)
        pre_click = inspect_page(page, boundary)
        save_probe(pre_click, directory / "pre-click")
        if (
            again != evidence
            or dialog_seen
            or page.frames != frames
            or len(page.context.pages) != 1
            or screen_identity(collected_status_projection(pre_click))
            != screen_identity(collected_status_projection(before))
            or not handle.evaluate("(a,b)=>a.isConnected&&a===b", current.element_handle())
        ):
            raise BrowserSafetyStop("collected_row_changed")
        persist_json(directory / "reserved.json", {"evidence": evidence, "max_clicks": 1, "answer_writes": 0})
        attempted = True
        handle.click(timeout=3000)
        for rule in boundary.frames:
            if rule.url == SIMULATION_URL:
                rule.required_text = []
        after = inspect_page(page, boundary)
        save_probe(after, directory / "stellar")
        if (
            dialog_seen
            or page.frames != frames
            or len(page.context.pages) != 1
            or after["ignored_frame_urls"]
            or outside(before) != outside(after)
        ):
            raise BrowserSafetyStop("collected_navigation_context_changed")
        mapping = map_stellar_capture(
            after,
            capture_sha256=screen_identity(after),
            allow_color_selection=True,
            allow_main_sequence_fields=True,
        )
        choices, _ = read_class_choices(frame)
        verify_detail_row(mapping, choices, evidence)
        receipt = {
            "star": mapping["star_name"],
            "action_source": "deterministic_navigation",
            "row_detail_values_verified": True,
            "classification_correctness_verified": False,
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
                "reason": str(exc)
                if isinstance(exc, BrowserSafetyStop)
                else "collected_navigation_uncertain",
                "click_may_have_occurred": attempted,
                "automatic_retry": False,
                "answer_writes": 0,
                "task_completed": False,
            },
        )
        if isinstance(exc, Exception) and not isinstance(exc, BrowserSafetyStop):
            raise BrowserSafetyStop("collected_navigation_uncertain") from None
        raise
    finally:
        page.remove_listener("dialog", dialog_handler)
