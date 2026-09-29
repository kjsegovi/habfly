"""Exclude the optional cosmetic mascot from every scientific/evidence image.

The temporary screenshot stylesheet changes only HabFly's reserved overlay;
it neither masks pixels nor changes chart content, pointer actions, or evidence.
"""

MASCOT_OVERLAY_ID = "habfly-mascot-overlay-v1"
MASCOT_OVERLAY_ATTRIBUTE = "data-habfly-presentation"
MASCOT_OVERLAY_MARKER = "mascot-v1"
MASCOT_OVERLAY_SELECTOR = f'div#{MASCOT_OVERLAY_ID}[{MASCOT_OVERLAY_ATTRIBUTE}="{MASCOT_OVERLAY_MARKER}"]'
EVIDENCE_SCREENSHOT_STYLE = f"{MASCOT_OVERLAY_SELECTOR} {{ visibility: hidden !important; }}"


def evidence_screenshot(target, **options):
    """Forward Page/Locator/ElementHandle options with the mascot hidden.

    Playwright restores its temporary stylesheet after capture, including when
    capture fails. Callers retain their original deadlines and screenshot types.
    """
    style = options.get("style")
    if style is not None and type(style) is not str:
        raise ValueError("Evidence screenshot style must be a string or None")
    options["style"] = f"{style}\n{EVIDENCE_SCREENSHOT_STYLE}" if style else EVIDENCE_SCREENSHOT_STYLE
    return target.screenshot(**options)
