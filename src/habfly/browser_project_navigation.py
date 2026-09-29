"""One guarded project-navigation click; no scientific or persistent writes.

Numbered tabs preserve detail/list context. Detail transitions require the same
visible star before and after. Only the two observed 40px header icons permit
list/starfield navigation. No retries, selectors from models, or hidden state.
"""

import re
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_collected import row_icon_exposed
from .browser_collected_revisit import collected_status_projection
from .browser_habitability import map_habitability_capture
from .browser_numeric import screen_identity
from .browser_planet import map_planet_capture
from .browser_probe import _visible_frame, inspect_page, save_probe
from .browser_setup import SetupStop, numbered_tab
from .browser_stellar import NUMBER, SIMULATION_URL, StellarMappingError, _atoms, map_stellar_capture

TABS = {"stellar": 1, "planet": 2, "habitability": 3}
LIST_LABELS = {
    "stellar": (
        'Parallax (")',
        "Peak λ (nm)",
        "Flux (W/m2)",
        "Distance (ly)",
        "Luminosity (Ls)",
        "Peak λ color",
        "Temp. (K)",
        "Classification",
        "Mass (Ms)",
        "Radius (Rs)",
        "Lifetime (years)",
    ),
    "planet": (
        "Doppler shift (nm)",
        "Brightness drop (%)",
        "Brightness drop period (days)",
        "Has planet?",
        "Orbital radius (au)",
        "Mass (ME)",
        "Radius (RE)",
        "Density (g/cm3)",
        "Classification",
    ),
    "habitability": (
        "Modeled Albedo",
        "Modeled Surface pressure (atm)",
        "Eq. Temp. (K)",
        "Trace Gases Present",
        "Greenhouse Effect",
        "Est. surface temp. (K)",
        "Water phase",
        "Habitable?",
    ),
}


def _compact(value):
    return re.sub(r"\s+", "", value).casefold()


def project_view(report):
    """Recognize only current, public screen contracts; no row/catalog reads."""
    frames = [f for f in report["frames"] if f["url"] == SIMULATION_URL]
    if len(frames) != 1:
        raise BrowserSafetyStop("ambiguous_project_simulation_frame")
    frame = frames[0]
    atoms = _atoms(frame["accessibility"])
    # The application retains a covered star tooltip in innerText when the
    # starfield is open. Only currently exposed AX text identifies this view;
    # raw before/after captures still preserve both representations.
    text = _compact(
        " ".join(
            value
            for key, value in atoms
            if key in {"text", "subscript", "superscript"} and isinstance(value, str)
        )
    )
    if "totalcollected" in text and "analyzeddata" in text and "funding" in text:
        kinds = [
            kind
            for kind, labels in LIST_LABELS.items()
            if all(text.count(_compact(label)) == 1 for label in labels)
        ]
        if len(kinds) != 1 or "yourreconstruction" in text:
            raise BrowserSafetyStop("unsupported_project_list")
        return {"surface": "list", "section": kinds[0], "star": None}
    if all(word in text for word in ("funding", "dataquality", "scavengerhunt")) and not any(
        word in text
        for word in ("observations", "reconstruction", "analyzeddata", "totalcollected", "viewstardata")
    ):
        return {"surface": "starfield", "section": None, "star": None}
    mappings = []
    for kind, mapper in (
        ("stellar", map_stellar_capture),
        ("planet", map_planet_capture),
        ("habitability", map_habitability_capture),
    ):
        try:
            kwargs = (
                {"allow_color_selection": True, "allow_main_sequence_fields": True}
                if kind == "stellar"
                else {}
            )
            mapping = mapper(report, capture_sha256=screen_identity(report), **kwargs)
            mappings.append((kind, mapping["star_name"]))
        except StellarMappingError:
            pass
    if len(mappings) != 1:
        raise BrowserSafetyStop("unsupported_or_ambiguous_project_detail")
    kind, star = mappings[0]
    names = [
        value
        for key, value in atoms
        if key == "text" and isinstance(value, str) and value.casefold() == star.casefold()
    ]
    if len(names) != 1:
        raise BrowserSafetyStop("ambiguous_project_star")
    return {"surface": "detail", "section": kind, "star": star}


def _target(frame, destination):
    if destination in TABS:
        try:
            target = numbered_tab(frame, TABS[destination])
        except SetupStop:
            raise BrowserSafetyStop("unavailable_project_numbered_tab") from None
        if not row_icon_exposed(target):
            raise BrowserSafetyStop("project_navigation_occluded")
        return target, {
            "number": TABS[destination],
            "box": target.bounding_box(),
            "accessibility": target.aria_snapshot(),
        }
    box = frame.frame_element().bounding_box()
    candidates = []
    for image in frame.get_by_role("img").all():
        bounds = image.bounding_box() if image.is_visible() else None
        if bounds and image.aria_snapshot() == "- img" and bounds["width"] == bounds["height"] == 40:
            candidates.append((bounds, image))
    candidates.sort(key=lambda item: item[0]["x"])
    if len(candidates) != 2 or not box:
        raise BrowserSafetyStop("unsupported_project_header_navigation")
    a, b = [entry[0] for entry in candidates]
    if (
        b["x"] - a["x"] != 50
        or a["y"] != b["y"]
        or not all(
            box["x"] <= bound["x"]
            and bound["x"] + 40 <= box["x"] + box["width"]
            and 0 <= bound["y"] - box["y"] <= 80
            for bound, _ in candidates
        )
        or not all(image.is_enabled() and row_icon_exposed(image) for _, image in candidates)
    ):
        raise BrowserSafetyStop("project_header_navigation_unexposed_or_changed")
    target = candidates[1 if destination == "list" else 0][1]
    return target, {"header_boxes": [a, b], "index": 1 if destination == "list" else 0}


def _outside(report):
    return report["outer_controls"], [f for f in report["frames"] if f["url"] != SIMULATION_URL]


def _required(report, view):
    phrases = {
        "stellar": ("Observations", "Reconstruction", "parallax"),
        "planet": ("Observations", "observe for", "has planet"),
        "habitability": ("Observations", "Modeled Albedo", "Water Phase"),
        "list": ("Funding", "Total Collected", "Analyzed Data"),
        "starfield": ("Funding", "Data Quality", "Scavenger Hunt"),
    }[view["section"] if view["surface"] == "detail" else view["surface"]]
    text = next(f["text"] for f in report["frames"] if f["url"] == SIMULATION_URL)
    values = []
    for phrase in phrases:
        match = re.search(r"\s+".join(map(re.escape, phrase.split())), text, re.IGNORECASE)
        if not match:
            raise BrowserSafetyStop("project_destination_labels_missing")
        values.append(match.group())
    return values


def navigation_status_projection(report, view):
    """Ignore only known autosave footer paint, not measurements or chart data.

    Snapshot files remain untouched. Detail Planet's notice is accepted only
    after its final ORBIT readout and immediately before Save. Text and AX are
    captured separately, so either may contain the transient footer alone.
    """
    if view["surface"] == "list":
        return collected_status_projection(report)
    if view["surface"] == "detail" and view["section"] == "stellar":
        from .browser_full_stellar import full_stellar_status_projection

        return full_stellar_status_projection(report)
    if view["surface"] != "detail" or view["section"] != "planet":
        return report
    value = deepcopy(report)
    for frame in value.get("frames", []):
        if frame.get("url") != SIMULATION_URL:
            continue
        raw, ax = frame["text"], frame["accessibility"]
        if raw.count("Data saved") == 1:
            frame["text"] = re.sub(rf"(\nORBIT \(years\)\n{NUMBER})\nData saved(?=\nSave$)", r"\1", raw)
        if ax.count("Data saved") == 1:
            frame["accessibility"] = re.sub(
                rf"(\n- text: STAR MASS \(Ms\) {NUMBER} STAR RADIUS \(Rs\) {NUMBER} "
                rf'ORBIT \(years\) {NUMBER}) Data saved(?=\n- button "Save"(?: \[disabled\])?$)',
                r"\1",
                ax,
            )
        # Save itself is not this adapter's target; its existing busy flag is
        # independent of whether an exposed List/tab control remains identical.
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


def navigate_project(page, config, output, destination, expected_star=None):
    """One visible click and verified destination; original config is unchanged.

    Returns suggested_required_text for the simulation frame, only after the
    complete destination check. Caller may apply that list to its own config.
    A failed call is never retried automatically, even if dispatch is uncertain.
    """
    if destination == "stellar-tab":
        destination = "stellar"
    if destination not in {*TABS, "list", "starfield"}:
        raise BrowserSafetyStop("unsupported_project_destination")
    if expected_star is not None and (
        not isinstance(expected_star, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", expected_star)
    ):
        raise BrowserSafetyStop("invalid_project_expected_star")
    boundary = config.model_copy(deep=True)
    rules = [rule for rule in boundary.frames if rule.url == SIMULATION_URL]
    if len(rules) != 1 or rules[0].count != 1:
        raise BrowserSafetyStop("unsupported_project_simulation_boundary")
    rules[0].required_text = []
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    frames = page.frames.copy()
    attempted, dialog_seen, popup_seen = False, [], []

    def dialog_handler(dialog):
        dialog_seen.append(True)
        dialog.dismiss()

    def popup_handler(_page):
        popup_seen.append(True)

    page.on("dialog", dialog_handler)
    page.context.on("page", popup_handler)
    try:

        def read():
            if dialog_seen or popup_seen or page.frames != frames or len(page.context.pages) != 1:
                raise BrowserSafetyStop("project_navigation_context_changed")
            report = inspect_page(page, boundary)
            sim = [f for f in page.frames if f.url == SIMULATION_URL and _visible_frame(f, page.main_frame)]
            if report["ignored_frame_urls"] or len(sim) != 1 or page.frames != frames:
                raise BrowserSafetyStop("project_navigation_frame_changed")
            return report, sim[0]

        before, frame = read()
        save_probe(before, directory / "before")
        start = project_view(before)
        if expected_star is not None and (
            start["star"] is None or start["star"].casefold() != expected_star.casefold()
        ):
            raise BrowserSafetyStop("project_navigation_star_mismatch")
        if destination in TABS and start["surface"] not in {"detail", "list"}:
            raise BrowserSafetyStop("numbered_tab_requires_detail_or_list")
        target, target_evidence = _target(frame, destination)
        handle = target.element_handle(timeout=2000)
        intent = {
            "mode": "bounded_project_navigation",
            "destination": destination,
            "from": start,
            "expected_star": expected_star,
            "target_evidence": target_evidence,
            "max_clicks": 1,
            "answer_writes": 0,
            "collection_clicks": 0,
            "save_clicks": 0,
            "assessment_clicks": 0,
            "submission_clicks": 0,
            "automatic_retry": False,
            "task_completed": False,
            "action_source": "deterministic_navigation",
        }
        persist_json(directory / "reserved.json", intent)
        current, same_frame = read()
        save_probe(current, directory / "pre-click")
        rebound, evidence = _target(same_frame, destination)
        # Only exact known autosave footer placement is transient. Star identity,
        # controls, scores, rows, chart content and navigation geometry remain.
        if (
            same_frame != frame
            or screen_identity(navigation_status_projection(current, start))
            != screen_identity(navigation_status_projection(before, start))
            or project_view(current) != start
            or evidence != target_evidence
            or not handle.evaluate("(a,b)=>a.isConnected&&a===b", rebound.element_handle(timeout=2000))
        ):
            raise BrowserSafetyStop("project_navigation_target_or_screen_changed")
        attempted = True
        handle.click(timeout=3000)
        after, same_frame = read()
        save_probe(after, directory / "after")
        end = project_view(after)
        if same_frame != frame or _outside(before) != _outside(after):
            raise BrowserSafetyStop("project_navigation_external_state_changed")
        if destination in TABS:
            if (
                end["surface"] != start["surface"]
                or end["section"] != destination
                or (start["surface"] == "detail" and end["star"].casefold() != start["star"].casefold())
            ):
                raise BrowserSafetyStop("project_navigation_destination_or_star_changed")
        elif end["surface"] != destination:
            raise BrowserSafetyStop("project_navigation_destination_not_reached")
        result = {
            **intent,
            "to": end,
            "navigation_clicks": 1,
            "destination_verified": True,
            "same_star_verified": start["surface"] == end["surface"] == "detail",
            "suggested_required_text": _required(after, end),
            "config_mutated": False,
        }
        persist_json(directory / "confirmed.json", result)
        return result
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "project_navigation_failed",
                "click_may_have_occurred": attempted,
                "automatic_retry": False,
                "answer_writes": 0,
                "task_completed": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("project_navigation_failed") from None
    finally:
        page.remove_listener("dialog", dialog_handler)
        page.context.remove_listener("page", popup_handler)
