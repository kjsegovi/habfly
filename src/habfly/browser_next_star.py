"""Bounded next-dot setup from an already-open, owned starfield.

No credentials, login, introductory navigation, answers, deletion, assessment,
save or submission. The selected pixels and fresh stellar readbacks are journaled.
This is deterministic navigation, not a learned scientific decision.
"""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import read_class_choices
from .browser_numeric import screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_setup import (
    BrowserSetup,
    SetupStop,
    numbered_tab,
    rendered_control,
    rendered_text,
    visible_star_point,
)
from .browser_stellar import SIMULATION_URL, map_stellar_capture


class NextStarPicker(BrowserSetup):
    def __init__(
        self,
        page,
        config,
        output,
        *,
        visited_stars,
        excluded_points,
        anchor=(0.4, 0.55),
        emit=lambda *_: None,
    ):
        if (
            not isinstance(visited_stars, (list, tuple))
            or len(visited_stars) > 500
            or any(
                not isinstance(s, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", s)
                for s in visited_stars
            )
            or len({s.casefold() for s in visited_stars}) != len(visited_stars)
            or len(excluded_points) != len(visited_stars)
            or not isinstance(anchor, (list, tuple))
            or len(anchor) != 2
            or any(type(v) not in {int, float} or not 0.15 <= v <= 0.85 for v in anchor)
        ):
            raise ValueError("An explicit one-point-per-visited-star inventory is required")
        self.directory = Path(output)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.visited_names = {s.casefold() for s in visited_stars}
        self.initial_frames = page.frames.copy()
        self.boundary_config = config.model_copy(deep=True)
        stellar_config = config.model_copy(deep=True)
        for rule in self.boundary_config.frames:
            if rule.url == SIMULATION_URL:
                rule.required_text = []
        for rule in stellar_config.frames:
            if rule.url == SIMULATION_URL:
                rule.required_text = []  # The strict stellar mapper validates labels/units.
        self.initial = inspect_page(page, self.boundary_config)
        if self.initial["ignored_frame_urls"]:
            raise BrowserSafetyStop("unknown_visible_frame")
        sim = [f for f in page.frames if f.url == SIMULATION_URL]
        if len(sim) != 1:
            raise BrowserSafetyStop("ambiguous_simulation_frame")
        text = rendered_text(sim[0]).upper()
        if not all(s in text for s in ("FUNDING", "DATA QUALITY")) or any(
            s in text for s in ("OBSERVATIONS", "RECONSTRUCTION", "ANALYZED DATA", "TOTAL COLLECTED")
        ):
            raise BrowserSafetyStop("next_star_requires_open_starfield")
        if any(
            rendered_control(e)
            for e in sim[0].get_by_text(re.compile(r"^VIEW STAR DATA$", re.IGNORECASE)).all()
        ):
            raise BrowserSafetyStop("next_star_existing_tooltip_visible")
        persist_json(
            self.directory / "scope.json",
            {
                "action_source": "deterministic_navigation",
                "visited_stars": list(visited_stars),
                "excluded_points": list(excluded_points),
                "starfield_anchor": list(anchor),
                "selection_basis": "rendered_dot_geometry_only_not_scientific_class",
                "max_star_clicks": 1,
                "max_view_clicks": 1,
                "max_seconds": self.MAX_SECONDS,
                "answer_writes": 0,
                "task_completed": False,
                "collection_count_verified": False,
            },
        )
        save_probe(self.initial, self.directory / "before")
        super().__init__(page, stellar_config, ("", ""), output=self.directory, emit=emit)
        self.excluded_star_points = tuple(excluded_points)
        self.starfield_anchor = tuple(anchor)

    def allowed(self, url):
        return self.config.allows(url)  # Never permit authentication/intro navigation.

    def advance(self):
        try:
            return self._advance()
        except (SetupStop, BrowserSafetyStop):
            self.close()
            raise
        except Exception:  # noqa: BLE001 - never expose driver URLs or credentials
            self.close()
            raise SetupStop("setup_browser_operation_failed") from None

    def _advance(self):
        self._guard()
        if self.page.frames != self.initial_frames:
            raise SetupStop("setup_simulation_frame_changed")
        capture = inspect_page(self.page, self.boundary_config)
        if capture["ignored_frame_urls"]:
            raise BrowserSafetyStop("unknown_visible_frame")
        if capture["outer_controls"] != self.initial["outer_controls"] or [
            f for f in capture["frames"] if f["url"] != SIMULATION_URL
        ] != [f for f in self.initial["frames"] if f["url"] != SIMULATION_URL]:
            raise BrowserSafetyStop("next_star_external_controls_changed")
        frame = next(f for f in self.page.frames if f.url == SIMULATION_URL)
        result = self._star(frame, rendered_text(frame))
        if result == "stellar":
            report = inspect_page(self.page, self.config)
            mapping = map_stellar_capture(report, capture_sha256=screen_identity(report))
            choices, _ = read_class_choices(frame)
            fields = mapping["observation"]["values"]["browser_field_map"]
            if mapping["star_name"].casefold() in self.visited_names or any(
                f["current_value"] != "" for f in fields.values()
            ):
                raise BrowserSafetyStop("next_star_not_fresh")
            save_probe(report, self.directory / "stellar")
            self.receipt = {
                "star": mapping["star_name"],
                "selected_point": self.star_point,
                "stellar_observations_verified": True,
                "fresh_blank_numeric_answers_verified": True,
                "painted_stellar_class": choices["selected"],
                "class_selection_verified": False,
                "action_source": "deterministic_navigation",
                "answer_writes": 0,
                "collection_count_verified": False,
                "task_completed": False,
            }
            persist_json(self.directory / "confirmed.json", self.receipt)
        return result


def open_next_star(
    page, config, output, *, visited_stars, excluded_points, anchor=(0.4, 0.55), emit=lambda *_: None
):
    picker = None
    try:
        picker = NextStarPicker(
            page,
            config,
            output,
            visited_stars=visited_stars,
            excluded_points=excluded_points,
            anchor=anchor,
            emit=emit,
        )
        while picker.advance() != "stellar":
            page.wait_for_timeout(100)
        return picker.receipt
    except BaseException as exc:
        if Path(output).exists():
            persist_json(
                Path(output) / "stopped.json",
                {
                    "reason": str(exc)
                    if isinstance(exc, (BrowserSafetyStop, SetupStop))
                    else "next_star_failed",
                    "star_click_may_have_occurred": bool(picker and picker.star_clicked),
                    "view_click_may_have_occurred": bool(picker and picker.view_clicked),
                    "automatic_retry": False,
                    "task_completed": False,
                },
            )
        raise
    finally:
        if picker:
            picker.close()


INITIAL_SELECTION_VERSION = 1
INITIAL_STARFIELD_IMAGE = "setup-starfield.png"


def validate_initial_selection(scope, png, point):
    """Reproduce owned rendered selection; legacy receipts imply default only.

    New metadata is never backfilled from a caller's chosen anchor. Historical
    non-default selections without recorded provenance remain unsupported.
    """
    if not isinstance(scope, dict) or hashlib.sha256(png).hexdigest() != scope.get("starfield_sha256"):
        raise BrowserSafetyStop("initial_star_selection_source_changed")
    keys = {"starfield_anchor", "excluded_points", "starfield_image"}
    if "selection_schema_version" in scope:
        if (
            type(scope["selection_schema_version"]) is not int
            or scope["selection_schema_version"] != INITIAL_SELECTION_VERSION
            or scope.get("starfield_image") != INITIAL_STARFIELD_IMAGE
            or type(scope.get("starfield_anchor")) is not list
            or type(scope.get("excluded_points")) is not list
            or scope["excluded_points"] != []
            or not keys <= scope.keys()
        ):
            raise BrowserSafetyStop("initial_star_invalid_selection_provenance")
        anchor = scope["starfield_anchor"]
    else:
        if keys & scope.keys():
            raise BrowserSafetyStop("initial_star_unversioned_selection_provenance")
        anchor = (0.4, 0.55)
    if visible_star_point(png, anchor=anchor) != point:
        raise BrowserSafetyStop("initial_star_selection_source_changed")
    return tuple(anchor)


def capture_initial_setup_star(setup, output):
    """Read-only handoff from completed login/intro/first-dot setup.

    Reuse the actual in-process setup owner, never credentials or a serialized
    caller assertion. This supplies the same fresh-star contract used after
    later starfield selections; no second dot or View Star Data click occurs.
    """
    if (
        not isinstance(setup, BrowserSetup)
        or isinstance(setup, NextStarPicker)
        or setup.stage != "stellar_screen_ready"
        or setup.closed is not True
        or setup.star_clicked is not True
        or setup.view_clicked is not True
        or not setup.ready_signature
        or setup.output is None
    ):
        raise BrowserSafetyStop("initial_star_requires_completed_setup")
    directory, source = Path(output), Path(setup.output)
    # The public rendered starfield is the only selection source. Authentication
    # screenshots, URLs and the original credential arguments are never copied.
    png = (source / INITIAL_STARFIELD_IMAGE).read_bytes()
    anchor, point = deepcopy(setup.starfield_anchor), deepcopy(setup.star_point)
    signature = setup.ready_signature
    if setup.excluded_star_points or visible_star_point(png, anchor=anchor) != point:
        raise BrowserSafetyStop("initial_star_selection_source_changed")
    directory.mkdir(parents=True, exist_ok=False)
    try:
        frames = setup.page.frames.copy()
        if len(setup.page.context.pages) != 1:
            raise BrowserSafetyStop("initial_star_unexpected_popup")
        before = inspect_page(setup.page, setup.config)
        if before["ignored_frame_urls"] or screen_identity(before) != signature:
            raise BrowserSafetyStop("initial_star_setup_screen_changed")
        mapping = map_stellar_capture(before, capture_sha256=screen_identity(before))
        frame = next(frame for frame in frames if frame.url == SIMULATION_URL)
        choices, handles = read_class_choices(frame)
        fields = mapping["observation"]["values"]["browser_field_map"]
        if any(field["current_value"] != "" for field in fields.values()):
            raise BrowserSafetyStop("initial_star_answers_not_blank")
        save_probe(before, directory / "before")
        after = inspect_page(setup.page, setup.config)
        newer, rebound = read_class_choices(frame)
        if (
            setup.page.frames != frames
            or len(setup.page.context.pages) != 1
            or screen_identity(before) != screen_identity(after)
            or choices != newer
            or setup.starfield_anchor != anchor
            or setup.star_point != point
            or setup.ready_signature != signature
            or setup.excluded_star_points
            or (source / INITIAL_STARFIELD_IMAGE).read_bytes() != png
            or any(
                not old.evaluate("(a,b)=>a.isConnected&&a===b", rebound[name])
                for name, old in handles.items()
            )
        ):
            raise BrowserSafetyStop("initial_star_changed_during_capture")
        save_probe(after, directory / "stellar")
        receipt = {
            "star": mapping["star_name"],
            "selected_point": point,
            "stellar_observations_verified": True,
            "fresh_blank_numeric_answers_verified": True,
            "painted_stellar_class": choices["selected"],
            "class_selection_verified": False,
            "action_source": "deterministic_navigation",
            "answer_writes": 0,
            "collection_count_verified": False,
            "task_completed": False,
        }
        scope = {
            "mode": "read_only_initial_setup_handoff",
            "browser_actions": 0,
            "selection_source": "completed_initial_setup_rendered_starfield",
            "starfield_sha256": hashlib.sha256(png).hexdigest(),
            "ready_screen_sha256": signature,
            "credentials_recorded": False,
            "automatic_retry": False,
            "selection_schema_version": INITIAL_SELECTION_VERSION,
            "starfield_anchor": list(anchor),
            "excluded_points": [],
            "starfield_image": INITIAL_STARFIELD_IMAGE,
        }
        validate_initial_selection(scope, png, point)
        # The already captured public starfield only: no login images or URLs.
        with (directory / INITIAL_STARFIELD_IMAGE).open("xb") as stream:
            stream.write(png)
        persist_json(directory / "scope.json", scope)
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "initial_star_capture_failed",
                "browser_actions": 0,
                "answer_writes": 0,
                "task_completed": False,
            },
        )
        if isinstance(exc, Exception) and not isinstance(exc, BrowserSafetyStop):
            raise BrowserSafetyStop("initial_star_capture_failed") from None
        raise


def reconcile_opened_star(page, config, failed_output, output):
    """Continue only the missing stellar-tab click; never reselect or reopen."""
    from .browser_classification import read_planet_class_choices
    from .browser_planet import map_planet_capture

    failed, directory = Path(failed_output), Path(output)
    read_json = lambda p: json.loads(p.read_text())
    stopped = read_json(failed / "stopped.json")
    if stopped != {
        "reason": "setup_ambiguous_or_unavailable_control",
        "star_click_may_have_occurred": True,
        "view_click_may_have_occurred": True,
        "automatic_retry": False,
        "task_completed": False,
    }:
        raise BrowserSafetyStop("unsupported_next_star_reconciliation")
    scope = read_json(failed / "scope.json")
    point = visible_star_point(
        (failed / "setup-starfield.png").read_bytes(),
        excluded=scope["excluded_points"],
        anchor=scope.get("starfield_anchor", (0.4, 0.55)),
    )
    boundary = config.model_copy(deep=True)
    for rule in boundary.frames:
        if rule.url == SIMULATION_URL:
            rule.required_text = []
    before = inspect_page(page, boundary)
    frames = page.frames.copy()
    if before["ignored_frame_urls"] or len(page.context.pages) != 1:
        raise BrowserSafetyStop("unexpected_next_star_context")
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    mapped = map_planet_capture(before, capture_sha256=screen_identity(before))
    values = mapped["observation"]["values"]
    if (
        mapped["star_name"].casefold() in {s.casefold() for s in scope["visited_stars"]}
        or values["has_planet"] is not None
        or any(
            v["current_value"] != ""
            for k, v in values["browser_field_map"].items()
            if k != "observation_days"
        )
    ):
        raise BrowserSafetyStop("next_star_reconciliation_not_blank")
    planet_paint = read_planet_class_choices(frame)[0]["selected"]
    source_before = read_json(failed / "before/observation.json")
    manifest = read_json(failed / "before/manifest.json")
    if (
        hashlib.sha256((failed / "before/observation.json").read_bytes()).hexdigest()
        != manifest["observation_sha256"]
    ):
        raise BrowserSafetyStop("next_star_source_hash_mismatch")
    outside = lambda report: (
        report["outer_controls"],
        [f for f in report["frames"] if f["url"] != SIMULATION_URL],
    )
    if outside(before) != outside(source_before):
        raise BrowserSafetyStop("next_star_external_controls_changed")
    # This source owns one immutable continuation; failed attempts are not retried.
    persist_json(
        failed / "stellar-tab-continuation-reserved.json",
        {
            "output": str(directory),
            "star": mapped["star_name"],
            "max_tab_clicks": 1,
            "star_click_retries": 0,
            "view_click_retries": 0,
        },
    )
    directory.mkdir(parents=True, exist_ok=False)
    save_probe(before, directory / "before")
    clicked = False
    try:
        handle = numbered_tab(frame, 1).element_handle(timeout=2000)
        again = inspect_page(page, boundary)
        if (
            page.frames != frames
            or screen_identity(again) != screen_identity(before)
            or not handle.evaluate("(a,b)=>a.isConnected&&a===b", numbered_tab(frame, 1).element_handle())
        ):
            raise BrowserSafetyStop("next_star_reconciliation_changed")
        clicked = True
        handle.click(timeout=3000)
        after = inspect_page(page, boundary)
        save_probe(after, directory / "stellar")
        fresh = map_stellar_capture(after, capture_sha256=screen_identity(after))
        if (
            page.frames != frames
            or outside(after) != outside(before)
            or fresh["star_name"].casefold() != mapped["star_name"].casefold()
            or any(
                f["current_value"] != "" for f in fresh["observation"]["values"]["browser_field_map"].values()
            )
        ):
            raise BrowserSafetyStop("next_star_reconciliation_side_effect")
        receipt = {
            "star": fresh["star_name"],
            "selected_point": point,
            "stellar_observations_verified": True,
            "fresh_blank_numeric_answers_verified": True,
            "painted_stellar_class": read_class_choices(frame)[0]["selected"],
            "planet_paint_before_tab": planet_paint,
            "class_selection_verified": False,
            "action_source": "deterministic_navigation",
            "answer_writes": 0,
            "collection_count_verified": False,
            "task_completed": False,
            "failed_run": str(failed),
            "original_failure_preserved": True,
            "star_click_retries": 0,
            "view_click_retries": 0,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "reason": str(exc)
                if isinstance(exc, (BrowserSafetyStop, SetupStop))
                else "next_star_continuation_failed",
                "tab_click_may_have_occurred": clicked,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        raise
