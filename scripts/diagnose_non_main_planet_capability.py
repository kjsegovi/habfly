"""Disposable public-UI capability diagnostic, NOT a scientific workflow.

One caller-chosen non-main class per fresh browser. Selecting Yes is an explicit
UI test, not transit evidence. No numeric copying, Save, assessment, score or
submission API is called. Credentials/preview URL use hidden terminal prompts.
"""

import argparse
import getpass
import hashlib
import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

from habfly.browser import BrowserSafetyStop
from habfly.browser_assessment_actions import persist_json
from habfly.browser_classification import StellarSelectionSession
from habfly.browser_next_star import capture_initial_setup_star
from habfly.browser_planet import BASE_FIELDS, PLANET_FIELDS
from habfly.browser_planet_numeric import PlanetNumericSession
from habfly.browser_planet_presence import DERIVED, presence_projection, preserved_paint_transition
from habfly.browser_probe import BrowserProbeConfig, inspect_page, save_probe
from habfly.browser_project_navigation import navigate_project
from habfly.browser_setup import BrowserSetup, SetupStop, rendered_control
from habfly.browser_stellar import SIMULATION_URL

CLASSES = ("white_dwarf", "red_giant", "supergiant")
MODE = "disposable_non_main_planet_ui_capability_v1"
FLAGS = {
    "diagnostic_only": True,
    "scientific_verified": False,
    "detected_planet": False,
    "reference_classification": False,
    "class_correctness_verified": False,
    "planet_presence_correctness_verified": False,
    "learned_policy": False,
    "training_label": False,
    "workflow_verified": False,
    "canonical_receipt": False,
    "task_completed": False,
    "project_completed": False,
    "submitted": False,
    "automatic_retry": False,
    "course_autosave_may_occur": True,
    "derived_numeric_writes": 0,
    "stellar_numeric_writes": 0,
    "planet_raw_numeric_writes": 0,
    "save_clicks": 0,
    "assessment_clicks": 0,
    "score_transfer_clicks": 0,
    "submission_clicks": 0,
}


class DiagnosticStop(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise DiagnosticStop(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_failure(error):
    if isinstance(error, DiagnosticStop):
        return (
            str(error)
            if re.fullmatch(r"diagnostic_[a-z0-9_]{1,100}", str(error))
            else "diagnostic_operation_failed"
        )
    if isinstance(error, (BrowserSafetyStop, SetupStop)):
        reason = str(error)
        if re.fullmatch(
            r"(?:setup_|initial_star_|planet_|project_navigation_|class_|fresh_star_|stellar_)[a-z0-9_:.-]{1,150}",
            reason,
        ):
            return reason
    return "diagnostic_operation_failed"


def field_summary(mapping):
    values = mapping["observation"]["values"]
    return {
        "star": mapping["star_name"],
        "has_planet": values["has_planet"],
        "visible_planet_field_map": deepcopy(values["browser_field_map"]),
        "stellar_inputs": deepcopy(values["stellar_inputs"]),
        "quantities_are_current_visible_reconstruction_readouts": True,
        "quantities_independently_validated": False,
        "quantity_origin_or_default_behavior_verified": False,
    }


def blank_planet(mapping, star, presence):
    values = mapping["observation"]["values"]
    fields = values["browser_field_map"]
    expected = BASE_FIELDS | (DERIVED if presence == "Yes" else set())
    units = dict(PLANET_FIELDS.values())
    require(mapping["star_name"].casefold() == star.casefold(), "diagnostic_star_changed")
    require(values["has_planet"] == presence, "diagnostic_presence_changed")
    require(set(fields) == expected, "diagnostic_conditional_fields_unsupported")
    require(
        all(
            spec["current_value"] == "" and spec["enabled"] is True and spec["unit"] == units[name]
            for name, spec in fields.items()
        ),
        "diagnostic_nonblank_or_unavailable_field",
    )


def native_presence(session, *, selected=None):
    control = session.frame.get_by_role("combobox")
    require(
        control.count() == 1
        and rendered_control(control)
        and control.is_enabled()
        and control.input_value() == (selected or "")
        and control.locator("option").all_text_contents() == ["", "Yes", "No"],
        "diagnostic_presence_control_not_verified",
    )
    handle = control.element_handle(timeout=2000)
    require(handle is not None, "diagnostic_presence_handle_missing")
    return handle


class CapabilityDiagnostic:
    MAX_SECONDS = 600

    def __init__(self, page, config, output, selected_class, fresh_star, *, started=None):
        require(selected_class in CLASSES, "diagnostic_unsupported_class")
        self.page, self.config, self.output = page, config, Path(output)
        self.selected_class, self.fresh_star = selected_class, Path(fresh_star)
        self.deadline = (time.monotonic() if started is None else started) + self.MAX_SECONDS
        self.frames, self.dialogs, self.popups = page.frames.copy(), [], []
        self.events, self.class_sessions, self.session = [], [], None
        self.closed = False
        self.report = {
            "mode": MODE,
            **FLAGS,
            "selected_class": selected_class,
            "capability_observed": False,
            "class_click_attempts": 0,
            "class_clicks_returned": 0,
            "yes_selection_attempted": False,
            "yes_selection_returned": False,
            "yes_readback_verified": False,
            "diagnostic_answer_write_attempts": 0,
        }
        self.dialog_handler = lambda dialog: self._dialog(dialog)
        self.popup_handler = lambda _: self.popups.append(True)
        page.on("dialog", self.dialog_handler)
        page.context.on("page", self.popup_handler)

    def _dialog(self, dialog):
        self.dialogs.append(True)
        dialog.dismiss()

    def check(self):
        self.page.wait_for_timeout(0)
        require(time.monotonic() < self.deadline, "diagnostic_time_limit")
        require(not self.dialogs and not self.popups, "diagnostic_unexpected_dialog_or_popup")
        require(len(self.page.context.pages) == 1, "diagnostic_context_changed")
        require(
            self.page.frames == self.frames and self.config.allows(self.page.url),
            "diagnostic_boundary_changed",
        )

    def emit(self, kind, payload):
        self.check()
        item = {"event": kind, "payload": payload, "mode": MODE, **FLAGS}
        persist_json(self.output / f"class-event-{len(self.events):02d}.json", item)
        self.events.append(item)
        self.check()  # Existing class adapter emits immediately before click.

    def _counts(self):
        self.report["class_click_attempts"] = sum(s._class_click_invoked for s in self.class_sessions)
        self.report["class_clicks_returned"] = sum(s._class_click_returned for s in self.class_sessions)
        self.report["diagnostic_answer_write_attempts"] = self.report["class_click_attempts"] + int(
            self.report["yes_selection_attempted"]
        )

    def run(self):
        try:
            self.check()
            session = StellarSelectionSession(self.page, self.config, self.emit, fresh_star=self.fresh_star)
            self.class_sessions.append(session)
            values = session.mapping["observation"]["values"]
            require(
                not values["conditional_fields_visible"] and values["color"]["selected"] is None,
                "diagnostic_fresh_stellar_state_required",
            )
            require(
                set(values["browser_field_map"]) == {"distance", "luminosity", "temperature"}
                and all(v["current_value"] == "" for v in values["browser_field_map"].values()),
                "diagnostic_blank_stellar_answers_required",
            )
            self.star = session.mapping["star_name"]
            self.report["star"] = self.star
            inherited = session.choices["selected"] == self.selected_class
            first = (
                ("red_giant" if self.selected_class == "white_dwarf" else "white_dwarf")
                if inherited
                else self.selected_class
            )
            save_probe(session.report, self.output / "stellar-before")
            persist_json(
                self.output / "diagnostic-class-intent.json",
                {
                    "mode": MODE,
                    **FLAGS,
                    "star": self.star,
                    "selected_class": self.selected_class,
                    "maximum_class_selections": 2 if inherited else 1,
                    "explicit_intermediate": first if inherited else None,
                    "rationale": "Deliberate UI capability test only; no inference about this star's real or course class.",
                },
            )
            self.check()
            session.select_class(first, source="reference_diagnostic")
            if inherited:
                save_probe(session.report, self.output / "stellar-intermediate")
                self.check()
                session = StellarSelectionSession(self.page, self.config, self.emit)
                self.class_sessions.append(session)
                session.select_class(
                    self.selected_class,
                    source="reference_diagnostic",
                    expected_previous=first,
                    revision_reason="Diagnostic UI capability test only; clear inherited paint and set the caller-selected non-main class. Not a reference classification, inferred class, or training label.",
                )
            self.check()
            require(session.choices["selected"] == self.selected_class, "diagnostic_class_readback_mismatch")
            values = session.mapping["observation"]["values"]
            require(
                session.mapping["star_name"].casefold() == self.star.casefold()
                and not values["conditional_fields_visible"]
                and set(values["browser_field_map"]) == {"distance", "luminosity", "temperature"}
                and all(v["current_value"] == "" for v in values["browser_field_map"].values()),
                "diagnostic_stellar_state_changed",
            )
            save_probe(session.report, self.output / "stellar-after")
            self.report["stellar_class_readback"] = deepcopy(session.choices)
            self.check()
            navigation = navigate_project(
                self.page, self.config, self.output / "to-planet", "planet", expected_star=self.star
            )
            self.check()
            for frame in self.config.frames:
                if frame.url == SIMULATION_URL:
                    frame.required_text = navigation["suggested_required_text"]
            self.select_yes()
            self.report["capability_observed"] = True
            self.report["outcome"] = "diagnostic_non_main_yes_panel_visible_not_workflow"
        except BaseException as error:  # noqa: BLE001 - sanitize uncertain driver failures, never retry
            self.report["failure_reason"] = safe_failure(error)
            self.report["outcome"] = "diagnostic_stopped_no_retry"
        finally:
            self._counts()
            self.close()
        return self.report

    def select_yes(self):
        self.check()
        self.session = PlanetNumericSession(
            self.page,
            self.config,
            self.output / "planet-read-guard",
            max_seconds=max(1, self.deadline - time.monotonic()),
            _allow_unset_planet=True,
        )
        before, mapping, choices, _ = self.session.current()
        blank_planet(mapping, self.star, None)
        self.report["before"] = field_summary(mapping)
        handle = native_presence(self.session)
        save_probe(before, self.output / "planet-before")
        self.report["before"].update(
            selected_class=self.selected_class,
            selected_class_capture_sha256=sha(self.output / "stellar-after/observation.json"),
            observation_sha256=sha(self.output / "planet-before/observation.json"),
        )
        persist_json(
            self.output / "diagnostic-yes-intent.json",
            {
                "mode": MODE,
                **FLAGS,
                "star": self.star,
                "selected_class": self.selected_class,
                "value": "Yes",
                "max_selections": 1,
                "rationale": "Reveal the visible reconstruction controls as a disposable UI capability test, not a detected-planet answer.",
                "before_observation_sha256": sha(self.output / "planet-before/observation.json"),
            },
        )
        self.session.current()
        require(
            handle.evaluate("(a,b)=>a.isConnected&&a===b", native_presence(self.session)),
            "diagnostic_presence_handle_replaced",
        )
        self.check()
        self.report["yes_selection_attempted"] = True
        handle.select_option(label="Yes", timeout=3000)
        self.report["yes_selection_returned"] = True
        try:
            after, newer, actual, _ = self.session.read()
        except Exception:
            # Preserve only a fresh, normally guarded public capture; this is
            # not a replacement mapping or authorization after a failed guard.
            self.check()
            raw = inspect_page(self.page, self.config)
            require(not raw["ignored_frame_urls"], "diagnostic_unknown_visible_frame")
            save_probe(raw, self.output / "planet-unmapped-after")
            raise
        save_probe(after, self.output / "planet-after")
        self.report["after"] = field_summary(newer)
        self.report["after"].update(
            selected_class=self.selected_class,
            selected_class_capture_sha256=sha(self.output / "stellar-after/observation.json"),
            observation_sha256=sha(self.output / "planet-after/observation.json"),
        )
        blank_planet(newer, self.star, "Yes")
        require(
            presence_projection(before, mapping) == presence_projection(after, newer)
            and preserved_paint_transition(choices, actual),
            "diagnostic_unexpected_presence_side_effect",
        )
        require(
            handle.evaluate("(a,b)=>a.isConnected&&a===b", native_presence(self.session, selected="Yes")),
            "diagnostic_presence_handle_replaced",
        )
        self.check()
        self.report["yes_readback_verified"] = True

    def close(self):
        if self.closed:
            return
        # Latch before driver cleanup: a closed target must never trigger a
        # second detach attempt or hide the original diagnostic result.
        self.closed = True
        for session in self.class_sessions:
            session.stopped = True
        operations = [
            (
                "diagnostic_dialog_listener_cleanup_failed",
                lambda: self.page.remove_listener("dialog", self.dialog_handler),
            ),
            (
                "diagnostic_popup_listener_cleanup_failed",
                lambda: self.page.context.remove_listener("page", self.popup_handler),
            ),
        ]
        if self.session is not None:
            operations.insert(0, ("diagnostic_planet_session_cleanup_failed", self.session.close))
        for reason, operation in operations:
            try:
                operation()
            except Exception:  # noqa: BLE001 - never expose driver text during cleanup
                self.report.setdefault("cleanup_failures", []).append(reason)


def cleanup(report, *, setup, probe, browser, browser_closed):
    """Best-effort shutdown with fixed codes; never suppress final reporting."""
    for name, resource in (("setup", setup), ("probe", probe)):
        if resource is not None:
            try:
                resource.close()
            except Exception:  # noqa: BLE001 - cleanup may follow a closed driver
                report.setdefault("cleanup_failures", []).append(f"diagnostic_{name}_cleanup_failed")
    if probe is not None:
        failures = report.setdefault("cleanup_failures", [])
        failures.extend(code for code in probe.report.get("cleanup_failures", []) if code not in failures)
    if browser is not None:
        try:
            browser.close()
            browser_closed = True
        except Exception:  # noqa: BLE001 - closure is not confirmed by an exception
            report.setdefault("cleanup_failures", []).append("diagnostic_browser_cleanup_failed")
            browser_closed = False
    report["browser_closed"] = browser_closed
    report["cleanup_completed"] = browser_closed and not report.get("cleanup_failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="selected_class", choices=CLASSES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    experiments = (Path.cwd() / "experiments").resolve()
    require(
        output.is_relative_to(experiments) and output != experiments and not output.exists(),
        "diagnostic_output_must_be_new_under_experiments",
    )
    require(sys.stdin.isatty(), "diagnostic_hidden_prompt_requires_terminal")
    # Driver debugging can include authentication actions. Do not inherit it
    # into a disposable login diagnostic or persist it beside the captures.
    for key in ("DEBUG", "PWDEBUG", "DEBUG_FILE"):
        os.environ.pop(key, None)
    url = getpass.getpass("Preview URL (hidden): ")
    email = getpass.getpass("Email (hidden): ")
    password = getpass.getpass("Password (hidden): ")
    try:
        raw = json.loads(Path("configs/browser_probe.example.json").read_bytes())
        raw["url"] = url
        config = BrowserProbeConfig.model_validate(raw)
    except Exception:  # noqa: BLE001 - configuration error may contain the hidden preview URL
        raise SystemExit("Invalid local preview configuration; no browser opened") from None
    output.mkdir(parents=True, exist_ok=False)
    persist_json(
        output / "scope.json",
        {
            "mode": MODE,
            **FLAGS,
            "selected_class": args.selected_class,
            "max_seconds": CapabilityDiagnostic.MAX_SECONDS,
            "setup_max_seconds": BrowserSetup.MAX_SECONDS,
            "maximum_class_selections": 2,
            "maximum_yes_selections": 1,
            "diagnostic_class_and_yes_are_answer_writes": True,
            "no_authentication_captures": True,
            "fresh_context": True,
            "helper_sha256": sha(__file__),
        },
    )
    from playwright.sync_api import sync_playwright

    report = {"mode": MODE, **FLAGS, "setup_ready": False, "selected_class": args.selected_class}
    browser, setup, probe, closed = None, None, None, False
    started = time.monotonic()
    try:
        with sync_playwright() as driver:
            browser = driver.chromium.launch(headless=False, timeout=30000)
            context = browser.new_context(viewport={"width": 1600, "height": 1100})
            page = context.new_page()
            setup_output = output / "setup"
            setup_output.mkdir()
            setup = BrowserSetup(page, config, (email, password), output=setup_output)
            email = password = url = ""
            page.goto(config.url, wait_until="domcontentloaded", timeout=30000)
            for _ in range(512):
                if setup.advance() == "stellar":
                    break
                page.wait_for_timeout(100)
            require(setup.closed and setup.stage == "stellar_screen_ready", "diagnostic_setup_not_complete")
            capture_initial_setup_star(setup, output / "initial-star")
            report["setup_ready"] = True
            for rule in config.frames:
                rule.required_text = []
            probe = CapabilityDiagnostic(
                page, config, output, args.selected_class, output / "initial-star", started=started
            )
            report.update(probe.run())
            browser.close()
            closed, browser = True, None
    except BaseException as error:  # noqa: BLE001 - preserve sanitized evidence and close browser
        report["failure_reason"] = safe_failure(error)
        report["outcome"] = "diagnostic_stopped_no_retry"
    finally:
        cleanup(report, setup=setup, probe=probe, browser=browser, browser_closed=closed)
        report["source_sha256"] = {
            str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and not p.is_symlink()
        }
        persist_json(output / "diagnostic-report.json", report)
        print(
            json.dumps(
                {
                    key: report.get(key)
                    for key in (
                        "mode",
                        "outcome",
                        "setup_ready",
                        "selected_class",
                        "star",
                        "capability_observed",
                        "diagnostic_answer_write_attempts",
                        "failure_reason",
                        "browser_closed",
                        "task_completed",
                        "project_completed",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
