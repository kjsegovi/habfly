"""Bounded visible gas-checkbox comparisons; no spectral or grading oracle.

Candidates are explicit. Ordinary checkbox actions update the course's model;
only chart crops, visible absorption and selected controls are recorded. No
curve paths, hidden gas lists, application state, or automatic best-fit label.
"""

import hashlib
import json
import re
import time
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import read_habitability_choices
from .browser_habitability import GASES, SUBSCRIPTS, map_habitability_capture
from .browser_numeric import comparable_screen, screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_setup import rendered_control
from .browser_stellar import NUMBER, SIMULATION_URL, _atoms, _control_atom
from .presentation_capture import evidence_screenshot


def _gas_letters(text):
    letters = re.sub(r"[\s,]", "", text.translate(SUBSCRIPTS))
    if re.sub("|".join(GASES), "", letters):
        raise BrowserSafetyStop("unexpected_gas_menu_text")


def gas_projection(report, mapping):
    """Exclude only gas widget state, its absorption, and exact Save busy text."""
    value = comparable_screen(report)
    frame = next(f for f in value["frames"] if f["url"] == SIMULATION_URL)
    target = next(c["id"] for c in mapping["observation"]["controls"] if c["label"] == "trace_gases")
    gas = next(c for c in frame["controls"] if c["id"] == target)
    atom = _control_atom(gas["accessibility"])
    atoms = _atoms(frame["accessibility"])
    starts = [i for i, item in enumerate(atoms) if item == atom]
    if len(starts) != 1:
        raise BrowserSafetyStop("ambiguous_gas_control_projection")
    start = starts[0]
    ends = [
        i for i in range(start + 1, len(atoms)) if atoms[i][0] == "text" and "Absorption %" in atoms[i][1]
    ]
    if len(ends) != 1:
        raise BrowserSafetyStop("ambiguous_gas_absorption_projection")
    end = ends[0]
    letters = []
    for key, text in atoms[start + 1 : end]:
        if key in {"text", "subscript"}:
            letters.append(str(text))
        elif (
            not re.fullmatch(r'checkbox "(' + "|".join(GASES) + r')"(?: \[checked\])?', key)
            or text is not None
        ):
            raise BrowserSafetyStop("unexpected_gas_widget_content")
    prefix, suffix = atoms[end][1].split("Absorption %", 1)
    letters.append(prefix)
    _gas_letters("".join(letters))
    atoms[end] = ("text", "Absorption %" + suffix)
    atoms[start:end] = [("combobox", ["<gas selection>"])]
    normalized = []
    for key, text in atoms:
        if key == "text":
            text = re.sub(rf"(Absorption % ){NUMBER}%", r"\1<result>%", text)
            text = text.removesuffix(" Data saved")
        if key == 'button "Save" [disabled]':
            key = 'button "Save"'
        normalized.append((key, text))
    frame["accessibility"] = normalized
    text = " ".join(frame["text"].split())
    pattern = rf"(TRACE GASES PRESENT)(.*?)(ABSORPTION %)\s+({NUMBER})%"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if len(matches) != 1:
        raise BrowserSafetyStop("ambiguous_gas_text_projection")
    _gas_letters(matches[0][2])
    frame["text"] = re.sub(pattern, r"\1 <gas selection> \3 <result>%", text, flags=re.IGNORECASE)
    frame["text"] = frame["text"].removesuffix(" Data saved Save") + (
        " Save" if frame["text"].endswith(" Data saved Save") else ""
    )
    controls = []
    for control in frame["controls"]:
        if control["role"] == "checkbox":
            continue  # The strict mapper has validated the exact seven names.
        if control["id"] == target:
            control["value"], control["accessibility"] = None, ("combobox", ["<gas selection>"])
        if control["role"] == "button" and control["accessibility"] in {
            '- button "Save"',
            '- button "Save" [disabled]',
        }:
            control["accessibility"], control["enabled"] = '- button "Save"', True
        control["id"] = f"{frame['id']}:gas-projection:{len(controls)}"
        controls.append(control)
    frame["controls"] = controls
    values = deepcopy(mapping["observation"]["values"])
    for key in ("selected_gas_label", "selected_gases"):
        values[key] = None
    values["readouts"]["absorption"] = None
    # Observation-local IDs can shift when the seven menu controls appear.
    values["equilibrium_temp"].pop("capture_target_id")
    return value, values


class GasComparisonSession:
    def __init__(self, page, config, output, *, max_writes=14, max_seconds=240):
        if type(max_writes) is not int or not 1 <= max_writes <= 14 or not 1 <= max_seconds <= 300:
            raise ValueError("Explicit bounded gas comparison budget required")
        self.page, self.config, self.output = page, config, Path(output)
        self.frames = page.frames.copy()
        self.deadline = time.monotonic() + max_seconds
        self.max_writes, self.writes = max_writes, 0
        self.stopped = self.unexpected_dialog = self.open_attempted = False
        self.frame = None
        self.report, self.mapping, self.choices = self.read()
        if self.mapping["observation"]["values"]["selected_gases"]:
            raise BrowserSafetyStop("gas_comparison_requires_empty_selection")
        self.output.mkdir(parents=True, exist_ok=False)
        save_probe(self.report, self.output / "initial")
        self.emit_scope()
        self.handles = {}
        page.on("dialog", self._dialog)

    def emit_scope(self):
        persist_json(
            self.output / "scope.json",
            {
                "star": self.mapping["star_name"],
                "max_checkbox_writes": self.max_writes,
                "selection_source": "explicit_reference_comparison_not_learned",
                "hidden_spectrum_data_read": False,
                "numeric_writes": 0,
                "assessment_clicks": 0,
                "task_completed": False,
            },
        )

    def _dialog(self, dialog):
        self.unexpected_dialog = True
        dialog.dismiss()

    def close(self):
        self.stopped = True
        self.page.remove_listener("dialog", self._dialog)

    def read(self):
        self.page.wait_for_timeout(0)
        if self.stopped or time.monotonic() >= self.deadline:
            raise BrowserSafetyStop("gas_session_stopped_or_timed_out")
        if self.unexpected_dialog or self.page.frames != self.frames or len(self.page.context.pages) != 1:
            raise BrowserSafetyStop("gas_context_changed")
        report = inspect_page(self.page, self.config)
        if report["ignored_frame_urls"]:
            raise BrowserSafetyStop("unknown_gas_frame")
        frames = [f for f in self.frames if f.url == SIMULATION_URL]
        if len(frames) != 1 or self.frame is not None and self.frame != frames[0]:
            raise BrowserSafetyStop("gas_frame_changed")
        self.frame = frames[0]
        mapping = map_habitability_capture(report, capture_sha256=screen_identity(report))
        return report, mapping, read_habitability_choices(self.frame)[0]

    def current(self):
        report, mapping, choices = self.read()
        if (
            gas_projection(report, mapping) != gas_projection(self.report, self.mapping)
            or mapping["observation"]["values"] != self.mapping["observation"]["values"]
            or choices != self.choices
        ):
            self.stopped = True
            raise BrowserSafetyStop("stale_gas_observation")
        return report, mapping, choices

    def open_menu(self):
        if self.open_attempted:
            raise BrowserSafetyStop("gas_menu_already_attempted")
        self.current()
        self.open_attempted = True
        menus = [e for e in self.frame.get_by_role("combobox").all() if e.is_visible()]
        if len(menus) != 3:
            raise BrowserSafetyStop("ambiguous_gas_menus")
        if not any(e.is_visible() for e in self.frame.get_by_role("checkbox").all()):
            overlay = [
                e
                for e in menus[0].locator("..").locator(":scope > div").all()
                if e.is_visible() and rendered_control(e)
            ]
            if len(overlay) != 1 or overlay[0].inner_text().strip():
                raise BrowserSafetyStop("unsupported_gas_menu_overlay")
            mb, ob = menus[0].bounding_box(), overlay[0].bounding_box()
            persist_json(self.output / "menu-open-reserved.json", {"clicks": 1, "checkbox_writes": 0})
            self.current()
            overlay[0].click(
                position={
                    "x": mb["x"] - ob["x"] + mb["width"] - 10,
                    "y": mb["y"] - ob["y"] + mb["height"] / 2,
                },
                timeout=3000,
            )
        report, mapping, choices = self.read()
        if (
            gas_projection(report, mapping) != gas_projection(self.report, self.mapping)
            or mapping["observation"]["values"]["selected_gases"]
            or choices != self.choices
        ):
            self.stopped = True
            raise BrowserSafetyStop("unexpected_gas_menu_open_side_effect")
        self.report, self.mapping = report, mapping
        for name in GASES:
            target = self.frame.get_by_role("checkbox", name=name, exact=True)
            if (
                target.count() != 1
                or not target.is_enabled()
                or not rendered_control(target)
                or target.is_checked()
            ):
                raise BrowserSafetyStop("unavailable_gas_checkbox")
            self.handles[name] = target.element_handle(timeout=1000)
        save_probe(report, self.output / "menu-opened")

    def toggle(self, name, checked):
        if name not in GASES or type(checked) is not bool or set(self.handles) != set(GASES):
            raise BrowserSafetyStop("explicit_gas_checkbox_required")
        if self.writes >= self.max_writes:
            raise BrowserSafetyStop("gas_write_budget_exhausted")
        before, mapping, choices = self.current()
        prior = mapping["observation"]["values"]["selected_gases"]
        if (name in prior) == checked:
            raise BrowserSafetyStop("gas_checkbox_already_in_requested_state")
        current = self.frame.get_by_role("checkbox", name=name, exact=True)
        if not rendered_control(current) or not self.handles[name].evaluate(
            "(a,b)=>a.isConnected&&a===b", current.element_handle()
        ):
            raise BrowserSafetyStop("gas_checkbox_replaced")
        number = self.writes + 1
        intent = {
            "gas": name,
            "checked": checked,
            "prior": prior,
            "star": mapping["star_name"],
            "action_source": "reference_diagnostic",
            "automatic_retry": False,
        }
        dispatched = False
        try:
            persist_json(self.output / f"write-{number:02d}-reserved.json", intent)
            self.current()
            self.writes += 1
            dispatched = True
            self.handles[name].set_checked(checked, timeout=3000)
            after, newer, actual_choices = self.read()
            save_probe(after, self.output / f"write-{number:02d}-after")
            expected = (set(prior) | {name}) if checked else (set(prior) - {name})
            if (
                gas_projection(before, mapping) != gas_projection(after, newer)
                or actual_choices != choices
                or set(newer["observation"]["values"]["selected_gases"]) != expected
            ):
                raise BrowserSafetyStop("unexpected_gas_checkbox_side_effect")
            for label, handle in self.handles.items():
                fresh = self.frame.get_by_role("checkbox", name=label, exact=True)
                if not handle.evaluate(
                    "(a,b)=>a.isConnected&&a===b", fresh.element_handle()
                ) or fresh.is_checked() != (label in expected):
                    raise BrowserSafetyStop("gas_checkbox_readback_mismatch")
            self.report, self.mapping = after, newer
            receipt = {
                **intent,
                "selected": sorted(expected),
                "absorption": newer["observation"]["values"]["readouts"]["absorption"],
                "readback_verified": True,
                "correctness_verified": False,
                "task_completed": False,
            }
            persist_json(self.output / f"write-{number:02d}-confirmed.json", receipt)
            return receipt
        except BaseException as exc:
            self.stopped = True
            persist_json(
                self.output / f"write-{number:02d}-stopped.json",
                {
                    "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "gas_write_uncertain",
                    "write_may_have_occurred": dispatched,
                    "automatic_retry": False,
                },
            )
            if isinstance(exc, Exception) and not isinstance(exc, BrowserSafetyStop):
                # Driver diagnostics can contain private page content. Preserve
                # the uncertain dispatch receipt, but expose only a safe reason.
                raise BrowserSafetyStop("gas_write_uncertain") from None
            raise

    def capture_spectrum(self, name):
        if not re.fullmatch(r"baseline|CH4|CO2|H2O|H2S|N2O|NH3|O3|combined", name):
            raise ValueError("Named gas chart artifact required")
        self.current()
        charts = [
            e
            for e in self.frame.get_by_role("img").all()
            if e.is_visible() and "Flux WAVELENGTH" in e.aria_snapshot()
        ]
        if len(charts) != 2:
            raise BrowserSafetyStop("ambiguous_terrestrial_spectra")
        charts.sort(key=lambda e: e.bounding_box()["x"])
        path = self.output / f"{name}.png"
        if path.exists():
            raise BrowserSafetyStop("gas_crop_already_recorded")
        # Only the left transit chart, not the right emission chart or page.
        evidence_screenshot(charts[0], path=str(path))
        self.current()
        return path


def compare_visible_gas_candidates(page, config, output, *, candidates):
    if (
        not isinstance(candidates, (list, tuple))
        or not candidates
        or len(set(candidates)) != len(candidates)
        or not set(candidates) <= set(GASES)
    ):
        raise ValueError("Explicit unique known gas candidates required")
    session = GasComparisonSession(page, config, output, max_writes=2 * len(candidates))
    try:
        session.open_menu()
        session.capture_spectrum("baseline")
        for name in candidates:
            session.toggle(name, True)
            session.capture_spectrum(name)
            session.toggle(name, False)
        result = {
            "star": session.mapping["star_name"],
            "candidates": list(candidates),
            "checkbox_writes": session.writes,
            "baseline_restored": True,
            "gas_selection_inferred": False,
            "learned_gas_identification": False,
            "task_completed": False,
            "chart_sha256": {
                name: hashlib.sha256((session.output / f"{name}.png").read_bytes()).hexdigest()
                for name in ("baseline", *candidates)
            },
        }
        persist_json(session.output / "report.json", result)
        return result
    finally:
        session.close()


def select_reference_gases(page, config, output, *, comparison, gases, rationale):
    """Apply an explicit visual reference judgment, not a learned gas label.

    Requires a successful, unchanged candidate comparison for this star and an
    empty live selection. Valid wrong choices remain wrong; no ranking, grading,
    automatic correction, rollback or repeat write is provided.
    """
    if (
        not isinstance(gases, (list, tuple))
        or not gases
        or len(set(gases)) != len(gases)
        or not set(gases) <= set(GASES)
        or not isinstance(rationale, str)
        or not 20 <= len(rationale.strip()) <= 2000
    ):
        raise ValueError("Explicit compared gas candidates and visual rationale required")
    comparison = Path(comparison)
    raw = (comparison / "report.json").read_bytes()
    evidence = json.loads(raw)
    candidates, hashes = evidence.get("candidates", []), evidence.get("chart_sha256", {})
    if (
        not isinstance(candidates, list)
        or len(set(candidates)) != len(candidates)
        or not set(candidates) <= set(GASES)
        or not set(gases) <= set(candidates)
        or evidence.get("baseline_restored") is not True
        or evidence.get("checkbox_writes") != 2 * len(candidates)
        or evidence.get("gas_selection_inferred") is not False
        or evidence.get("learned_gas_identification") is not False
        or evidence.get("task_completed") is not False
        or set(hashes) != {"baseline", *candidates}
        or any(
            hashlib.sha256((comparison / f"{name}.png").read_bytes()).hexdigest() != digest
            for name, digest in hashes.items()
        )
    ):
        raise BrowserSafetyStop("incompatible_gas_comparison_evidence")
    session = GasComparisonSession(page, config, output, max_writes=len(gases))
    try:
        if session.mapping["star_name"] != evidence.get("star"):
            raise BrowserSafetyStop("gas_comparison_star_changed")
        initial = comparison / "initial/observation.json"
        initial_raw = initial.read_bytes()
        manifest = json.loads((comparison / "initial/manifest.json").read_bytes())
        if hashlib.sha256(initial_raw).hexdigest() != manifest.get("observation_sha256"):
            raise BrowserSafetyStop("gas_comparison_capture_changed")
        before = json.loads(initial_raw)
        mapping = map_habitability_capture(before, capture_sha256=screen_identity(before))
        if gas_projection(before, mapping) != gas_projection(session.report, session.mapping):
            raise BrowserSafetyStop("gas_comparison_conditions_changed")
        intent = {
            "star": evidence["star"],
            "gases": list(gases),
            "rationale": rationale,
            "action_source": "explicit_visual_reference_not_learned",
            "comparison_sha256": hashlib.sha256(raw).hexdigest(),
            "chart_sha256": hashes,
            "correctness_verified": False,
            "automatic_retry": False,
            "task_completed": False,
        }
        persist_json(session.output / "selection.json", intent)
        session.open_menu()
        for name in gases:
            session.toggle(name, True)
        session.capture_spectrum("combined")
        result = {
            **intent,
            "readback_verified": True,
            "checkbox_writes": session.writes,
            "absorption": session.mapping["observation"]["values"]["readouts"]["absorption"],
        }
        persist_json(session.output / "report.json", result)
        return result
    except BaseException as exc:
        persist_json(
            session.output / "selection-stopped.json",
            {
                "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "gas_selection_uncertain",
                "checkbox_writes_attempted": session.writes,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        raise
    finally:
        session.close()
