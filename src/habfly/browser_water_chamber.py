"""Read water phase through the course's visible pressure-temperature tool.

The caller supplies conditions. This is UI transport, not a local phase oracle,
an answer selector, or permission to inspect the helper's application state.
Every invocation opens and closes its own help dialog once; uncertain actions
are recorded, never retried. The task's controls must be unchanged afterward.
"""

import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_numeric import comparable_screen
from .browser_planet_chart import VISIBLE_TOOLTIP
from .browser_probe import inspect_page, save_probe
from .browser_setup import RENDER_VISIBILITY_JS
from .browser_stellar import NUMBER
from .presentation_capture import evidence_screenshot

CHAMBER_URL = "https://s3.us-west-2.amazonaws.com/etx-habworlds/phases-of-matter/index.html"
HELP_BUTTON = "Pressure-Temperature Chamber Additional Information"
PHASES = ("liquid", "solid", "gas")


def chamber_control_exposed(control):
    """A child caption/icon is part of its button, not a sibling occluder."""
    return control.evaluate(
        "e=>{"
        + RENDER_VISIBILITY_JS
        + """
        if(!styled(e)) return false;
        const r=e.getBoundingClientRect();
        let left=Math.max(0,r.left),right=Math.min(innerWidth,r.right);
        let top=Math.max(0,r.top),bottom=Math.min(innerHeight,r.bottom);
        for(let p=e.parentElement;p;p=p.parentElement){
            const s=getComputedStyle(p),b=p.getBoundingClientRect();
            if(['hidden','clip','scroll','auto'].includes(s.overflowX)){left=Math.max(left,b.left);right=Math.min(right,b.right);}
            if(['hidden','clip','scroll','auto'].includes(s.overflowY)){top=Math.max(top,b.top);bottom=Math.min(bottom,b.bottom);}
        }
        if(right<=left||bottom<=top) return false;
        const hit=document.elementFromPoint((left+right)/2,(top+bottom)/2);
        return !!hit&&(hit===e||e.contains(hit));
    }"""
    )


def condition_number(text):
    if not isinstance(text, str) or len(text) > 64 or not re.fullmatch(NUMBER, text):
        raise BrowserSafetyStop("invalid_chamber_condition")
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise BrowserSafetyStop("invalid_chamber_condition") from None
    if not value.is_finite() or value <= 0:
        raise BrowserSafetyStop("invalid_chamber_condition")
    return value


class WaterChamberSession:
    """One bounded visible-tool query. No persistent browser/profile ownership."""

    def __init__(self, page, config, output: Path, emit=lambda *_: None, *, max_seconds=40):
        if not 1 <= max_seconds <= 90:
            raise ValueError("Chamber queries require a 1..90 second budget")
        self.page, self.config, self.output, self.emit = page, config, Path(output), emit
        self.max_seconds = max_seconds
        self.attempted = self.closed = self.unexpected_dialog = False
        self.close_attempted = False
        self.dialog = self.chamber = None
        self.dialog_handle = None

    def _dialog(self, dialog):
        self.unexpected_dialog = True
        dialog.dismiss()

    def _guard(self):
        if self.unexpected_dialog:
            raise BrowserSafetyStop("chamber_unexpected_dialog")
        if time.monotonic() >= self.deadline:
            raise BrowserSafetyStop("chamber_time_limit")
        if len(self.page.context.pages) != 1 or not self.config.allows(self.page.url):
            raise BrowserSafetyStop("chamber_navigation_changed")
        if any(f not in self.page.frames or f.url != url for f, url in self.base_frames):
            raise BrowserSafetyStop("chamber_base_frame_changed")
        dialogs = [x for x in self.page.get_by_role("dialog").all() if x.is_visible()]
        if len(dialogs) != 1 or not self.dialog_handle.evaluate(
            "(a,b)=>a.isConnected&&a===b", dialogs[0].element_handle(timeout=1000)
        ):
            raise BrowserSafetyStop("chamber_help_dialog_changed")
        if any(x.is_visible() for x in self.page.locator('input[type="password"]').all()):
            raise BrowserSafetyStop("chamber_authentication_lost")
        extra = [f for f in self.page.frames if f not in [p for p, _ in self.base_frames]]
        if len(extra) > 1:
            raise BrowserSafetyStop("chamber_unknown_frame")
        for frame in extra:
            if frame.url not in {CHAMBER_URL, "about:blank"} or not self.dialog_handle.evaluate(
                "(d,e)=>d.contains(e)", frame.frame_element()
            ):
                raise BrowserSafetyStop("chamber_unknown_frame")
            if self.chamber is not None and frame != self.chamber:
                raise BrowserSafetyStop("chamber_frame_replaced")
        if self.chamber is not None and (self.chamber not in extra or self.chamber.url != CHAMBER_URL):
            raise BrowserSafetyStop("chamber_frame_changed")

    @staticmethod
    def _one(locator):
        visible = [x for x in locator.all() if x.is_visible()]
        if len(visible) != 1 or not visible[0].is_enabled() or not chamber_control_exposed(visible[0]):
            raise BrowserSafetyStop("chamber_control_unavailable")
        return visible[0]

    def _conditions(self):
        self._guard()
        # Only the visible material button is evidence of the selected material.
        self._one(self.chamber.get_by_role("button", name="Water (H2O)", exact=True))
        if any(
            x.is_visible()
            for x in self.chamber.locator('input[type="password"], [role="dialog"], dialog[open]').all()
        ):
            raise BrowserSafetyStop("chamber_unexpected_state")
        for unit, handle in self.inputs.items():
            current = self._one(self.chamber.get_by_role("textbox", name=unit, exact=True))
            if not handle.evaluate("(a,b)=>a.isConnected&&a===b", current.element_handle(timeout=1000)):
                raise BrowserSafetyStop("chamber_input_replaced")

    def _phase(self):
        self._conditions()
        observed = []
        for name in PHASES:
            images = self.chamber.get_by_role("img", name=name, exact=True)
            if images.count() != 1:
                raise BrowserSafetyStop("chamber_ambiguous_phase_icons")
            icon = images.first
            paint = icon.evaluate("""e=>{
                let opacity=1; for(let p=e;p;p=p.parentElement) opacity*=Number(getComputedStyle(p).opacity);
                const s=getComputedStyle(e);return {opacity,display:s.display,visibility:s.visibility};
            }""")
            observed.append({"phase": name, "paint": paint, "fully_exposed": icon.evaluate(VISIBLE_TOOLTIP)})
        visible = [x["phase"] for x in observed if x["paint"]["opacity"] == 1 and x["fully_exposed"]]
        if len(visible) != 1 or any(
            x["paint"]["opacity"] != 0 for x in observed if x["phase"] not in visible
        ):
            return None, observed
        return visible[0], observed

    def query(self, pressure: str, temperature: str, *, pressure_unit="atm", temperature_unit="K", source):
        expected = {"atm": condition_number(pressure), "K": condition_number(temperature)}
        if (pressure_unit, temperature_unit) != ("atm", "K") or source not in {
            "checkpoint",
            "reference_diagnostic",
        }:
            raise BrowserSafetyStop("invalid_chamber_request")
        if self.attempted or self.closed:
            raise BrowserSafetyStop("chamber_session_already_used")
        before = inspect_page(self.page, self.config)
        if before["ignored_frame_urls"] or len(self.page.context.pages) != 1:
            raise BrowserSafetyStop("chamber_unsafe_initial_context")
        if any(f.url == CHAMBER_URL for f in self.page.frames):
            raise BrowserSafetyStop("chamber_already_open")
        button = self._one(self.page.get_by_role("button", name=HELP_BUTTON, exact=True))
        self.output.mkdir(parents=True, exist_ok=False)
        save_probe(before, self.output / "task-before")
        self.base_frames = [(f, f.url) for f in self.page.frames]
        self.deadline = time.monotonic() + self.max_seconds
        self.attempted = True
        intent = {
            "pressure": pressure,
            "pressure_unit": "atm",
            "temperature": temperature,
            "temperature_unit": "K",
            "action_source": source,
            "task_answer_write": False,
        }
        persist_json(self.output / "reserved.json", intent)
        self.page.on("dialog", self._dialog)
        try:
            self.emit("action_proposed", {"tool": "water_phase_chamber", **intent})
            button.click(timeout=3000)
            # A dialog is a container, not a click target. The course centers
            # its content inside a larger transparent dialog whose center may
            # not hit the container. Verify its actual Close/input targets
            # separately; retain identity first so failures can safely close it.
            self.page.get_by_role("dialog").first.wait_for(state="visible", timeout=3000)
            dialogs = [x for x in self.page.get_by_role("dialog").all() if x.is_visible()]
            if len(dialogs) != 1:
                raise BrowserSafetyStop("chamber_help_dialog_changed")
            self.dialog = dialogs[0]
            self.dialog_handle = self.dialog.element_handle(timeout=1000)
            self._one(self.dialog.get_by_role("button", name="Close", exact=True))
            while True:
                self._guard()
                matches = [f for f in self.page.frames if f.url == CHAMBER_URL]
                if (
                    len(matches) == 1
                    and matches[0].get_by_role("textbox", name="atm", exact=True).count() == 1
                ):
                    self.chamber = matches[0]
                    break
                self.page.wait_for_timeout(100)
            # The first textbox can exist before the helper finishes rendering
            # its controls. Poll only readiness; never retry a dispatched input.
            while True:
                self._guard()
                controls = {u: self.chamber.get_by_role("textbox", name=u, exact=True) for u in ("atm", "K")}
                if any(c.count() > 1 for c in controls.values()):
                    raise BrowserSafetyStop("chamber_ambiguous_inputs")
                if all(
                    c.count() == 1 and c.is_visible() and c.is_enabled() and chamber_control_exposed(c)
                    for c in controls.values()
                ):
                    self.inputs = {u: c.element_handle(timeout=1000) for u, c in controls.items()}
                    break
                self.page.wait_for_timeout(100)
            self._conditions()
            for unit, text, key in (("atm", pressure, "Enter"), ("K", temperature, "Tab")):
                self._conditions()
                handle = self.inputs[unit]
                handle.fill(text, timeout=3000)
                if handle.input_value() != text:
                    raise BrowserSafetyStop("chamber_exact_fill_failed")
                handle.press(key, timeout=3000)
                self._conditions()
                if condition_number(handle.input_value()) != expected[unit]:
                    raise BrowserSafetyStop("chamber_condition_readback_mismatch")
            committed_at = time.monotonic()
            stable, stable_since = None, None
            while True:
                phase, icons = self._phase()
                if any(condition_number(self.inputs[u].input_value()) != expected[u] for u in expected):
                    raise BrowserSafetyStop("chamber_condition_changed")
                if (
                    phase is not None
                    and phase == stable
                    and time.monotonic() - stable_since >= 0.5
                    and time.monotonic() - committed_at >= 2
                ):
                    break
                if phase != stable:
                    stable, stable_since = phase, time.monotonic()
                self.page.wait_for_timeout(100)
            result = {
                **intent,
                "phase": phase,
                "icons": icons,
                "conditions_verified": True,
                "visible_readback": {u: h.input_value() for u, h in self.inputs.items()},
                "source": "visible_chamber_indicator",
                "task_completed": False,
            }
            persist_json(self.output / "observed.json", result)
            evidence_screenshot(
                self.chamber.frame_element(), path=str(self.output / "chamber.png"), timeout=3000
            )
            self._close_owned_dialog()
            after = inspect_page(self.page, self.config)
            save_probe(after, self.output / "task-after")
            if (
                comparable_screen(before) != comparable_screen(after)
                or [(f, f.url) for f in self.page.frames] != self.base_frames
            ):
                raise BrowserSafetyStop("chamber_task_changed")
            persist_json(self.output / "confirmed.json", result)
            self.emit("action_result", {"tool": "water_phase_chamber", **result})
            return result
        except BaseException as exc:
            persist_json(
                self.output / "stopped.json",
                {
                    "reason": str(exc)
                    if isinstance(exc, BrowserSafetyStop)
                    else "chamber_operation_uncertain",
                    "retry_allowed": False,
                    "task_completed": False,
                },
            )
            if self.chamber is not None and not self.closed:
                try:
                    if (
                        self.chamber.is_detached()
                        or self.chamber.url != CHAMBER_URL
                        or not self.config.allows(self.page.url)
                        or any(c.is_visible() for c in self.chamber.locator('input[type="password"]').all())
                    ):
                        raise BrowserSafetyStop("chamber_diagnostics_unsafe")
                    # Diagnostic visible control geometry only; not a fallback
                    # answer and never used to bypass a failed readiness check.
                    details = []
                    for unit in ("atm", "K"):
                        for c in self.chamber.get_by_role("textbox", name=unit, exact=True).all():
                            if c.is_visible():
                                details.append(
                                    {
                                        "unit": unit,
                                        "box": c.bounding_box(),
                                        "enabled": c.is_enabled(),
                                        "exposed": chamber_control_exposed(c),
                                    }
                                )
                    persist_json(self.output / "unconfirmed-controls.json", {"controls": details})
                    evidence_screenshot(
                        self.chamber.frame_element(),
                        path=str(self.output / "unconfirmed-helper.png"),
                        timeout=2000,
                    )
                except Exception:  # noqa: BLE001 - diagnostics must not replace the original failure
                    self.emit("error", {"message": "chamber_failure_evidence_unavailable"})
            raise
        finally:
            # Only our unchanged dialog may be dismissed. Never close an
            # unexpected replacement or navigate in an attempt to recover.
            if self.dialog_handle is not None and not self.close_attempted:
                try:
                    self._close_owned_dialog()
                except Exception:  # noqa: BLE001 - preserve original failure, redact browser details
                    persist_json(
                        self.output / "cleanup-stopped.json",
                        {
                            "reason": "owned_dialog_not_confirmed_closed",
                            "retry_allowed": False,
                        },
                    )
            self.closed = True
            self.page.remove_listener("dialog", self._dialog)

    def _close_owned_dialog(self):
        if self.close_attempted:
            raise BrowserSafetyStop("chamber_close_already_attempted")
        if not self.config.allows(self.page.url) or not self.dialog_handle.evaluate("e=>e.isConnected"):
            raise BrowserSafetyStop("chamber_close_identity_changed")
        matches = [x for x in self.page.get_by_role("dialog").all() if x.is_visible()]
        if len(matches) != 1 or not self.dialog_handle.evaluate(
            "(a,b)=>a===b", matches[0].element_handle(timeout=1000)
        ):
            raise BrowserSafetyStop("chamber_close_identity_changed")
        close = self._one(matches[0].get_by_role("button", name="Close", exact=True))
        self.close_attempted = True
        persist_json(self.output / "close-reserved.json", {"retry_allowed": False})
        close.click(timeout=3000)
        self.dialog.wait_for(state="hidden", timeout=3000)
        self.closed = True
