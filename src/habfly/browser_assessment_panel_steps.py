"""Bounded panel navigation only: never assess, spend, save, score or submit.

The exact buttons and panel contents are grounded in saved public captures.
Selection is proved by exposed panel text, not CSS state or labels in rows.
"""

import hashlib
import math
import re
import time
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import assessment_config, persist_json, rows_hash
from .browser_collected import row_icon_exposed
from .browser_no_planet_workflow import _Evidence
from .browser_numeric import screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_project_inventory import VISIBLE_TEXT, _capture_inventory, _outer_exposed, _visible_rows
from .browser_project_navigation import _outside, navigation_status_projection, project_view
from .browser_stellar import SIMULATION_URL, _atoms
from .contracts import RuntimeEvent
from .project_assessment import AssessmentError, parse_assessment_text
from .project_inventory_source import load_inventory_source

MODE = "cooperative_assessment_panel_navigation"
LABELS = {"data_quality": "DATA QUALITY", "scavenger_hunt": "SCAVENGER HUNT"}
ALLOW = {"ASSESSMENT", *LABELS.values()}
ZERO = (
    "assessment_clicks",
    "answer_writes",
    "save_clicks",
    "score_transfer_clicks",
    "submission_clicks",
    "simulation_dollars_spent",
    "journal_writes",
)


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("assessment_panel_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _header(text, total, buttons):
    """Only text before Observations can establish an active assessment panel."""
    sections = re.split(r"\bOBSERVATIONS\b", text, flags=re.IGNORECASE)
    _require(len(sections) == 2, "ambiguous_observation_boundary")
    header = " ".join(sections[0].upper().split())
    base_pattern = (
        r"\bFUNDING\s*\$([0-9]+)\s+DATA QUALITY\s+([0-9]+(?:\.[0-9]+)?)%\s+"
        r"SCAVENGER HUNT\s+([0-9]+)/8\b"
    )
    found = re.findall(base_pattern, header)
    _require(len(found) == 1, "missing_current_funding_or_scores")
    funding, quality, hunt = found[0]
    base = {
        "funding": int(funding),
        "data_quality_percent": float(quality),
        "scavenger_found": int(hunt),
        "collected": total,
    }
    _require(
        0 <= base["data_quality_percent"] <= 100 and 0 <= base["scavenger_found"] <= 8,
        "invalid_current_scores",
    )
    try:
        panel = parse_assessment_text(header + f" TOTAL COLLECTED {total}")
    except AssessmentError as exc:
        # No active panel is accepted only for the narrow exposed closed header.
        remainder = re.sub(r"^1 2 3\s+", "", re.sub(base_pattern, "", header).strip())
        _require(
            str(exc) == "unknown_assessment_mode"
            and remainder in {"", "EDIT DATA ASSESSMENT"}
            and "ASSESSMENT" in buttons
            and not ({"ASSESS", *LABELS.values()} & set(buttons)),
            "unsupported_or_incomplete_panel",
        )
        return base, None
    _require(
        panel.mode in LABELS
        and panel.acknowledgement is None
        and panel.cost == 100
        and {"ASSESS", "ASSESSMENT", *LABELS.values()} <= set(buttons),
        "unsupported_or_acknowledged_panel",
    )
    return base, panel


def _public_panel(report, *, inventory_kind="legacy_single_page"):
    """Offline public-AX mapping, independently checked against exposed text live."""
    frames = [f for f in report["frames"] if f["url"] == SIMULATION_URL]
    _require(len(frames) == 1, "ambiguous_simulation")
    frame = frames[0]
    atoms = _atoms(frame["accessibility"])
    texts = [
        value
        for key, value in atoms
        if key in {"text", "subscript", "superscript"} and isinstance(value, str)
    ]
    _require(inventory_kind in {"legacy_single_page", "live_paginated"}, "unsupported_inventory_kind")
    if inventory_kind == "live_paginated":
        from .browser_project_paginated_inventory_steps import parse_inventory_page

        counts, rows = parse_inventory_page(report)
    else:
        counts, rows = _capture_inventory(report)
    buttons = []
    for control in frame["controls"]:
        if control["role"] != "button":
            continue
        match = re.fullmatch(r'- button "([^"\n]+)"(?: \[disabled\])?', control["accessibility"])
        if match:
            buttons.append(match[1].upper())
    _require(all(buttons.count(label) <= 1 for label in {*ALLOW, "ASSESS"}), "ambiguous_panel_buttons")
    baseline, panel = _header(" ".join(texts), counts["total"], buttons)
    return baseline, panel, counts, rows, buttons


class AssessmentPanelSteps:
    """Offline constructor; <=3 advances and <=2 exact menu/tab clicks.

    A prior inventory fixes collection/rows/context. Funding and current scores
    are pinned at the first current read, allowing a separate legitimate charged
    assessment between two navigator instances. This component never charges.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        inventory_dir,
        inventory_sha256,
        kind,
        max_seconds=60,
        emit=lambda _: None,
        cancelled=lambda: False,
    ):
        _require(kind in LABELS, "unsupported_kind")
        _require(
            type(max_seconds) in {int, float} and math.isfinite(max_seconds) and 0 < max_seconds <= 180,
            "invalid_deadline",
        )
        _require(callable(emit) and callable(cancelled), "invalid_callback")
        self.book = _Evidence(run_history)
        source = Path(inventory_dir)
        self.inventory_dir = self.book.path(source if source.is_absolute() else self.book.history / source)
        _require(
            isinstance(inventory_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", inventory_sha256),
            "invalid_inventory_hash",
        )
        _require(
            _sha(self.book.read(self.inventory_dir / "confirmed.json")) == inventory_sha256,
            "inventory_hash_changed",
        )
        self.inventory_source = load_inventory_source(
            self.book, self.inventory_dir, expected_sha256=inventory_sha256
        )
        self.inventory = self.inventory_source.receipt
        self.source = self.inventory_source.anchor
        if self.inventory_source.kind == "live_paginated":
            from .browser_project_paginated_inventory_steps import parse_inventory_page

            self.anchor_counts, self.rows = parse_inventory_page(self.source)
            _require(
                self.anchor_counts["start"] == 1
                and self.anchor_counts["total"] == self.inventory["total_collected"]
                and self.rows == self.inventory_source.rows[: len(self.rows)],
                "invalid_paginated_anchor",
            )
        else:
            self.anchor_counts, self.rows = _capture_inventory(self.source)
        self.row_hash = rows_hash(
            next(f["text"] for f in self.source["frames"] if f["url"] == SIMULATION_URL)
        )
        self.output = self.book.path(output)
        _require(
            self.output != self.book.history
            and not self.output.is_relative_to(self.inventory_dir)
            and not self.inventory_dir.is_relative_to(self.output),
            "output_overlaps_source",
        )
        self.page, self.config = page, assessment_config(config)
        self._config = self.config.model_dump(mode="json")
        self.kind, self.max_seconds, self._callback, self._cancelled = kind, max_seconds, emit, cancelled
        self._started = time.monotonic()
        self._sequence = self.advances = self.clicks = self.dispatch_attempts = 0
        self._busy = self._finalizing = self._forward_failed = self._listeners = False
        self._frames, self._frame, self._baseline, self._previous = None, None, None, None
        self._dialogs, self._popups = [], []
        self.finished, self.report, self.failure, self.panel = False, None, None, None
        self.phase = "inspect_panel"
        self.output.mkdir(parents=True, exist_ok=False)
        self._stream = (self.output / "events.jsonl").open("x", encoding="utf-8")
        self.scope = {
            "schema_version": 1,
            "mode": MODE,
            "kind": kind,
            "inventory_path": str((self.inventory_dir / "confirmed.json").relative_to(self.book.history)),
            "inventory_sha256": inventory_sha256,
            "project_rows_sha256": self.row_hash,
            "collected": self.inventory["total_collected"],
            "max_advances": 3,
            "max_navigation_clicks": 2,
            "max_seconds": max_seconds,
            "pause_counts_toward_deadline": True,
            "automatic_retry": False,
            "browser_owned_by_caller": True,
            "scientific_verified": False,
            "assessment_confirmed": False,
            "task_completed": False,
            "project_completed": False,
            **dict.fromkeys(ZERO, 0),
            **self.inventory_source.binding_fields(),
        }
        persist_json(self.output / "scope.json", self.scope)
        self.book.read(self.output / "scope.json")
        try:
            self._emit("hello", {"protocol_version": 1, **self.scope})
            self._check()
        except Exception as exc:  # noqa: BLE001 - redact callback/source errors
            self._stop(exc)

    def _emit(self, event, payload):
        value = RuntimeEvent(event=event, payload=payload, sequence=self._sequence, run_id=self.output.name)
        self._sequence += 1
        self._stream.write(value.model_dump_json() + "\n")
        self._stream.flush()
        if not self._forward_failed:
            try:
                self._callback(value.model_dump(mode="json"))
            except Exception:  # noqa: BLE001 - callback text is never a protocol diagnostic
                self._forward_failed = True
                raise BrowserSafetyStop("assessment_panel_event_forwarding_failed") from None

    def _save_capture(self, report, directory):
        save_probe(report, directory)
        captured = self.book.capture(directory)
        _require(screen_identity(captured) == screen_identity(report), "capture_changed_during_recording")

    def _record(self, path, value):
        persist_json(path, value)
        _require(self.book.json(path) == value, "record_changed_during_recording")

    def _check(self):
        _require(not self.finished and not self._cancelled(), "cancelled_or_stopped")
        _require(time.monotonic() - self._started < self.max_seconds, "time_limit")
        _require(self.config.model_dump(mode="json") == self._config, "boundary_changed")
        self.book.unchanged()

    def _dialog(self, dialog):
        self._dialogs.append(True)  # Do not accept/dismiss an unexpected dialog as another UI action.

    def _popup(self, page):
        self._popups.append(True)

    def _context(self):
        self._check()
        if not self._listeners:
            self.page.on("dialog", self._dialog)
            self.page.context.on("page", self._popup)
            self._listeners = True
            self._frames = self.page.frames.copy()
        _require(
            not self._dialogs
            and not self._popups
            and len(self.page.context.pages) == 1
            and self.page.frames == self._frames,
            "context_changed",
        )

    def _read(self):
        from .browser_project_paginated_inventory_steps import read_settled_inventory

        report = read_settled_inventory(
            self.page,
            self.config,
            check=self._context,
            deadline=self._started + self.max_seconds,
            clock=time.monotonic,
            inspector=inspect_page,
        )
        _require(
            report["ignored_frame_urls"] == []
            and isinstance(report.get("outer_url"), str)
            and report["outer_url"] == self.source.get("outer_url")
            and _outside(report) == _outside(self.source),
            "outside_context_changed",
        )
        frames = [f for f in self.page.frames if f.url == SIMULATION_URL]
        _require(
            len(frames) == 1
            and frames[0].parent_frame == self.page.main_frame
            and (self._frame is None or frames[0] == self._frame),
            "frame_changed",
        )
        self._frame = frames[0]
        baseline, panel, counts, rows, buttons = _public_panel(
            report, inventory_kind=self.inventory_source.kind
        )
        _require(rows == self.rows and counts == self.anchor_counts, "collection_changed")
        visible = next(f for f in report["frames"] if f["url"] == SIMULATION_URL)
        if rows_hash(visible["text"]) != self.row_hash:
            self._save_capture(report, self.output / "changed-row-diagnostic")
            raise BrowserSafetyStop("assessment_panel_rows_hash_changed")
        if self.inventory_source.kind == "live_paginated":
            from .browser_project_paginated_inventory_steps import visible_inventory_page

            visible_inventory_page(self._frame, counts, rows)
        else:
            _visible_rows(self._frame, counts, rows)
        nodes = self._frame.locator("body").evaluate(VISIBLE_TEXT)
        _require(
            _outer_exposed(self._frame, [box for n in nodes for box in n["boxes"]]), "panel_outer_occluded"
        )
        native_base, native_panel = _header(" ".join(n["text"] for n in nodes), counts["total"], buttons)
        _require(
            native_base == baseline and self._panel_key(native_panel) == self._panel_key(panel),
            "panel_text_not_fully_exposed",
        )
        if self._baseline is None:
            self._baseline = baseline
        _require(baseline == self._baseline, "funding_or_scores_changed")
        self._check()
        return report, panel

    @staticmethod
    def _panel_key(panel):
        return panel.model_dump(exclude={"visible_text_sha256"}) if panel else None

    def _button(self, label):
        _require(label in ALLOW, "forbidden_button")
        controls = [
            e for e in self._frame.get_by_role("button", name=label, exact=True).all() if e.is_visible()
        ]
        _require(
            len(controls) == 1
            and controls[0].is_enabled()
            and controls[0].evaluate("e=>e.tagName==='BUTTON'")
            and row_icon_exposed(controls[0]),
            "button_unavailable_or_occluded",
        )
        box = controls[0].evaluate(
            "e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}}"
        )
        _require(
            box
            and _outer_exposed(
                self._frame,
                [box],
            ),
            "button_outer_occluded",
        )
        return controls[0].element_handle(timeout=2000)

    def state(self):
        return {
            **self.scope,
            "phase": self.phase,
            "finished": self.finished,
            "panel_verified": self.phase == "panel_verified",
            "failure_reason": self.failure,
            "advances": self.advances,
            "navigation_clicks": self.clicks,
            "navigation_dispatch_attempts": self.dispatch_attempts,
            "assessment": self.panel.model_dump(mode="json") if self.panel else None,
        }

    def advance(self):
        if self.finished:
            return self.state()
        if self._busy:
            self._stop(BrowserSafetyStop("assessment_panel_reentrant_advance"))
            return self.state()
        self._busy = True
        try:
            self._check()
            _require(self.advances < 3, "advance_limit")
            self.advances += 1
            before, panel = self._read()
            if self._previous is not None:
                _require(self._screen(before) == self._screen(self._previous), "view_changed_between_steps")
            if panel and panel.mode == self.kind:
                after, again = self._read()
                _require(
                    self._screen(before) == self._screen(after)
                    and self._panel_key(panel) == self._panel_key(again),
                    "panel_changed_during_readback",
                )
                self._save_capture(after, self.output / "verified")
                self.panel = again
                self._finish()
            elif self.phase == "inspect_panel":
                self._save_capture(before, self.output / "initial")
                self._previous = before
                self.phase = "open_assessment" if panel is None else "select_panel"
                self._emit("state", self.state())
                self._check()
            else:
                label = "ASSESSMENT" if panel is None else LABELS[self.kind]
                _require(
                    self.clicks < 2 and self.dispatch_attempts == self.clicks,
                    "navigation_budget_or_uncertainty",
                )
                handle = self._button(label)
                prefix = f"navigation-{self.dispatch_attempts:02d}"
                self._save_capture(before, self.output / (prefix + "-before"))
                intent = {
                    "kind": "CLICK",
                    "visible_label": label,
                    "requested_panel": self.kind,
                    "max_clicks": 1,
                    "project_rows_sha256": self.row_hash,
                    "assessment_clicks": 0,
                }
                self._record(self.output / (prefix + "-reserved.json"), intent)
                self._emit("action_proposed", intent)
                current, current_panel = self._read()
                rebound = self._button(label)
                _require(
                    self._screen(current) == self._screen(before)
                    and self._panel_key(current_panel) == self._panel_key(panel)
                    and handle.evaluate("(a,b)=>a.isConnected&&a===b", rebound),
                    "changed_before_dispatch",
                )
                self._check()
                self.dispatch_attempts += 1
                self._record(self.output / (prefix + "-dispatch.json"), intent)
                handle.click(timeout=3000)
                after, selected = self._read()
                _require(
                    selected is not None and (label == "ASSESSMENT" or selected.mode == self.kind),
                    "requested_panel_not_reached",
                )
                self._save_capture(after, self.output / (prefix + "-after"))
                self.clicks += 1
                receipt = {
                    **intent,
                    "panel_mode": selected.mode,
                    "panel_selected_verified": True,
                    "funding_unchanged": True,
                    "collection_unchanged": True,
                    "assessment_confirmed": False,
                }
                self._record(self.output / (prefix + "-confirmed.json"), receipt)
                self._emit("action_result", receipt)
                self._check()
                self._previous = after
                if selected.mode == self.kind:
                    current, again = self._read()
                    _require(
                        self._screen(current) == self._screen(after)
                        and self._panel_key(again) == self._panel_key(selected),
                        "panel_changed_during_readback",
                    )
                    self._save_capture(current, self.output / "verified")
                    self.panel = again
                    self._finish()
                else:
                    self.phase = "select_panel"
                    self._emit("state", self.state())
                    self._check()
        except BaseException as exc:
            self._stop(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            self._busy = False
        return self.state()

    @staticmethod
    def _screen(report):
        return screen_identity(navigation_status_projection(report, project_view(report)))

    def _finish(self):
        self._check()
        self.phase = "panel_verified"
        self._emit("episode_summary", {**self.state(), "finished": True})
        self._check()
        capture = self.book.capture(self.output / "verified")
        self.book.read(self.output / "events.jsonl")
        self.book.unchanged()
        self.finished = True
        self.report = {
            **self.state(),
            "source_sha256": dict(self.book.hashes),
            "current_screen_sha256": screen_identity(capture),
        }
        persist_json(self.output / "confirmed.json", self.report)
        self._close()

    def _stop(self, exc):
        if self.finished or self._finalizing:
            return
        self._finalizing = True
        code = str(exc)
        self.failure = (
            code
            if isinstance(exc, (BrowserSafetyStop, AssessmentError))
            and re.fullmatch(r"[a-z][a-z0-9_]{0,140}", code)
            else "assessment_panel_operation_failed"
        )
        self.finished, self.phase = True, "stopped"
        try:
            self.report = self.state()
            persist_json(self.output / "stopped.json", self.report)
            if not self._forward_failed:
                self._emit("error", {"message": self.failure, "type": "AssessmentPanelStop"})
                self._emit("state", self.state())
        finally:
            self._close()
            self._finalizing = False

    def _close(self):
        if self._listeners:
            self.page.remove_listener("dialog", self._dialog)
            self.page.context.remove_listener("page", self._popup)
            self._listeners = False
        self._stream.close()

    def abort(self):
        self._stop(BrowserSafetyStop("assessment_panel_operator_aborted"))
        return self.state()

    def close(self):
        return self.abort()
