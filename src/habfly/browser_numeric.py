"""Human-stepped numeric transport diagnostic, NOT a learned browser agent.

Only native numeric text fields in the observed v1.5.2 stellar screen can be
filled. No generic selector/action API, click, Enter, Save, score or submission.
All handles are resolved from current visible evidence and are never persisted.
"""

import copy
import hashlib
import json
import re
import time
import uuid
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_probe import BrowserProbeConfig, _visible_frame, inspect_page, save_probe
from .browser_stellar import (
    FIELDS,
    NUMBER,
    SIMULATION_URL,
    StellarMappingError,
    _atoms,
    _control_atom,
    map_stellar_capture,
    plan_numeric_copy,
)
from .contracts import Observation, RuntimeEvent, StepResult
from .knowledge import LocalCalculator, load_knowledge_pack

MODE = "human_stepped_numeric_diagnostic"
DISPLAY_POLICY = {
    "id": "controlled_commit_decimal_rounding_v1",
    "minimum_significant_digits": 4,
    "scope": "after_exact_copy_and_controlled_tab_only",
    "course_grading_tolerance": False,
    "evidence_run": "browser-numeric-002",
    "evidence_events_sha256": "e16af79da894cdd354055ad72d3f7345cfd20c6bb26103788d0283eb1e0a532e",
}
SCREEN_COMPARISON_POLICY = {
    "id": "stellar_footer_autosave_notice_v1",
    "scope": "exact_simulation_footer_in_both_text_and_accessibility",
    "ignored_notice": "Data saved",
    "verifies_persistence": False,
}


def _without_autosave_notice(frame):
    """Normalize only the paired footer notice observed in the live diagnostic.

    Not a general whitespace/status filter: a changed measurement, answer,
    control, error, other frame, or a notice in another position still differs.
    Called on a deep copy; raw observations and evidence remain untouched.
    """
    if frame.get("url") != SIMULATION_URL:
        return
    text, accessibility = frame.get("text", ""), frame.get("accessibility", "")
    if not isinstance(text, str) or not isinstance(accessibility, str):
        return
    text_suffix = "\n1 Rs\nData saved\nSave"
    ax_suffix = (
        "\n- text: mass, radius and lifetime are only relevant for main sequence stars "
        'main sequence red giant supergiant white dwarf 1 Rs Data saved\n- button "Save"'
    )
    if (
        text.count("Data saved") == accessibility.count("Data saved") == 1
        and text.endswith(text_suffix)
        and accessibility.endswith(ax_suffix)
    ):
        frame["text"] = text.removesuffix(text_suffix) + text_suffix.replace("Data saved\n", "")
        frame["accessibility"] = accessibility.removesuffix(ax_suffix) + ax_suffix.replace(" Data saved", "")


def committed_display(exact: str, displayed: str):
    """Accept exact spelling/value or nearest-decimal rounding retaining >=4 digits.

    This is an explicit transport policy, not a claim about the site's formatter
    or grading. No relative-error epsilon, truncation, unit change or coarse
    rounding is accepted. Both conventional tie rules are permitted at exact ties.
    """
    result = {"exact_copied": exact, "display_value": displayed, "policy": DISPLAY_POLICY["id"]}
    try:
        if any(
            not isinstance(s, str) or len(s) > 64 or not re.fullmatch(NUMBER, s) for s in (exact, displayed)
        ):
            raise ValueError
        original, visible = Decimal(exact), Decimal(displayed)
        if any(not n.is_finite() or n <= 0 or not -324 <= n.adjusted() <= 308 for n in (original, visible)):
            raise ValueError
        if exact == displayed:
            return {**result, "format": "exact"}
        if visible == original:
            return {**result, "format": "equivalent_numeric_spelling"}
        with localcontext() as ctx:
            ctx.prec = 400
            for precision in range(min(len(original.as_tuple().digits) - 1, 17), 3, -1):
                quantum = Decimal(1).scaleb(original.adjusted() - precision + 1)
                for rounding in (ROUND_HALF_EVEN, ROUND_HALF_UP):
                    if original.quantize(quantum, rounding=rounding) == visible:
                        return {
                            **result,
                            "format": "decimal_rounding",
                            "matched_rounding_significant_digits": precision,
                        }
    except (ValueError, InvalidOperation):
        pass
    raise BrowserSafetyStop("unapproved_committed_display")


def numeric_value_projection(report, mapping, destination):
    """Exclude only the one permitted native/AX value, preserving label and role.

    Except for the paired footer autosave notice, every other visible byte/atom
    stays comparable, including protected widgets and the other answer fields.
    """
    projected = comparable_screen(report)
    target = mapping["observation"]["values"]["browser_field_map"][destination]["capture_target_id"]
    frame = next(f for f in projected["frames"] if f["id"] == mapping["simulation_frame"])
    numeric = [c for c in frame["controls"] if c["role"] in {"textbox", "spinbutton"}]
    ordinal = next(i for i, control in enumerate(numeric) if control["id"] == target)
    control = numeric[ordinal]
    control["value"] = None
    key, _ = _control_atom(control["accessibility"])
    control["accessibility"] = (key, None)
    atoms, position = _atoms(frame["accessibility"]), 0
    for index, (key, value) in enumerate(atoms):
        if key.split(" ", 1)[0] in {"textbox", "spinbutton"}:
            if position == ordinal:
                atoms[index] = (key, None)
            position += 1
    frame["accessibility"] = atoms
    return projected


# Read rendered text adjacency independently of AX inventory order. No app state,
# data-* attributes, hidden text, scripts, raw HTML or generated selector lookup.
# Stop on ambiguous/nested unsupported markup rather than guess a field index.
VISIBLE_PREFIX = """target => {
    function visible(element) {
        for (let p = element; p; p = p.parentElement) {
            const s = getComputedStyle(p);
            if (s.display === 'none' || s.visibility !== 'visible' ||
                s.opacity === '0' || p.getAttribute('aria-hidden') === 'true' || p.hidden) return false;
        }
        return true;
    }
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let text = '', node;
    while ((node = walker.nextNode())) {
        if (node === target) return text;
        if (node.nodeType === Node.ELEMENT_NODE) {
            if (visible(node) && node.matches('input,textarea,select,button,a,[role="textbox"],[role="spinbutton"],[role="combobox"],[role="button"]')) text = '';
        } else if (visible(node.parentElement) && !node.parentElement.closest('script,style,template,select,button,a')) {
            const range = document.createRange(); range.selectNodeContents(node);
            if (range.getClientRects().length) text += node.textContent;
        }
    }
    return null;
}"""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def comparable_screen(report):
    value = copy.deepcopy(report)
    value.pop("captured_at", None)
    value.pop("setup_mode", None)
    for frame in value.get("frames", []):
        _without_autosave_notice(frame)
    return value


def screen_identity(report):
    return digest(comparable_screen(report))


def screen_changes(before, after, *, limit=40):
    """Bounded, deterministic JSON-pointer diff of already-approved visible data.

    Ignore the same metadata and paired autosave notice as screen_identity.
    Long changed strings show context around their first differing character;
    the separate evidence file retains both complete snapshots without clipping.
    """
    if not 1 <= limit <= 100:
        raise ValueError("Invalid diagnostic change limit")
    changes, total = [], 0

    def display_pair(left, right):
        offset = 0
        if isinstance(left, str) and isinstance(right, str):
            for a, b in zip(left, right):
                if a != b:
                    break
                offset += 1
        result = {}
        for name, value in (("before", left), ("after", right)):
            if isinstance(value, str) and len(value) > 240:
                start = max(0, offset - 60) if isinstance(left, str) and isinstance(right, str) else 0
                result[name] = {
                    "excerpt": value[start : start + 240],
                    "start": start,
                    "length": len(value),
                    "sha256": hashlib.sha256(value.encode()).hexdigest(),
                    "truncated": True,
                }
            elif isinstance(value, (dict, list)):
                encoded = json.dumps(value, sort_keys=True)
                result[name] = {
                    "excerpt": encoded[:240],
                    "length": len(encoded),
                    "truncated": len(encoded) > 240,
                }
            else:
                result[name] = value
        return result

    def visit(left, right, path="", left_present=True, right_present=True):
        nonlocal total
        if left_present and right_present and type(left) is type(right):
            if isinstance(left, dict):
                for key in sorted(left.keys() | right.keys()):
                    pointer = key.replace("~", "~0").replace("/", "~1")
                    visit(left.get(key), right.get(key), f"{path}/{pointer}", key in left, key in right)
                return
            if isinstance(left, list):
                for index in range(max(len(left), len(right))):
                    visit(
                        left[index] if index < len(left) else None,
                        right[index] if index < len(right) else None,
                        f"{path}/{index}",
                        index < len(left),
                        index < len(right),
                    )
                return
            if left == right:
                return
        total += 1
        if len(changes) < limit:
            changes.append(
                {
                    "path": path or "/",
                    "before_present": left_present,
                    "after_present": right_present,
                    **display_pair(left, right),
                }
            )

    visit(comparable_screen(before), comparable_screen(after))
    return {"change_count": total, "changes": changes, "truncated": total > len(changes)}


class NumericJournal:
    """Flush each event before side effects; retain failures without browser secrets."""

    def __init__(self, output: Path, *, learned_policy=False, provenance=None):
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        self.stream = (output / "events.jsonl").open("x")
        self.sequence = 0
        self.run_id = uuid.uuid4().hex
        self.screen_diagnostics = []
        self.numeric_readbacks = {}
        self.learned_policy = learned_policy
        self.provenance = provenance or {}
        self.emit("hello", {"mode": MODE, "learned_policy": learned_policy, "allow_submission": False})
        self.emit(
            "state",
            {
                "mode": MODE,
                "status": "paused",
                "environment": "browser",
                "policy": (
                    "checkpoint_autonomous"
                    if self.provenance.get("browser_execution") == "autonomous"
                    else "checkpoint_human_confirmed"
                )
                if learned_policy
                else "human_selected",
                "task": "numeric_transport",
                "checkpoint": None,
            },
        )

    def emit(self, event, payload):
        if self.learned_policy:
            payload = {**payload, "learned_policy": True, "mode": "learned_numeric_transport"}
            if payload.get("action_source") == "human_selected":
                payload["action_source"] = "checkpoint_selected_human_confirmed"
        message = RuntimeEvent(event=event, sequence=self.sequence, run_id=self.run_id, payload=payload)
        self.stream.write(message.model_dump_json() + "\n")
        self.stream.flush()
        self.sequence += 1

    def record_screen_change(self, reason, before, after):
        """Only receives inspect_page reports; does not re-read a possibly unsafe page."""
        differences = screen_changes(before, after)
        evidence = {
            "schema_version": 1,
            "mode": "numeric_screen_change_evidence",
            "reason": reason,
            "screen_comparison_policy": SCREEN_COMPARISON_POLICY,
            "before_screen_sha256": screen_identity(before),
            "after_screen_sha256": screen_identity(after),
            **differences,
            "before": before,
            "after": after,
        }
        filename = f"screen-change-{len(self.screen_diagnostics) + 1:03d}.json"
        encoded = (json.dumps(evidence, indent=2, ensure_ascii=False) + "\n").encode()
        with (self.output / filename).open("xb") as stream:
            stream.write(encoded)
        reference = {
            "path": filename,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "reason": reason,
            "change_count": differences["change_count"],
            "truncated": differences["truncated"],
        }
        self.screen_diagnostics.append(reference)
        self.emit("state", {"mode": MODE, "screen_change": {**reference, **differences}})

    def finish(self, *, outcome, attempts, verified, pack_hash):
        report = {
            "schema_version": 1,
            "mode": "learned_numeric_transport" if self.learned_policy else MODE,
            "outcome": outcome,
            "write_attempts": attempts,
            "verified_fields": verified,
            "numeric_transport_passed": outcome == "numeric_transport_verified"
            and set(verified) == {"distance", "luminosity", "temperature"},
            "task_completed": False,
            "browser_acceptance_passed": False,
            "learned_policy": self.learned_policy,
            "provenance": self.provenance,
            "allow_submission": False,
            "knowledge_pack_hash": pack_hash,
            "calculation_scope": "unclassified_common_only",
            "setup_mode": "independent_test_session",
            "screen_diagnostics": self.screen_diagnostics,
            "numeric_readbacks": self.numeric_readbacks,
            "display_policy": DISPLAY_POLICY,
            "screen_comparison_policy": SCREEN_COMPARISON_POLICY,
        }
        self.emit("episode_summary", report)
        self.stream.close()
        report["events_sha256"] = hashlib.sha256((self.output / "events.jsonl").read_bytes()).hexdigest()
        (self.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


class NumericSession:
    ALLOW_COLOR_SELECTION = False
    MAX_WRITES = 3
    MAX_CALCULATIONS = 6
    MAX_SECONDS = 900

    def __init__(self, page, config: BrowserProbeConfig, journal: NumericJournal):
        self.page, self.config, self.journal = page, config, journal
        self.calculator = LocalCalculator(load_knowledge_pack())
        self.results = {}
        self.verified = []
        self.attempts = self.calculations = self.revision = 0
        self.started = time.monotonic()
        self.stopped = False
        self.frame = None
        self.unexpected_dialog = False
        page.on("dialog", self._dialog)

    def _dialog(self, dialog):
        # Never accept browser confirmations or record their potentially private text.
        self.unexpected_dialog = True
        dialog.dismiss()

    def _guard(self):
        if self.stopped:
            raise BrowserSafetyStop("diagnostic_stopped")
        if time.monotonic() - self.started > self.MAX_SECONDS:
            raise BrowserSafetyStop("diagnostic_time_limit")
        # Human terminal pauses block the driver's navigation event processing.
        self.page.wait_for_timeout(0)
        if self.unexpected_dialog:
            raise BrowserSafetyStop("unexpected_browser_dialog")
        if len(self.page.context.pages) != 1:
            raise BrowserSafetyStop("unexpected_popup")

    def _require_same_screen(self, before, after, reason):
        if screen_identity(before) != screen_identity(after):
            self.journal.record_screen_change(reason, before, after)
            raise BrowserSafetyStop(reason)

    def _read(self):
        self._guard()
        report = inspect_page(self.page, self.config)
        if report["ignored_frame_urls"]:
            raise BrowserSafetyStop("unknown_visible_frame")
        mapping = map_stellar_capture(
            report, capture_sha256=screen_identity(report), allow_color_selection=self.ALLOW_COLOR_SELECTION
        )
        frames = [
            f for f in self.page.frames if f.url == SIMULATION_URL and _visible_frame(f, self.page.main_frame)
        ]
        if len(frames) != 1 or (self.frame is not None and frames[0] != self.frame):
            raise BrowserSafetyStop("simulation_frame_changed")
        frame = frames[0]
        handles = {}
        fields = mapping["observation"]["values"]["browser_field_map"]
        for role in ("textbox", "spinbutton"):
            for locator in frame.get_by_role(role).all():
                if not locator.is_visible() or locator.get_attribute("aria-hidden") == "true":
                    continue
                handle = locator.element_handle(timeout=2000)
                if handle is None:
                    raise BrowserSafetyStop("missing_native_field")
                native = handle.evaluate(
                    "e => e.tagName === 'INPUT' && ['text','number',''].includes(e.type) && e.isConnected"
                )
                if not native or not handle.is_editable():
                    raise BrowserSafetyStop("unsupported_or_readonly_field")
                prefix = handle.evaluate(VISIBLE_PREFIX)
                if prefix is None:
                    raise BrowserSafetyStop("detached_numeric_field")
                label = "".join(prefix.casefold().split()).split("yourreconstruction")[-1]
                if label not in FIELDS:
                    raise BrowserSafetyStop("unverified_visible_field_label")
                name, unit = FIELDS[label]
                if name in handles or name not in fields or unit != fields[name]["unit"]:
                    raise BrowserSafetyStop("ambiguous_live_field")
                if handle.input_value() != fields[name]["current_value"]:
                    raise BrowserSafetyStop("inconsistent_live_value")
                handles[name] = handle
        if set(handles) != set(fields):
            raise BrowserSafetyStop("incomplete_live_field_map")
        # Re-read to reject DOM/value changes during the multi-call binding pass.
        self._require_same_screen(
            report, inspect_page(self.page, self.config), "screen_changed_during_binding"
        )
        self._guard()
        return report, mapping, handles, frame

    def start(self):
        self._guard()
        initial = inspect_page(self.page, self.config)
        initial["setup_mode"] = "independent_test_session"
        # Preserve allowed visible evidence even when a stricter live binding fails.
        save_probe(initial, self.journal.output / "capture")
        self.report, self.mapping, self.handles, self.frame = self._read()
        self._require_same_screen(initial, self.report, "screen_changed_during_start")
        self._emit_observation()

    def _current(self):
        report, mapping, handles, frame = self._read()
        self._require_same_screen(self.report, report, "stale_numeric_observation")
        for name, old in self.handles.items():
            if not old.evaluate("(old, current) => old.isConnected && old === current", handles[name]):
                raise BrowserSafetyStop("numeric_control_replaced")
        return report, mapping, handles, frame

    def observation(self):
        observation = copy.deepcopy(self.mapping["observation"])
        observation["revision"] = self.revision
        observation["progress"]["task"] = MODE
        observation["values"]["diagnostic_results"] = self.results
        observation["values"]["verified_numeric_fields"] = list(self.verified)
        observation["values"]["numeric_readbacks"] = copy.deepcopy(self.journal.numeric_readbacks)
        for control, name in zip(observation["controls"], observation["values"]["browser_field_map"]):
            control["id"] = f"live:{self.revision}:{name}"
            control["actions"] = [] if name in self.verified else ["TYPE"]
        return Observation.model_validate(observation)

    def _emit_observation(self):
        self.journal.emit("observation", self.observation().model_dump(mode="json"))

    def sources(self):
        measurements = self.mapping["observation"]["values"]["measurements"]
        return {**measurements, **{key: item["result"] for key, item in self.results.items()}}

    def calculate(self, operation, selections):
        """Only operator-selected bindings; no expert repair or answer generation."""
        self._current()
        if self.calculations >= self.MAX_CALCULATIONS:
            raise BrowserSafetyStop("calculation_limit")
        self.calculations += 1
        sources = self.sources()
        if not isinstance(selections, dict) or any(key not in sources for key in selections.values()):
            raise StellarMappingError("unknown_binding_source")
        bindings = {
            name: {"value": sources[key]["value"], "unit": sources[key]["unit"]}
            for name, key in selections.items()
        }
        result = self.calculator.execute_unclassified_common(operation, bindings)
        ident = f"result:{self.calculations}"
        self.journal.emit(
            "state",
            {
                "mode": MODE,
                "operation": operation,
                "bindings": selections,
                "result_id": ident,
                "result": result.model_dump(),
            },
        )
        if not result.ok:
            raise StellarMappingError(result.error)
        self.results[ident] = {"operation": operation, "bindings": selections, "result": result.model_dump()}
        self._emit_observation()
        return ident, result

    def copy(self, result_id, destination, *, confirm, authorization="human_confirmation"):
        """One confirmed fill then Tab/readback; no retries, Enter, Save or rollback."""
        from .knowledge import CalculationResult

        if authorization not in {"human_confirmation", "autonomous_opt_in"} or (
            authorization == "autonomous_opt_in"
            and self.journal.provenance.get("browser_execution") != "autonomous"
        ):
            raise BrowserSafetyStop("invalid_copy_authorization")
        self._current()
        if self.attempts >= self.MAX_WRITES or destination in self.verified:
            raise BrowserSafetyStop("numeric_write_limit")
        if result_id not in self.results:
            raise StellarMappingError("unknown_result")
        result = CalculationResult.model_validate(self.results[result_id]["result"])
        intent = plan_numeric_copy(
            self.mapping, result, destination, capture_sha256=self.mapping["capture_sha256"]
        )
        action = {
            "kind": "TYPE",
            "target": f"live:{self.revision}:{destination}",
            "value": intent["value"],
            "observation_revision": self.revision,
        }
        commit_action = {**action, "kind": "KEYPRESS", "value": "Tab"}
        self.journal.emit(
            "action_proposed",
            {
                "action": action,
                "mode": MODE,
                "action_source": "checkpoint_selected_autonomous"
                if authorization == "autonomous_opt_in"
                else "human_selected",
                "copy_authorization": authorization,
                "result_id": result_id,
                "unit": result.unit,
                "commit_action": commit_action,
                "display_policy": DISPLAY_POLICY,
            },
        )
        if not confirm(
            f"Write {intent['value']} {result.unit} to {destination}, then Tab to verify its displayed value?"
        ):
            return False
        # Revalidate AFTER the human pause; use the exact native handle, not nth().
        self._current()
        self.attempts += 1
        self.journal.emit("state", {"mode": MODE, "write_attempt": self.attempts, "destination": destination})
        try:
            self.handles[destination].fill(intent["value"], timeout=3000)
            # First and second readback catch immediate input/change formatting.
            report, mapping, handles, _frame = self._read()
            self.journal.emit("state", {"mode": MODE, "post_write_observation": mapping["observation"]})
            self._validate_transition(
                self.report,
                self.mapping,
                report,
                mapping,
                handles,
                destination,
                intent["value"],
                "unrelated_simulation_state_changed",
            )
            # Do not leave a previously typed field uncommitted until the next fill.
            # Focus is public UI state. No hidden application/model state is read.
            if not handles[destination].evaluate("e => document.activeElement === e"):
                raise BrowserSafetyStop("focus_changed_before_commit")
            self._guard()
            self.journal.emit(
                "state", {"mode": MODE, "commit_action": commit_action, "exact_readback": intent["value"]}
            )
            handles[destination].press("Tab", timeout=3000)
            committed_report, committed_mapping, committed_handles, committed_frame = self._read()
            self.journal.emit(
                "state", {"mode": MODE, "post_commit_observation": committed_mapping["observation"]}
            )
            committed_values = committed_mapping["observation"]["values"]
            display = committed_values["browser_field_map"][destination]["current_value"]
            formatting = self._validate_transition(
                report,
                mapping,
                committed_report,
                committed_mapping,
                committed_handles,
                destination,
                display,
                "unrelated_state_changed_during_commit",
                exact_copied=intent["value"],
            )
            receipt = {
                "result_id": result_id,
                "unit": result.unit,
                "exact_input_verified": True,
                "commit_key": "Tab",
                **formatting,
            }
            self.report, self.mapping, self.handles, self.frame = (
                committed_report,
                committed_mapping,
                committed_handles,
                committed_frame,
            )
            self.journal.numeric_readbacks[destination] = receipt
            self.verified.append(destination)
            self.revision += 1
            self.journal.emit(
                "action_result",
                {
                    **StepResult(observation=self.observation(), steps=self.attempts).model_dump(mode="json"),
                    "action": action,
                    "commit_action": commit_action,
                    "readback": display,
                    "exact_readback": intent["value"],
                    "numeric_readback": receipt,
                    "numeric_copy_verified": True,
                    "copy_authorization": authorization,
                    "task_completed": False,
                },
            )
            self._emit_observation()
            return True
        except Exception:
            # A failed readback may follow a real mutation. Never retry/undo it.
            self.stopped = True
            raise

    def _validate_transition(
        self,
        before_report,
        before_mapping,
        report,
        mapping,
        handles,
        destination,
        expected_value,
        mismatch_reason,
        *,
        exact_copied=None,
    ):
        try:
            self._validate_fields(
                mapping, handles, before_mapping["observation"]["values"], destination, expected_value
            )
            if numeric_value_projection(
                before_report, before_mapping, destination
            ) != numeric_value_projection(report, mapping, destination):
                raise BrowserSafetyStop(mismatch_reason)
            if exact_copied is not None:
                return committed_display(exact_copied, expected_value)
        except BrowserSafetyStop as exc:
            self.journal.record_screen_change(str(exc), before_report, report)
            raise

    def _validate_fields(self, mapping, handles, before, destination, expected_value):
        after = mapping["observation"]["values"]
        for name, old in self.handles.items():
            if not old.evaluate("(old, current) => old.isConnected && old === current", handles[name]):
                raise BrowserSafetyStop("numeric_control_replaced_after_write")
        if after["star_name"] != before["star_name"] or after["measurements"] != before["measurements"]:
            raise BrowserSafetyStop("star_changed_after_write")
        for name, field in after["browser_field_map"].items():
            previous = before["browser_field_map"][name]
            expected = expected_value if name == destination else previous["current_value"]
            if field["current_value"] != expected:
                raise BrowserSafetyStop(
                    "numeric_readback_mismatch" if name == destination else "unrelated_field_changed"
                )
            if {k: v for k, v in field.items() if k != "current_value"} != {
                k: v for k, v in previous.items() if k != "current_value"
            }:
                raise BrowserSafetyStop("numeric_field_identity_changed")


def run_numeric_diagnostic(page, config, output, *, prompt, confirm, echo):
    """Bounded interactive shell; same owned context for capture and all writes."""
    from playwright.sync_api import Error as PlaywrightError

    journal = NumericJournal(output)
    session = NumericSession(page, config, journal)
    outcome = "cancelled"
    try:
        session.start()
        echo(f"Selected star: {session.mapping['star_name']}. No star class is assumed.")
        while len(session.verified) < 3:
            operation = prompt("Calculation: distance / luminosity / temperature (or stop)")
            if operation == "stop":
                break
            if operation not in {"distance", "luminosity", "temperature"}:
                raise StellarMappingError("unsupported_diagnostic_operation")
            spec = session.calculator.pack.operation(operation)
            echo(spec.description)
            echo(json.dumps(session.sources(), indent=2))
            selections = {
                name: prompt(f"Source ID for {name} ({item.unit})") for name, item in spec.inputs.items()
            }
            result_id, result = session.calculate(operation, selections)
            echo(f"{result_id}: {result.value!r} {result.unit}")
            destination = prompt("Answer destination: distance / luminosity / temperature")
            if not session.copy(result_id, destination, confirm=confirm):
                break
            receipt = session.journal.numeric_readbacks[destination]
            echo(
                f"{destination}: exact copy {receipt['exact_copied']}; displayed after Tab {receipt['display_value']} ({receipt['format']}). Save was not clicked."
            )
        if len(session.verified) == 3:
            session._current()
            outcome = "numeric_transport_verified"
    except (BrowserSafetyStop, StellarMappingError, PlaywrightError) as exc:
        outcome = (
            str(exc)
            if isinstance(exc, (BrowserSafetyStop, StellarMappingError))
            else "browser_operation_failed"
        )
        journal.emit("error", {"reason": outcome, "writes_may_have_occurred": session.attempts > 0})
        echo(f"Stopped: {outcome}. No retry or automatic restoration. Keep the recorded evidence.")
        if journal.screen_diagnostics:
            reference = journal.screen_diagnostics[-1]
            # Render through JSON so visible page text cannot inject terminal controls.
            evidence = json.loads((output / reference["path"]).read_text())
            echo(
                f"Visible-state changes: {reference['change_count']}. Full evidence: {output / reference['path']}"
            )
            for change in evidence["changes"][:5]:
                echo(json.dumps(change, ensure_ascii=True))
            if reference["change_count"] > 5:
                echo("Additional changes are preserved in the evidence file and event log.")
    except BaseException as exc:
        outcome = (
            "interrupted"
            if isinstance(exc, (KeyboardInterrupt, EOFError)) or type(exc).__name__ == "Abort"
            else "internal_error"
        )
        journal.emit("error", {"reason": outcome, "writes_may_have_occurred": session.attempts > 0})
        raise
    finally:
        page.remove_listener("dialog", session._dialog)
        report = journal.finish(
            outcome=outcome,
            attempts=session.attempts,
            verified=session.verified,
            pack_hash=session.calculator.pack.checksum,
        )
    return report
