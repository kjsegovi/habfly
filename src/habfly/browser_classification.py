"""Narrow stellar choice transport using painted controls, never hidden inputs.

The v1.5.2 class circle is a LABEL with a painted ::after dot. Its adjacent
visible class text identifies it. We do not read its hidden radio, CSS class,
data attributes, event handlers, or the simulation's model. This adapter accepts
a caller's selection; it neither infers scientific class nor repairs choices.
"""

import hashlib
import json
import re
import time
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_numeric import comparable_screen, digest, screen_changes, screen_identity
from .browser_probe import _visible_frame, inspect_page
from .browser_setup import rendered_control
from .browser_stellar import CLASSES, SIMULATION_URL, _atoms, map_stellar_capture

RENDERING = """e => {
    const s=getComputedStyle(e), p=getComputedStyle(e,'::after');
    return {tag:e.tagName, width:s.width,height:s.height,border:s.border,
      display:s.display,opacity:s.opacity,content:p.content,dotWidth:p.width,
      dotHeight:p.height,dotDisplay:p.display,dotOpacity:p.opacity,dotColor:p.backgroundColor};
}"""

_RENDERING_PROPERTIES = (
    "tag",
    "width",
    "height",
    "border",
    "display",
    "opacity",
    "content",
    "dotWidth",
    "dotHeight",
    "dotDisplay",
    "dotOpacity",
    "dotColor",
)
_READ_PHASES = {
    "initial_read",
    "current_read",
    "pre_selection_read",
    "post_click_read",
    "pre_dispatch_native_binding",
    "pre_prefix_read",
    "post_prefix_read",
}
_LENGTH = r"(?:[0-9]{1,4}(?:\.[0-9]{1,12})?(?:px|em|rem|%)|auto)"
_COLOR = r"rgba?\([0-9]{1,3},\s*[0-9]{1,3},\s*[0-9]{1,3}(?:,\s*(?:0|1)(?:\.[0-9]{1,16})?)?\)"
_COMMON_RENDERING = {
    "tag": "LABEL",
    "width": "26px",
    "height": "26px",
    "display": "block",
    "opacity": "1",
    "content": '""',
    "dotWidth": "14px",
    "dotHeight": "14px",
    "dotDisplay": "block",
}
_POST_CLICK_PAINT_SECONDS = 2.0
_POST_CLICK_PAINT_PROBES = 40
_OPACITY = re.compile(r"(?:0|[1-9][0-9]?)(?:\.[0-9]{1,16})?(?:[eE](?P<exponent>[+-]?[0-9]{1,2}))?")


def _diagnostic_property(name, value):
    """Only bounded computed-style grammar, never arbitrary CSS strings."""
    if not isinstance(value, str) or len(value) > 128:
        return None
    if name == "tag":
        valid = value in {"LABEL", "DIV", "SPAN", "BUTTON", "INPUT"}
    elif name in {"width", "height", "dotWidth", "dotHeight"}:
        valid = re.fullmatch(_LENGTH, value)
    elif name in {"display", "dotDisplay"}:
        valid = value in {
            "block",
            "inline",
            "inline-block",
            "none",
            "flex",
            "inline-flex",
            "grid",
            "inline-grid",
            "contents",
        }
    elif name in {"opacity", "dotOpacity"}:
        # Chromium serializes small computed float opacities as, for example,
        # "1e-07" and the minimum positive float32 value "1.4013e-45".
        # Bounded numeric diagnostics are not accepted final painted states.
        number = _OPACITY.fullmatch(value)
        valid = (
            number is not None
            and abs(int(number.group("exponent") or "0")) <= 45
            and 0 <= Decimal(value) <= 1
        )
    elif name == "content":
        valid = value in {'""', "''", "none", "normal"}
    elif name == "dotColor":
        valid = re.fullmatch(_COLOR, value)
    else:
        valid = re.fullmatch(
            _LENGTH + r" (?:none|hidden|solid|dotted|dashed|double|groove|ridge|inset|outset) " + _COLOR,
            value,
        )
    return value if valid else None


class ClassCircleRenderingStop(BrowserSafetyStop):
    """Failure evidence only; the original acceptance rule/reason is unchanged."""

    def __init__(self, choice, rendering):
        super().__init__("unsupported_class_circle_rendering")
        rendering = rendering if isinstance(rendering, dict) else {}
        self.diagnostic = {
            "schema_version": 1,
            "reason": "unsupported_class_circle_rendering",
            "affected_choice": choice
            if choice in {*CLASSES, "gas_giant", "ice_giant", "terrestrial", "not_habitable", "habitable"}
            else None,
            "visible_rendering": {
                name: _diagnostic_property(name, rendering.get(name)) for name in _RENDERING_PROPERTIES
            },
            "validation_stage": "common_geometry"
            if any(rendering.get(k) != v for k, v in _COMMON_RENDERING.items())
            else "paint_state",
            "source": "existing_computed_visible_rendering_read",
            "additional_dom_queries": 0,
            "paint_acceptance_changed": False,
        }

    def at_read(self, phase, *, click_invoked=False, click_returned=False):
        if (
            phase not in _READ_PHASES
            or type(click_invoked) is not bool
            or type(click_returned) is not bool
            or click_returned
            and not click_invoked
        ):
            raise ValueError("Invalid rendering diagnostic phase")
        self.diagnostic.update(
            failure_phase=phase,
            native_class_click_invoked=click_invoked,
            native_class_click_returned=click_returned,
            readback_verified=False,
        )
        return self


def _stellar_status(report):
    # Lazy import: full_stellar also consumes the visible class reader.
    from .browser_full_stellar import full_stellar_status_projection

    return full_stellar_status_projection(report)


class ClassScreenMismatch(BrowserSafetyStop):
    """The two already-read public captures; no follow-up browser query."""

    def __init__(
        self, reason, before, after, *, phase, class_invoked, class_returned, prefix_invoked, prefix_returned
    ):
        super().__init__(reason)
        self.before, self.after = deepcopy(before), deepcopy(after)
        self.diagnostic = {
            "schema_version": 1,
            "reason": reason,
            "failure_phase": phase,
            "native_class_click_invoked": class_invoked,
            "native_class_click_returned": class_returned,
            "native_prefix_selection_invoked": prefix_invoked,
            "native_prefix_selection_returned": prefix_returned,
            "comparison_projection": "full_stellar_status_projection",
            "before_screen_sha256": screen_identity(before),
            "after_screen_sha256": screen_identity(after),
            "before_projected_sha256": screen_identity(_stellar_status(before)),
            "after_projected_sha256": screen_identity(_stellar_status(after)),
            "public_screen_changes": screen_changes(before, after),
            "projected_screen_changes": screen_changes(_stellar_status(before), _stellar_status(after)),
            "additional_dom_queries": 0,
            "readback_verified": False,
            "task_completed": False,
            "automatic_retry": False,
        }


def painted_selection(rendering):
    if any(rendering.get(k) != v for k, v in _COMMON_RENDERING.items()):
        raise BrowserSafetyStop("unsupported_class_circle_rendering")
    state = (rendering.get("border"), rendering.get("dotOpacity"), rendering.get("dotColor"))
    if state == ("1px solid rgb(0, 200, 220)", "0", "rgba(0, 0, 0, 0)"):
        return False
    if state == ("1px solid rgb(255, 255, 255)", "1", "rgb(255, 255, 255)"):
        return True
    raise BrowserSafetyStop("unsupported_class_circle_rendering")


def _read_choice_circles(frame, names):
    handles, rendering, selected = {}, {}, []
    for name in names:
        caption = re.compile(r"^\s*" + r"\s+".join(map(re.escape, name.split("_"))) + r"\s*$", re.IGNORECASE)
        labels = [e for e in frame.get_by_text(caption).all() if e.is_visible()]
        if len(labels) != 1:
            raise BrowserSafetyStop("ambiguous_class_label")
        label = labels[0]
        parent = label.locator("..")
        if " ".join(parent.inner_text().split()).casefold() != name.replace("_", " "):
            raise BrowserSafetyStop("unverified_class_label_container")
        circles = [e for e in parent.locator(":scope > label").all() if e.is_visible()]
        if len(circles) != 1 or not rendered_control(circles[0]):
            raise BrowserSafetyStop("unavailable_class_circle")
        circle = circles[0]
        box, caption = circle.bounding_box(), label.bounding_box()
        if (
            not box
            or not caption
            or not (
                0 <= caption["y"] - box["y"] - box["height"] <= 12
                and abs(box["x"] + box["width"] / 2 - caption["x"] - caption["width"] / 2) < 2
            )
        ):
            raise BrowserSafetyStop("class_circle_not_above_label")
        rendering[name] = circle.evaluate(RENDERING)
        try:
            if painted_selection(rendering[name]):
                selected.append(name)
        except BrowserSafetyStop as exc:
            if str(exc) != "unsupported_class_circle_rendering":
                raise
            raise ClassCircleRenderingStop(name, rendering[name]) from None
        handles[name] = circle.element_handle(timeout=2000)
    if len(selected) > 1:
        raise BrowserSafetyStop("multiple_selected_classes")
    return {"selected": selected[0] if selected else None, "rendering": rendering}, handles


def read_class_choices(frame):
    return _read_choice_circles(frame, CLASSES)


def read_planet_class_choices(frame):
    """Same observed painted widget; no planet class is inferred or selected."""
    return _read_choice_circles(frame, ("gas_giant", "ice_giant", "terrestrial"))


def read_habitability_choices(frame):
    """Read the painted default/current choice; never infer a policy decision."""
    return _read_choice_circles(frame, ("not_habitable", "habitable"))


def _stable_mapping(mapping):
    values = deepcopy(mapping["observation"]["values"])
    fields = values["browser_field_map"]
    for name in ("mass", "radius", "lifetime"):
        fields.pop(name, None)
    for field in fields.values():
        field.pop("capture_target_id")
    values["visible_numeric_fields"] = list(fields)
    values["color"].pop("capture_target_id")
    values.pop("conditional_fields_visible")
    values.pop("lifetime_prefix")
    return values


def class_transition_projection(report, mapping):
    """Remove only the known blank conditional panel, not arbitrary changed text."""
    value = comparable_screen(_stellar_status(report))
    frame = next(f for f in value["frames"] if f["url"] == SIMULATION_URL)
    values = mapping["observation"]["values"]
    remove = {
        f["capture_target_id"]
        for name, f in values["browser_field_map"].items()
        if name in {"mass", "radius", "lifetime"}
    }
    if values["lifetime_prefix"]:
        remove.add(values["lifetime_prefix"]["capture_target_id"])
    frame["controls"] = [c for c in frame["controls"] if c["id"] not in remove]
    # The capture inventory re-numbers the color control after fields appear.
    for index, control in enumerate(frame["controls"]):
        control["id"] = f"{frame['id']}:projected:{index}"
    atoms = _atoms(frame["accessibility"])
    if values["conditional_fields_visible"]:
        start = next(i for i, (k, v) in enumerate(atoms) if k == "text" and v == "mass (M")
        finish = next(i for i in range(start, len(atoms)) if atoms[i][0].split(" ", 1)[0] == "combobox")
        atoms[start : finish + 1] = []
    warning = "mass, radius and lifetime are only relevant for main sequence stars "
    frame["accessibility"] = [
        (k, v.removeprefix(warning) if k == "text" and isinstance(v, str) else v) for k, v in atoms
    ]
    text = re.sub(r"\s+", " ", frame["text"]).strip()
    text = re.sub(re.escape(warning), "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"mass \(MS\)\s*radius \(RS\)\s*lifetime \(years\)\s*ka Ma Ga Ta ", "", text, flags=re.IGNORECASE
    )
    frame["text"] = text
    return value


def prefix_transition_projection(report, target, *, initializes_lifetime=False):
    value = comparable_screen(_stellar_status(report))
    frame = next(f for f in value["frames"] if f["url"] == SIMULATION_URL)
    control = next(c for c in frame["controls"] if c["id"] == target)
    atom = _atoms(control["accessibility"])[0]
    strip = lambda items: [s.removesuffix(" [selected]") for s in items]
    control["accessibility"] = (atom[0], strip(atom[1]))
    control["value"] = None
    frame["accessibility"] = [
        (k, strip(v)) if (k, v) == atom else (k, v) for k, v in _atoms(frame["accessibility"])
    ]
    if initializes_lifetime:
        lifetime = [
            c
            for c in frame["controls"]
            if c["role"] == "textbox" and _atoms(c["accessibility"])[0][0] == 'textbox "0.00000"'
        ]
        if len(lifetime) != 1:
            raise BrowserSafetyStop("ambiguous_lifetime_initialization")
        lifetime[0]["accessibility"] = ('textbox "0.00000"', None)
        lifetime[0]["value"] = None
        frame["accessibility"] = [
            (k, None if k == 'textbox "0.00000"' else v) for k, v in frame["accessibility"]
        ]
    return value


class StellarSelectionSession:
    """At most one class choice and one prefix choice, with no rollback/retry."""

    def __init__(self, page, config, emit, *, fresh_star=None):
        self.page, self.config, self.emit = page, config, emit
        self.stopped = False
        self._rendering_phase = "initial_read"
        self._class_click_invoked = self._class_click_returned = False
        self._prefix_selection_invoked = self._prefix_selection_returned = False
        self._pending_class = None
        self.writes = set()
        self.frame = None
        self.fresh_star = None
        if fresh_star is not None:
            path = Path(fresh_star)
            receipt = json.loads((path / "confirmed.json").read_text())
            raw = (path / "stellar/observation.json").read_bytes()
            manifest = json.loads((path / "stellar/manifest.json").read_text())
            if (
                receipt.get("fresh_blank_numeric_answers_verified") is not True
                or receipt.get("class_selection_verified") is not False
                or receipt.get("answer_writes") != 0
                or receipt.get("action_source") != "deterministic_navigation"
                or hashlib.sha256(raw).hexdigest() != manifest.get("observation_sha256")
                or (path / "class-selection-reserved.json").exists()
            ):
                raise BrowserSafetyStop("invalid_fresh_star_class_evidence")
            self.fresh_star = {
                "path": path,
                "receipt": receipt,
                "report": json.loads(raw),
                "hash": manifest["observation_sha256"],
            }
        self.report, self.mapping, self.choices, self.handles = self.read()

    def _fresh_initial_paint(self, report, mapping, choices):
        if self.fresh_star is None or "class" in self.writes:
            return False
        from .browser_full_stellar import full_stellar_status_projection

        proof = self.fresh_star
        values = mapping["observation"]["values"]
        valid = (
            mapping["star_name"] == proof["receipt"]["star"]
            and choices["selected"] == proof["receipt"]["painted_stellar_class"]
            and not values["conditional_fields_visible"]
            and values["color"]["selected"] is None
            and all(f["current_value"] == "" for f in values["browser_field_map"].values())
            and screen_identity(full_stellar_status_projection(report))
            == screen_identity(full_stellar_status_projection(proof["report"]))
        )
        if not valid:
            raise BrowserSafetyStop("fresh_star_class_evidence_changed")
        return True

    def read(self):
        # Keep only the already-approved public capture if this read fails.
        # In particular an inspect_page failure has no fabricated replacement.
        self._read_stage, self._read_public_report = "inspect_page", None
        try:
            return self._read_once()
        except Exception as exc:
            try:
                exc._habfly_class_read_context = {
                    "read_phase": self._rendering_phase,
                    "read_stage": self._read_stage,
                    "native_class_click_invoked": self._class_click_invoked,
                    "native_class_click_returned": self._class_click_returned,
                    "native_prefix_selection_invoked": self._prefix_selection_invoked,
                    "native_prefix_selection_returned": self._prefix_selection_returned,
                    "report": deepcopy(self._read_public_report),
                }
            except Exception:  # noqa: BLE001,S110 - never replace failure or log unsafe exception text
                pass
            raise

    def _read_once(self):
        if self.stopped:
            raise BrowserSafetyStop("stellar_selection_stopped")
        if len(self.page.context.pages) != 1:
            raise BrowserSafetyStop("unexpected_popup")
        report = inspect_page(self.page, self.config)
        self._read_stage = "frame_validation"
        if report["ignored_frame_urls"]:
            raise BrowserSafetyStop("unknown_visible_frame")
        self._read_public_report = deepcopy(report)
        frames = [
            f for f in self.page.frames if f.url == SIMULATION_URL and _visible_frame(f, self.page.main_frame)
        ]
        if len(frames) != 1 or self.frame is not None and frames[0] != self.frame:
            raise BrowserSafetyStop("simulation_frame_changed")
        self.frame = frames[0]
        self._read_stage = "map_stellar_capture"
        mapping = map_stellar_capture(
            report,
            capture_sha256=screen_identity(report),
            allow_color_selection=True,
            allow_main_sequence_fields=True,
        )
        self._read_stage = "read_class_choices"
        try:
            choices, handles = read_class_choices(self.frame)
        except ClassCircleRenderingStop as exc:
            try:
                choices, handles = self._settle_post_click_paint(exc)
            except ClassCircleRenderingStop as final:
                raise final.at_read(
                    self._rendering_phase,
                    click_invoked=self._class_click_invoked,
                    click_returned=self._class_click_returned,
                ) from None
        self._read_stage = "fresh_source_validation"
        unconfirmed = self._fresh_initial_paint(report, mapping, choices)
        if (
            mapping["observation"]["values"]["conditional_fields_visible"]
            != (choices["selected"] == "main_sequence")
            and not unconfirmed
        ):
            raise BrowserSafetyStop("class_conditional_fields_disagree")
        self._read_stage = "confirming_public_capture"
        after = inspect_page(self.page, self.config)
        if screen_identity(_stellar_status(report)) != screen_identity(_stellar_status(after)):
            raise self._screen_mismatch("screen_changed_during_class_read", report, after)
        self._rendering_phase = "current_read"
        return report, mapping, choices, handles

    def _screen_mismatch(self, reason, before, after):
        return ClassScreenMismatch(
            reason,
            before,
            after,
            phase=self._rendering_phase,
            class_invoked=self._class_click_invoked,
            class_returned=self._class_click_returned,
            prefix_invoked=self._prefix_selection_invoked,
            prefix_returned=self._prefix_selection_returned,
        )

    def _settle_post_click_paint(self, first):
        """Read only the known dot animation; never accept an intermediate paint.

        The surrounding read still verifies the full public snapshot and the
        caller still verifies scientific values and its original deadline.
        The small paint budget adds no click, new reservation, or retry owner.
        """

        def eligible(error):
            visible = error.diagnostic["visible_rendering"]
            opacity = visible.get("dotOpacity")
            return (
                self._rendering_phase == "post_click_read"
                and self._class_click_returned
                and self._pending_class in CLASSES
                and error.diagnostic["affected_choice"] in {self._pending_class, self.choices["selected"]}
                and all(visible.get(k) == v for k, v in _COMMON_RENDERING.items())
                and isinstance(opacity, str)
                and 0 < float(opacity) < 1
                and (visible["border"], visible["dotColor"])
                in {
                    ("1px solid rgb(0, 200, 220)", "rgba(0, 0, 0, 0)"),
                    ("1px solid rgb(255, 255, 255)", "rgb(255, 255, 255)"),
                }
            )

        def context():
            if self.stopped:
                raise BrowserSafetyStop("stellar_selection_stopped")
            if len(self.page.context.pages) != 1 or self.page.context.pages[0] != self.page:
                raise BrowserSafetyStop("unexpected_popup")
            frames = [
                frame
                for frame in self.page.frames
                if frame.url == SIMULATION_URL and _visible_frame(frame, self.page.main_frame)
            ]
            if frames != [self.frame]:
                raise BrowserSafetyStop("simulation_frame_changed")
            if any(not handle.evaluate("e => e.isConnected") for handle in self.handles.values()):
                raise BrowserSafetyStop("class_control_replaced")

        if not eligible(first):
            raise first
        deadline, error = time.monotonic() + _POST_CLICK_PAINT_SECONDS, first
        for probe in range(1, _POST_CLICK_PAINT_PROBES + 1):
            context()
            if not eligible(error):
                raise error
            error.diagnostic["post_click_settling"] = {
                "max_seconds": _POST_CLICK_PAINT_SECONDS,
                "max_probes": _POST_CLICK_PAINT_PROBES,
                "probes": probe - 1,
                "settled": False,
                "additional_clicks": 0,
            }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise error
            self.page.wait_for_timeout(min(50, remaining * 1000))
            context()
            if time.monotonic() >= deadline:
                raise error
            try:
                choices, handles = read_class_choices(self.frame)
            except ClassCircleRenderingStop as next_error:
                error = next_error
                continue
            context()
            if time.monotonic() >= deadline:
                raise error
            for name, old in self.handles.items():
                if not old.evaluate("(a,b) => a.isConnected && a===b", handles[name]):
                    raise BrowserSafetyStop("class_control_replaced")
            return choices, handles
        error.diagnostic["post_click_settling"] = {
            "max_seconds": _POST_CLICK_PAINT_SECONDS,
            "max_probes": _POST_CLICK_PAINT_PROBES,
            "probes": _POST_CLICK_PAINT_PROBES,
            "settled": False,
            "additional_clicks": 0,
        }
        raise error

    def current(self):
        report, mapping, choices, handles = self.read()
        if screen_identity(_stellar_status(report)) != screen_identity(_stellar_status(self.report)):
            raise self._screen_mismatch("stale_class_observation", self.report, report)
        if choices != self.choices:
            raise BrowserSafetyStop("stale_class_observation")
        for name, old in self.handles.items():
            if not old.evaluate("(a,b) => a.isConnected && a===b", handles[name]):
                raise BrowserSafetyStop("class_control_replaced")
        return report, mapping, choices, handles

    def select_class(self, name, *, source, expected_previous=None, revision_reason=None):
        if name not in CLASSES or source not in {"reference_diagnostic", "checkpoint"}:
            raise BrowserSafetyStop("invalid_class_selection")
        self._rendering_phase = "pre_selection_read"
        self._class_click_invoked = self._class_click_returned = False
        self.current()
        revising = expected_previous is not None
        if revising:
            values = self.mapping["observation"]["values"]
            if (
                source != "reference_diagnostic"
                or expected_previous not in CLASSES
                or expected_previous == name
                or self.choices["selected"] != expected_previous
                or not isinstance(revision_reason, str)
                or not revision_reason.strip()
                or len(revision_reason) > 1000
                or self.fresh_star is not None
                or values["color"]["selected"] is not None
                or any(f["current_value"] != "" for f in values["browser_field_map"].values())
            ):
                raise BrowserSafetyStop("invalid_blank_class_revision")
        elif revision_reason is not None:
            raise BrowserSafetyStop("invalid_blank_class_revision")
        if "class" in self.writes or (
            self.choices["selected"] is not None and self.fresh_star is None and not revising
        ):
            raise BrowserSafetyStop("class_write_limit")
        previous = self.choices["selected"]
        if self.fresh_star is not None:
            from .browser_assessment_actions import persist_json

            persist_json(
                self.fresh_star["path"] / "class-selection-reserved.json",
                {
                    "star": self.mapping["star_name"],
                    "source_capture_sha256": self.fresh_star["hash"],
                    "previous_unconfirmed_paint": previous,
                    "selected_class": name,
                    "action_source": source,
                    "max_clicks": 1,
                    "automatic_retry": False,
                },
            )
        self.writes.add("class")
        self.emit(
            "action_proposed",
            {
                "kind": "SELECT",
                "target": "stellar_class",
                "value": name,
                "action_source": source,
                "correctness_verified": False,
                **({"previous": expected_previous, "revision_reason": revision_reason} if revising else {}),
            },
        )
        try:
            self._pending_class = name
            self._class_click_invoked = True
            self.handles[name].click(timeout=3000)
            self._class_click_returned = True
            self._rendering_phase = "post_click_read"
            report, mapping, choices, handles = self.read()
            if choices["selected"] != name:
                raise BrowserSafetyStop("class_readback_mismatch")
            for other in CLASSES:
                expected = (
                    self.choices["rendering"][name]
                    if previous != name and other == previous
                    else self.choices["rendering"][other]
                )
                if other != name and choices["rendering"][other] != expected:
                    raise BrowserSafetyStop("unrelated_class_changed")
            if _stable_mapping(mapping) != _stable_mapping(self.mapping):
                raise BrowserSafetyStop("stellar_data_changed_by_class")
            if class_transition_projection(self.report, self.mapping) != class_transition_projection(
                report, mapping
            ):
                raise self._screen_mismatch("unrelated_state_changed_by_class", self.report, report)
            fields = mapping["observation"]["values"]["browser_field_map"]
            if name == "main_sequence" and any(
                fields[n]["current_value"] != "" for n in ("mass", "radius", "lifetime")
            ):
                raise BrowserSafetyStop("preexisting_conditional_answers")
            self.report, self.mapping, self.choices, self.handles = report, mapping, choices, handles
            receipt = {
                "selected_class": name,
                "selection_source": source,
                "readback_verified": True,
                "correctness_verified": False,
                "rendering_sha256": digest(choices),
                "task_completed": False,
            }
            if self.fresh_star is not None:
                receipt["fresh_star_capture_sha256"] = self.fresh_star["hash"]
                receipt["previous_unconfirmed_paint"] = previous
            if revising:
                receipt.update(previous=expected_previous, revision_reason=revision_reason)
            self.emit("action_result", receipt)
            return receipt
        except Exception:
            self.stopped = True
            raise

    def select_prefix(self, prefix):
        from .browser_stellar import LIFETIME_PREFIXES

        self._rendering_phase = "pre_prefix_read"
        self._class_click_invoked = self._class_click_returned = False
        self._prefix_selection_invoked = self._prefix_selection_returned = False
        self.current()
        values = self.mapping["observation"]["values"]
        if prefix not in LIFETIME_PREFIXES or self.choices["selected"] != "main_sequence":
            raise BrowserSafetyStop("invalid_lifetime_prefix_selection")
        if "prefix" in self.writes or values["lifetime_prefix"]["selected"] is not None:
            raise BrowserSafetyStop("prefix_write_limit")
        controls = [e for e in self.frame.get_by_role("combobox").all() if e.is_visible()]
        matches = [
            e
            for e in controls
            if e.aria_snapshot()
            == next(
                c["accessibility"]
                for f in self.report["frames"]
                if f["url"] == SIMULATION_URL
                for c in f["controls"]
                if c["id"] == values["lifetime_prefix"]["capture_target_id"]
            )
        ]
        if len(matches) != 1 or not matches[0].is_enabled() or not rendered_control(matches[0]):
            raise BrowserSafetyStop("unavailable_lifetime_prefix")
        handle = matches[0].element_handle(timeout=2000)
        if not handle.evaluate("e => e.tagName==='SELECT' && !e.multiple"):
            raise BrowserSafetyStop("unsupported_prefix_control")
        self.writes.add("prefix")
        self.emit(
            "action_proposed",
            {
                "kind": "SELECT",
                "target": "lifetime_prefix",
                "value": prefix,
                "action_source": "explicit_unit_transport",
            },
        )
        try:
            self._prefix_selection_invoked = True
            handle.select_option(label=prefix, timeout=3000)
            self._prefix_selection_returned = True
            self._rendering_phase = "post_prefix_read"
            report, mapping, choices, handles = self.read()
            new = mapping["observation"]["values"]
            if choices != self.choices or new["lifetime_prefix"]["selected"] != prefix:
                raise BrowserSafetyStop("prefix_readback_mismatch")
            if _stable_mapping(mapping) != _stable_mapping(self.mapping):
                raise BrowserSafetyStop("stellar_data_changed_by_prefix")
            for name in ("mass", "radius", "lifetime"):
                if (
                    new["browser_field_map"][name]["current_value"]
                    != values["browser_field_map"][name]["current_value"]
                ) and not (
                    name == "lifetime"
                    and values["browser_field_map"][name]["current_value"] == ""
                    and new["browser_field_map"][name]["current_value"] == "0.000"
                ):
                    raise BrowserSafetyStop("numeric_data_changed_by_prefix")
            target = values["lifetime_prefix"]["capture_target_id"]
            initializes = (
                values["browser_field_map"]["lifetime"]["current_value"] == ""
                and new["browser_field_map"]["lifetime"]["current_value"] == "0.000"
            )
            if prefix_transition_projection(
                self.report, target, initializes_lifetime=initializes
            ) != prefix_transition_projection(report, target, initializes_lifetime=initializes):
                raise self._screen_mismatch("unrelated_state_changed_by_prefix", self.report, report)
            self.report, self.mapping, self.choices, self.handles = report, mapping, choices, handles
            receipt = {
                "lifetime_prefix": prefix,
                "readback_verified": True,
                "task_completed": False,
                "blank_lifetime_initialized_to_zero": initializes,
            }
            self.emit("action_result", receipt)
            return receipt
        except Exception:
            self.stopped = True
            raise
