"""Hover one caller-selected, rendered spectral excursion marker.

The FluxChartSession supplies the same activity/answer/budget guards. This
sensor never reads the moving-line transform or hidden wavelength properties.
"""

import re
import time
from decimal import Decimal, localcontext
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_planet_chart import VISIBLE_TOOLTIP

_SETTLE_SECONDS = 2.0
_POLL_MILLISECONDS = 50
GEOMETRIC_PAIR_MODE = "visible_geometric_spectrum_pair_v1"
_VALUE_ERRORS = {
    "invalid_visible_wavelength",
    "invalid_spectrum_excursion_order",
    "asymmetric_visible_spectrum_excursion",
}
_SAFE_REASONS = {
    "ambiguous_rest_wavelength",
    "unsupported_spectral_strip",
    "ambiguous_visible_excursion_markers",
    "excursion_marker_outside_visible_strip",
    "excursion_marker_occluded",
    "missing_or_ambiguous_spectrum_tooltip",
    "invalid_visible_spectrum_tooltip",
    "spectrum_tooltip_unsettled",
    "spectrum_tooltip_wrong_side",
    "spectrum_tooltip_duplicate_endpoint",
    "spectrum_marker_pair_changed",
    "spectrum_tooltip_unexposed",
    "spectrum_selected_marker_not_hovered",
    "spectrum_event_forwarding_failed",
    "spectrum_capture_star_changed",
    "unexpected_chart_context",
    "ambiguous_simulation_frame",
    "ambiguous_flux_chart",
    "not_planet_observations",
    "ambiguous_visible_star_name",
    "chart_session_stopped",
    "chart_time_limit",
    "navigation_outside_activity",
    "chart_frame_changed",
    "chart_frame_hidden_or_replaced",
    "previously_hidden_frame_became_visible",
    "chart_context_or_answers_changed",
    "chart_action_limit",
    "chart_document_body_replaced",
    "chart_replaced",
    "chart_container_changed",
    "authentication_required",
    "unexpected_modal",
} | {"spectrum_" + value for value in _VALUE_ERRORS}
_BOUND_MARKER = """(line,a)=>{
  const r=line.getBoundingClientRect(),s=a.strip.getBoundingClientRect();
  const same=(r,b)=>r.x===b.x&&r.y===b.y&&r.width===b.width&&r.height===b.height;
  return {bound:line.isConnected&&a.strip.isConnected&&a.parent.isConnected&&
    line.parentElement===a.strip&&a.strip.parentElement===a.parent&&same(r,a.lineBox)&&same(s,a.stripBox),
    hovered:line.matches(':hover'),
    hit:document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===line};
}"""


def _settled_reading(session, parent, line, bound, side, *, previous_wavelength=None, bound_guard=None):
    """Read-only settling after ONE native move; never repair an endpoint."""
    deadline = min(time.monotonic() + _SETTLE_SECONDS, session.started + session.max_seconds)
    previous, diagnostic = None, session._spectrum_diagnostic
    last_issue = "missing_or_ambiguous_spectrum_tooltip"
    while time.monotonic() < deadline:
        session._guard()
        if bound_guard is not None:
            bound_guard()
        if time.monotonic() >= deadline:
            break
        identity = line.evaluate(_BOUND_MARKER, bound)
        diagnostic["marker_readback"] = {
            key: identity.get(key) is True for key in ("bound", "hovered", "hit")
        }
        if identity.get("bound") is not True or identity.get("hit") is not True:
            raise BrowserSafetyStop("spectrum_selected_marker_not_hovered")
        if not line.evaluate(VISIBLE_TOOLTIP):
            raise BrowserSafetyStop("excursion_marker_occluded")
        tips = [
            e
            for e in parent.get_by_text(re.compile(r"^\d+(?:\.\d+)?nm$")).all()
            if e.is_visible() and e.evaluate("e=>getComputedStyle(e).backgroundColor==='rgb(250, 250, 220)'")
        ]
        if len(tips) > 1:
            raise BrowserSafetyStop("missing_or_ambiguous_spectrum_tooltip")
        if tips:
            if not tips[0].evaluate(VISIBLE_TOOLTIP):
                raise BrowserSafetyStop("spectrum_tooltip_unexposed")
            text = tips[0].inner_text()
            if not re.fullmatch(r"\d{1,8}(?:\.\d{1,30})?nm", text):
                raise BrowserSafetyStop("invalid_visible_spectrum_tooltip")
            diagnostic["last_visible_wavelength_text"] = text
            value = Decimal(text.removesuffix("nm"))
            if side in {"left", "right"}:
                # Geometry selects the marker; only its displayed number tells
                # us the wavelength side. Never synthesize the other endpoint.
                correct_side = value > 0 and value != Decimal("656.3")
            else:
                correct_side = 0 < value < Decimal("656.3") if side == "blue" else value > Decimal("656.3")
            duplicate = previous_wavelength is not None and value == Decimal(
                previous_wavelength.removesuffix("nm")
            )
            if duplicate:
                correct_side = False
            if correct_side:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                handle = tips[0].element_handle(timeout=min(2000, max(1, remaining * 1000)))
                if (
                    previous is not None
                    and previous[0] == text
                    and previous[1].evaluate("(a,b)=>a.isConnected&&a===b", handle)
                ):
                    session._guard()
                    if bound_guard is not None:
                        bound_guard()
                    identity = line.evaluate(_BOUND_MARKER, bound)
                    if not all(identity.get(key) is True for key in ("bound", "hit")):
                        raise BrowserSafetyStop("spectrum_selected_marker_not_hovered")
                    if time.monotonic() >= deadline:
                        break
                    return text
                previous = (text, handle)
                last_issue = "spectrum_tooltip_unsettled"
            else:
                # A stale red tooltip is not evidence about a requested blue marker.
                previous = None
                last_issue = (
                    "spectrum_tooltip_duplicate_endpoint" if duplicate else "spectrum_tooltip_wrong_side"
                )
        else:
            previous = None
            last_issue = "missing_or_ambiguous_spectrum_tooltip"
        remaining = deadline - time.monotonic()
        if remaining > 0:
            session.page.wait_for_timeout(min(_POLL_MILLISECONDS, remaining * 1000))
    session._guard()
    raise BrowserSafetyStop(last_issue)


def normalize_geometric_spectrum_pair(left_text, right_text):
    """Interpret two actual labels; geometry is not a wavelength-direction claim."""
    from .planet_charts import spectrum_excursion

    for text in (left_text, right_text):
        if not isinstance(text, str) or not re.fullmatch(r"[0-9]{1,8}(?:\.[0-9]{1,30})?nm", text):
            raise ValueError("invalid_visible_wavelength")
    ordered = sorted((left_text, right_text), key=lambda text: Decimal(text.removesuffix("nm")))
    # This independently rejects zero, equal, same-side and asymmetric labels.
    # The supported visible label domain is up to 38 digits. Default Decimal
    # precision could round an asymmetric last digit into apparent agreement.
    with localcontext() as context:
        context.prec = 64
        result = spectrum_excursion("656.3nm", *ordered)
    return {
        "readings": dict(zip(("blue", "red"), ordered, strict=True)),
        "result": result,
        "marker_readings": {"left": left_text, "right": right_text},
    }


def read_geometric_spectrum_pair(session):
    """Two native moves to one pinned pair; polarity comes from visible labels.

    The old blue/red API below intentionally retains its original strict side
    checks. This new recipe records left/right events and never repairs, retries,
    or manufactures a measurement when the independent pair fails validation.
    """
    try:
        session._guard()
        label = session.frame.get_by_text("656.3nm", exact=True)
        if label.count() != 1 or not label.is_visible() or not label.evaluate(VISIBLE_TOOLTIP):
            raise BrowserSafetyStop("ambiguous_rest_wavelength")
        label_handle = label.element_handle(timeout=2000)
        parent_locator = label.locator("..")
        parent = parent_locator.element_handle(timeout=2000)
        strips = [
            element
            for element in parent.query_selector_all(":scope > div")
            if element.is_visible()
            and element.evaluate("e=>getComputedStyle(e).backgroundColor==='rgb(193, 58, 44)'")
        ]
        if len(strips) != 1:
            raise BrowserSafetyStop("unsupported_spectral_strip")
        strip = strips[0]
        lines = [
            element
            for element in strip.query_selector_all(":scope > div")
            if element.is_visible()
            and element.evaluate(
                "e=>getComputedStyle(e).backgroundColor==='rgb(0, 0, 0)'&&e.getBoundingClientRect().width===1"
            )
        ]
        if len(lines) != 2:
            raise BrowserSafetyStop("ambiguous_visible_excursion_markers")
        lines.sort(key=lambda element: element.bounding_box()["x"])
        boxes, strip_box = [element.bounding_box() for element in lines], strip.bounding_box()
        if boxes[0]["x"] + boxes[0]["width"] > boxes[1]["x"]:
            raise BrowserSafetyStop("ambiguous_visible_excursion_markers")
        client_box = (
            "e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}}"
        )
        bindings = [
            {
                "strip": strip,
                "parent": parent,
                "lineBox": line.evaluate(client_box),
                "stripBox": strip.evaluate(client_box),
            }
            for line in lines
        ]
        for box in boxes:
            if not (
                strip_box["x"] <= box["x"]
                and box["x"] + box["width"] <= strip_box["x"] + strip_box["width"]
                and strip_box["y"] <= box["y"]
                and box["y"] + box["height"] <= strip_box["y"] + strip_box["height"]
            ):
                raise BrowserSafetyStop("excursion_marker_outside_visible_strip")

        def pair_guard():
            session._guard()
            if label.count() != 1 or not parent_locator.evaluate(
                """(p,a)=>{
                  if(p!==a.parent||!a.label.isConnected||a.label.parentElement!==p||
                    a.label.innerText!=='656.3nm'||a.strip.parentElement!==p||
                    getComputedStyle(a.strip).backgroundColor!=='rgb(193, 58, 44)')return false;
                  const lines=[...a.strip.querySelectorAll(':scope > div')].filter(e=>{
                    const r=e.getBoundingClientRect(),s=getComputedStyle(e);
                    return r.width===1&&r.height>0&&s.display!=='none'&&s.visibility==='visible'&&
                      s.backgroundColor==='rgb(0, 0, 0)';});
                  return lines.length===2&&lines.includes(a.lines[0])&&lines.includes(a.lines[1]);
                }""",
                {"parent": parent, "label": label_handle, "strip": strip, "lines": lines},
            ):
                raise BrowserSafetyStop("spectrum_marker_pair_changed")
            for line, box, bound in zip(lines, boxes, bindings, strict=True):
                identity = line.evaluate(_BOUND_MARKER, bound)
                if identity.get("bound") is not True or line.bounding_box() != box:
                    raise BrowserSafetyStop("spectrum_marker_pair_changed")
                if identity.get("hit") is not True or not line.evaluate(VISIBLE_TOOLTIP):
                    raise BrowserSafetyStop("excursion_marker_occluded")
            if not label.evaluate(VISIBLE_TOOLTIP) or label.inner_text() != "656.3nm":
                raise BrowserSafetyStop("ambiguous_rest_wavelength")

        pair_guard()
        readings = {}
        for marker, line, box, bound in zip(("left", "right"), lines, boxes, bindings, strict=True):
            pair_guard()
            session._begin("HOVER", {"surface": "spectrum", "marker": marker})
            # A callback may abort or replace the page. Recheck before moving.
            pair_guard()
            session._spectrum_diagnostic = {
                "source": "rendered_geometry_and_visible_tooltip",
                "marker_mode": GEOMETRIC_PAIR_MODE,
                "marker": marker,
                "strip_box": strip_box,
                "marker_boxes": boxes,
                "requested_pointer": {"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2},
                "marker_readback": None,
                "last_visible_wavelength_text": None,
                "learned_perception": False,
                "scientific_verified": False,
            }
            session.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            # Use the retained parent handle for identity; the locator resolves
            # tooltip text only and is checked against that handle on each poll.
            text = _settled_reading(
                session,
                parent_locator,
                line,
                bound,
                marker,
                previous_wavelength=readings.get("left"),
                bound_guard=pair_guard,
            )
            readings[marker] = text
            session.emit(
                "action_result",
                {
                    "sequence": session.actions,
                    "spectrum_sample": {
                        "marker": marker,
                        "wavelength_text": text,
                        "source": "visible_spectrum_tooltip",
                    },
                    "task_completed": False,
                },
            )
            pair_guard()
        return normalize_geometric_spectrum_pair(readings["left"], readings["right"])
    except ValueError as exc:
        session.stopped = True
        if str(exc) in _VALUE_ERRORS:
            raise BrowserSafetyStop("spectrum_" + str(exc)) from None
        raise
    except BaseException:
        session.stopped = True
        raise


def _failure_kind(exc):
    """Fixed categories only; driver messages and arbitrary class names are private."""
    if isinstance(exc, BrowserSafetyStop):
        return "browser_safety_stop"
    return next(
        (
            name
            for kind, name in (
                (ValueError, "value_error"),
                (TypeError, "type_error"),
                (TimeoutError, "timeout_error"),
                (OSError, "io_error"),
                (KeyboardInterrupt, "keyboard_interrupt"),
                (SystemExit, "system_exit"),
            )
            if isinstance(exc, kind)
        ),
        "unexpected_exception",
    )


def _failure_reason(exc):
    return (
        str(exc)
        if isinstance(exc, BrowserSafetyStop) and str(exc) in _SAFE_REASONS
        else "spectrum_capture_failed"
    )


def hover_spectrum_marker(session, side):
    if side not in {"blue", "red"}:
        raise ValueError("Select blue or red excursion marker explicitly")
    session._begin("HOVER", {"surface": "spectrum", "marker": side})
    try:
        label = session.frame.get_by_text("656.3nm", exact=True)
        if label.count() != 1 or not label.is_visible():
            raise BrowserSafetyStop("ambiguous_rest_wavelength")
        parent = label.locator("..")
        strips = [
            e
            for e in parent.locator(":scope > div").all()
            if e.is_visible() and e.evaluate("e=>getComputedStyle(e).backgroundColor==='rgb(193, 58, 44)'")
        ]
        if len(strips) != 1:
            raise BrowserSafetyStop("unsupported_spectral_strip")
        strip = strips[0]
        lines = [
            e
            for e in strip.locator(":scope > div").all()
            if e.is_visible()
            and e.evaluate(
                "e=>getComputedStyle(e).backgroundColor==='rgb(0, 0, 0)'&&e.getBoundingClientRect().width===1"
            )
        ]
        if len(lines) != 2:
            raise BrowserSafetyStop("ambiguous_visible_excursion_markers")
        lines.sort(key=lambda e: e.bounding_box()["x"])
        line = lines[0 if side == "blue" else 1]
        b = line.bounding_box()
        s = strip.bounding_box()
        session._spectrum_diagnostic = {
            "source": "rendered_geometry_and_visible_tooltip",
            "marker": side,
            "strip_box": s,
            "marker_boxes": [candidate.bounding_box() for candidate in lines],
            "requested_pointer": {"x": b["x"] + b["width"] / 2, "y": b["y"] + b["height"] / 2},
            "marker_readback": None,
            "last_visible_wavelength_text": None,
            "learned_perception": False,
            "scientific_verified": False,
        }
        if not (
            s["x"] <= b["x"]
            and b["x"] + b["width"] <= s["x"] + s["width"]
            and s["y"] <= b["y"]
            and b["y"] + b["height"] <= s["y"] + s["height"]
        ):
            raise BrowserSafetyStop("excursion_marker_outside_visible_strip")
        if not line.evaluate(VISIBLE_TOOLTIP):
            raise BrowserSafetyStop("excursion_marker_occluded")
        bound = {
            "strip": strip.element_handle(timeout=2000),
            "parent": parent.element_handle(timeout=2000),
            # Client-frame geometry is used only to verify the same native paint;
            # the pointer above uses Playwright's page-relative bounding box.
            "lineBox": line.evaluate(
                "e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}}"
            ),
            "stripBox": strip.evaluate(
                "e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}}"
            ),
        }
        line_handle = line.element_handle(timeout=2000)
        session.page.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
        text = _settled_reading(session, parent, line_handle, bound, side)
        reading = {
            "marker": side,
            "wavelength_text": text,
            "source": "visible_spectrum_tooltip",
        }
        session.emit(
            "action_result",
            {"sequence": session.actions, "spectrum_sample": reading, "task_completed": False},
        )
        return reading
    except BaseException:
        session.stopped = True
        raise


def capture_spectrum_excursion(
    page, config, output, *, expected_star, max_seconds=120, emit=lambda *_: None, marker_mode=None
):
    """Record exactly two ordinary hovers in the existing raster-input format.

    This is reference sensing, not learned perception. No browser state is
    restored after cancellation; partial evidence remains incomplete. The caller
    owns the page and passes the same star established by its window evidence.
    """
    from .browser_planet_chart import FluxChartSession
    from .planet_charts import spectrum_excursion

    if (
        not isinstance(expected_star, str)
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9 '-]{1,79}", expected_star)
        or type(max_seconds) not in {int, float}
        or not 1 <= max_seconds <= 120
        or not callable(emit)
        or marker_mode not in (None, GEOMETRIC_PAIR_MODE)
    ):
        raise ValueError("Explicit star and bounded spectrum capture required")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    events, session = [], None
    stage = "scope"

    def forward(kind, payload):
        event = {"kind": kind, "payload": payload}
        persist_json(directory / f"event-{len(events):02d}.json", event)
        events.append(event)
        try:
            emit(kind, payload)
        except Exception:  # noqa: BLE001 - callbacks cannot authorize a later hover after failure
            raise BrowserSafetyStop("spectrum_event_forwarding_failed") from None

    try:
        persist_json(
            directory / "scope.json",
            {
                "mode": "visible_spectrum_reference",
                "star": expected_star,
                "max_hovers": 2,
                "max_seconds": max_seconds,
                "answer_writes": 0,
                "learned_perception": False,
                "training_label": False,
                "automatic_retry": False,
                "tooltip_settle_seconds": _SETTLE_SECONDS,
                "stable_tooltip_reads": 2,
                "hover_evidence": "ordinary_native_hover_and_visible_tooltip",
                "native_event_delivery_verified": False,
                **({"marker_mode": marker_mode} if marker_mode is not None else {}),
            },
        )
        stage = "session_initialization"
        session = FluxChartSession(page, config, forward, max_actions=2, max_seconds=max_seconds)
        if session.star.casefold() != expected_star.casefold():
            raise BrowserSafetyStop("spectrum_capture_star_changed")
        if marker_mode == GEOMETRIC_PAIR_MODE:
            stage = "geometric_pair_hover"
            pair = read_geometric_spectrum_pair(session)
            blue, red = ({"wavelength_text": pair["readings"][side]} for side in ("blue", "red"))
        else:
            stage = "blue_hover"
            blue = hover_spectrum_marker(session, "blue")
            stage = "red_hover"
            red = hover_spectrum_marker(session, "red")
        stage = "final_context_guard"
        session._guard()
        stage = "excursion_validation"
        try:
            excursion = (
                pair["result"]
                if marker_mode == GEOMETRIC_PAIR_MODE
                else spectrum_excursion("656.3nm", blue["wavelength_text"], red["wavelength_text"])
            )
        except ValueError as exc:
            if str(exc) in _VALUE_ERRORS:
                raise BrowserSafetyStop("spectrum_" + str(exc)) from None
            raise
        result = {
            "events": events,
            "result": excursion,
            "visibility": "full_glyph_and_occlusion_checked",
            **({"marker_mode": marker_mode} if marker_mode is not None else {}),
        }
        stage = "spectrum_persistence"
        persist_json(directory / "spectrum.json", result)
        return result
    except BaseException as exc:
        reason = _failure_reason(exc)
        persist_json(
            directory / "stopped.json",
            {
                "reason": reason,
                "failure_stage": stage,
                "failure_kind": _failure_kind(exc),
                "visible_diagnostic": getattr(session, "_spectrum_diagnostic", None),
                "hover_actions_started": session.actions if session else 0,
                "answer_writes": 0,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        if isinstance(exc, Exception) and (not isinstance(exc, BrowserSafetyStop) or str(exc) != reason):
            raise BrowserSafetyStop(reason) from None
        raise
    finally:
        if session:
            session.close()
