"""One-shot public outcome evidence after an uncertain Submit, never a receipt.

Action guards are deliberately not reused as an outcome-shape recognizer: the
known application frames may have been replaced or removed, and an HTML dialog
may be the response. Privacy boundaries, however, remain strict before and after
every read. Native dialog messages/default values are never read.
"""

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from .browser_probe import BrowserProbeConfig, _url_identity, _visible_frame, public_url
from .presentation_capture import evidence_screenshot

MAX_FRAMES = 8
MAX_IMAGE_BYTES = 16_000_000
_AUTH_INPUTS = (
    'input[type="password"], input[type="email"], input[autocomplete="username"], '
    'input[autocomplete="current-password"], input[autocomplete="new-password"], '
    'input[autocomplete="one-time-code"]'
)
_AUTH_BUTTON = re.compile(r"^(?:sign[ -]?in|log[ -]?in|login)$", re.IGNORECASE)
_DIALOG_TYPES = {"alert", "confirm", "prompt", "beforeunload"}


class _Refused(Exception):
    pass


def native_dialog_disposition(dialog):
    """Only the public dialog type is safe without validated page context."""
    try:
        kind = dialog.type
    except Exception:  # noqa: BLE001 - never expose a driver exception or prompt value
        kind = "unknown"
    return {
        "type": kind if isinstance(kind, str) and kind in _DIALOG_TYPES else "unknown",
        "message_disposition": "withheld_unverified_native_dialog_context",
    }


def _write(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def preserve_submission_outcome(
    page,
    config,
    output,
    *,
    deadline,
    original_failure,
    native_dialogs=(),
    popup_observed=False,
    cancelled=lambda: False,
    _clock=time.monotonic,
):
    """Persist one diagnostic/disposition under the caller's existing deadline.

    Only the caller can authorize invocation after its durable Submit dispatch
    marker. A fresh exclusive directory prevents overwrites. Returned metadata has
    no visible content; files inside remain diagnostic evidence, never model input
    or authoritative acknowledgement. No retry, action, input value or hidden app
    state is read here.
    """
    path = Path(output).absolute()
    if path != path.resolve() or not path.parent.is_dir():
        raise ValueError("invalid_outcome_diagnostic_directory")
    path.mkdir(exist_ok=False)
    reason = (
        original_failure
        if isinstance(original_failure, str) and re.fullmatch(r"[a-z][a-z0-9_:.-]{0,150}", original_failure)
        else "submission_failure"
    )
    record = {
        "schema_version": 1,
        "mode": "failure_only_public_submission_outcome",
        "original_failure": reason,
        "consistency_unverified": True,
        "authority": "diagnostic_only_not_submission_acknowledgement",
        "browser_actions": 0,
        "submission_verified": False,
        "acknowledgement_interpreted": False,
        "submitted": False,
        "task_completed": False,
        "project_completed": False,
        "automatic_retry": False,
        "content_saved": False,
        "source_sha256": {},
    }
    pins = []

    def timeout():
        if cancelled():
            raise _Refused("cancelled")
        remaining = deadline - _clock()
        if remaining <= 0:
            raise _Refused("deadline_exhausted")
        return max(1, min(3000, int(remaining * 1000)))

    def guard():
        timeout()
        if native_dialogs:
            raise _Refused("native_dialog_content_withheld")
        popup = popup_observed() if callable(popup_observed) else popup_observed
        if popup or len(page.context.pages) != 1 or page.context.pages[0] is not page:
            raise _Refused("unexpected_popup")
        if not config.allows(page.url):
            raise _Refused("navigation_outside_activity")
        frames = list(page.frames)
        if not 1 <= len(frames) <= MAX_FRAMES or page.main_frame not in frames:
            raise _Refused("frame_budget_or_identity")
        visible = [
            frame for frame in frames if frame is page.main_frame or _visible_frame(frame, page.main_frame)
        ]
        allowed = {_url_identity(rule.url) for rule in config.frames}
        # Inspect every visible URL before reading ANY body, including the main
        # body, so an unexpected embedded login cannot leak into the viewport.
        for frame in visible:
            if frame is not page.main_frame:
                try:
                    known = _url_identity(frame.url) in allowed
                except ValueError:
                    known = False
                if not known:
                    raise _Refused("unknown_visible_frame")
        for frame in visible:
            inputs = frame.locator(_AUTH_INPUTS).all()
            buttons = frame.get_by_role("button", name=_AUTH_BUTTON).all()
            links = frame.get_by_role("link", name=_AUTH_BUTTON).all()
            if len(inputs) + len(buttons) + len(links) > 32 or any(
                item.is_visible() for item in inputs + buttons + links
            ):
                raise _Refused("authentication_required")
        timeout()
        # Protocol reads above can deliver navigation/frame/popup events. Do not
        # act on a cached safe URL or a privacy check for replaced frame objects.
        popup = popup_observed() if callable(popup_observed) else popup_observed
        if native_dialogs:
            raise _Refused("native_dialog_content_withheld")
        if popup or len(page.context.pages) != 1 or page.context.pages[0] is not page:
            raise _Refused("unexpected_popup")
        if not config.allows(page.url):
            raise _Refused("navigation_outside_activity")
        if list(page.frames) != frames:
            raise _Refused("privacy_context_changed")
        if [
            frame for frame in frames if frame is page.main_frame or _visible_frame(frame, page.main_frame)
        ] != visible:
            raise _Refused("privacy_context_changed")
        for frame in visible:
            if frame is not page.main_frame and _url_identity(frame.url) not in allowed:
                raise _Refused("unknown_visible_frame")
        return visible

    def content_guard():
        current = guard()
        if current != [frame for frame, _identity, _url in pins] or any(
            _url_identity(frame.url, preview=frame is page.main_frame and config.mode == "preview")
            != identity
            for frame, identity, _url in pins
        ):
            raise _Refused("capture_context_changed")

    def content_read(operation):
        content_guard()
        value = operation(timeout())
        content_guard()
        return value

    try:
        if (
            not isinstance(config, BrowserProbeConfig)
            or type(deadline) not in {int, float}
            or not math.isfinite(deadline)
        ):
            raise _Refused("invalid_diagnostic_scope")
        timeout()
        if native_dialogs:
            # A blocking native dialog prevents reliable authentication checks.
            # Do not query its message/default value, accept/dismiss, or read DOM.
            raise _Refused("native_dialog_content_withheld")
        visible = guard()
        pins = [
            (
                frame,
                _url_identity(frame.url, preview=frame is page.main_frame and config.mode == "preview"),
                frame.url,
            )
            for frame in visible
        ]
        frames = []
        budget = 0
        for frame, _identity, captured_url in pins:
            body = frame.locator("body")
            text = content_read(lambda milliseconds, body=body: body.inner_text(timeout=milliseconds))
            ax = content_read(lambda milliseconds, body=body: body.aria_snapshot(timeout=milliseconds))
            if not isinstance(text, str) or not isinstance(ax, str):
                raise _Refused("invalid_public_text")
            budget += len(text) + len(ax)
            if budget > config.max_text_chars:
                raise _Refused("public_text_budget")
            # URL queries may carry preview/session identifiers. Other text is
            # student-visible content; never use it as a receipt or instruction.
            for known_url in (config.url, captured_url):
                text, ax = (
                    text.replace(known_url, public_url(known_url)),
                    ax.replace(known_url, public_url(known_url)),
                )
            frames.append(
                {
                    "surface": "outer" if frame is page.main_frame else "known_frame",
                    "url": public_url(captured_url),
                    "text": text,
                    "accessibility": ax,
                }
            )
        png = content_read(
            lambda milliseconds: evidence_screenshot(page, full_page=False, timeout=milliseconds)
        )
        if (
            not isinstance(png, bytes)
            or not png.startswith(b"\x89PNG\r\n\x1a\n")
            or len(png) > MAX_IMAGE_BYTES
        ):
            raise _Refused("invalid_viewport_image")
        content_guard()
        # Changing public outcomes need not have identical text across reads.
        # All bytes remain in memory until the final privacy check succeeds.
        raw = (json.dumps({"frames": frames, "consistency_unverified": True}, indent=2) + "\n").encode()
        _write(path / "visible-outcome.json", raw)
        _write(path / "viewport.png", png)
        record.update(
            disposition="public_outcome_captured",
            content_saved=True,
            source_sha256={
                "visible-outcome.json": hashlib.sha256(raw).hexdigest(),
                "viewport.png": hashlib.sha256(png).hexdigest(),
            },
        )
    except _Refused as exc:
        record["disposition"] = str(exc)
    except Exception:  # noqa: BLE001 - no private driver exception text or URLs in dispositions
        record["disposition"] = "public_capture_unavailable"
    record["native_dialogs"] = [
        {
            "type": item["type"]
            if isinstance(item.get("type"), str) and item["type"] in _DIALOG_TYPES
            else "unknown",
            "message_disposition": "withheld_unverified_native_dialog_context",
        }
        for item in native_dialogs[:MAX_FRAMES]
        if isinstance(item, dict)
    ]
    # A filesystem failure can leave a safe but partial diagnostic artifact.
    # Describe what actually exists rather than silently claiming no content.
    record["source_sha256"] = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in ("visible-outcome.json", "viewport.png")
        if (path / name).is_file()
    }
    record["content_saved"] = bool(record["source_sha256"])
    _write(path / "disposition.json", (json.dumps(record, indent=2) + "\n").encode())
    return record
