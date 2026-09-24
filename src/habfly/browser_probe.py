"""Read-only, frame-scoped preflight. Deliberately has no action execution API.

This is not a bridge from the local stellar checkpoint to HabWorlds. Only rendered
text/accessibility information is recorded; unknown frame bodies are never read.
"""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from pydantic import Field, model_validator

from .browser import BrowserSafetyStop, _activity_path, normalized_control_label
from .contracts import Contract


def _url_identity(url: str, *, preview: bool = False):
    parsed = urlsplit(url)
    _ = parsed.port  # Validate malformed/out-of-range ports before launching a browser.
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(ord(char) < 33 for char in url)
    ):
        raise ValueError("Expected a credential-free HTTP(S) activity URL")
    query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    if query and (
        not preview
        or len(query) != 1
        or query[0][0] != "preview_sequence_id"
        or not re.fullmatch(r"[A-Za-z0-9_:-]{1,128}", query[0][1])
    ):
        raise ValueError("Only one explicit preview_sequence_id query value is supported")
    return parsed.scheme, parsed.netloc, _activity_path(parsed.path or "/"), tuple(query)


def public_url(url: str) -> str:
    """Do not persist session queries, fragments, or URL credentials."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return "[non-http frame]"
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        port = ""
    return urlunsplit((parsed.scheme, host + port, parsed.path, "", ""))


class ProbeFrame(Contract):
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    url: str
    count: int = Field(default=1, ge=1, le=10)
    required_text: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_url(self):
        _url_identity(self.url)
        return self


class BrowserProbeConfig(Contract):
    mode: Literal["preview", "delivery"] = "preview"
    url: str
    frames: list[ProbeFrame] = Field(min_length=1)
    allow_submission: Literal[False] = False
    max_text_chars: int = Field(default=40000, ge=100, le=100000)
    max_controls: int = Field(default=250, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_boundary(self):
        _url_identity(self.url, preview=self.mode == "preview")
        if urlsplit(self.url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("The outer activity must use a local loopback URL")
        if len({item.name for item in self.frames}) != len(self.frames):
            raise ValueError("Frame names must be unique")
        if len({item.url for item in self.frames}) != len(self.frames):
            raise ValueError("Declare same-URL siblings with count, not duplicate rules")
        return self

    def allows(self, url: str) -> bool:
        try:
            return _url_identity(url, preview=self.mode == "preview") == _url_identity(
                self.url, preview=self.mode == "preview"
            )
        except ValueError:
            return False


def boundary_diagnostic(config: BrowserProbeConfig, url: str) -> dict:
    """Explain an address mismatch without logging session/query/credential values."""
    expected = _url_identity(config.url, preview=config.mode == "preview")
    try:
        actual = _url_identity(url, preview=config.mode == "preview")
        changed = [
            label
            for label, left, right in zip(
                ("scheme", "host_or_port", "path", "preview_sequence_id"), expected, actual
            )
            if left != right
        ]
    except ValueError:
        changed = ["unsupported_url_components"]
    try:
        current = public_url(url)
    except ValueError:
        current = "[unparseable URL]"
    return {
        "expected": public_url(config.url),
        "current": current,
        "different_components": changed,
        "query_values": "redacted",
    }


def capture_interactively(
    page, config: BrowserProbeConfig, *, confirm, echo, attempts: int = 3, new_test_session: bool = False
) -> dict | None:
    """Allow a bounded human correction without changing the pinned boundary.

    There are no UI actions here. The human may finish login or return to the
    exact requested URL, then ask for another read. No artifact is saved on error.
    """
    if not 1 <= attempts <= 3:
        raise ValueError("Capture attempts must be between one and three")
    recoverable = {
        "navigation_outside_activity",
        "authentication_required",
        "unexpected_modal",
    }
    for attempt in range(attempts):
        if not confirm("Is the selected star's stellar tab visible and ready for read-only capture?"):
            return None
        # stdin confirmation blocks the sync driver's event loop. page.url and
        # context.pages are cached: drain queued navigation/frame events with a
        # zero-duration protocol round trip before reading either. This is not a
        # loading delay; inspect_page still enforces exact URL/frame readiness.
        page.wait_for_timeout(0)
        if len(page.context.pages) != 1:
            raise BrowserSafetyStop("unexpected_popup")
        try:
            report = inspect_page(page, config)
            report["setup_mode"] = "independent_test_session" if new_test_session else "existing_state"
            return report
        except BrowserSafetyStop as exc:
            reason = str(exc)
            if reason not in recoverable and not reason.startswith(
                ("frame_count_mismatch:", "frame_not_ready:")
            ):
                raise
            echo(f"No capture saved: {reason}")
            if reason == "navigation_outside_activity":
                echo(json.dumps(boundary_diagnostic(config, page.url), indent=2))
                echo(
                    "After signing in, paste the ORIGINAL --url into this Chromium tab. "
                    "Do not create another preview URL from the dashboard. If the original link cannot "
                    "open the intended activity, answer no and report the mismatch. "
                    "The configured URL has not been changed."
                )
            if attempt + 1 == attempts:
                raise BrowserSafetyStop("capture_attempt_limit") from None
            echo(
                f"Chromium stays open. Finish sign-in, return to the selected star's stellar tab, "
                f"or close unexpected panels before retrying ({attempts - attempt - 1} attempts left). "
                "Do not reset, edit answers, assess, save, update score or submit."
            )
            if not new_test_session:
                echo("Existing-state mode: do not collect a replacement star.")
    return None


PROTECTED_LABELS = {
    normalized_control_label(label)
    for label in ("Submit Project", "I am ready to submit project.", "Update Score", "Assess", "Delete star")
}


def _visible_frame(frame, main_frame) -> bool:
    """Check the complete embedding chain, including hidden parent iframes."""
    while frame != main_frame:
        if frame.parent_frame is None or not frame.frame_element().is_visible():
            return False
        frame = frame.parent_frame
    return True


def _check_auth_and_modals(frame):
    if any(item.is_visible() for item in frame.locator('input[type="password"]').all()):
        raise BrowserSafetyStop("authentication_required")
    if any(item.is_visible() for item in frame.get_by_role("dialog").all()):
        raise BrowserSafetyStop("unexpected_modal")


def _read_controls(frame, frame_id: str, config: BrowserProbeConfig):
    controls = []
    for role in ("button", "link", "textbox", "spinbutton", "combobox", "checkbox", "radio"):
        for item in frame.get_by_role(role).all():
            if not item.is_visible() or item.get_attribute("aria-hidden") == "true":
                continue
            snapshot = item.aria_snapshot(timeout=3000)
            label = snapshot.splitlines()[0] if snapshot else "Unlabelled control"
            controls.append(
                {
                    "id": f"{frame_id}:c{len(controls)}",
                    "role": role,
                    "accessibility": snapshot,
                    "enabled": item.is_enabled(),
                    # Evidence inventory only: these IDs cannot be executed by TorusBrowser.
                    "actions": [],
                    "protected": False,
                }
            )
            normalized = normalized_control_label(label)
            controls[-1]["protected"] = any(name in normalized for name in PROTECTED_LABELS)
            if role in {"textbox", "spinbutton", "combobox"}:
                # Rendered native value is distinct from an accessible name/placeholder.
                # Do not read custom widgets' hidden state or arbitrary properties.
                tag = item.evaluate("element => element.tagName.toLowerCase()")
                controls[-1]["value"] = item.input_value() if tag in {"input", "textarea", "select"} else None
            if len(controls) > config.max_controls:
                raise BrowserSafetyStop("control_budget_exceeded")
    return controls


def _read_frame(frame, frame_id: str, config: BrowserProbeConfig):
    _check_auth_and_modals(frame)
    body = frame.locator("body")
    # Visible text/AX only: no scripts, app state, data-* attributes or answer keys.
    text, accessibility = body.inner_text(timeout=3000), body.aria_snapshot(timeout=3000)
    if max(len(text), len(accessibility)) > config.max_text_chars:
        raise BrowserSafetyStop("observation_budget_exceeded")
    return {
        "id": frame_id,
        "url": public_url(frame.url),
        "text": text,
        "accessibility": accessibility,
        "controls": _read_controls(frame, frame_id, config),
    }


def inspect_page(page, config: BrowserProbeConfig) -> dict:
    """Capture one current screen without navigation, clicks, typing, or scoring.

    Authentication/bootstrap is owned by the human. Only known frame URLs are
    eligible for content extraction; frame identity/count/readiness is fail-closed.
    No screenshot, browser trace, cookies or storage state is exported.
    """
    if not config.allows(page.url):
        raise BrowserSafetyStop("navigation_outside_activity")
    _check_auth_and_modals(page.main_frame)
    outer_controls = _read_controls(page.main_frame, "outer", config)
    identities = {item.name: _url_identity(item.url) for item in config.frames}
    matches = {item.name: [] for item in config.frames}
    ignored = []
    for frame in page.frames:
        if frame == page.main_frame or not _visible_frame(frame, page.main_frame):
            continue
        try:
            identity = _url_identity(frame.url)
        except ValueError:
            identity = None
        rule = next((item for item in config.frames if identities[item.name] == identity), None)
        if rule is None:
            ignored.append(public_url(frame.url))
        else:
            matches[rule.name].append(frame)
    for rule in config.frames:
        if len(matches[rule.name]) != rule.count:
            raise BrowserSafetyStop(f"frame_count_mismatch:{rule.name}")
    captured = []
    for rule in config.frames:
        for index, frame in enumerate(matches[rule.name]):
            captured_frame = _read_frame(frame, f"{rule.name}-{index}", config)
            if any(text not in captured_frame["text"] for text in rule.required_text):
                raise BrowserSafetyStop(f"frame_not_ready:{rule.name}")
            captured.append(captured_frame)
    if not config.allows(page.url):
        raise BrowserSafetyStop("navigation_outside_activity")
    _check_auth_and_modals(page.main_frame)
    for rule in config.frames:
        for frame in matches[rule.name]:
            if frame.is_detached() or _url_identity(frame.url) != identities[rule.name]:
                raise BrowserSafetyStop("frame_changed_during_observation")
            _check_auth_and_modals(frame)
    return {
        "schema_version": 1,
        "mode": "read_only_browser_preflight",
        "session_mode": config.mode,
        "captured_at": datetime.now(UTC).isoformat(),
        "outer_url": public_url(page.url),
        "outer_controls": outer_controls,
        "frames": captured,
        "ignored_frame_urls": ignored,
        "actions_executed": 0,
        "allow_submission": False,
        "browser_acceptance_passed": False,
        "pending_gates": [
            "Verify visible stellar measurement and answer mappings in managed Chromium",
            "Obtain course H-R classification rules; classification is not learned",
            "Resolve wavelength-color boundaries and lifetime magnitude prefixes",
            "Implement and fixture-test the local-calculation/browser control bridge",
            "Verify one-star completion and persistence independently of submission",
        ],
    }


def save_probe(report: dict, output: Path) -> dict:
    """Create a new evidence directory; never overwrite an earlier inspection."""
    output.mkdir(parents=True, exist_ok=False)
    encoded = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode()
    (output / "observation.json").write_bytes(encoded)
    manifest = {
        "schema_version": 1,
        "observation_sha256": hashlib.sha256(encoded).hexdigest(),
        "mode": report["mode"],
        "actions_executed": 0,
        "browser_acceptance_passed": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
