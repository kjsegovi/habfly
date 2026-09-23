"""Read-only application adapter: all mutations happen through ordinary UI actions."""

import hashlib
import json
import logging
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

from playwright.sync_api import Error as PlaywrightError
from pydantic import Field, model_validator

from .contracts import Action, ActionKind, Contract, Control, Observation, StepResult, validate_action

logger = logging.getLogger(__name__)


def _activity_path(path):
    """Decode paths before checking the boundary; reject ambiguous traversal."""
    for _ in range(5):
        decoded = unquote(path, errors="strict")
        if decoded == path:
            break
        path = decoded
    else:
        raise ValueError("Too many URL encoding layers")
    if not path.startswith("/") or "\\" in path or any(ord(char) < 32 for char in path):
        raise ValueError("Invalid activity path")
    if any(segment in {".", ".."} for segment in path.split("/")):
        raise ValueError("Path traversal is outside the activity boundary")
    return path


class BrowserConfig(Contract):
    url: str
    allowed_path: str = "/"
    max_steps: int = Field(default=300, gt=0)
    max_seconds: float = Field(default=600, gt=0)
    action_interval: float = Field(default=0.25, ge=0)
    repeat_limit: int = Field(default=4, gt=0)
    allow_submission: bool = False
    submission_labels: list[str] = Field(
        default_factory=lambda: ["Submit Project", "I am ready to submit project"]
    )
    completion_text: str = "Project submitted"
    chart_selector: str = "canvas, svg[role=img], [role=img][aria-label*=lightcurve]"
    tooltip_selector: str = '[role="tooltip"]'
    allowed_dialog_labels: list[str] = Field(default_factory=list)
    artifact_dir: Path = Path("experiments/browser")
    storage_state: Path | None = None
    headless: bool = False

    @model_validator(mode="after")
    def local_target(self):
        url = urlsplit(self.url)
        if url.scheme not in {"http", "https"} or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Browser acceptance targets must use a local loopback URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Use a credential-free URL without query or fragment")
        if any(char in self.allowed_path for char in "?#"):
            raise ValueError("allowed_path must not contain a query or fragment")
        _activity_path(self.allowed_path)
        if not self.submission_labels:
            raise ValueError("Declare final submission labels")
        if not self.allows(self.url):
            raise ValueError("Start URL is outside allowed_path")
        return self

    def allows(self, url):
        try:
            target, base = urlsplit(url), urlsplit(self.url)
            path = _activity_path(target.path or "/")
            prefix = _activity_path(self.allowed_path).rstrip("/")
        except (ValueError, UnicodeError):
            return False
        if target.username is not None or target.password is not None:
            return False
        return (target.scheme, target.netloc) == (base.scheme, base.netloc) and (
            not prefix or path == prefix or path.startswith(prefix + "/")
        )


class BrowserSafetyStop(RuntimeError):
    pass


class TorusBrowser:
    def __init__(self, config: BrowserConfig, page=None):
        self.config, self.page = config, page
        self._owner = page is None
        self.playwright = self.browser = self.context = None
        self.revision = self.steps = 0
        self.total_reward = 0.0
        self.last_time = 0.0
        self.started = time.monotonic()
        self.repeat = 0
        self.previous_signature = None
        self.safety_stop = None
        self.observation = None
        self.targets = {}
        self.run_dir = None
        self._tracing = False
        self._closed = False

    def start(self):
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        import uuid

        self.run_dir = self.config.artifact_dir / uuid.uuid4().hex
        self.run_dir.mkdir()
        try:
            if self._owner:
                from playwright.sync_api import sync_playwright

                self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(headless=self.config.headless)
                self.context = self.browser.new_context(storage_state=self.config.storage_state)
                self.page = self.context.new_page()
            else:
                self.context = self.page.context
            self.context.tracing.start(screenshots=True, snapshots=True, sources=False)
            self._tracing = True
            self.page.on("dialog", self._dialog)
            self.context.on("page", self._popup)
            # Context routes also see the initial request from a popup.
            self.context.route("**/*", self._route)
            self.started = time.monotonic()
            if self._owner:
                self.page.goto(self.config.url, wait_until="domcontentloaded")
            self.observation = self.observe()
            return self.observation
        except Exception as exc:
            if isinstance(exc, BrowserSafetyStop):
                self.safety_stop = str(exc)
            self.close()
            raise

    def _route(self, route):
        req = route.request
        if req.is_navigation_request():
            try:
                frame = req.frame
            except PlaywrightError:
                # Playwright has no frame for an initial popup navigation yet.
                self.safety_stop = "unexpected_popup"
                route.abort()
                return
            if frame.parent_frame is None:
                if frame != self.page.main_frame:
                    self.safety_stop = "unexpected_popup"
                elif not self.config.allows(req.url):
                    self.safety_stop = "navigation_outside_activity"
                if self.safety_stop:
                    route.abort()
                    return
        route.continue_()

    def _popup(self, popup):
        if popup != self.page:
            self.safety_stop = "unexpected_popup"
            popup.close()

    def _dialog(self, dialog):
        self.safety_stop = "unexpected_browser_dialog"
        dialog.dismiss()

    def _guard(self):
        if self.safety_stop:
            raise BrowserSafetyStop(self.safety_stop)
        if not self.config.allows(self.page.url):
            raise BrowserSafetyStop("navigation_outside_activity")
        if time.monotonic() - self.started >= self.config.max_seconds:
            raise BrowserSafetyStop("time_limit")
        for item in self.page.locator('input[type="password"]').all():
            if item.is_visible():
                raise BrowserSafetyStop("authentication_required")
        for item in self.page.get_by_role("dialog").all():
            if item.is_visible() and self._label(item) not in self.config.allowed_dialog_labels:
                raise BrowserSafetyStop("unexpected_modal")

    def _label(self, locator):
        for attr in ("aria-label", "title"):
            value = locator.get_attribute(attr)
            if value:
                return value.strip()
        # Playwright's accessibility snapshot resolves labelledby and native labels.
        snapshot = locator.aria_snapshot()
        import re

        match = re.match(r'- [\w -]+ "((?:[^"\\]|\\.)*)"', snapshot)
        if match:
            try:
                return json.loads('"' + match.group(1) + '"')
            except json.JSONDecodeError:
                return match.group(1)
        return locator.inner_text().strip() or locator.get_attribute("placeholder") or "Unlabelled control"

    def observe(self):
        self._guard()
        controls, targets = [], {}
        specs = [
            ("button", self.page.get_by_role("button"), [ActionKind.CLICK]),
            ("link", self.page.get_by_role("link"), [ActionKind.CLICK]),
            ("textbox", self.page.get_by_role("textbox"), [ActionKind.TYPE, ActionKind.KEYPRESS]),
            ("spinbutton", self.page.get_by_role("spinbutton"), [ActionKind.TYPE, ActionKind.KEYPRESS]),
            ("combobox", self.page.get_by_role("combobox"), [ActionKind.SELECT]),
            ("checkbox", self.page.get_by_role("checkbox"), [ActionKind.CLICK]),
            ("radio", self.page.get_by_role("radio"), [ActionKind.CLICK]),
            (
                "chart",
                self.page.locator(self.config.chart_selector),
                [ActionKind.HOVER, ActionKind.DRAG, ActionKind.SCROLL],
            ),
        ]
        for role, locator, actions in specs:
            for item in locator.all():
                if not item.is_visible() or item.get_attribute("aria-hidden") == "true":
                    continue
                label = self._label(item)
                key = f"{self.revision}:{len(controls)}"
                value, options = "", []
                if role in {"textbox", "spinbutton", "combobox"}:
                    tag = item.evaluate("element => element.tagName.toLowerCase()")
                    if role == "combobox" and tag != "select":
                        raise BrowserSafetyStop("unsupported_custom_combobox")
                    if tag in {"input", "textarea", "select"}:
                        value = item.input_value()
                    elif item.get_attribute("contenteditable") in {"true", ""}:
                        value = item.inner_text()
                    else:
                        raise BrowserSafetyStop("unsupported_custom_textbox")
                if role in {"checkbox", "radio"}:
                    value = str(item.is_checked()).lower()
                if role == "combobox":
                    for option in item.locator("option").all():
                        disabled_group = option.locator("xpath=ancestor::optgroup[@disabled]").count()
                        if (
                            option.get_attribute("disabled") is None
                            and not disabled_group
                            and option.get_attribute("hidden") is None
                            and option.get_attribute("aria-hidden") != "true"
                        ):
                            options.append(option.inner_text().strip())
                controls.append(
                    Control(
                        id=key,
                        label=label,
                        role=role,
                        value=value,
                        options=options,
                        enabled=item.is_enabled(),
                        actions=actions,
                    )
                )
                targets[key] = item
        visible_text = self.page.locator("body").inner_text()
        tooltip = " ".join(
            t.inner_text() for t in self.page.locator(self.config.tooltip_selector).all() if t.is_visible()
        )
        chart = {"tooltip_text": tooltip} if tooltip else {}
        crop = None
        chart_control = next((c for c in controls if c.role == "chart"), None)
        if chart_control and not tooltip and self.run_dir:
            crop_path = self.run_dir / f"chart-{self.revision:05}.png"
            targets[chart_control.id].screenshot(path=str(crop_path))
            crop = str(crop_path.resolve())
        modalities = ["text", "controls"] + (["chart_pixels"] if crop else [])
        observation = Observation(
            revision=self.revision,
            instruction=visible_text,
            controls=controls,
            feedback=" ".join(
                x.inner_text()
                for x in self.page.locator('[role="status"], [role="alert"]').all()
                if x.is_visible()
            ),
            chart=chart,
            chart_crop=crop,
            modalities=modalities,
            progress={"submitted": self.config.completion_text in visible_text},
        )
        self.targets = targets
        return observation

    @staticmethod
    def fingerprint(observation):
        obj = observation.model_dump(exclude={"revision", "chart_crop"})
        for control in obj["controls"]:
            control.pop("id")
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

    def step(self, action):
        action = Action.model_validate(action)
        if self.observation is None:
            raise RuntimeError("start the browser first")
        reason, terminated, truncated = None, False, False
        before = self.observation
        try:
            self._guard()
            if self.steps >= self.config.max_steps:
                raise BrowserSafetyStop("step_limit")
            refreshed = self.observe()
            if self.fingerprint(refreshed) != self.fingerprint(before):
                raise BrowserSafetyStop("stale_page_observation")
            control = validate_action(before, action)
            if not self.config.allow_submission:
                final = {label.casefold() for label in self.config.submission_labels}
                if control and (
                    control.label.casefold() in final
                    or (
                        action.kind == ActionKind.KEYPRESS
                        and action.value == "Enter"
                        and any(c.label.casefold() in final for c in before.controls)
                    )
                ):
                    raise BrowserSafetyStop("submission_disabled")
            delay = self.config.action_interval - (time.monotonic() - self.last_time)
            if delay > 0:
                self.page.wait_for_timeout(delay * 1000)
            if action.kind == ActionKind.STOP:
                raise BrowserSafetyStop("agent_stopped")
            if action.kind != ActionKind.WAIT:
                item = self.targets[action.target]
                if action.kind == ActionKind.CLICK:
                    if item.get_attribute("target") == "_blank":
                        raise BrowserSafetyStop("unexpected_popup")
                    item.click(timeout=3000)
                elif action.kind == ActionKind.TYPE:
                    item.fill(action.value, timeout=3000)
                elif action.kind == ActionKind.SELECT:
                    item.select_option(label=action.value, timeout=3000)
                elif action.kind == ActionKind.KEYPRESS:
                    if action.value not in {"Enter", "Tab", "ArrowUp", "ArrowDown", "Escape"}:
                        raise BrowserSafetyStop("key_not_allowed")
                    item.press(action.value, timeout=3000)
                else:
                    bounds = item.bounding_box()
                    if not bounds:
                        raise BrowserSafetyStop("target_has_no_bounds")
                    x = bounds["x"] + (action.x if action.x is not None else 0.5) * bounds["width"]
                    y = bounds["y"] + (action.y if action.y is not None else 0.5) * bounds["height"]
                    self.page.mouse.move(x, y)
                    if action.kind == ActionKind.SCROLL:
                        self.page.mouse.wheel((action.dx or 0) * 400, (action.dy or 0) * 400)
                    elif action.kind == ActionKind.DRAG:
                        self.page.mouse.down()
                        try:
                            end_x = max(
                                bounds["x"],
                                min(
                                    bounds["x"] + bounds["width"] - 1, x + (action.dx or 0) * bounds["width"]
                                ),
                            )
                            end_y = max(
                                bounds["y"],
                                min(
                                    bounds["y"] + bounds["height"] - 1,
                                    y + (action.dy or 0) * bounds["height"],
                                ),
                            )
                            self.page.mouse.move(end_x, end_y, steps=5)
                        finally:
                            self.page.mouse.up()
                # Deliver queued popup/dialog events and asynchronous pointer updates
                # before treating the resulting observation as safe to act upon.
                self.page.wait_for_timeout(50)
            self.steps += 1
            self.last_time = time.monotonic()
            self.revision += 1
            self.observation = self.observe()
            signature = self.fingerprint(self.observation)
            self.repeat = self.repeat + 1 if signature == self.fingerprint(before) else 0
            if self.repeat >= self.config.repeat_limit:
                raise BrowserSafetyStop("repeated_unchanged_state")
            terminated = self.observation.progress["submitted"]
            if self.steps >= self.config.max_steps and not terminated:
                raise BrowserSafetyStop("step_limit")
        except (BrowserSafetyStop, ValueError, PlaywrightError, OSError) as exc:
            reason = self.safety_stop or (
                str(exc) if isinstance(exc, (BrowserSafetyStop, ValueError)) else type(exc).__name__
            )
            self.safety_stop, truncated = reason, True
            # Never capture a login form or storage state as a failure artifact.
            if reason != "authentication_required" and self.run_dir:
                try:
                    self.page.screenshot(path=str(self.run_dir / "failure.png"))
                except (PlaywrightError, OSError) as screenshot_error:
                    logger.warning(
                        "Unable to capture browser failure evidence: %s", type(screenshot_error).__name__
                    )
        result = StepResult(
            observation=self.observation,
            terminated=terminated,
            truncated=truncated,
            failure_reason=reason,
            steps=self.steps,
        )
        if self.run_dir:
            with (self.run_dir / "actions.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "observation": before.model_dump(mode="json"),
                            "action": action.model_dump(mode="json"),
                            "result": result.model_dump(mode="json"),
                        }
                    )
                    + "\n"
                )
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.context and self._tracing:
            try:
                if self.safety_stop == "authentication_required":
                    # Authentication pages can contain saved credentials. Discard
                    # the entire in-memory trace rather than exporting its frames.
                    self.context.tracing.stop()
                else:
                    self.context.tracing.stop(path=str(self.run_dir / "trace.zip"))
            except (PlaywrightError, OSError) as exc:
                logger.warning("Unable to finish browser trace: %s", type(exc).__name__)
            self._tracing = False
        if self.context:
            try:
                self.context.unroute("**/*", self._route)
                self.context.remove_listener("page", self._popup)
            except PlaywrightError as exc:
                logger.warning("Unable to remove browser context callbacks: %s", type(exc).__name__)
        if self.page:
            try:
                self.page.remove_listener("dialog", self._dialog)
            except PlaywrightError as exc:
                logger.warning("Unable to remove browser page callbacks: %s", type(exc).__name__)
        if self._owner:
            if self.browser:
                try:
                    self.browser.close()
                except PlaywrightError as exc:
                    logger.warning("Unable to close browser: %s", type(exc).__name__)
            if self.playwright:
                try:
                    self.playwright.stop()
                except PlaywrightError as exc:
                    logger.warning("Unable to stop Playwright: %s", type(exc).__name__)
