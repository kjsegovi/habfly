"""Bounded, deterministic UI setup, separate from policy inference and its data.

No credentials, authentication-page snapshots, cookies, or application internals
are recorded. Only explicitly recognized UI transitions may be performed.
"""

import hashlib
import io
import os
import re
import time
from urllib.parse import urlsplit, urlunsplit

from .browser import BrowserSafetyStop
from .browser_numeric import screen_identity
from .browser_probe import _url_identity, _visible_frame, inspect_page
from .browser_stellar import SIMULATION_URL, StellarMappingError, map_stellar_capture
from .presentation_capture import evidence_screenshot


class SetupStop(RuntimeError):
    """Carries only an allowlisted local reason, never driver exception text."""


_LOGIN_OPERATIONS = frozenset(
    {
        "inspect_cookie_preferences",
        "close_cookie_preferences",
        "inspect_cookie_notice",
        "close_cookie_notice",
        "inspect_modal",
        "locate_email",
        "locate_password",
        "locate_submit",
        "validate_form_initial",
        "inspect_field_types",
        "signing_in_callback",
        "validate_form_before_email",
        "fill_email",
        "validate_form_after_email",
        "fill_password",
        "validate_form_after_password",
        "inspect_login_exposure",
        "waiting_for_login_ui_callback",
        "validate_form_before_submit",
        "submit_click",
    }
)


def _login_failure_code(operation, exception):
    """Classify without inspecting exception text, names, arguments or causes.

    These are fixed class-family labels, not arbitrary exception class names.
    In particular a submit-click timeout does not prove whether its request was
    sent, navigation began, or authentication succeeded. Setup still stops.
    """
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    if not isinstance(operation, str) or operation not in _LOGIN_OPERATIONS:
        operation = "unclassified"
    classes = (
        (PlaywrightTimeoutError, "playwright_timeout_error"),
        (PlaywrightError, "playwright_error"),
        (TimeoutError, "timeout_error"),
        (OSError, "os_error"),
        (ValueError, "value_error"),
        (TypeError, "type_error"),
        (RuntimeError, "runtime_error"),
    )
    label = next((name for kind, name in classes if isinstance(exception, kind)), "unknown_error")
    return f"setup_login_{operation}_{label}"


def _submit_timeout_stage(exception):
    """Discard a known driver's log after extracting only fixed progress labels.

    This is diagnostic, not evidence of authentication or permission to retry.
    Unknown/subclassed exceptions are never stringified. No matched line or
    dynamic pointer description is returned, stored, or emitted.
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    if type(exception) is not PlaywrightTimeoutError:
        return "unknown"
    lines = {re.sub(r"^\d+ ×\s*", "", line.strip().lstrip("- ")) for line in str(exception).splitlines()}
    if {"click action done", "waiting for scheduled navigations to finish"} <= lines:
        return "navigation_wait"
    if any(line.endswith(" intercepts pointer events") for line in lines):
        return "pointer_interception"
    if "performing click action" in lines:
        return "click_started"
    if lines & {
        "waiting for element to be visible, enabled and stable",
        "element is not visible",
        "element is not enabled",
        "element is not stable",
    }:
        return "actionability_wait"
    return "unknown"


# innerText/is_visible alone include opacity-zero, clipped and covered panels.
# Read only text whose painted location can actually be seen in this frame.
RENDER_VISIBILITY_JS = """
function styled(e) {
    if (!e || e.closest('script,style,template,[hidden],[aria-hidden="true"]')) return false;
    for (let p=e; p; p=p.parentElement) {
        const s=getComputedStyle(p);
        if (s.display==='none' || s.visibility!=='visible' || Number(s.opacity)===0) return false;
    }
    return true;
}
function exposed(rect, e, text=false) {
    let left=Math.max(0,rect.left), right=Math.min(innerWidth,rect.right);
    let top=Math.max(0,rect.top), bottom=Math.min(innerHeight,rect.bottom);
    let behind=false;
    for (let p=e.parentElement; p; p=p.parentElement) {
        const s=getComputedStyle(p), r=p.getBoundingClientRect();
        if (['hidden','clip','scroll','auto'].includes(s.overflowX)) {
            left=Math.max(left,r.left);right=Math.min(right,r.right);
        }
        if (['hidden','clip','scroll','auto'].includes(s.overflowY)) {
            top=Math.max(top,r.top);bottom=Math.min(bottom,r.bottom);
        }
        if (Number(s.zIndex)<0) behind=true;
    }
    if (right<=left || bottom<=top) return false;
    const hit=document.elementFromPoint((left+right)/2,(top+bottom)/2);
    // Non-interactive labels may intentionally pass pointer events to their
    // container. A sibling overlay is still an occluder, never a visible label.
    return hit===e || (text && !behind && !(Number(getComputedStyle(e).zIndex)<0) &&
        getComputedStyle(e).pointerEvents==='none' && hit && hit.contains(e));
}
"""


def rendered_text(frame):
    return frame.locator("body").evaluate(
        "root => {"
        + RENDER_VISIBILITY_JS
        + """
        const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT), text=[];
        while (walker.nextNode()) {
            const node=walker.currentNode, e=node.parentElement;
            if (!styled(e)) continue;
            const range=document.createRange();range.selectNodeContents(node);
            if ([...range.getClientRects()].some(r=>exposed(r,e,true))) text.push(node.textContent);
        }
        return text.join(' ').replace(/\\s+/g,' ').trim();
        }"""
    )


def rendered_control(control):
    return control.evaluate(
        "e => {" + RENDER_VISIBILITY_JS + "return styled(e) && exposed(e.getBoundingClientRect(),e); }"
    )


def visible_star_point(png, *, excluded=(), anchor=(0.4, 0.55)):
    """Pick a small bright dot from rendered pixels, never a hidden star catalog.

    Deliberately excludes the header, edges and footer. No measurement-dependent
    selection and no full-page image is sent to the model. Unknown layouts stop.
    """
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(png)).convert("RGB")
    width, height = image.size
    if not 400 <= width <= 2400 or not 300 <= height <= 1800:
        raise SetupStop("setup_unsupported_starfield_size")
    if (
        not isinstance(anchor, (tuple, list))
        or len(anchor) != 2
        or any(type(v) not in {int, float} or not 0.15 <= v <= 0.85 for v in anchor)
    ):
        raise SetupStop("setup_invalid_starfield_anchor")
    # Exclusions come from prior visible pixel selections in this unchanged
    # viewport, never from a hidden catalog or an inferred scientific class.
    if (
        not isinstance(excluded, (list, tuple))
        or len(excluded) > 500
        or any(
            not isinstance(point, dict)
            or set(point) != {"x", "y", "width", "height"}
            or point["width"] != width
            or point["height"] != height
            or any(
                type(point[k]) not in {int, float} or not 0 <= point[k] < bound
                for k, bound in (("x", width), ("y", height))
            )
            for point in excluded
        )
    ):
        raise SetupStop("setup_invalid_star_exclusion")
    pixels = np.asarray(image, dtype=np.int16)
    # The real CSS-scale capture has dim 1–2px cores (none reached the old
    # 165/channel cutoff). Require local contrast as well as compact shape.
    intensity = pixels.min(2)
    mask = (intensity >= 100) & (pixels.max(2) - intensity <= 65)
    mask[: max(100, height // 5)] = False
    mask[-max(40, height // 10) :] = False
    mask[:, : max(25, width // 25)] = False
    mask[:, -max(25, width // 25) :] = False
    if np.count_nonzero(mask) > width * height * 0.05:
        # Reject bright loading surfaces without flood-filling millions of pixels.
        raise SetupStop("setup_no_visible_star_candidate")
    candidates = []
    for y, x in zip(*np.where(mask)):
        if not mask[y, x]:
            continue
        stack, cluster = [(int(x), int(y))], []
        mask[y, x] = False
        while stack:
            cx, cy = stack.pop()
            cluster.append((cx, cy))
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx]:
                    mask[ny, nx] = False
                    stack.append((nx, ny))
        xs, ys = zip(*cluster)
        dx, dy = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        # Reject letters, panels, long lines and isolated compression specks.
        if 2 <= len(cluster) <= 80 and 1 <= dx <= 12 and 1 <= dy <= 12 and max(dx, dy) / min(dx, dy) <= 2:
            surrounding = intensity[
                max(0, min(ys) - 4) : min(height, max(ys) + 5), max(0, min(xs) - 4) : min(width, max(xs) + 5)
            ]
            contrast = min(intensity[py, px] for px, py in cluster) - float(np.median(surrounding))
            if contrast >= 60:
                candidates.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2))
    candidates = [
        p
        for p in candidates
        if all((p[0] - old["x"]) ** 2 + (p[1] - old["y"]) ** 2 > 12**2 for old in excluded)
    ]
    if not candidates:
        raise SetupStop("setup_no_visible_star_candidate")
    # Leave room above/right for the tooltip, away from sticky page headers.
    x, y = min(
        candidates,
        key=lambda p: ((p[0] - width * anchor[0]) ** 2 + (p[1] - height * anchor[1]) ** 2, p[1], p[0]),
    )
    return {"x": x, "y": y, "width": width, "height": height}


def consume_credentials():
    email = os.environ.pop("HABFLY_LOGIN_EMAIL", "")
    password = os.environ.pop("HABFLY_LOGIN_PASSWORD", "")
    if not email or not password:
        raise SetupStop("setup_credentials_missing")
    return email, password


def visible_star_link(png, star):
    """Locate the cyan action line in the black tooltip above the clicked dot.

    HabWorlds draws this link in the canvas, so it has no DOM/accessibility
    label. This is deliberately a narrow visual adapter, not general OCR.
    Ambiguous or changed layouts return no candidate and never trigger a click.
    """
    import numpy as np
    from PIL import Image

    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)
    height, width = pixels.shape[:2]
    if (width, height) != (star["width"], star["height"]):
        raise SetupStop("setup_starfield_changed")
    # Tooltip in the observed layout is immediately above/right of the dot.
    x0, x1 = max(0, int(star["x"]) - 8), min(width, int(star["x"]) + 300)
    y0, y1 = max(0, int(star["y"]) - 180), max(0, int(star["y"]) - 8)
    crop = pixels[y0:y1, x0:x1]
    cyan = (crop[:, :, 0] < 110) & (crop[:, :, 1] > 160) & (crop[:, :, 2] > 160)
    rows = np.where(cyan.sum(1) >= 8)[0]
    bands = []
    for row in rows:
        if not bands or row - bands[-1][-1] > 2:
            bands.append([])
        bands[-1].append(int(row))
    candidates = []
    for band in bands:
        a, b = band[0], band[-1] + 1
        columns = np.where(cyan[a:b].any(0))[0]
        left, right = int(columns[0]), int(columns[-1]) + 1
        w, h = right - left, b - a
        area = crop[max(0, a - 3) : b + 3, max(0, left - 3) : right + 3]
        if 65 <= w <= 260 and 5 <= h <= 26 and w / h >= 5 and (area.max(2) < 35).mean() > 0.50:
            candidates.append({"x": x0 + (left + right) / 2, "y": y0 + (a + b) / 2})
    return candidates[0] if len(candidates) == 1 else None


def numbered_tab(frame, number):
    """Rendered numbered tab, including an unnamed image with visible AX text."""
    if number not in {1, 2, 3}:
        raise SetupStop("setup_unknown_numbered_tab")
    snapshots = {f'- img "{number}"', f'- img: "{number}"'}
    controls = [
        e for e in frame.get_by_role("img").all() if e.is_visible() and e.aria_snapshot() in snapshots
    ]
    if len(controls) != 1 or not controls[0].is_enabled():
        raise SetupStop("setup_ambiguous_or_unavailable_tab")
    return controls[0]


class BrowserSetup:
    MAX_SECONDS = 90
    # A fresh public login page exposed Sign in before its known cookie dialog
    # arrived about one second later. Let the UI settle cooperatively before
    # credentials; this consumes the existing absolute setup budget.
    LOGIN_UI_SETTLE_SECONDS = 1.5
    LOGIN_UI_OCCLUSION_SECONDS = 3.0
    STARFIELD_WAIT_SECONDS = 30
    STARFIELD_POLL_SECONDS = 0.5
    # Observed 2026-09-24: intro -> instructions -> points -> warning -> simulation.
    SCREENS = (
        ("Are we alone?", "Instructions"),
        ("You will be presented with a starfield", "Next"),
        ("Grade Breakdown", "Next"),
        ("There is a known issue in the Project", "Next"),
    )

    def __init__(self, page, config, credentials, *, emit=lambda *_: None, output=None):
        self.page, self.config, self.emit = page, config, emit
        self._email, self._password = credentials
        parsed = urlsplit(config.url)
        self.login_url = urlunsplit((parsed.scheme, parsed.netloc, "/authors/log_in", "", ""))
        self.landing_url = urlunsplit((parsed.scheme, parsed.netloc, "/workspaces/course_author", "", ""))
        self.started = time.monotonic()
        self.stage = "opening_preview"
        self.visited = set()
        self.submitted_login = False
        self.login_ui_since = None
        self.login_occluded_since = None
        self.login_credentials_started = False
        self.login_cookie_closing = None
        self.post_login_refresh_attempted = False
        self.post_login_refresh_verified = False
        self.returned_to_preview = False
        self.stop_reason = None
        self.closed = False
        self.output = output
        self.star_clicked = self.view_clicked = self.stellar_tab_clicked = False
        self.star_click_time = None
        self.star_point = None
        self.simulation_frame = None
        self.ready_signature = None
        self.starfield_started = None
        self.starfield_next_poll = 0.0
        self.starfield_candidate = None
        self.starfield_loading_recorded = False
        self.excluded_star_points = ()
        self.starfield_anchor = (0.4, 0.55)
        self.page.on("dialog", self._dialog)
        self.page.route("**/*", self._route)

    def _dialog(self, dialog):
        self.stop_reason = "setup_unexpected_dialog"
        dialog.dismiss()

    def allowed(self, url):
        try:
            return self.config.allows(url) or _url_identity(url) in {
                _url_identity(self.login_url),
                _url_identity(self.landing_url),
            }
        except ValueError:
            return False

    def _route(self, route):
        request = route.request
        if (
            request.is_navigation_request()
            and request.frame == self.page.main_frame
            and not self.allowed(request.url)
        ):
            self.stop_reason = "setup_navigation_outside_boundary"
            route.abort()
        else:
            route.fallback()

    def _guard(self):
        self.page.wait_for_timeout(0)
        if self.closed:
            raise SetupStop("setup_closed")
        if time.monotonic() - self.started > self.MAX_SECONDS:
            raise SetupStop("setup_time_limit")
        if self.stop_reason:
            raise SetupStop(self.stop_reason)
        if len(self.page.context.pages) != 1:
            raise SetupStop("setup_unexpected_popup")
        if not self.allowed(self.page.url):
            raise SetupStop("setup_navigation_outside_boundary")

    def _stage(self, name):
        if self.stage != name:
            self.stage = name
            self.emit("state", {"setup_stage": name, "action_source": "deterministic_setup"})

    def _one(self, locator):
        matches = [item for item in locator.all() if item.is_visible()]
        if len(matches) != 1 or not matches[0].is_enabled():
            raise SetupStop("setup_ambiguous_or_unavailable_control")
        return matches[0]

    def advance(self):
        try:
            return self._advance()
        except SetupStop:
            self.close()
            raise
        except Exception:  # noqa: BLE001 - never leak Playwright call logs or credential values
            self.close()
            raise SetupStop("setup_browser_operation_failed") from None

    def _advance(self):
        self._guard()
        if self.page.url == self.login_url:
            if self.post_login_refresh_attempted:
                raise SetupStop("setup_authentication_lost_after_refresh")
            return self._login()
        if self.page.url == self.landing_url:
            if self.returned_to_preview or self.post_login_refresh_attempted:
                raise SetupStop("setup_preview_redirect_loop")
            self.returned_to_preview = True
            self._stage("returning_to_configured_preview")
            self.page.goto(self.config.url, wait_until="domcontentloaded", timeout=10000)
            return "waiting"
        # Only the exact pinned activity is readable after authentication.
        if not self.config.allows(self.page.url):
            raise SetupStop("setup_navigation_outside_boundary")
        self._email = self._password = ""
        if any(x.is_visible() for x in self.page.locator('input[type="password"]').all()):
            raise SetupStop("setup_unexpected_login_form")
        if any(x.is_visible() for x in self.page.get_by_role("dialog").all()):
            raise SetupStop("setup_unexpected_modal")
        if self.submitted_login and not self.post_login_refresh_attempted:
            return self._refresh_after_login()
        if self.post_login_refresh_attempted:
            # Toast text need not be an interactive hit target (nested spans or
            # pointer-events:none). Inspect painted text, not control geometry.
            if re.search(r"\bwelcome\s+back\b", rendered_text(self.page.main_frame), re.IGNORECASE):
                raise SetupStop("setup_welcome_back_persisted_after_refresh")
            if not self.post_login_refresh_verified:
                self.post_login_refresh_verified = True
                self._stage("post_login_refresh_verified")
                return "waiting"
        frames = [
            f for f in self.page.frames if f.url == SIMULATION_URL and _visible_frame(f, self.page.main_frame)
        ]
        if len(frames) > 1:
            raise SetupStop("setup_ambiguous_simulation")
        if frames:
            # Visible content only. No JS application globals, catalogs or grading state.
            text = rendered_text(frames[0])
            if (
                ("FUNDING" in text.upper() and "DATA QUALITY" in text.upper())
                or self.star_clicked
                or "YOUR RECONSTRUCTION" in text.upper()
            ):
                return self._star(frames[0], text)
            self._stage("waiting_for_simulation")
            return "waiting"
        body = self.page.locator("body").inner_text(timeout=3000)
        for index, (marker, label) in enumerate(self.SCREENS):
            if marker.casefold() in body.casefold():
                if index in self.visited:
                    return "waiting"  # Never click Next twice while a transition is pending.
                button = self._one(
                    self.page.get_by_role("button", name=re.compile(f"^{label}$", re.IGNORECASE))
                )
                self._guard()
                self._stage(f"intro_screen_{index + 1}")
                button.click(timeout=3000)
                self.visited.add(index)
                return "waiting"
        self._stage("waiting_for_preview_content")
        return "waiting"

    def _refresh_after_login(self):
        # Restart only the document, never its authenticated context. A new
        # browser would discard the local preview attempt. This one-shot refresh
        # consumes the login flash before any intro/star action or policy work.
        if self.visited or self.star_clicked or self.view_clicked or self.stellar_tab_clicked:
            raise SetupStop("setup_post_login_refresh_after_activity")
        before = self.page.url
        self.post_login_refresh_attempted = True
        self._stage("refreshing_authenticated_preview")
        self._guard()  # Callback cancellation, popup, boundary and original deadline.
        if self.page.url != before:
            raise SetupStop("setup_post_login_refresh_page_changed")
        if any(x.is_visible() for x in self.page.locator('input[type="password"]').all()):
            raise SetupStop("setup_unexpected_login_form")
        if any(x.is_visible() for x in self.page.get_by_role("dialog").all()):
            raise SetupStop("setup_unexpected_modal")
        remaining_ms = int((self.MAX_SECONDS - (time.monotonic() - self.started)) * 1000)
        if remaining_ms <= 0:
            raise SetupStop("setup_time_limit")
        try:
            self.page.reload(wait_until="domcontentloaded", timeout=min(10000, remaining_ms))
        except Exception:  # noqa: BLE001 - no driver logs, authentication state or retries
            raise SetupStop("setup_post_login_refresh_failed") from None
        self._guard()
        if self.page.url == self.login_url:
            raise SetupStop("setup_authentication_lost_after_refresh")
        if not self.config.allows(self.page.url):
            raise SetupStop("setup_post_login_refresh_outside_preview")
        # Verification is a separate scheduled read before intro/star actions.
        return "waiting"

    def _login(self):
        # Diagnostic tags identify only the operation being attempted. No
        # authentication-page capture, input readback or driver log is added.
        self._login_operation = "inspect_cookie_preferences"
        try:
            return self._login_native()
        except SetupStop:
            raise
        except Exception as exc:  # noqa: BLE001 - never expose credential-bearing driver text
            reason = _login_failure_code(self._login_operation, exc)
            if reason == "setup_login_submit_click_playwright_timeout_error":
                reason += f"_at_{self._login_location()}_{_submit_timeout_stage(exc)}"
            raise SetupStop(reason) from None
        finally:
            self._login_operation = None

    def _login_location(self):
        # Only a fixed category escapes; do not snapshot the authentication page
        # or record its URL, query, inputs, DOM, screenshot, or page title.
        try:
            current = self.page.url
            if current == self.login_url:
                return "login"
            if current == self.landing_url:
                return "landing"
            return "preview" if self.config.allows(current) else "outside"
        except Exception:  # noqa: BLE001 - a diagnostic read must not replace the original stop
            return "unavailable"

    def _close_login_cookie(self):
        # The native sign-in form and cookie notice were inspected through the UI.
        if self.login_cookie_closing is not None:
            pending = self.login_cookie_closing
            self._login_operation = "inspect_cookie_" + pending["kind"]
            self._guard()
            for dialog in self.page.get_by_role("dialog").all():
                if dialog.is_visible() and not any(
                    dialog.evaluate("(e, original) => e === original", handle)
                    for handle in pending["dialogs"]
                ):
                    raise SetupStop("setup_unexpected_modal")
            # Retain the actual known notice node across its dismissal animation.
            # A replacement notice is not authority for another Close click.
            candidates = self.page.get_by_role(
                "dialog" if pending["kind"] == "preferences" else "heading",
                name="Cookie Preferences" if pending["kind"] == "preferences" else "We use cookies",
                exact=True,
            )
            visible = [item for item in candidates.all() if item.is_visible()]
            if len(visible) > 1 or (
                visible and not visible[0].evaluate("(e, original) => e === original", pending["notice"])
            ):
                raise SetupStop("setup_cookie_notice_changed_during_close")
            if time.monotonic() >= pending["deadline"]:
                raise SetupStop("setup_cookie_notice_close_unresolved")
            if pending["notice"].is_visible():
                return True
            self.login_cookie_closing = None
            return False
        self._login_operation = "inspect_cookie_preferences"
        preferences = self.page.get_by_role("dialog", name="Cookie Preferences", exact=True)
        if any(x.is_visible() for x in preferences.all()):
            self._login_operation = "close_cookie_preferences"
            self._guard()
            notice = self._one(preferences)
            close = self._one(notice.get_by_role("button", name="Close", exact=True))
            close_handle = self._begin_cookie_close(notice, "preferences", close)
            close_handle.click(timeout=3000)
            self.login_ui_since = None
            self._stage("cookie_notice_closed")
            return True
        self._login_operation = "inspect_cookie_notice"
        cookie = self.page.get_by_role("heading", name="We use cookies", exact=True)
        if any(x.is_visible() for x in cookie.all()):
            # Scope Close to the visible cookie container, not the authentication alert.
            self._login_operation = "close_cookie_notice"
            container = self._one(cookie)
            notice = container
            for _ in range(3):
                container = container.locator("..")
                buttons = container.get_by_role("button", name="Close", exact=True)
                if buttons.count() == 1:
                    self._guard()
                    close = self._one(buttons)
                    close_handle = self._begin_cookie_close(notice, "notice", close)
                    close_handle.click(timeout=3000)
                    break
            else:
                raise SetupStop("setup_cookie_notice_needs_manual_close")
            self.login_ui_since = None
            self._stage("cookie_notice_closed")
            return True
        return False

    def _begin_cookie_close(self, notice, kind, close):
        handle = notice.element_handle(timeout=3000)
        close_handle = close.element_handle(timeout=3000)
        if (
            handle is None
            or close_handle is None
            or not notice.evaluate("(e, original) => e === original", handle)
            or not close.evaluate("(e, original) => e === original", close_handle)
        ):
            raise SetupStop("setup_cookie_notice_changed_before_close")
        dialogs = []
        for dialog in self.page.get_by_role("dialog").all():
            if not dialog.is_visible():
                continue
            original = dialog.element_handle(timeout=3000)
            if original is None or not dialog.evaluate(
                "(e, notice) => e === notice || e.contains(notice)", handle
            ):
                raise SetupStop("setup_unexpected_modal")
            if not dialog.evaluate("(e, original) => e === original", original):
                raise SetupStop("setup_cookie_notice_changed_before_close")
            dialogs.append(original)
        self._guard()
        self.login_cookie_closing = {
            "kind": kind,
            "notice": handle,
            "dialogs": dialogs,
            "deadline": time.monotonic() + self.LOGIN_UI_OCCLUSION_SECONDS,
        }
        return close_handle

    def _login_modal_guard(self):
        self._login_operation = "inspect_modal"
        if any(x.is_visible() for x in self.page.get_by_role("dialog").all()):
            raise SetupStop("setup_unexpected_modal")

    def _login_native(self):
        if self._close_login_cookie():
            return "waiting"
        self._login_modal_guard()
        if self.submitted_login:
            self._stage("waiting_for_login_result")
            return "waiting"  # Single submission only; deadline covers bad credentials.
        self._login_operation = "locate_email"
        email = self._one(self.page.get_by_placeholder("Email", exact=True))
        self._login_operation = "locate_password"
        password = self._one(self.page.get_by_placeholder("Password", exact=True))
        self._login_operation = "locate_submit"
        submit = self._one(self.page.get_by_role("button", name="Sign in", exact=True))

        # Inspect form destinations, not hidden CSRF fields or input values.
        def validate_form():
            self._guard()
            forms = [
                x.evaluate("e => e.form ? {action:e.form.action, method:e.form.method} : null")
                for x in (email, password, submit)
            ]
            if any(form != {"action": self.login_url, "method": "post"} for form in forms):
                raise SetupStop("setup_untrusted_login_form")
            if submit.get_attribute("formaction") or submit.get_attribute("formmethod"):
                raise SetupStop("setup_untrusted_login_form")

        self._login_operation = "validate_form_initial"
        validate_form()
        self._login_operation = "inspect_field_types"
        if password.get_attribute("type") != "password" or email.get_attribute("type") != "email":
            raise SetupStop("setup_unexpected_login_fields")

        def exposed():
            self._login_operation = "inspect_login_exposure"
            self._guard()
            if not all(rendered_control(control) for control in (email, password, submit)):
                # A known cookie can appear during the read. Only its ordinary
                # Close control is permitted; unknown overlays remain a stop.
                if self._close_login_cookie():
                    return False
                self._login_modal_guard()
                if self.login_credentials_started:
                    raise SetupStop("setup_login_control_not_exposed")
                # The observed cookie backdrop paints before its heading/modal
                # appears. Waiting is read-only and never authorizes an unknown
                # overlay action; persistent cover stops within this fixed bound.
                if self.login_occluded_since is None:
                    self.login_occluded_since = time.monotonic()
                if time.monotonic() - self.login_occluded_since >= self.LOGIN_UI_OCCLUSION_SECONDS:
                    raise SetupStop("setup_login_control_not_exposed")
                self.login_ui_since = None
                self._stage("waiting_for_login_ui")
                self._guard()
                return False
            self.login_occluded_since = None
            return True

        def ready_for_write():
            self._guard()
            if self._close_login_cookie():
                return False
            self._login_modal_guard()
            return exposed()

        if not exposed():
            return "waiting"
        if self.login_ui_since is None:
            self.login_ui_since = time.monotonic()
        if time.monotonic() - self.login_ui_since < self.LOGIN_UI_SETTLE_SECONDS:
            self._login_operation = "waiting_for_login_ui_callback"
            self._stage("waiting_for_login_ui")
            self._guard()
            return "waiting"
        self._login_operation = "signing_in_callback"
        self._stage("signing_in")
        if not ready_for_write():
            return "waiting"
        self._login_operation = "validate_form_before_email"
        validate_form()
        self._login_operation = "fill_email"
        self.login_credentials_started = True
        email.fill(self._email, timeout=3000)
        self._login_operation = "validate_form_after_email"
        validate_form()
        if not ready_for_write():
            return "waiting"
        self._login_operation = "fill_password"
        password.fill(self._password, timeout=3000)
        self._login_operation = "validate_form_after_password"
        validate_form()
        # Filling and form validation may span arrival of a late cookie panel.
        # A pre-dispatch dismissal is a separate tick, never a submit retry.
        if not ready_for_write():
            return "waiting"
        self._login_operation = "validate_form_before_submit"
        validate_form()
        self._login_operation = "submit_click"
        submit.click(timeout=3000)
        self.submitted_login = True
        self._email = self._password = ""
        return "waiting"

    def _star(self, frame, text):
        if self.simulation_frame is not None and frame != self.simulation_frame:
            raise SetupStop("setup_simulation_frame_changed")
        self.simulation_frame = frame
        if any(x.is_visible() for x in frame.get_by_role("dialog").all()):
            raise SetupStop("setup_unexpected_modal")
        if any(x.is_visible() for x in frame.locator('input[type="password"]').all()):
            raise SetupStop("setup_unexpected_login_form")
        upper = text.upper()
        if self.view_clicked:
            # VIEW STAR DATA may retain another tab. The visible tab image is 1.
            if not self.stellar_tab_clicked and ("OBSERVE FOR" in upper or "TERRESTRIAL" in upper):
                numbered_tab(frame, 1).click(timeout=3000)
                self.stellar_tab_clicked = True
                self.ready_signature = None
                self._stage("opening_stellar_tab")
                return "waiting"
            return self._stellar_ready(frame)
        view = frame.get_by_text(re.compile(r"^VIEW STAR DATA$", re.IGNORECASE))
        if any(rendered_control(x) for x in view.all()):
            self._guard()
            self._stage("opening_star_data")
            self._one(view).click(timeout=3000)
            self.view_clicked = True
            return "waiting"
        if self.star_clicked:
            handle = frame.frame_element()
            png = evidence_screenshot(handle, timeout=5000, scale="css")
            link = visible_star_link(png, self.star_point)
            if link:
                self._guard()
                if (
                    visible_star_link(evidence_screenshot(handle, timeout=5000, scale="css"), self.star_point)
                    != link
                ):
                    raise SetupStop("setup_starfield_changed")
                self._record_pixels("setup-star-link.png", png, link)
                self._stage("opening_star_data")
                self._click_pixel(handle, link)
                self.view_clicked = True
                return "waiting"
            if time.monotonic() - self.star_click_time > 8:
                raise SetupStop("setup_star_selection_not_confirmed")
            return "waiting"
        if any(word in upper for word in ("ANALYZED DATA", "OBSERVATIONS", "RECONSTRUCTION")) or any(
            rendered_control(x) for x in frame.locator("input, select, textarea").all()
        ):
            self.starfield_candidate = None
            self._stage("waiting_for_starfield")
            return "waiting"
        now = time.monotonic()
        if self.starfield_started is None:
            self.starfield_started = now
        if now < self.starfield_next_poll:
            return "waiting"
        self.starfield_next_poll = now + self.STARFIELD_POLL_SECONDS
        handle = frame.frame_element()
        handle.hover(timeout=3000)
        png = evidence_screenshot(handle, timeout=5000, scale="css")
        self._guard()
        if time.monotonic() - self.starfield_started >= self.STARFIELD_WAIT_SECONDS:
            self._record_pixels("setup-starfield-rejected.png", png, None)
            raise SetupStop("setup_starfield_render_timeout")
        try:
            point = visible_star_point(png, excluded=self.excluded_star_points, anchor=self.starfield_anchor)
        except SetupStop as exc:
            if str(exc) == "setup_no_visible_star_candidate":
                self.starfield_candidate = None
                if not self.starfield_loading_recorded:
                    self._record_pixels("setup-starfield-loading.png", png, None)
                    self.starfield_loading_recorded = True
                self._stage("waiting_for_starfield_render")
                return "waiting"
            # Unknown layout is not a loading condition; stop immediately.
            self._record_pixels("setup-starfield-rejected.png", png, None)
            raise
        self._guard()
        if rendered_text(frame) != text:
            self.starfield_candidate = None
            self._stage("waiting_for_starfield_render")
            return "waiting"
        # Two matching observations, separated by the poll interval. Waiting
        # never clicks, reloads, or retries a previously selected star.
        candidate = (point, text)
        if self.starfield_candidate != candidate:
            self.starfield_candidate = candidate
            self._stage("waiting_for_stable_starfield")
            return "waiting"
        self._record_pixels("setup-starfield.png", png, point)
        self._stage("selecting_visible_star")
        self._click_pixel(handle, point)
        self.star_point = point
        self.star_clicked = True
        self.star_click_time = time.monotonic()
        return "waiting"

    def _stellar_ready(self, frame):
        # Readiness requires an actual open action AND exposed editable fields.
        # Text hit-testing is useful on the star map, but not a reliable gate
        # for decorative/non-interactive headings inside a scrollable screen.
        # Reuse the strict DOM/AX measurement, label, unit and identity mapping;
        # never fall back to hidden app state or bypass NumericSession's binding.
        controls = [
            x for x in frame.locator('input:not([type="hidden"]), textarea').all() if rendered_control(x)
        ]
        if len(controls) != 3 or not all(x.is_editable() for x in controls):
            self.ready_signature = None
            self._stage("waiting_for_stellar_controls")
            return "waiting"
        frame_counts = [
            sum(f.url == rule.url and _visible_frame(f, self.page.main_frame) for f in self.page.frames)
            for rule in self.config.frames
        ]
        if frame_counts != [r.count for r in self.config.frames]:
            self.ready_signature = None
            self._stage("waiting_for_stellar_frames")
            return "waiting"
        self._guard()
        try:
            report = inspect_page(self.page, self.config)
        except BrowserSafetyStop as exc:
            if str(exc) in {f"frame_not_ready:{rule.name}" for rule in self.config.frames}:
                self.ready_signature = None
                self._stage("waiting_for_stellar_measurements")
                return "waiting"
            # The message is not persisted here; the setup boundary sanitizes it.
            raise
        if report["ignored_frame_urls"]:
            raise SetupStop("setup_unknown_visible_frame")
        signature = screen_identity(report)
        try:
            map_stellar_capture(report, capture_sha256=signature)
        except StellarMappingError:
            self.ready_signature = None
            self._stage("waiting_for_stellar_measurements")
            return "waiting"
        self._guard()
        if self.ready_signature == signature:
            self._stage("stellar_screen_ready")
            self.close()
            return "stellar"
        self.ready_signature = signature
        self._stage("verifying_stellar_screen")
        return "waiting"

    def _record_pixels(self, name, png, point):
        if self.output is not None:
            with (self.output / name).open("xb") as stream:
                stream.write(png)
        self.emit(
            "state",
            {
                "setup_evidence": name,
                "action_source": "deterministic_setup",
                "starfield_sha256": hashlib.sha256(png).hexdigest(),
                "visible_point": point,
            },
        )

    def _click_pixel(self, handle, point):
        border = handle.evaluate(
            "e => ({x:parseFloat(getComputedStyle(e).borderLeftWidth)||0, y:parseFloat(getComputedStyle(e).borderTopWidth)||0})"
        )
        handle.click(position={"x": point["x"] - border["x"], "y": point["y"] - border["y"]}, timeout=3000)

    def close(self):
        self._email = self._password = ""
        if not self.closed:
            self.page.remove_listener("dialog", self._dialog)
            self.page.unroute("**/*", self._route)
            self.closed = True
