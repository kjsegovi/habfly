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


class SetupStop(RuntimeError):
    """Carries only an allowlisted local reason, never driver exception text."""


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


def visible_star_point(png):
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
    if not candidates:
        raise SetupStop("setup_no_visible_star_candidate")
    # Leave room above/right for the tooltip, away from sticky page headers.
    x, y = min(
        candidates, key=lambda p: ((p[0] - width * 0.4) ** 2 + (p[1] - height * 0.55) ** 2, p[1], p[0])
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


class BrowserSetup:
    MAX_SECONDS = 90
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
            return self._login()
        if self.page.url == self.landing_url:
            if self.returned_to_preview:
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

    def _login(self):
        # The native sign-in form and cookie notice were inspected through the UI.
        preferences = self.page.get_by_role("dialog", name="Cookie Preferences", exact=True)
        if any(x.is_visible() for x in preferences.all()):
            self._one(self._one(preferences).get_by_role("button", name="Close", exact=True)).click(
                timeout=3000
            )
            self._stage("cookie_notice_closed")
            return "waiting"
        cookie = self.page.get_by_role("heading", name="We use cookies", exact=True)
        if any(x.is_visible() for x in cookie.all()):
            # Scope Close to the visible cookie container, not the authentication alert.
            container = self._one(cookie)
            for _ in range(3):
                container = container.locator("..")
                buttons = container.get_by_role("button", name="Close", exact=True)
                if buttons.count() == 1:
                    self._one(buttons).click(timeout=3000)
                    break
            else:
                raise SetupStop("setup_cookie_notice_needs_manual_close")
            self._stage("cookie_notice_closed")
            return "waiting"
        if any(x.is_visible() for x in self.page.get_by_role("dialog").all()):
            raise SetupStop("setup_unexpected_modal")
        if self.submitted_login:
            self._stage("waiting_for_login_result")
            return "waiting"  # Single submission only; deadline covers bad credentials.
        email = self._one(self.page.get_by_placeholder("Email", exact=True))
        password = self._one(self.page.get_by_placeholder("Password", exact=True))
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

        validate_form()
        if password.get_attribute("type") != "password" or email.get_attribute("type") != "email":
            raise SetupStop("setup_unexpected_login_fields")
        self._stage("signing_in")
        validate_form()
        email.fill(self._email, timeout=3000)
        validate_form()
        password.fill(self._password, timeout=3000)
        validate_form()
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
                self._one(frame.get_by_role("img", name="1", exact=True)).click(timeout=3000)
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
            png = handle.screenshot(timeout=5000, scale="css")
            link = visible_star_link(png, self.star_point)
            if link:
                self._guard()
                if visible_star_link(handle.screenshot(timeout=5000, scale="css"), self.star_point) != link:
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
        png = handle.screenshot(timeout=5000, scale="css")
        self._guard()
        if time.monotonic() - self.starfield_started >= self.STARFIELD_WAIT_SECONDS:
            self._record_pixels("setup-starfield-rejected.png", png, None)
            raise SetupStop("setup_starfield_render_timeout")
        try:
            point = visible_star_point(png)
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
