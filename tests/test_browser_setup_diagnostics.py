"""Injected native-call failures; never launches or contacts a browser."""

import json
from types import SimpleNamespace

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from habfly.browser_setup import (
    _LOGIN_OPERATIONS,
    BrowserSetup,
    SetupStop,
    _login_failure_code,
    _submit_timeout_stage,
)

EMAIL, PASSWORD = "fixture-secret@example.invalid", "fixture-private-password"
PRIVATE = f"driver URL http://localhost/private?token=hidden input {EMAIL} {PASSWORD}"


class Controls:
    def __init__(self, page, entries):
        self.page, self.entries = page, entries

    def all(self):
        self.page.trip("all")
        return self.entries

    def count(self):
        self.page.trip("count")
        return len(self.entries)


class Control:
    def __init__(self, page, name):
        self.page, self.name = page, name

    def is_visible(self):
        self.page.trip("is_visible")
        if self.name == "cookie":
            return self.page.cookie
        if self.name == "preferences":
            return self.page.preferences
        return True

    def is_enabled(self):
        self.page.trip("is_enabled")
        return True

    def locator(self, query):
        assert query == ".."
        self.page.trip("cookie_parent")
        return self

    def get_by_role(self, role, *, name, exact):
        assert role == "button" and name == "Close" and exact
        self.page.trip("cookie_close_lookup")
        return Controls(self.page, [Control(self.page, "cookie_close")])

    def element_handle(self, *, timeout):
        assert timeout == 3000
        self.page.trip("notice_handle")
        return self

    def evaluate(self, source, argument=None):
        if source == "(e, original) => e === original":
            self.page.trip("notice_identity")
            return self.name == argument.name
        if "getBoundingClientRect" in source:
            self.page.trip("exposure")
            return True
        assert source == "e => e.form ? {action:e.form.action, method:e.form.method} : null"
        self.page.trip("form_metadata")
        return {"action": self.page.form_action, "method": "post"}

    def get_attribute(self, name):
        self.page.trip("attribute_" + name)
        if name == "type":
            return self.name
        assert name in {"formaction", "formmethod"}
        return None

    def fill(self, value, *, timeout):
        self.page.fills.append((self.name, timeout))
        assert value == (EMAIL if self.name == "email" else PASSWORD)
        self.page.trip("fill")

    def click(self, *, timeout):
        self.page.clicks.append((self.name, timeout))
        self.page.trip("click")
        if self.name == "cookie_close":
            self.page.cookie = self.page.preferences = False


class Page:
    url = "http://localhost/authors/log_in"

    def __init__(self):
        self.context = SimpleNamespace(pages=[self])
        self.form_action = self.url
        self.fills, self.clicks, self.calls, self.events = [], [], [], []
        self.fail_operation, self.failure = None, None
        self.setup = None
        self.preferences = self.cookie = False

    def trip(self, operation):
        stage = getattr(self.setup, "_login_operation", None)
        self.calls.append((stage, operation))
        if stage == self.fail_operation and self.failure is not None:
            raise self.failure

    def on(self, *args):
        pass

    def route(self, *args):
        pass

    def remove_listener(self, *args):
        pass

    def unroute(self, *args):
        pass

    def wait_for_timeout(self, milliseconds):
        assert milliseconds == 0
        self.trip("guard")

    def get_by_role(self, role, *, name=None, exact=False):
        self.trip("lookup_" + role)
        if role == "dialog":
            if name == "Cookie Preferences":
                return Controls(self, [Control(self, "preferences")] if self.preferences else [])
            assert name is None
            return Controls(self, [])
        if role == "heading":
            assert name == "We use cookies" and exact
            return Controls(self, [Control(self, "cookie")] if self.cookie else [])
        assert role == "button" and name == "Sign in" and exact
        return Controls(self, [Control(self, "submit")])

    def get_by_placeholder(self, name, *, exact):
        self.trip("lookup_" + name.casefold())
        assert name in {"Email", "Password"} and exact
        return Controls(self, [Control(self, name.casefold())])

    def emit(self, kind, payload):
        self.trip("callback")
        self.events.append((kind, payload))


@pytest.fixture
def setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(BrowserSetup, "LOGIN_UI_SETTLE_SECONDS", 0)
    page = Page()
    config = SimpleNamespace(
        url="http://localhost/activity?preview_sequence_id=fixture-only",
        allows=lambda url: url == "http://localhost/activity?preview_sequence_id=fixture-only",
    )
    setup = BrowserSetup(page, config, (EMAIL, PASSWORD), emit=page.emit, output=tmp_path)
    page.setup = setup
    yield page, setup, tmp_path
    setup.close()


@pytest.mark.parametrize("operation", sorted(_LOGIN_OPERATIONS))
def test_every_fixed_login_operation_has_sanitized_failure_and_no_retry(setup_case, operation):
    page, setup, output = setup_case
    page.fail_operation, page.failure = operation, RuntimeError(PRIVATE)
    page.preferences = operation == "close_cookie_preferences"
    page.cookie = operation == "close_cookie_notice"
    if operation == "waiting_for_login_ui_callback":
        setup.LOGIN_UI_SETTLE_SECONDS = 1.5
    with pytest.raises(SetupStop) as raised:
        setup.advance()
    assert str(raised.value) == f"setup_login_{operation}_runtime_error"
    assert raised.value.__suppress_context__ is True
    assert setup.closed and setup._email == setup._password == ""
    assert setup._login_operation is None
    assert not list(output.iterdir())
    public = str(raised.value) + json.dumps(page.events)
    assert all(secret not in public for secret in (EMAIL, PASSWORD, "http://", "hidden", PRIVATE))
    counts = len(page.fills), len(page.clicks)
    with pytest.raises(SetupStop, match="^setup_closed$"):
        setup.advance()
    assert (len(page.fills), len(page.clicks)) == counts


@pytest.mark.parametrize(
    "exception,label",
    [
        (PlaywrightTimeoutError, "playwright_timeout_error"),
        (PlaywrightError, "playwright_error"),
        (TimeoutError, "timeout_error"),
        (OSError, "os_error"),
        (ValueError, "value_error"),
        (TypeError, "type_error"),
        (RuntimeError, "runtime_error"),
        (Exception, "unknown_error"),
    ],
)
@pytest.mark.parametrize("operation", ["fill_email", "fill_password", "submit_click"])
def test_class_families_are_fixed_without_driver_values(setup_case, exception, label, operation):
    page, setup, _ = setup_case
    page.fail_operation, page.failure = operation, exception(PRIVATE)
    suffix = (
        "_at_login_unknown" if operation == "submit_click" and exception is PlaywrightTimeoutError else ""
    )
    with pytest.raises(SetupStop, match=f"^setup_login_{operation}_{label}{suffix}$"):
        setup.advance()
    assert setup.closed and setup._email == setup._password == ""


def test_unknown_exception_class_and_string_conversion_are_never_observed(setup_case):
    page, setup, _ = setup_case

    def forbidden(self):
        pytest.fail("Exception stringification could disclose private driver values")

    secret_class = type("private_" + PASSWORD, (Exception,), {"__str__": forbidden, "__repr__": forbidden})
    page.fail_operation, page.failure = "submit_click", secret_class(PRIVATE)
    with pytest.raises(SetupStop, match="^setup_login_submit_click_unknown_error$"):
        setup.advance()


@pytest.mark.parametrize("operation", [None, [], {}, PRIVATE, 7])
def test_unknown_operation_values_never_enter_the_reason(operation):
    assert _login_failure_code(operation, RuntimeError(PRIVATE)) == "setup_login_unclassified_runtime_error"


def test_existing_guard_reason_and_deadline_remain_unchanged(setup_case):
    page, setup, _ = setup_case
    page.form_action = "http://outside.invalid/do-not-submit"
    with pytest.raises(SetupStop, match="^setup_untrusted_login_form$"):
        setup.advance()
    assert not page.fills and not page.clicks and setup.MAX_SECONDS == 90


def test_success_keeps_existing_form_order_timeouts_and_single_submit(setup_case):
    page, setup, output = setup_case
    assert setup.advance() == "waiting"
    assert page.fills == [("email", 3000), ("password", 3000)]
    assert page.clicks == [("submit", 3000)]
    form_stages = [stage for stage, call in page.calls if call == "form_metadata"]
    assert form_stages == [
        name
        for name in (
            "validate_form_initial",
            "validate_form_before_email",
            "validate_form_after_email",
            "validate_form_after_password",
            "validate_form_before_submit",
        )
        for _ in range(3)
    ]
    assert setup.submitted_login and setup._email == setup._password == ""
    assert setup.advance() == "waiting" and setup.stage == "waiting_for_login_result"
    assert page.clicks == [("submit", 3000)] and not list(output.iterdir())


def test_submit_timeout_is_not_reported_as_authenticated_or_safe_to_retry(setup_case):
    page, setup, _ = setup_case
    page.fail_operation, page.failure = "submit_click", PlaywrightTimeoutError(PRIVATE)
    with pytest.raises(
        SetupStop, match="^setup_login_submit_click_playwright_timeout_error_at_login_unknown$"
    ):
        setup.advance()
    assert page.clicks == [("submit", 3000)] and setup.closed
    assert setup.stage == "signing_in" and not setup.submitted_login
    assert all(
        "authenticated" not in payload and "retry_allowed" not in payload for _, payload in page.events
    )


@pytest.mark.parametrize(
    "log,stage",
    [
        ("  - click action done\n  - waiting for scheduled navigations to finish", "navigation_wait"),
        ("  - waiting for scheduled navigations to finish", "unknown"),
        ("  - click action done", "unknown"),
        ("  - <private description> intercepts pointer events", "pointer_interception"),
        ("  - performing click action", "click_started"),
        ("  - element is not stable", "actionability_wait"),
        ("  2 × waiting for element to be visible, enabled and stable", "actionability_wait"),
        ("selector text='click action done; waiting for scheduled navigations to finish'", "unknown"),
    ],
)
def test_only_known_driver_lines_produce_fixed_timeout_stage(log, stage):
    assert _submit_timeout_stage(PlaywrightTimeoutError(PRIVATE + "\nCall log:\n" + log)) == stage


def test_unknown_timeout_subclasses_are_not_stringified():
    def forbidden(self):
        pytest.fail("Arbitrary exception text must not be inspected")

    for base in (Exception, PlaywrightTimeoutError):
        private = type("private_" + PASSWORD, (base,), {"__str__": forbidden})
        assert _submit_timeout_stage(private(PRIVATE)) == "unknown"


@pytest.mark.parametrize("category", ["login", "landing", "preview", "outside", "unavailable"])
def test_submit_timeout_fixed_location_and_navigation_stage_only(setup_case, monkeypatch, category):
    page, setup, output = setup_case
    page.fail_operation = "submit_click"
    page.failure = PlaywrightTimeoutError(
        PRIVATE + "\nCall log:\n  - click action done\n  - waiting for scheduled navigations to finish"
    )
    urls = {
        "login": setup.login_url,
        "landing": setup.landing_url,
        "preview": setup.config.url,
        "outside": "http://outside.invalid/private?password=" + PASSWORD,
        "unavailable": "http://outside.invalid/private?password=" + PASSWORD,
    }
    original = page.trip

    def trip(operation):
        if operation == "click":
            page.url = urls[category]
            if category == "unavailable":

                def fail(_):
                    raise RuntimeError(PRIVATE)

                setup.config.allows = fail
        original(operation)

    monkeypatch.setattr(page, "trip", trip)
    with pytest.raises(SetupStop) as raised:
        setup.advance()
    assert str(raised.value) == (
        "setup_login_submit_click_playwright_timeout_error_at_" + category + "_navigation_wait"
    )
    assert setup.closed and setup._email == setup._password == ""
    assert page.clicks == [("submit", 3000)] and not list(output.iterdir())
    assert all(
        value not in str(raised.value) + json.dumps(page.events) for value in (PASSWORD, EMAIL, "http")
    )


def test_non_login_failure_keeps_legacy_generic_code(setup_case, monkeypatch):
    _, setup, _ = setup_case

    def failure():
        raise RuntimeError(PRIVATE)

    monkeypatch.setattr(setup, "_advance", failure)
    with pytest.raises(SetupStop, match="^setup_browser_operation_failed$"):
        setup.advance()
    assert setup.closed and setup._email == setup._password == ""
