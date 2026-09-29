"""Injected public surfaces only; no Playwright process or network is started."""

import hashlib
import json
from types import SimpleNamespace

import pytest

import habfly.browser_submission_outcome_diagnostic as module
from habfly.browser_probe import BrowserProbeConfig


@pytest.fixture
def public_page(tmp_path):
    state = SimpleNamespace(clock=0, calls=[], hook=None, popup=False, cancelled=False, dialogs=[])
    config = BrowserProbeConfig(
        url="http://localhost/activity?preview_sequence_id=q%3A123%3A946",
        frames=[{"name": "known", "url": "https://fixture.invalid/known", "count": 2}],
    )

    def called(operation):
        state.calls.append(operation)
        if state.hook:
            state.hook(operation)

    class Surface:
        def __init__(self, url, parent=None):
            self.url, self.parent_frame = url, parent
            self.auth, self.visible, self.login_button, self.login_link = False, True, False, False
            self.text, self.ax = "Visible outcome", '- dialog "Visible outcome"'

        def frame_element(self):
            return SimpleNamespace(is_visible=lambda: self.visible)

        def locator(self, selector):
            if selector == module._AUTH_INPUTS:
                called("auth_check")
                return SimpleNamespace(all=lambda: [SimpleNamespace(is_visible=lambda: self.auth)])
            assert selector == "body"  # No input values, scripts, selectors for hidden data.
            called("body_binding")

            def text(**kwargs):
                assert 0 < kwargs["timeout"] <= 3000
                called("text")
                return self.text

            def ax(**kwargs):
                assert 0 < kwargs["timeout"] <= 3000
                called("ax")
                return self.ax

            return SimpleNamespace(inner_text=text, aria_snapshot=ax)

        def get_by_role(self, role, **kwargs):
            assert role in {"button", "link"} and kwargs["name"] is module._AUTH_BUTTON
            return SimpleNamespace(
                all=lambda: [
                    SimpleNamespace(
                        is_visible=lambda: self.login_button if role == "button" else self.login_link
                    )
                ]
            )

    outer = Surface(config.url)
    known = Surface(config.frames[0].url, outer)
    page = SimpleNamespace(url=config.url, main_frame=outer, frames=[outer, known])
    page.context = SimpleNamespace(pages=[page])
    state.image = b"\x89PNG\r\n\x1a\nsynthetic diagnostic image"

    def screenshot(**options):
        assert options["full_page"] is False and 0 < options["timeout"] <= 3000
        called("image")
        return state.image

    page.screenshot = screenshot
    return SimpleNamespace(
        root=tmp_path, page=page, config=config, state=state, outer=outer, known=known, Surface=Surface
    )


def capture(fixture, **options):
    return module.preserve_submission_outcome(
        fixture.page,
        fixture.config,
        fixture.root / options.pop("output", "diagnostic"),
        deadline=options.pop("deadline", 180),
        original_failure=options.pop("original_failure", "project_submission_context_changed"),
        native_dialogs=fixture.state.dialogs,
        popup_observed=lambda: fixture.state.popup,
        cancelled=lambda: fixture.state.cancelled,
        _clock=lambda: fixture.state.clock,
        **options,
    )


def assert_no_content(fixture, result, expected):
    assert result["disposition"] == expected
    assert result["content_saved"] is False and result["source_sha256"] == {}
    assert {path.name for path in (fixture.root / "diagnostic").iterdir()} == {"disposition.json"}
    raw = (fixture.root / "diagnostic/disposition.json").read_text()
    assert "PRIVATE" not in raw and "preview_sequence_id" not in raw


@pytest.mark.parametrize("count", [0, 1, 3, 7])
def test_current_known_frames_and_visible_html_dialog_are_diagnostic_only(public_page, count):
    fixture = public_page
    fixture.page.frames = [fixture.outer] + [
        fixture.Surface(fixture.known.url, fixture.outer) for _ in range(count)
    ]
    fixture.outer.text = "Outcome at " + fixture.page.url
    result = capture(fixture)
    assert result["disposition"] == "public_outcome_captured"
    assert result["consistency_unverified"] and result["content_saved"]
    assert all(
        result[key] is False
        for key in (
            "submission_verified",
            "acknowledgement_interpreted",
            "submitted",
            "task_completed",
            "project_completed",
            "automatic_retry",
        )
    )
    assert result["browser_actions"] == 0
    for name, digest in result["source_sha256"].items():
        assert hashlib.sha256((fixture.root / "diagnostic" / name).read_bytes()).hexdigest() == digest
    saved = json.loads((fixture.root / "diagnostic/visible-outcome.json").read_bytes())
    assert len(saved["frames"]) == count + 1 and saved["consistency_unverified"]
    assert saved["frames"][0]["accessibility"].startswith("- dialog")
    assert "preview_sequence_id" not in json.dumps(saved)
    calls = list(fixture.state.calls)
    with pytest.raises(FileExistsError):
        capture(fixture)
    assert fixture.state.calls == calls  # Never replace artifacts or repeat capture.


@pytest.mark.parametrize("surface", ["outer", "known"])
@pytest.mark.parametrize("auth", ["auth", "login_button", "login_link"])
def test_any_visible_authentication_surface_blocks_all_content(public_page, surface, auth):
    setattr(getattr(public_page, surface), auth, True)
    result = capture(public_page)
    assert_no_content(public_page, result, "authentication_required")
    assert not {"text", "ax", "image"} & set(public_page.state.calls)


def test_authentication_labels_are_exact_and_do_not_match_logout():
    assert module._AUTH_BUTTON.fullmatch("Sign in")
    assert module._AUTH_BUTTON.fullmatch("LOGIN")
    assert not module._AUTH_BUTTON.fullmatch("Logout")
    assert not module._AUTH_BUTTON.fullmatch("Log out")
    assert not module._AUTH_BUTTON.fullmatch("Project login instructions")


@pytest.mark.parametrize("change", ["url", "unknown_frame", "popup", "frame_budget"])
def test_unsafe_context_never_reads_any_body(public_page, change):
    fixture = public_page
    if change == "url":
        fixture.page.url = "https://private.invalid/PRIVATE"
    elif change == "unknown_frame":
        fixture.known.url = "https://private.invalid/PRIVATE"
    elif change == "popup":
        fixture.state.popup = True  # Sticky event even if the popup already closed.
    else:
        fixture.page.frames += [fixture.known] * 7
    expected = {
        "url": "navigation_outside_activity",
        "unknown_frame": "unknown_visible_frame",
        "popup": "unexpected_popup",
        "frame_budget": "frame_budget_or_identity",
    }[change]
    assert_no_content(fixture, capture(fixture), expected)
    assert not {"text", "ax", "image"} & set(fixture.state.calls)


@pytest.mark.parametrize("when", ["text", "ax", "image"])
@pytest.mark.parametrize("change", ["auth", "unknown_frame", "url", "popup", "native_dialog"])
def test_late_privacy_change_discards_all_in_memory_content(public_page, when, change):
    fixture = public_page

    def hook(operation):
        if operation != when:
            return
        if change == "auth":
            fixture.known.auth = True
        elif change == "unknown_frame":
            fixture.known.url = "https://private.invalid/PRIVATE"
        elif change == "url":
            fixture.page.url = "https://private.invalid/PRIVATE"
        elif change == "popup":
            fixture.state.popup = True
        else:
            fixture.state.dialogs.append({"type": "prompt", "message": "PRIVATE"})

    fixture.state.hook = hook
    expected = {
        "auth": "authentication_required",
        "unknown_frame": "unknown_visible_frame",
        "url": "navigation_outside_activity",
        "popup": "unexpected_popup",
        "native_dialog": "native_dialog_content_withheld",
    }[change]
    assert_no_content(fixture, capture(fixture), expected)


def test_frame_replacement_during_authentication_guard_is_refused(public_page):
    fixture = public_page

    def hook(operation):
        if operation == "auth_check":
            fixture.page.frames = [fixture.outer, fixture.Surface(fixture.known.url, fixture.outer)]

    fixture.state.hook = hook
    assert_no_content(fixture, capture(fixture), "privacy_context_changed")
    assert not {"text", "ax", "image"} & set(fixture.state.calls)


@pytest.mark.parametrize("when", ["body_binding", "text", "ax", "image"])
@pytest.mark.parametrize("change", ["hidden", "replaced", "allowed_url"])
def test_capture_never_reads_stale_hidden_or_rebound_frame_content(public_page, when, change):
    fixture = public_page
    alternate = "https://fixture.invalid/alternate"
    fixture.config.frames.append(
        fixture.config.frames[0].model_copy(update={"name": "alternate", "url": alternate})
    )

    def hook(operation):
        if operation != when:
            return
        fixture.state.hook = None
        if change == "hidden":
            fixture.known.visible = False
        elif change == "replaced":
            fixture.page.frames = [fixture.outer, fixture.Surface(fixture.known.url, fixture.outer)]
        else:
            fixture.known.url = alternate

    fixture.state.hook = hook
    assert_no_content(fixture, capture(fixture), "capture_context_changed")
    if when == "body_binding":
        assert not {"text", "ax", "image"} & set(fixture.state.calls)


def test_hidden_unknown_frame_is_not_read_or_used_as_evidence(public_page):
    fixture = public_page
    hidden = fixture.Surface("https://private.invalid/PRIVATE", fixture.outer)
    hidden.visible, hidden.auth = False, True
    fixture.page.frames.append(hidden)
    result = capture(fixture)
    assert result["disposition"] == "public_outcome_captured"
    saved = (fixture.root / "diagnostic/visible-outcome.json").read_text()
    assert "PRIVATE" not in saved


@pytest.mark.parametrize("kind", ["alert", "confirm", "prompt", "beforeunload", "PRIVATE", None])
def test_native_dialog_preserves_type_only_without_touching_message_or_dom(public_page, kind):
    class Dialog:
        type = kind

        @property
        def message(self):
            pytest.fail("Native dialog text is not safe in an unverified context")

        @property
        def default_value(self):
            pytest.fail("Never read a native prompt input")

        def accept(self):
            pytest.fail("Never accept a dialog")

        dismiss = accept

    disposition = module.native_dialog_disposition(Dialog())
    public_page.state.dialogs.append(disposition)
    result = capture(public_page)
    assert_no_content(public_page, result, "native_dialog_content_withheld")
    assert result["native_dialogs"][0]["type"] == (kind if kind in module._DIALOG_TYPES else "unknown")
    assert not public_page.state.calls


@pytest.mark.parametrize("case", ["expired", "cancelled", "invalid_deadline", "late_deadline"])
def test_original_deadline_and_cancellation_never_expand_for_diagnostics(public_page, case):
    fixture = public_page
    options = {}
    if case == "expired":
        fixture.state.clock = 180
    elif case == "cancelled":
        fixture.state.cancelled = True
    elif case == "invalid_deadline":
        options["deadline"] = float("nan")
    else:
        fixture.state.hook = lambda operation: (
            setattr(fixture.state, "clock", 180) if operation == "text" else None
        )
    expected = {"cancelled": "cancelled", "invalid_deadline": "invalid_diagnostic_scope"}.get(
        case, "deadline_exhausted"
    )
    assert_no_content(fixture, capture(fixture, **options), expected)
    assert "image" not in fixture.state.calls


def test_changing_safe_text_is_retained_without_a_consistency_claim(public_page):
    fixture = public_page
    fixture.state.hook = lambda operation: (
        setattr(fixture.outer, "ax", '- dialog "Later public outcome"') if operation == "text" else None
    )
    result = capture(fixture)
    assert result["disposition"] == "public_outcome_captured" and result["consistency_unverified"]


@pytest.mark.parametrize("failure", ["text_budget", "invalid_png", "driver_error"])
def test_bounded_or_failed_capture_keeps_sanitized_disposition_only(public_page, failure):
    fixture = public_page
    if failure == "text_budget":
        fixture.outer.text = "x" * (fixture.config.max_text_chars + 1)
    elif failure == "invalid_png":
        fixture.state.image = b"PRIVATE invalid image"
    else:

        def failed(_operation):
            raise RuntimeError("PRIVATE DRIVER DETAILS")

        fixture.state.hook = failed
    expected = {
        "text_budget": "public_text_budget",
        "invalid_png": "invalid_viewport_image",
        "driver_error": "public_capture_unavailable",
    }[failure]
    assert_no_content(fixture, capture(fixture, original_failure="PRIVATE failure"), expected)
    assert (
        json.loads((fixture.root / "diagnostic/disposition.json").read_bytes())["original_failure"]
        == "submission_failure"
    )
