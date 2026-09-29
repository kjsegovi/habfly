"""Opt-in fresh native handles, with intercepted parity and mutation fixtures.

No actual preview, answers, network responses, or model are used. Pure cases
track role re-resolution rather than claiming real-browser timing gains.
"""

from collections import Counter
from types import SimpleNamespace

import pytest
import test_browser_control_batch as fixtures
import test_browser_probe as probe_fixtures
from playwright.sync_api import Error as PlaywrightError

import habfly.browser_probe as module
from habfly.browser import BrowserSafetyStop

batch_browser, batch_page = fixtures.batch_browser, fixtures.batch_page
probe_page = probe_fixtures.page
CONFIG = SimpleNamespace(max_controls=256)


@pytest.fixture(scope="module")
def chromium(batch_browser):
    # Reuse the active synchronous driver instead of aliasing its fixture under
    # a second name (which would enter a second Playwright event loop).
    return batch_browser


class NativeSnapshot:
    def __init__(self, locator, owner, role, index):
        self.locator, self.owner, self.role, self.index = locator, owner, role, index

    def aria_snapshot(self, **kwargs):
        value = self.locator.aria_snapshot(**kwargs)
        if self.owner.mutation and not self.owner.changed:
            self.owner.changed = True
            self.owner.mutation()
        return value


class NativeRole:
    def __init__(self, locator, owner, role):
        self.locator, self.owner, self.role = locator, owner, role

    def element_handles(self):
        handles = self.locator.element_handles()
        for handle in handles:
            original = handle.dispose

            def dispose(handle=handle, original=original):
                self.owner.disposed.append(handle)
                return original()

            self.owner.monkeypatch.setattr(handle, "dispose", dispose)
            self.owner.handles.append(handle)
        return handles

    def nth(self, index):
        return NativeSnapshot(self.locator.nth(index), self.owner, self.role, index)

    def evaluate_all(self, expression, argument):
        self.owner.compared.append(self.role)
        return self.locator.evaluate_all(expression, argument)


class NativeFrame:
    """Instrument real objects; no production validator or result is replaced."""

    def __init__(self, frame, monkeypatch, mutation=None):
        self.frame, self.monkeypatch, self.mutation = frame, monkeypatch, mutation
        self.handles, self.disposed, self.compared = [], [], []
        self.changed = False

    def get_by_role(self, role):
        return NativeRole(self.frame.get_by_role(role), self, role)


def test_native_pinned_inventory_is_exact_legacy_and_current_default(batch_page, monkeypatch):
    frame = NativeFrame(batch_page.main_frame, monkeypatch)
    before = batch_page.locator("body").inner_html()
    legacy = fixtures.legacy_controls(batch_page.main_frame, "test", CONFIG)
    default = module._read_controls(batch_page.main_frame, "test", CONFIG)
    pinned = module._read_controls_pinned(frame, "test", CONFIG)
    assert pinned == default == legacy
    assert frame.compared == list(fixtures.ROLES)
    assert frame.handles and frame.disposed == frame.handles
    assert batch_page.locator("body").inner_html() == before
    assert batch_page.evaluate("[fixtureReads,fixtureWrites,fixtureEvents,fixtureMutations]") == [
        [],
        [],
        [],
        [],
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "document.querySelector('#submit').replaceWith(document.querySelector('#submit').cloneNode(true))",
        "document.body.insertBefore(document.querySelectorAll('button')[1],document.querySelector('#submit'))",
        "document.querySelector('#zero').style.display='none'",
        "document.querySelector('#zero').setAttribute('aria-hidden','true')",
        "document.querySelector('#zero').remove()",
        "document.querySelector('#zero').setAttribute('role','combobox')",
        "document.body.insertAdjacentHTML('beforeend','<button>New button</button>')",
    ],
    ids=["replace", "reorder", "css-hide", "aria-hide", "detach", "role-change", "add"],
)
def test_native_mid_capture_inventory_mutation_cannot_be_accepted(batch_page, monkeypatch, mutation):
    frame = NativeFrame(batch_page.main_frame, monkeypatch, lambda: batch_page.evaluate(mutation))
    with pytest.raises((BrowserSafetyStop, PlaywrightError)):
        module._read_controls_pinned(frame, "test", CONFIG)
    assert frame.changed and frame.disposed == frame.handles
    assert batch_page.evaluate("[fixtureWrites,fixtureEvents]") == [[], []]


def test_native_pinned_budget_stop_disposes_every_enumerated_handle(batch_page, monkeypatch):
    frame = NativeFrame(batch_page.main_frame, monkeypatch)
    with pytest.raises(BrowserSafetyStop, match="^control_budget_exceeded$"):
        module._read_controls_pinned(frame, "test", SimpleNamespace(max_controls=1))
    assert frame.handles and frame.disposed == frame.handles
    assert batch_page.evaluate("[fixtureReads,fixtureWrites,fixtureEvents,fixtureMutations]") == [
        [],
        [],
        [],
        [],
    ]


@pytest.mark.parametrize("configured", [False, True])
def test_native_complete_page_frame_auth_and_output_contract_is_unchanged(probe_page, configured):
    config = probe_fixtures.config()
    old = module.inspect_page(probe_page, config)
    config.pinned_control_capture = configured
    new = module.inspect_page(probe_page, config, pin_controls=not configured)
    assert {k: v for k, v in old.items() if k != "captured_at"} == {
        k: v for k, v in new.items() if k != "captured_at"
    }
    probe_page.frames[1].locator("body").evaluate(
        "node=>node.insertAdjacentHTML('beforeend','<div role=dialog>Unexpected</div>')"
    )
    with pytest.raises(BrowserSafetyStop, match="^unexpected_modal$"):
        module.inspect_page(probe_page, config, pin_controls=not configured)


class Handle:
    def __init__(self, frame, role="button", *, visible=True, hidden=False, native=False):
        self.frame, self.role = frame, role
        self.visible, self.hidden, self.native = visible, hidden, native
        self.connected, self.disposed, self.value_reads = True, 0, 0

    def is_visible(self):
        self.frame.calls["handle_visible"] += 1
        return self.visible

    def evaluate(self, expression, argument):
        assert argument is (self.role in fixtures.VALUE_ROLES)
        self.frame.calls["handle_metadata"] += 1
        if self.hidden:
            return {"hidden": True}
        value = None
        if argument and self.native:
            self.value_reads += 1
            value = "0"
        return {"hidden": False, "value": value}

    def is_enabled(self):
        self.frame.calls["handle_enabled"] += 1
        return True

    def dispose(self):
        self.disposed += 1
        self.frame.calls["dispose"] += 1
        if self.frame.dispose_error:
            raise PlaywrightError("fixture closed driver")


class Role:
    def __init__(self, frame, role):
        self.frame, self.role = frame, role

    def element_handles(self):
        self.frame.calls["enumerate"] += 1
        if self.role == self.frame.enumeration_failure:
            raise BrowserSafetyStop("fixture_enumeration_failed")
        return list(self.frame.controls[self.role])

    def nth(self, index):
        def snapshot(**kwargs):
            assert kwargs == {"timeout": 3000}
            self.frame.calls["locator_ax"] += 1
            if self.frame.snapshot_error:
                raise BrowserSafetyStop("fixture_snapshot_failed")
            return f'{self.role} "Visible control"'

        return SimpleNamespace(aria_snapshot=snapshot)

    def evaluate_all(self, expression, argument):
        self.frame.calls["inventory_compare"] += 1
        current = self.frame.controls[self.role]
        return len(current) == len(argument) and all(
            a is b and a.connected and b.connected for a, b in zip(current, argument)
        )


class Frame:
    def __init__(self):
        self.calls, self.controls = Counter(), {role: [] for role in fixtures.ROLES}
        self.snapshot_error = self.dispose_error = False
        self.enumeration_failure = None

    def get_by_role(self, role):
        return Role(self, role)

    def add(self, role="button", **kwargs):
        handle = Handle(self, role, **kwargs)
        self.controls[role].append(handle)
        return handle


def test_pure_fresh_enumeration_keeps_ax_but_avoids_repeated_native_role_resolution():
    frame = Frame()
    handles = [frame.add() for _ in range(35)] + [frame.add("textbox", native=True) for _ in range(5)]
    first = module._read_controls_pinned(frame, "test", CONFIG)
    assert len(first) == 40
    assert frame.calls == Counter(
        enumerate=7,
        locator_ax=40,
        inventory_compare=7,
        handle_visible=40,
        handle_metadata=40,
        handle_enabled=40,
        dispose=40,
    )
    assert all(handle.disposed == 1 for handle in handles)
    module._read_controls_pinned(frame, "test", CONFIG)
    assert frame.calls["enumerate"] == 14  # No per-role/cross-capture cache.
    assert all(handle.disposed == 2 for handle in handles)


@pytest.mark.parametrize(
    "visible,hidden,native", [(False, False, True), (True, True, True), (True, False, False)]
)
def test_pure_hidden_or_custom_handle_value_is_never_read(visible, hidden, native):
    frame = Frame()
    handle = frame.add("textbox", visible=visible, hidden=hidden, native=native)
    controls = module._read_controls_pinned(frame, "test", CONFIG)
    assert handle.value_reads == 0 and handle.disposed == 1
    if visible and not hidden:
        assert controls[0]["value"] is None
    else:
        assert controls == []


@pytest.mark.parametrize("where", ["snapshot", "enumeration"])
def test_pure_handles_dispose_on_partial_capture_without_masking_original_failure(where):
    frame = Frame()
    handle = frame.add()
    frame.dispose_error = True
    if where == "snapshot":
        frame.snapshot_error = True
    else:
        frame.enumeration_failure = "link"
    with pytest.raises(BrowserSafetyStop, match=f"fixture_{where}_failed"):
        module._read_controls_pinned(frame, "test", CONFIG)
    assert handle.disposed == 1


@pytest.mark.parametrize("value", [1, 0, None, "true", [], {}])
def test_pure_pin_optin_is_strict_before_any_page_or_frame_access(value):
    with pytest.raises(ValueError, match="pin_controls"):
        module.inspect_page(None, None, pin_controls=value)
    with pytest.raises(ValueError, match="pin_controls"):
        module._read_frame(None, "test", None, pin_controls=value)


@pytest.mark.parametrize("optin", [False, True])
def test_pure_frame_routes_only_explicit_optin(monkeypatch, optin):
    calls = []
    monkeypatch.setattr(module, "_check_auth_and_modals", lambda _: None)
    monkeypatch.setattr(module, "_read_controls", lambda *_: calls.append("legacy") or [])
    monkeypatch.setattr(module, "_read_controls_pinned", lambda *_: calls.append("pinned") or [])
    body = SimpleNamespace(inner_text=lambda **_: "Visible", aria_snapshot=lambda **_: "Visible AX")
    frame = SimpleNamespace(url=fixtures.URL, locator=lambda _: body)
    result = module._read_frame(frame, "test", SimpleNamespace(max_text_chars=100), pin_controls=optin)
    assert result["controls"] == []
    assert calls == ["pinned" if optin else "legacy"]
