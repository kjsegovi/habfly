"""Read-only control batching parity and explicit protocol-cost accounting.

Native cases are fully intercepted and require the shared browser-idle gate.
The latency model counts API calls; it is not a real-browser timing claim.
"""

import os
from collections import Counter
from types import SimpleNamespace

import pytest
from playwright.sync_api import sync_playwright

from habfly.browser import BrowserSafetyStop, normalized_control_label
from habfly.browser_probe import PROTECTED_LABELS, _read_controls

ROLES = ("button", "link", "textbox", "spinbutton", "combobox", "checkbox", "radio")
VALUE_ROLES = {"textbox", "spinbutton", "combobox"}
URL = "https://control-batch.fixture.invalid/"


def legacy_controls(frame, frame_id, config):
    """Exact pre-batch implementation, kept as a fixture parity oracle."""
    controls = []
    for role in ROLES:
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
                    "actions": [],
                    "protected": False,
                }
            )
            normalized = normalized_control_label(label)
            controls[-1]["protected"] = any(name in normalized for name in PROTECTED_LABELS)
            if role in VALUE_ROLES:
                tag = item.evaluate("element => element.tagName.toLowerCase()")
                controls[-1]["value"] = item.input_value() if tag in {"input", "textarea", "select"} else None
            if len(controls) > config.max_controls:
                raise BrowserSafetyStop("control_budget_exceeded")
    return controls


HTML = """<!doctype html><html><body>
  <button id=submit>Submit Project</button><button disabled>Disabled button</button>
  <a href="#example">Example link</a>
  <label>Zero<input id=zero value="0"></label>
  <label>Blank<input id=blank value=""></label>
  <label>Text area<textarea id=area>first\nsecond</textarea></label>
  <label>Numeric<input id=number type=number value="1.25"></label>
  <label>Options<select id=select><option value="different-value">Displayed option</option></select></label>
  <label><input type=checkbox checked>I am ready to submit project.</label>
  <label><input type=radio name=one checked>Radio one</label>
  <fieldset disabled>
    <legend><label>Legend exception<input id=legend value="legend"></label></legend>
    <label>Disabled fieldset<input id=disabled-field value="disabled"></label>
  </fieldset>
  <div role=group aria-disabled=true><button>ARIA disabled button</button></div>
  <div id=custom-text role=textbox tabindex=0 aria-label="Custom text">Rendered text</div>
  <div id=custom-spin role=spinbutton tabindex=0 aria-label="Custom spin" style="width:100px;height:20px"
       aria-valuenow=2 aria-valuemin=0 aria-valuemax=10></div>
  <div id=custom-combo role=combobox tabindex=0 aria-label="Custom combo"
       aria-expanded=false>Rendered option</div>
  <label style="display:none">CSS hidden<input id=css-hidden value="private-css"></label>
  <label aria-hidden=true>ARIA hidden<input id=aria-hidden value="private-aria" aria-hidden=true></label>
  <script>
    window.fixtureReads=[]; window.fixtureWrites=[];
    for(const id of ['css-hidden','aria-hidden','custom-text','custom-spin','custom-combo']) {
      Object.defineProperty(document.getElementById(id),'value',{
        get(){window.fixtureReads.push(id);throw new Error('fixture value getter must not run')},
        set(){window.fixtureWrites.push(id);throw new Error('fixture value setter must not run')}
      });
    }
    window.fixtureEvents=[];
    for(const kind of ['input','change','click','submit'])
      document.addEventListener(kind,()=>window.fixtureEvents.push(kind),true);
    window.fixtureMutations=[];
    window.fixtureObserver=new MutationObserver(xs=>window.fixtureMutations.push(...xs.map(x=>x.type)));
    fixtureObserver.observe(document.body,{subtree:true,attributes:true,childList:true,characterData:true});
  </script>
</body></html>"""


@pytest.fixture(scope="module")
def batch_browser():
    if os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1":
        pytest.skip("intercepted native browser requires an explicit idle window")
    with sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def batch_page(batch_browser):
    context = batch_browser.new_context()
    context.route(
        "**/*",
        lambda route: (
            route.fulfill(status=200, content_type="text/html", body=HTML)
            if route.request.url == URL
            else route.abort()
        ),
    )
    page = context.new_page()
    page.goto(URL, wait_until="load")
    # The inline observer starts before the HTML parser inserts the whitespace
    # following </script>. Exclude that load-time childList record, then keep
    # the same observer running across both readers and the no-write assertion.
    page.evaluate("() => { fixtureObserver.takeRecords(); fixtureMutations.length=0; }")
    try:
        yield page
    finally:
        context.close()


def test_native_inventory_exactly_matches_legacy_and_does_not_write(batch_page):
    frame, config = batch_page.main_frame, SimpleNamespace(max_controls=256)
    before = batch_page.locator("body").inner_html()
    expected = legacy_controls(frame, "test", config)
    actual = _read_controls(frame, "test", config)
    assert actual == expected
    assert {item["role"] for item in actual} == set(ROLES)
    assert [item["id"] for item in actual] == [f"test:c{n}" for n in range(len(actual))]
    assert all(item["actions"] == [] for item in actual)
    protected = [item for item in actual if item["protected"]]
    assert len(protected) == 2
    assert all(
        "Submit Project" in x["accessibility"] or "ready to submit" in x["accessibility"] for x in protected
    )
    assert batch_page.locator("body").inner_html() == before
    assert batch_page.evaluate("[fixtureReads,fixtureWrites,fixtureEvents,fixtureMutations]") == [
        [],
        [],
        [],
        [],
    ]


def test_native_values_and_playwright_enabled_semantics_are_preserved(batch_page):
    actual = _read_controls(batch_page.main_frame, "test", SimpleNamespace(max_controls=256))

    def named(label):
        return next(item for item in actual if f'"{label}"' in item["accessibility"])

    for label, value in {
        "Zero": "0",
        "Blank": "",
        "Text area": "first\nsecond",
        "Numeric": "1.25",
        "Options": "different-value",
        "Legend exception": "legend",
        "Disabled fieldset": "disabled",
        "Custom text": None,
        "Custom spin": None,
        "Custom combo": None,
    }.items():
        assert named(label)["value"] == value
    assert named("Legend exception")["enabled"] is True
    assert named("Disabled fieldset")["enabled"] is False
    assert named("Disabled button")["enabled"] is False
    assert named("ARIA disabled button")["enabled"] is False
    assert not any("private-" in item["accessibility"] for item in actual)
    assert batch_page.evaluate("fixtureReads") == []


class HiddenRoleFrame:
    """Force an actual aria-hidden node into the role list to exercise the guard.

    Normal Playwright role enumeration excludes it. This explicit seam models
    a node becoming aria-hidden between enumeration and metadata capture.
    """

    def __init__(self, frame):
        self.frame = frame

    def get_by_role(self, role):
        items = [self.frame.locator("#aria-hidden")] if role == "textbox" else []
        return SimpleNamespace(all=lambda: items)


def test_native_aria_hidden_early_return_never_reads_native_value(batch_page):
    frame = HiddenRoleFrame(batch_page.main_frame)
    assert _read_controls(frame, "test", SimpleNamespace(max_controls=256)) == []
    assert batch_page.evaluate("fixtureReads") == []


def test_native_control_budget_rejection_remains_identical(batch_page):
    for reader in (legacy_controls, _read_controls):
        with pytest.raises(BrowserSafetyStop, match="^control_budget_exceeded$"):
            reader(batch_page.main_frame, "test", SimpleNamespace(max_controls=2))
    assert batch_page.evaluate("[fixtureWrites,fixtureEvents]") == [[], []]


class Cost:
    """Synthetic fixed per-call delay, without real sleeps or driver execution."""

    def __init__(self):
        self.calls = Counter()
        self.seconds = 0.0

    def call(self, method):
        self.calls[method] += 1
        self.seconds += 0.025


class FakeControl:
    def __init__(self, cost, role, *, visible=True, hidden=False, native=True):
        self.cost, self.role = cost, role
        self.visible, self.hidden, self.native = visible, hidden, native
        self.value_reads = 0

    def is_visible(self):
        self.cost.call("is_visible")
        return self.visible

    def get_attribute(self, name):
        assert name == "aria-hidden"
        self.cost.call("get_attribute")
        return "true" if self.hidden else None

    def aria_snapshot(self, **kwargs):
        assert kwargs == {"timeout": 3000}
        self.cost.call("aria_snapshot")
        return f'{self.role} "Visible control"'

    def is_enabled(self):
        self.cost.call("is_enabled")
        return True

    def input_value(self):
        self.cost.call("input_value")
        self.value_reads += 1
        assert self.visible and not self.hidden and self.native
        return "0"

    def evaluate(self, expression, argument=None):
        self.cost.call("evaluate")
        if expression == "element => element.tagName.toLowerCase()":
            return "input" if self.native else "div"
        if self.hidden:
            return {"hidden": True}
        value = None
        if self.role in VALUE_ROLES and self.native:
            self.value_reads += 1
            value = "0"
        return {"hidden": False, "value": value}


class FakeFrame:
    def __init__(self, cost, controls):
        self.cost, self.controls = cost, controls

    def get_by_role(self, role):
        def all_items():
            self.cost.call("all")
            return [control for control in self.controls if control.role == role]

        return SimpleNamespace(all=all_items)


def test_batched_api_call_count_and_modeled_latency_decrease_without_removed_guards():
    # Exact saved No-screen cardinality:40 visible controls,5 native value controls.
    inventories, costs = [], []
    for reader in (legacy_controls, _read_controls):
        cost = Cost()
        controls = [FakeControl(cost, "button") for _ in range(35)]
        controls.extend(FakeControl(cost, "textbox") for _ in range(5))
        inventories.append(reader(FakeFrame(cost, controls), "test", SimpleNamespace(max_controls=256)))
        costs.append(cost)
    assert inventories[0] == inventories[1]
    old, new = costs
    assert old.calls["get_attribute"] == 40 and new.calls["get_attribute"] == 35
    assert old.calls["evaluate"] == new.calls["evaluate"] == 5
    assert old.calls["input_value"] == 5 and new.calls["input_value"] == 0
    for guard in ("is_visible", "aria_snapshot", "is_enabled"):
        assert old.calls[guard] == new.calls[guard] == 40
    assert old.calls["all"] == new.calls["all"] == 7
    assert sum(old.calls.values()) - sum(new.calls.values()) == 10
    assert old.seconds - new.seconds == pytest.approx(0.25)


@pytest.mark.parametrize(
    "visible,hidden,native", [(False, False, True), (True, True, True), (True, False, False)]
)
def test_injected_hidden_or_custom_controls_never_read_value(visible, hidden, native):
    cost = Cost()
    control = FakeControl(cost, "textbox", visible=visible, hidden=hidden, native=native)
    result = _read_controls(FakeFrame(cost, [control]), "test", SimpleNamespace(max_controls=256))
    assert control.value_reads == 0
    if not visible or hidden:
        assert result == []
    else:
        assert result[0]["value"] is None
    if not visible:
        assert cost.calls == Counter({"all": 7, "is_visible": 1})
    elif hidden:
        assert cost.calls == Counter({"all": 7, "is_visible": 1, "evaluate": 1})


def test_injected_budget_still_counts_only_visible_exposed_inventory():
    cost = Cost()
    controls = [FakeControl(cost, "textbox", hidden=True), FakeControl(cost, "button")]
    result = _read_controls(FakeFrame(cost, controls), "test", SimpleNamespace(max_controls=1))
    assert len(result) == 1 and result[0]["id"] == "test:c0"
