"""Native fixtures are opt-in and entirely intercepted, never a live project.

Do not set HABFLY_INTERCEPTED_BROWSER_IDLE=1 while the root owns an active native
browser run. The module marker and launch fixture both enforce this idle gate.
No class/source validator or production adapter is patched in these tests.
"""

import json
import os
import shutil

import pytest
from playwright.sync_api import sync_playwright
from test_browser_full_stellar import full_html
from test_browser_numeric import OUTER, WIDGET, config
from test_browser_setup import canvas_html

from habfly.browser import BrowserSafetyStop
from habfly.browser_class_setup_steps import FreshStarClassSteps
from habfly.browser_classification import read_class_choices
from habfly.browser_next_star import capture_initial_setup_star
from habfly.browser_numeric import screen_identity
from habfly.browser_probe import inspect_page
from habfly.browser_setup import BrowserSetup
from habfly.browser_star_preflight import validate_star_class_source
from habfly.browser_stellar import SIMULATION_URL, map_stellar_capture
from habfly.runtime import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Native intercepted fixture requires an explicitly granted idle browser window",
)

RATIONALE = (
    "This disposable fixture supplies an explicit class decision for transport testing. "
    "The decision is not inferred from the simulated measurements and is not a scientific or training label."
)


@pytest.fixture(scope="module")
def isolated_chromium():
    # Keep a second check next to launch even if a caller overrides test markers.
    if os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1":
        pytest.skip("No idle browser window granted")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def native_fresh_star(isolated_chromium, tmp_path):
    """Real visible navigation/capture; handlers belong only to these fixtures."""
    contexts = []

    def create(*, inherited_main, inherited_class=None):
        assert inherited_class in {None, "white_dwarf"}
        paint = inherited_class or ("main_sequence" if inherited_main else None)
        history = tmp_path / (
            "inherited-" + inherited_class
            if inherited_class
            else "inherited-main"
            if inherited_main
            else "fresh-non-main"
        )
        history.mkdir()
        context = isolated_chromium.new_context(
            viewport={"width": 1600, "height": 1100}, service_workers="block"
        )
        contexts.append(context)
        native_actions, requested_urls = [], []
        detail = full_html().split("<script>")[0]
        if inherited_main:
            detail = detail.replace(
                '<label onclick="choose(this,true)">',
                '<label class="selected" onclick="choose(this,true)">',
                1,
            )
        if inherited_class:
            caption = inherited_class.replace("_", " ")
            detail = detail.replace(
                f'<label onclick="choose(this,false)"></label><div>{caption}</div>',
                f'<label class="selected" onclick="choose(this,false)"></label><div>{caption}</div>',
                1,
            )
        documents = {
            OUTER: (
                '<label><input type="checkbox">I am ready to submit project.</label>'
                f'<iframe style="width:950px;height:600px;border:0;display:block" src="{SIMULATION_URL}"></iframe>'
                f'<iframe src="{WIDGET}"></iframe><iframe src="{WIDGET}"></iframe>'
            ),
            SIMULATION_URL: canvas_html(semantic=True, detail=detail),
            WIDGET: "<button>Update Score</button><button>Submit Project</button>",
        }

        def intercept(route):
            requested_urls.append(route.request.url)
            if route.request.url in documents:
                route.fulfill(
                    status=200, content_type="text/html; charset=utf-8", body=documents[route.request.url]
                )
            else:
                route.abort()  # No request is ever continued to a network origin.

        context.route("**/*", intercept)
        page = context.new_page()
        page.expose_function("recordFixtureSelection", lambda event: native_actions.append(event))
        page.goto(OUTER, wait_until="load")
        frame = page.frame(url=SIMULATION_URL)
        # Fixture-owned event instrumentation records actual native dispatches.
        # The production component never reads these functions or event handlers.
        frame.evaluate("""() => {
            window.choose = (element, main) => {
                window.recordFixtureSelection({target:'stellar_class',
                    value:element.nextElementSibling.innerText.trim().replaceAll(' ','_')});
                if (element.classList.contains('selected')) return;
                document.querySelectorAll('.choice label').forEach(x=>x.classList.remove('selected'));
                element.classList.add('selected');
                document.querySelector('#conditional').style.display=main?'block':'none';
                document.querySelector('#warning').style.display=main?'none':'block';
            };
            document.addEventListener('change', event => {
                if (event.target.id === 'prefix') {
                    document.querySelector('#lifetime').value = '0.000';
                    window.recordFixtureSelection({target:'lifetime_prefix',
                        value:event.target.selectedOptions[0].innerText});
                }
            });
        }""")
        setup_output = history / "setup"
        setup_output.mkdir()
        setup = BrowserSetup(page, config(), ("", ""), output=setup_output)
        for _ in range(80):
            if setup.advance() == "stellar":
                break
            page.wait_for_timeout(100)
        assert setup.closed and setup.stage == "stellar_screen_ready"
        fresh = history / "initial-star"
        receipt = capture_initial_setup_star(setup, fresh)
        assert receipt["fresh_blank_numeric_answers_verified"]
        assert receipt["painted_stellar_class"] == paint
        assert not receipt["class_selection_verified"] and receipt["answer_writes"] == 0
        assert not native_actions
        assert set(requested_urls) <= set(documents)
        return page, frame, history, fresh, native_actions

    yield create
    for context in contexts:
        context.close()


def create_steps(page, history, fresh, selected, prefix, events, output="class-setup"):
    return FreshStarClassSteps(
        page,
        config(),
        history / output,
        run_history=history,
        fresh_star=fresh,
        selected_class=selected,
        reference_rationale=RATIONALE,
        lifetime_prefix=prefix,
        max_seconds=180,
        max_advances=4,
        emit=events.append,
    )


def assert_exact_events(steps, forwarded):
    events = read_trace(steps.output / "events.jsonl")
    assert [e.model_dump(mode="json") for e in events] == forwarded
    class_events = [
        json.loads(path.read_bytes()) for path in sorted((steps.output / "class").glob("event-*.json"))
    ]
    assert class_events == [
        {"event": e.event, "payload": e.payload}
        for e in events
        if e.event in {"action_proposed", "action_result"}
        and (e.payload.get("target") == "stellar_class" or "selected_class" in e.payload)
    ]
    assert all(e.version == 1 for e in events)
    assert not any(e.payload.get("task_completed") for e in events)


def test_native_inherited_white_dwarf_clears_through_red_giant_without_prefix(native_fresh_star):
    page, _, history, fresh, native_actions = native_fresh_star(
        inherited_main=False,
        inherited_class="white_dwarf",
    )
    events = []
    steps = create_steps(page, history, fresh, "white_dwarf", None, events)
    steps.advance()
    assert native_actions == [{"target": "stellar_class", "value": "red_giant"}]
    assert not steps.finished
    steps.advance()
    assert steps.report["setup_verified"], steps.report
    assert native_actions == [
        {"target": "stellar_class", "value": "red_giant"},
        {"target": "stellar_class", "value": "white_dwarf"},
    ]
    assert not (steps.output / "prefix").exists()
    assert not (steps.output / "class-operation-failure").exists()
    assert_exact_events(steps, events)


def test_native_inherited_main_has_separate_class_clicks_and_ga_prefix(native_fresh_star):
    page, frame, history, fresh, native_actions = native_fresh_star(inherited_main=True)
    forwarded = []
    before = screen_identity(inspect_page(page, config()))
    steps = create_steps(page, history, fresh, "main_sequence", "Ga", forwarded)
    assert not native_actions and screen_identity(inspect_page(page, config())) == before
    try:
        steps.advance()
        assert native_actions == [{"target": "stellar_class", "value": "white_dwarf"}]
        assert read_class_choices(frame)[0]["selected"] == "white_dwarf"
        assert not steps.finished and not (steps.output / "class/confirmed.json").exists()

        steps.advance()
        assert native_actions[-1] == {"target": "stellar_class", "value": "main_sequence"}
        assert len(native_actions) == 2 and steps.phase == "prefix_pending"
        source = validate_star_class_source(history, steps.output / "class", "Althinagon", "main_sequence")
        assert source["class_clicks"] == 2 and source == steps.class_source
        assert all(
            control.input_value() == ""
            for control in frame.get_by_role("textbox").all()
            if control.is_visible()
        )
        class_after = (steps.output / "class/after/observation.json").read_bytes()

        steps.advance()
        assert native_actions[-1] == {"target": "lifetime_prefix", "value": "Ga"}
        assert len(native_actions) == 3 and steps.report["setup_verified"]
        assert frame.locator("#lifetime").input_value() == "0.000"
        assert frame.locator("#prefix").input_value() == "Ga"
        assert all(
            frame.locator(f"#{name}").input_value() == ""
            for name in ("distance", "luminosity", "temperature", "mass", "radius")
        )
        assert (steps.output / "class/after/observation.json").read_bytes() == class_after
        assert (
            validate_star_class_source(history, steps.output / "class", "Althinagon", "main_sequence")
            == source
        )
        prefix = json.loads((steps.output / "prefix/confirmed.json").read_bytes())
        assert prefix["readback_verified"] and prefix["blank_lifetime_initialized_to_zero"]
        assert not prefix["task_completed"] and prefix["numeric_answer_writes"] == 0
        assert (fresh / "lifetime-prefix-reserved.json").exists()
        assert not page.get_by_role("checkbox").is_checked()
        assert_exact_events(steps, forwarded)
        steps.advance()
        assert len(native_actions) == 3
        with pytest.raises(BrowserSafetyStop):
            create_steps(page, history, fresh, "main_sequence", "Ga", [], output="retry")
        assert len(native_actions) == 3
    finally:
        steps.close()


def test_native_fresh_non_main_rejects_swapped_source_and_unsupported_class(native_fresh_star):
    page, frame, history, fresh, native_actions = native_fresh_star(inherited_main=False)
    swapped = history / "swapped-source"
    shutil.copytree(fresh, swapped)
    raw = swapped / "stellar/observation.json"
    raw.write_bytes(raw.read_bytes() + b"\n")  # Immutable observation bytes no longer match their manifest.
    with pytest.raises(BrowserSafetyStop):
        create_steps(page, history, swapped, "white_dwarf", None, [], output="bad-source")
    with pytest.raises(BrowserSafetyStop, match="invalid_explicit_class"):
        create_steps(page, history, fresh, "unsupported_class", None, [], output="bad-class")
    assert not native_actions and not (fresh / "class-selection-reserved.json").exists()

    forwarded = []
    steps = create_steps(page, history, fresh, "white_dwarf", None, forwarded)
    try:
        assert not native_actions
        steps.advance()
        assert native_actions == [{"target": "stellar_class", "value": "white_dwarf"}]
        assert steps.report["setup_verified"] and not steps.report["task_completed"]
        assert not steps.report["scientific_verified"] and not steps.report["classification_learned"]
        source = validate_star_class_source(history, steps.output / "class", "Althinagon", "white_dwarf")
        assert source == steps.class_source and source["class_clicks"] == 1
        report = inspect_page(page, config())
        mapping = map_stellar_capture(report, capture_sha256=screen_identity(report))
        assert set(mapping["observation"]["values"]["browser_field_map"]) == {
            "distance",
            "luminosity",
            "temperature",
        }
        assert read_class_choices(frame)[0]["selected"] == "white_dwarf"
        assert (
            not (steps.output / "prefix").exists() and not (fresh / "lifetime-prefix-reserved.json").exists()
        )
        assert not page.get_by_role("checkbox").is_checked()
        assert_exact_events(steps, forwarded)
        steps.advance()
        assert len(native_actions) == 1
    finally:
        steps.close()
