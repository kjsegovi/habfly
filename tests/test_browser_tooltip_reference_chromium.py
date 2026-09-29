"""Opt-in intercepted tooltip-reference transport, never native perception proof.

The three zoomed diagnostic bundles are explicitly synthetic source artifacts
from the existing pure builder. Their original/before PNG and axis records are
bound to the genuine intercepted overview capture. No production evidence,
reload, freshness, native mapping, copy, or receipt validator is patched.

Every request is fulfilled locally. The idle-browser flag is checked both at
collection and at the Chromium launch site. Root owns authorizing execution;
this file must not be interpreted as proof against the actual HabWorlds page.
"""

# ruff: noqa: F811 - imported pytest fixture dependencies

import json
import os
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright
from test_browser_numeric import OUTER, WIDGET, config, stellar_html
from test_browser_planet_window_choice import window_page  # noqa: F401
from test_browser_raster_planet_evidence import raster_page, read, sha, sources, write  # noqa: F401
from test_planet_tooltip_reference import fixture as synthetic_diagnostic

import habfly.browser_raster_planet_evidence as transport
from habfly.browser import BrowserSafetyStop
from habfly.browser_raster_steps import RasterInputSteps
from habfly.browser_stellar import SIMULATION_URL
from habfly.planet_tooltip_reference import DIAGNOSTIC_FILES, SENSOR_FLAGS, load_tooltip_reference

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must grant an idle window for intercepted tooltip transport fixtures",
)


@pytest.fixture(scope="module")
def chromium():
    if os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1":
        pytest.skip("No idle browser window granted")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def page(chromium):
    context = chromium.new_context(service_workers="block")
    pages = {
        OUTER: (
            '<label><input type="checkbox">I am ready to submit project.</label>'
            f'<iframe src="{SIMULATION_URL}"></iframe>'
            f'<iframe src="{WIDGET}"></iframe><iframe src="{WIDGET}"></iframe>'
        ),
        SIMULATION_URL: stellar_html(),
        WIDGET: "<button>Update Score</button><button>Submit Project</button>",
    }
    requests, unexpected = [], []

    def fulfill(route):
        url = route.request.url
        requests.append(url)
        if url not in pages:
            unexpected.append(url)
        route.fulfill(
            status=200 if url in pages else 404,
            content_type="text/html; charset=utf-8",
            body=pages.get(url, "Unexpected intercepted fixture request"),
        )

    context.route("**/*", fulfill)
    current = context.new_page()
    try:
        current.goto(OUTER, wait_until="load")
        yield current
        assert requests and not unexpected
        assert current.url == OUTER
    finally:
        context.close()


def bound_tooltip_sources(surface, history):
    """Explicit synthetic tooltip seam; genuine current overview and spectrum.

    This does not make the synthetic HOVER records native. It lets the real
    shared validator and all current-chart guards exercise their complete
    source/hash contract without replacing any production validation function.
    """
    source = sources(surface, history)
    original = read(source["window_report"])
    png = (source["window_report"].parent / "chart.png").read_bytes()
    diagnostics = []
    for index in range(3):
        spec = synthetic_diagnostic(
            history, f"synthetic-diagnostic-{index}", 1000 * (index + 1), star=original["star"]
        )
        directory = history / spec["directory"]
        scope = read(directory / "scope.json")
        overview = history / scope["source_dir"]
        (overview / "chart.png").write_bytes(png)
        (overview / "report.json").write_bytes(source["window_report"].read_bytes())
        source_hashes = {name: sha(overview / name) for name in ("report.json", "chart.png")}
        scope["source_hashes"] = source_hashes
        write(directory / "scope.json", scope)
        report = read(directory / "report.json")
        report["source_hashes"] = source_hashes
        write(directory / "report.json", report)
        (directory / "before.png").write_bytes(png)
        write(
            directory / "before.json",
            {
                "time_axis_labels": original["time_axis_labels"],
                "flux_axis_labels": original["flux_axis_labels"],
                "chart_sha256": original["chart_sha256"],
                **SENSOR_FLAGS,
            },
        )
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        assert events[-1]["event"] == "episode_summary"
        events[-1]["payload"] = report
        (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        diagnostics.append(
            {"directory": directory.name, "files": {name: sha(directory / name) for name in DIAGNOSTIC_FILES}}
        )
    source["tooltip_reference"] = {"diagnostics": diagnostics, "expected_star": original["star"]}
    return source


@pytest.fixture
def tooltip_case(raster_page, tmp_path):
    page, frame = raster_page
    # Casefold identity must preserve the native mapper's display spelling.
    frame.locator("div").first.evaluate("e=>{e.textContent='Dulat';e.style.textTransform='uppercase'}")
    source = bound_tooltip_sources(raster_page, tmp_path)
    result = load_tooltip_reference(
        tmp_path, source["tooltip_reference"]["diagnostics"], expected_star="DULAT"
    )
    assert result["mode"] == transport.TOOLTIP_MODE and not result["answer_authorized"]
    assert result["period_days"]["value"] == "1000"
    assert result["brightness_drop_percent"]["value"] == "0.762"
    yield page, frame, tmp_path, source


def choose(case, *, output="presence", source=None):
    page, _, root, original = case
    return transport.select_raster_detected_planet(
        page, config(), root / output, run_history=root, **(original if source is None else source)
    )


def copier(case, *, output="raw", emit=lambda _: None):
    page, _, root, _ = case
    receipt = root / "presence/confirmed.json"
    return RasterInputSteps(
        page,
        config(),
        root / output,
        run_history=root,
        presence_path=receipt,
        presence_sha256=sha(receipt),
        emit=emit,
    )


def assert_no_answers(frame):
    assert all(frame.locator("#" + name).input_value() == "" for name in transport.RAW)


def finish_raw(case):
    page, frame, root, source = case
    owner = transport._Owned(root)
    evidence = transport._evidence(owner, **source)
    assert evidence["mode"] == transport.TOOLTIP_MODE
    transport._reload(owner, evidence)
    receipt = choose(case)
    assert receipt["readback_verified"] and receipt["numeric_writes"] == 0
    assert receipt["evidence"] == evidence and frame.evaluate("window.fixtureSelections") == 1
    assert_no_answers(frame)
    for name, recorded in (("fresh", receipt["fresh"]), ("preselect", receipt["preselect"])):
        transport._fresh_evidence(owner, root / "presence" / name, recorded, evidence)
    emitted = []
    component = copier(case, emit=emitted.append)
    try:
        assert not component.finished and not component.session.attempted
        for index, name in enumerate(transport.RAW, 1):
            component.advance()
            assert list(component.session.verified) == list(transport.RAW[:index]), component.report
            assert frame.locator("#" + name).input_value() == evidence["measurements"][name]["value"]
            assert all(frame.locator("#" + n).input_value() == "" for n in transport.RAW[index:])
            # Inspecting paused state cannot add an event or a native write.
            before = (root / "raw/events.jsonl").read_bytes()
            component.state()
            assert (root / "raw/events.jsonl").read_bytes() == before
        report = component.report
        assert component.finished and report["raw_measurement_transport_verified"]
        assert report["numeric_writes"] == 3 and report["mode"] == transport.TOOLTIP_MODE
        assert report["action_source"] == transport._action_source(evidence)
        assert report["evidence"] == evidence
        for key, value in transport._flags_for(evidence).items():
            assert report[key] == value
        assert all(not report[key] for key in ("task_completed", "saved", "assessed", "submitted"))
        assert report["evidence"]["measurements"]["brightness_drop"]["physical_bounds"] is None
        assert "lower" not in report["evidence"]["measurements"]["brightness_drop"]
        assert (
            report["evidence"]["measurements"]["period_days"]["compatibility_interval"]["endpoints"] == "open"
        )
        for name, spec in evidence["measurements"].items():
            assert report["verified_fields"][name]["unit"] == spec["unit"]
            assert report["verified_fields"][name]["value"] == spec["value"]
        for index, recorded in enumerate(report["precopy"], 1):
            transport._fresh_evidence(owner, root / "raw" / f"precopy-{index}", recorded, evidence)
        transport._fresh_evidence(owner, root / "raw/fresh", report["fresh"], evidence)
        assert emitted == [json.loads(line) for line in (root / "raw/events.jsonl").read_text().splitlines()]
        assert len(list((root / "raw/native-copies").glob("copy-*-confirmed.json"))) == 3
        assert all(frame.locator("#" + name).input_value() == "" for name in transport.DERIVED)
        assert not page.get_by_role("checkbox").is_checked()
        transport._reload(owner, evidence)
        return report
    finally:
        component.close()


def test_real_tooltip_validators_and_three_cooperative_native_copies(tooltip_case):
    finish_raw(tooltip_case)


def test_wrong_unit_never_reserves_or_types_native_input(tooltip_case):
    _, frame, root, _ = tooltip_case
    choose(tooltip_case)
    component = copier(tooltip_case)
    try:
        assert not component.finished, component.report
        with pytest.raises(BrowserSafetyStop, match="invalid_planet_copy_request"):
            component.session.copy("line_shift", "0.00004788", "day", source="reference_diagnostic")
        assert not component.session.attempted and not component.session.verified
        assert not list((root / "raw/native-copies").glob("copy-*-reserved.json"))
        assert_no_answers(frame)
    finally:
        component.abort()


@pytest.mark.parametrize("effect", ["wrong_readback", "other_field", "replaced_handle"])
def test_readback_side_effect_keeps_uncertainty_and_blocks_later_copies(tooltip_case, effect):
    _, frame, root, _ = tooltip_case
    choose(tooltip_case)
    script = {
        "wrong_readback": "e.value='8'",
        "other_field": "document.querySelector('#period_days').value='5'",
        "replaced_handle": "e.replaceWith(e.cloneNode(true))",
    }[effect]
    frame.locator("#line_shift").evaluate('(e,body)=>e.onblur=()=>{new Function("e",body)(e)}', script)
    component = copier(tooltip_case)
    try:
        component.advance()
        assert component.finished and component.session.attempted == {"line_shift"}
        assert not component.session.verified and not component.report["raw_measurement_transport_verified"]
        assert (root / "raw/native-copies/copy-01-reserved.json").exists()
        stopped = read(root / "raw/native-copies/copy-01-stopped.json")
        assert stopped["write_may_have_occurred"] and not stopped["retry_allowed"]
        assert not list((root / "raw/native-copies").glob("copy-*-confirmed.json"))
        before = frame.get_by_role("textbox").evaluate_all("es=>es.map(e=>e.value)")
        component.advance()
        assert frame.get_by_role("textbox").evaluate_all("es=>es.map(e=>e.value)") == before
        retry = copier(tooltip_case, output="new-output")
        try:
            assert retry.finished and retry.report["outcome"] == "raster_planet_inputs_already_reserved"
        finally:
            retry.close()
    finally:
        component.close()


@pytest.mark.parametrize("first_mode", ["tooltip", "raster"])
def test_mode_switch_cannot_retry_presence_in_new_output(tooltip_case, first_mode):
    _, frame, root, source = tooltip_case
    raster = {key: value for key, value in source.items() if key != "tooltip_reference"}
    first, second = (source, raster) if first_mode == "tooltip" else (raster, source)
    receipt = choose(tooltip_case, source=first)
    assert receipt["mode"] == (transport.TOOLTIP_MODE if first_mode == "tooltip" else transport.MODE)
    with pytest.raises(BrowserSafetyStop, match="presence_already_reserved"):
        choose(tooltip_case, output="changed-mode", source=second)
    assert frame.evaluate("window.fixtureSelections") == 1
    assert_no_answers(frame)
    assert len(list((root / "planet-raster-presence-reservations").glob("*.json"))) == 1
    assert not read(root / "changed-mode/stopped.json")["write_may_have_occurred"]


@pytest.mark.parametrize("effect", ["source_mutation", "chart_mutation", "wrong_visible_unit"])
def test_changed_source_or_current_view_never_selects_yes(tooltip_case, effect):
    _, frame, root, source = tooltip_case
    if effect == "source_mutation":
        directory = root / source["tooltip_reference"]["diagnostics"][0]["directory"]
        (directory / "readable.json").write_text("{}")
    elif effect == "chart_mutation":
        frame.locator(".dip").first.evaluate("e=>e.setAttribute('y2','50.5')")
    else:
        frame.locator("#line_shift").evaluate("e=>e.previousElementSibling.textContent='doppler shift (day)'")
    with pytest.raises(BrowserSafetyStop):
        choose(tooltip_case)
    assert frame.evaluate("window.fixtureSelections") == 0
    assert_no_answers(frame)
    assert not read(root / "presence/stopped.json")["write_may_have_occurred"]


def test_source_change_after_native_reservation_blocks_fill(tooltip_case):
    _, frame, root, source = tooltip_case
    choose(tooltip_case)
    diagnostic = root / source["tooltip_reference"]["diagnostics"][0]["directory"] / "readable.json"

    def emit(event):
        if event["event"] == "action_proposed":
            diagnostic.write_text("{}")

    component = copier(tooltip_case, emit=emit)
    try:
        component.advance()
        assert component.finished and component.session.attempted == {"line_shift"}
        assert not component.session.verified and not component.report["raw_measurement_transport_verified"]
        assert_no_answers(frame)
        assert (root / "raw/native-copies/copy-01-reserved.json").exists()
    finally:
        component.close()


@pytest.mark.skipif(
    not os.environ.get("HABFLY_PLANET_PILOT") or not os.environ.get("HABFLY_PLANET_FINAL"),
    reason="Additional explicit opt-in for existing frozen model; no training or final cases opened",
)
def test_optional_frozen_derived_child_after_tooltip_raw_transport(tooltip_case):
    from habfly.browser_planet_steps import PlanetDerivedSteps
    from habfly.data import load_graph

    page, frame, root, _ = tooltip_case
    report = finish_raw(tooltip_case)
    raw = {name: frame.locator("#" + name).input_value() for name in transport.RAW}
    component = PlanetDerivedSteps(
        page,
        config(),
        root / "derived",
        pilot=Path(os.environ["HABFLY_PLANET_PILOT"]),
        final_evaluation=Path(os.environ["HABFLY_PLANET_FINAL"]),
        graph=load_graph("data/processed/graphs-v2/graph-2000"),
        supplied_star_class="main_sequence",
    )
    try:
        for _ in range(129):
            if component.finished:
                break
            component.advance()
        assert component.finished and component.report["planet_transport_verified"], component.report
        assert component.report["sources_unchanged"] and component.report["optimizer_updates"] == 0
        assert not component.report["task_completed"]
        assert len(list((root / "derived/native-copies").glob("copy-*-confirmed.json"))) == 4
        assert raw == {name: frame.locator("#" + name).input_value() for name in transport.RAW}
        assert read(root / "raw/report.json") == report
        assert not page.get_by_role("checkbox").is_checked()
    finally:
        component.close()
