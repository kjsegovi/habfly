"""Opt-in fully intercepted two-feature native transfer, not perception proof.

The overview and spectrum are captured through genuine guarded browser reads.
Two zoomed HOVER event bundles are explicitly authored synthetic fixtures whose
source, before-capture, axes and exact-two hint indices match that overview.
No production source, reload, freshness, mapping or copy validator is patched.
Native tooltip perception has a separate sensor test; this file does not claim
live HabWorlds or scientific acceptance, model inference, Save, or submission.
"""

# ruff: noqa: F811 - imported pytest fixture dependencies

import json
import os
from copy import deepcopy

import pytest
from test_browser_raster_planet_evidence import read, sha, sources, write
from test_browser_tooltip_reference_chromium import (  # noqa: F401
    assert_no_answers,
    choose,
    chromium,
    copier,
    page,
    raster_page,
    window_page,
)
from test_planet_tooltip_reference import fixture as synthetic_diagnostic

import habfly.browser_raster_planet_evidence as transport
from habfly.browser import BrowserSafetyStop
from habfly.browser_shallow_transit_probe import (
    EXACT_TWO_HINT_POLICY,
    candidate_columns,
    overview_hint_metadata,
)
from habfly.planet_tooltip_reference import DIAGNOSTIC_FILES, SENSOR_FLAGS, TWO_MODE, load_tooltip_reference

pytestmark = pytest.mark.skipif(
    os.environ.get("HABFLY_INTERCEPTED_BROWSER_IDLE") != "1",
    reason="Root must grant an idle window for intercepted two-tooltip transport fixtures",
)


def bound_two_sources(surface, history):
    """Synthetic sample events bound to the unmodified current screenshot."""
    source = sources(surface, history)
    original = read(source["window_report"])
    png = (source["window_report"].parent / "chart.png").read_bytes()
    groups = candidate_columns(png, original["flux_axis_labels"], hint_policy=EXACT_TWO_HINT_POLICY)
    assert groups == [[76], [168]]
    ticks = sorted((float(row["center_x"]), float(row["value"])) for row in original["time_axis_labels"])
    days_per_pixel = (ticks[-1][1] - ticks[0][1]) / (ticks[-1][0] - ticks[0][0])
    diagnostics = []
    for index, (group, center) in enumerate(zip(groups, (1000, 3000), strict=True)):
        spec = synthetic_diagnostic(
            history, f"synthetic-two-{index}", center, star=original["star"], linked=True
        )
        directory = history / spec["directory"]
        scope, report = read(directory / "scope.json"), read(directory / "report.json")
        overview = history / scope["source_dir"]
        (overview / "chart.png").write_bytes(png)
        (overview / "report.json").write_bytes(source["window_report"].read_bytes())
        pins = {name: sha(overview / name) for name in ("report.json", "chart.png")}
        interval = [
            max(0.0, ticks[0][1] + (group[0] - 0.5 - ticks[0][0]) * days_per_pixel),
            min(5000.0, ticks[0][1] + (group[-1] + 1.5 - ticks[0][0]) * days_per_pixel),
        ]
        metadata = overview_hint_metadata(EXACT_TWO_HINT_POLICY)
        scope.update(
            **metadata,
            source_hashes=pins,
            candidate_index=index,
            candidate_columns=group,
            source_hint_day_interval=interval,
        )
        report.update(**metadata, source_hashes=pins, source_hint_day_interval=interval)
        write(directory / "scope.json", scope)
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
        (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
        diagnostics.append(
            {"directory": directory.name, "files": {name: sha(directory / name) for name in DIAGNOSTIC_FILES}}
        )
    source["tooltip_reference"] = {
        "expected_star": original["star"],
        "diagnostics": diagnostics,
        "mode": TWO_MODE,
    }
    return source


@pytest.fixture
def two_case(raster_page, tmp_path):
    page, frame = raster_page
    # Synthetic controlled UI: keep only two one-pixel shallow overview hints.
    # No production DOM manipulation or recorded PNG repair is performed.
    frame.evaluate("""()=>{
      for(const node of document.querySelectorAll('#chart .dip'))node.remove();
      const chart=document.querySelector('#chart');
      for(const x of [76,168]){
        const hint=document.createElementNS('http://www.w3.org/2000/svg','rect');
        hint.setAttribute('x',String(x));hint.setAttribute('y','21');
        hint.setAttribute('width','1');hint.setAttribute('height','1');
        hint.setAttribute('fill','rgb(55,83,96)');chart.append(hint);
      }
    }""")
    frame.locator("div").first.evaluate("e=>{e.textContent='Dulat';e.style.textTransform='uppercase'}")
    source = bound_two_sources(raster_page, tmp_path)
    result = load_tooltip_reference(
        tmp_path, source["tooltip_reference"]["diagnostics"], expected_star="DULAT", mode=TWO_MODE
    )
    assert result["period_days"]["value"] == "2000"
    assert result["brightness_drop_percent"]["value"] == "0.762"
    assert [feature["overview_candidate_index"] for feature in result["features"]] == [0, 1]
    assert not result["answer_authorized"]
    return page, frame, tmp_path, source


def test_native_yes_and_three_exact_copies_rebuild_two_mode_at_every_boundary(two_case):
    page, frame, root, source = two_case
    owner = transport._Owned(root)
    evidence = transport._evidence(owner, **source)
    receipt = choose(two_case)
    assert receipt["value"] == "Yes" and receipt["readback_verified"]
    assert receipt["numeric_writes"] == 0 and frame.evaluate("window.fixtureSelections") == 1
    assert_no_answers(frame)
    assert receipt["evidence"] == evidence
    for stage in ("fresh", "preselect"):
        assert receipt[stage]["measurement_mode"] == TWO_MODE
        transport._fresh_evidence(owner, root / "presence" / stage, receipt[stage], evidence)
    emitted = []
    component = copier(two_case, emit=emitted.append)
    try:
        assert not component.finished and not component.session.attempted, component.report
        for index, name in enumerate(transport.RAW, 1):
            component.advance()
            assert list(component.session.verified) == list(transport.RAW[:index]), component.report
            assert frame.locator("#" + name).input_value() == evidence["measurements"][name]["value"]
            assert all(frame.locator("#" + n).input_value() == "" for n in transport.RAW[index:])
        report = component.report
        assert component.finished and report["raw_measurement_transport_verified"]
        assert report["numeric_writes"] == 3 and report["mode"] == TWO_MODE
        assert report["evidence"] == evidence
        for key, expected in transport.TWO_TOOLTIP_FLAGS.items():
            assert type(receipt[key]) is type(report[key]) is type(expected)
            assert receipt[key] == report[key] == expected
        for name, spec in evidence["measurements"].items():
            assert report["verified_fields"][name]["unit"] == spec["unit"]
            assert report["verified_fields"][name]["value"] == spec["value"]
        for stage, recorded in [
            ("fresh", report["fresh"]),
            *((f"precopy-{index}", value) for index, value in enumerate(report["precopy"], 1)),
        ]:
            assert recorded["measurement_mode"] == TWO_MODE
            transport._fresh_evidence(owner, root / "raw" / stage, recorded, evidence)
        transport._reload(owner, evidence)
        assert emitted == [json.loads(line) for line in (root / "raw/events.jsonl").read_text().splitlines()]
        assert len(list((root / "raw/native-copies").glob("copy-*-confirmed.json"))) == 3
        assert all(event["provenance"] == TWO_MODE for event in report["native_events"])
        assert all(frame.locator("#" + name).input_value() == "" for name in transport.DERIVED)
        assert all(
            report[key] is False
            for key in ("task_completed", "scientific_verified", "saved", "assessed", "submitted")
        )
        assert not page.get_by_role("checkbox").is_checked()
    finally:
        component.close()


def test_invalid_explicit_mode_stops_before_presence_or_numeric_answers(two_case):
    _, frame, root, source = two_case
    bad = deepcopy(source)
    bad["tooltip_reference"]["mode"] = "unrecognized_two_tooltip_mode"
    with pytest.raises(BrowserSafetyStop, match="invalid_tooltip_descriptor"):
        choose(two_case, source=bad)
    assert frame.evaluate("window.fixtureSelections") == 0
    assert frame.get_by_role("combobox").input_value() == ""
    assert_no_answers(frame)
    assert not list(root.glob("planet-raster-*-reservations"))
    stopped = read(root / "presence/stopped.json")
    assert not stopped["write_may_have_occurred"] and not stopped["reservation_created"]
