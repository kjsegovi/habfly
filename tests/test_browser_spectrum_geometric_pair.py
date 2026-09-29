"""Intercepted native spectrum-pair fixtures; no real preview or catalog.

All requests use the existing locally fulfilled fixture. Labels/handlers below
are explicit synthetic UI, not claims about Azryisil's unobserved second label.
"""

import json

import pytest
import test_browser_planet_chart as chart_fixtures
import test_browser_planet_spectrum as spectrum_fixtures
import test_browser_planet_window_choice as window_fixtures
import test_browser_raster_planet_evidence as raster_fixtures
from test_browser_numeric import config

import habfly.browser_planet_spectrum as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import capture_observation_progress
from habfly.planet_charts import spectrum_excursion

BLUE = "656.29999868nm"
RED = "656.30000132nm"
chart_page, chromium, page = chart_fixtures.chart_page, chart_fixtures.chromium, chart_fixtures.page
session, spectrum_page = chart_fixtures.session, spectrum_fixtures.spectrum_page
window_page, raster_page = window_fixtures.window_page, raster_fixtures.raster_page


def install_pair(
    spectrum_page, *, left=BLUE, right=RED, exact_geometry=False, delayed_right=False, marker_separation=None
):
    _, frame = spectrum_page
    parent = frame.get_by_text("656.3nm", exact=True).locator("..")
    if exact_geometry:
        frame.frame_element().evaluate(
            "e=>{e.style.position='absolute';e.style.left='325px';e.style.top='100px';e.style.border='0'}"
        )
        frame.locator("body").evaluate("e=>e.style.margin='0'")
    parent.evaluate(
        """(p,a)=>{
          const strip=p.children[0],tip=p.children[1];
          if(a.exact){
            p.style.position='absolute';p.style.left='32px';p.style.top='136.5px';
            // A normal CSS scale gives the saved sub-layout-pixel geometry;
            // no assumption about the real page's transform is made here.
            p.style.width='570px';p.style.height='200px';p.style.fontSize='32px';
            p.style.transform='scale(0.5)';p.style.transformOrigin='0 0';
            strip.style.width='570px';strip.style.height='80px';
            tip.style.top='90px';tip.style.left='120px';p.children[2].style.top='150px';
            strip.children[0].style.left='283.75px';
            strip.children[1].style.left='286.234375px';
            if(a.separation!==null)strip.children[1].style.left=(283.75+2*a.separation)+'px';
            for(const line of strip.children){line.style.top='4px';line.style.width='2px';line.style.height='76px';}
          }
          const show=text=>{tip.style.display='block';tip.innerText=text;};
          strip.children[0].onmousemove=()=>show(a.left);
          strip.children[1].onmousemove=()=>{
            if(a.delay)setTimeout(()=>show(a.right),150);else show(a.right);
          };
        }""",
        {
            "left": left,
            "right": right,
            "exact": exact_geometry,
            "delay": delayed_right,
            "separation": marker_separation,
        },
    )
    return parent


def record_moves(page, monkeypatch):
    calls, move = [], page.mouse.move

    def wrapped(x, y, **kwargs):
        calls.append((x, y))
        return move(x, y, **kwargs)

    monkeypatch.setattr(page.mouse, "move", wrapped)
    return calls


@pytest.mark.parametrize("reversed_labels", [False, True])
@pytest.mark.parametrize("exact_geometry", [False, True])
def test_native_geometric_pair_uses_actual_labels_not_left_polarity(
    spectrum_page, monkeypatch, reversed_labels, exact_geometry
):
    left, right = (RED, BLUE) if reversed_labels else (BLUE, RED)
    parent = install_pair(spectrum_page, left=left, right=right, exact_geometry=exact_geometry)
    boxes = [
        entry.bounding_box() for entry in parent.locator(":scope > div").first.locator(":scope > div").all()
    ]
    if exact_geometry:
        expected = [
            {"x": 498.875, "y": 238.5, "width": 1, "height": 38},
            {"x": 500.1171875, "y": 238.5, "width": 1, "height": 38},
        ]
        lines = parent.locator(":scope > div").first.locator(":scope > div").all()
        for line, box, recorded in zip(lines, boxes, expected, strict=True):
            client = line.evaluate(
                "e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}}"
            )
            assert {**client, "x": client["x"] + 325, "y": client["y"] + 100} == recorded
            # CDP page boxes can round an exact DOM 1/128px coordinate to1/64.
            assert box == pytest.approx(recorded, abs=1 / 64)
        exposed = all(line.evaluate(module.VISIBLE_TOOLTIP) for line in lines)
    moves = record_moves(spectrum_page[0], monkeypatch)
    s, events = session(spectrum_page, max_actions=2, max_seconds=10)
    try:
        if exact_geometry and not exposed:
            # This headless layout may hit the neighbor at the left center,
            # unlike the live diagnostic's hit=true. Do not fake equivalent
            # exposure, weaken the guard, or call this a successful pair read.
            with pytest.raises(BrowserSafetyStop, match="excursion_marker_occluded"):
                module.read_geometric_spectrum_pair(s)
            assert not moves and s.actions == 0 and s.stopped
            return
        result = module.read_geometric_spectrum_pair(s)
        assert result == {
            "readings": {"blue": BLUE, "red": RED},
            "marker_readings": {"left": left, "right": right},
            "result": spectrum_excursion("656.3nm", BLUE, RED),
        }
        assert moves == [(box["x"] + 0.5, box["y"] + box["height"] / 2) for box in boxes]
        assert s.actions == 2
        proposed = [p for kind, p in events if kind == "action_proposed"]
        samples = [p["spectrum_sample"] for kind, p in events if kind == "action_result"]
        assert [p["marker"] for p in proposed] == ["left", "right"]
        assert [p["sequence"] for p in proposed] == [1, 2]
        assert [(p["marker"], p["wavelength_text"]) for p in samples] == [("left", left), ("right", right)]
        assert spectrum_page[1].get_by_role("textbox").nth(1).input_value() == ""
    finally:
        s.close()


def test_native_second_endpoint_waits_for_distinct_label_without_rehover(spectrum_page, monkeypatch):
    install_pair(spectrum_page, left=RED, right=BLUE, delayed_right=True)
    moves = record_moves(spectrum_page[0], monkeypatch)
    s, _ = session(spectrum_page, max_actions=2, max_seconds=10)
    try:
        result = module.read_geometric_spectrum_pair(s)
        assert result["marker_readings"] == {"left": RED, "right": BLUE}
        assert len(moves) == s.actions == 2
    finally:
        s.close()


def test_native_nearby_subpixel_pair_with_exposed_three_pixel_separation(spectrum_page, monkeypatch):
    # Preserve the saved left subpixel position but deliberately widen the
    # pair. This is synthetic success coverage, not an exact live reproduction.
    parent = install_pair(spectrum_page, left=RED, right=BLUE, exact_geometry=True, marker_separation=3)
    lines = parent.locator(":scope > div").first.locator(":scope > div").all()
    assert [line.bounding_box() for line in lines] == [
        {"x": 498.875, "y": 238.5, "width": 1, "height": 38},
        {"x": 501.875, "y": 238.5, "width": 1, "height": 38},
    ]
    assert all(line.evaluate(module.VISIBLE_TOOLTIP) for line in lines)
    moves = record_moves(spectrum_page[0], monkeypatch)
    s, _ = session(spectrum_page, max_actions=2, max_seconds=10)
    try:
        result = module.read_geometric_spectrum_pair(s)
        assert result["readings"] == {"blue": BLUE, "red": RED}
        assert result["marker_readings"] == {"left": RED, "right": BLUE}
        assert moves == [(499.375, 257.5), (502.375, 257.5)]
        assert s.actions == 2
    finally:
        s.close()


@pytest.mark.parametrize(
    "left,right",
    [(RED, RED), (RED, "656.30000264nm"), ("0nm", RED), ("656.3nm", RED), (BLUE, "656.30000264nm")],
)
def test_native_equal_same_side_zero_center_or_asymmetric_labels_never_produce_pair(
    spectrum_page, monkeypatch, left, right
):
    install_pair(spectrum_page, left=left, right=right)
    moves = record_moves(spectrum_page[0], monkeypatch)
    s, events = session(spectrum_page, max_actions=2, max_seconds=10)
    try:
        with pytest.raises(BrowserSafetyStop):
            module.read_geometric_spectrum_pair(s)
        assert s.stopped and len(moves) <= 2 and s.actions <= 2
        assert all(p.get("task_completed") is False for k, p in events if k == "action_result")
        assert spectrum_page[1].get_by_role("textbox").nth(1).input_value() == ""
    finally:
        s.close()


@pytest.mark.parametrize("change", ["replace_right", "replace_parent", "cover_right", "move_right"])
def test_native_pair_keeps_original_second_marker_and_context_bound(spectrum_page, monkeypatch, change):
    parent = install_pair(spectrum_page)
    moves = record_moves(spectrum_page[0], monkeypatch)
    s, events = session(spectrum_page, max_actions=2, max_seconds=10)
    original_emit = s.emit

    def mutate(kind, payload):
        original_emit(kind, payload)
        if kind == "action_result" and payload.get("sequence") == 1:
            parent.evaluate(
                """(p,change)=>{
                  const right=p.children[0].children[1];
                  if(change==='replace_right')right.replaceWith(right.cloneNode(true));
                  if(change==='replace_parent')p.replaceWith(p.cloneNode(true));
                  if(change==='move_right')right.style.left='175px';
                  if(change==='cover_right'){
                    const cover=document.createElement('div'),r=right.getBoundingClientRect();
                    cover.style=`position:fixed;left:${r.x-2}px;top:${r.y}px;width:5px;height:${r.height}px;background:white;z-index:9999`;
                    document.body.append(cover);
                  }
                }""",
                change,
            )

    s.emit = mutate
    try:
        with pytest.raises(BrowserSafetyStop):
            module.read_geometric_spectrum_pair(s)
        assert s.stopped and len(moves) == 1
        assert len([p for k, p in events if k == "action_result"]) == 1
    finally:
        s.close()


def test_native_callback_cancellation_prevents_second_move(spectrum_page, monkeypatch):
    install_pair(spectrum_page)
    moves = record_moves(spectrum_page[0], monkeypatch)
    s, _ = session(spectrum_page, max_actions=2, max_seconds=10)
    original_emit = s.emit

    def abort(kind, payload):
        original_emit(kind, payload)
        if kind == "action_result" and payload.get("sequence") == 1:
            s.stopped = True

    s.emit = abort
    try:
        with pytest.raises(BrowserSafetyStop, match="chart_session_stopped"):
            module.read_geometric_spectrum_pair(s)
        assert len(moves) == 1 and s.actions == 1
    finally:
        s.close()


def test_native_geometric_capture_preserves_raw_events_and_explicit_mode(spectrum_page, tmp_path):
    install_pair(spectrum_page, left=RED, right=BLUE)
    directory = tmp_path / "geometric"
    report = module.capture_spectrum_excursion(
        spectrum_page[0], config(), directory, expected_star="Jyremis", marker_mode=module.GEOMETRIC_PAIR_MODE
    )
    assert report["marker_mode"] == "visible_geometric_spectrum_pair_v1"
    assert report == json.loads((directory / "spectrum.json").read_bytes())
    assert report["result"] == spectrum_excursion("656.3nm", BLUE, RED)
    assert [event["kind"] for event in report["events"]] == [
        "observation",
        "action_proposed",
        "action_result",
        "action_proposed",
        "action_result",
    ]
    assert report["events"][2]["payload"]["spectrum_sample"]["wavelength_text"] == RED
    assert report["events"][4]["payload"]["spectrum_sample"]["wavelength_text"] == BLUE
    assert not (directory / "stopped.json").exists()


def test_native_reversed_pair_survives_every_fresh_check_and_three_raw_copies(raster_page, tmp_path):
    page, frame = raster_page
    left, right = "656.30004788nm", "656.29995212nm"
    frame.locator("#spectrum-block").evaluate(
        """(p,labels)=>{
          const strip=p.children[0],tip=p.children[1];
          for(const [index,text] of labels.entries())strip.children[index].onmousemove=()=>{
            tip.style.display='block';tip.innerText=text;
          };
        }""",
        [left, right],
    )
    spectrum_dir = tmp_path / "geometric-source"
    module.capture_spectrum_excursion(
        page, config(), spectrum_dir, expected_star="Jyremis", marker_mode=module.GEOMETRIC_PAIR_MODE
    )
    capture_observation_progress(page, config(), tmp_path / "source", requested_days=5000)
    window_report, spectrum_path = tmp_path / "source/report.json", spectrum_dir / "spectrum.json"
    source = {
        "window_report": window_report,
        "window_report_sha256": raster_fixtures.sha(window_report),
        "spectrum_path": spectrum_path,
        "spectrum_sha256": raster_fixtures.sha(spectrum_path),
    }
    receipt = raster_fixtures.select(raster_page, tmp_path, source)
    assert receipt["readback_verified"] and receipt["value"] == "Yes"
    assert receipt["numeric_writes"] == 0 and frame.evaluate("window.fixtureSelections") == 1
    result = raster_fixtures.copy(raster_page, tmp_path)
    assert result["raw_measurement_transport_verified"] and result["numeric_writes"] == 3
    assert list(result["verified_fields"]) == list(raster_fixtures.module.RAW)
    assert result["evidence"]["measurements"]["line_shift"]["value"] == "0.00004788"
    for name, expected in result["evidence"]["measurements"].items():
        assert frame.locator("#" + name).input_value() == expected["value"]
        assert result["verified_fields"][name]["unit"] == expected["unit"]
    assert all(frame.locator("#" + name).input_value() == "" for name in raster_fixtures.module.DERIVED)
    assert not page.get_by_role("checkbox").is_checked()
    assert result["saved"] is result["assessed"] is result["submitted"] is False
    assert result["task_completed"] is result["learned_perception"] is result["scientific_verified"] is False
    artifact_paths = [spectrum_path] + [
        tmp_path / relative / "spectrum.json"
        for relative in (
            "presence/fresh",
            "presence/preselect",
            "copy/fresh",
            "copy/precopy-1",
            "copy/precopy-2",
            "copy/precopy-3",
        )
    ]
    for path in artifact_paths:
        artifact = raster_fixtures.read(path)
        assert artifact["marker_mode"] == module.GEOMETRIC_PAIR_MODE
        assert artifact["result"] == spectrum_excursion("656.3nm", right, left)
        samples = [
            event["payload"]["spectrum_sample"]
            for event in artifact["events"]
            if event["kind"] == "action_result"
        ]
        assert [(sample["marker"], sample["wavelength_text"]) for sample in samples] == [
            ("left", left),
            ("right", right),
        ]
        assert len(artifact["events"]) == 5


def test_native_stale_pair_capture_keeps_failure_without_spectrum_receipt(spectrum_page, tmp_path):
    install_pair(spectrum_page, left=RED, right=RED)
    directory = tmp_path / "stale"
    with pytest.raises(BrowserSafetyStop):
        module.capture_spectrum_excursion(
            spectrum_page[0],
            config(),
            directory,
            expected_star="Jyremis",
            marker_mode=module.GEOMETRIC_PAIR_MODE,
        )
    assert not (directory / "spectrum.json").exists()
    stopped = json.loads((directory / "stopped.json").read_bytes())
    assert stopped["hover_actions_started"] == 2
    assert stopped["answer_writes"] == 0 and stopped["automatic_retry"] is False
    assert stopped["task_completed"] is False


def test_native_legacy_capture_default_still_rejects_reversed_labels(spectrum_page, tmp_path):
    install_pair(spectrum_page, left=RED, right=BLUE)
    directory = tmp_path / "legacy"
    with pytest.raises(BrowserSafetyStop, match="^spectrum_tooltip_wrong_side$"):
        module.capture_spectrum_excursion(spectrum_page[0], config(), directory, expected_star="Jyremis")
    scope = json.loads((directory / "scope.json").read_bytes())
    assert "marker_mode" not in scope
    assert json.loads((directory / "stopped.json").read_bytes())["hover_actions_started"] == 1
    assert not (directory / "spectrum.json").exists()
