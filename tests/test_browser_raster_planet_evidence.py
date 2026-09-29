"""Fully intercepted fixtures; no live website, grading oracle, or training."""
# ruff: noqa: F811

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401

import habfly.browser_raster_planet_evidence as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import capture_observation_progress
from habfly.browser_planet_chart import FluxChartSession
from habfly.browser_planet_spectrum import hover_spectrum_marker
from habfly.planet_charts import spectrum_excursion


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def raster_page(window_page):
    page, frame = window_page
    frame.evaluate("""()=>{
      const chart=document.querySelector('#chart');
      for(const x of [60.5,100.5,140.5,180.5,220.5]){
        const line=document.createElementNS('http://www.w3.org/2000/svg','line');
        line.setAttribute('class','dip');line.setAttribute('x1',x);line.setAttribute('x2',x);
        line.setAttribute('y1','20.5');line.setAttribute('y2','30.5');
        line.setAttribute('stroke','rgb(80,132,154)');chart.append(line);
      }
      const label=[...document.querySelectorAll('span')].find(e=>e.textContent.includes('656.3nm'));
      label.outerHTML=`<div id='spectrum-block' style='position:relative;width:285px;height:115px'>
        <div id='strip' style='position:relative;width:285px;height:40px;background:rgb(193,58,44)'>
          <div id='blue-marker' style='position:absolute;left:100px;width:1px;height:38px;background:black'
            onmousemove="this.parentElement.nextElementSibling.style.display='block';this.parentElement.nextElementSibling.innerText='656.29995212nm'"></div>
          <div id='red-marker' style='position:absolute;left:160px;width:1px;height:38px;background:black'
            onmousemove="this.parentElement.nextElementSibling.style.display='block';this.parentElement.nextElementSibling.innerText='656.30004788nm'"></div>
        </div>
        <div id='spectrum-tip' style='background:rgb(250,250,220);display:none;position:absolute;top:42px;left:60px'></div>
        <div style='padding-top:25px'>656.3nm</div><div>656.299997nm</div><div>656.300003nm</div><span>observe for</span>
      </div>`;
    }""")
    return page, frame


def sources(raster_page, history):
    page, _ = raster_page
    events = []
    session = FluxChartSession(
        page, config(), lambda kind, payload: events.append({"kind": kind, "payload": payload}), max_actions=2
    )
    try:
        blue = hover_spectrum_marker(session, "blue")["wavelength_text"]
        red = hover_spectrum_marker(session, "red")["wavelength_text"]
    finally:
        session.close()
    spectrum_path = history / "spectrum.json"
    write(
        spectrum_path,
        {
            "visibility": "full_glyph_and_occlusion_checked",
            "events": events,
            "result": spectrum_excursion("656.3nm", blue, red),
        },
    )
    capture_observation_progress(page, config(), history / "source", requested_days=5000)
    path = history / "source/report.json"
    return {
        "window_report": path,
        "window_report_sha256": sha(path),
        "spectrum_path": spectrum_path,
        "spectrum_sha256": sha(spectrum_path),
    }


def select(raster_page, history, source, **kwargs):
    page, _ = raster_page
    return module.select_raster_detected_planet(
        page, config(), history / "presence", run_history=history, **source, **kwargs
    )


def copy(raster_page, history, *, output="copy"):
    page, _ = raster_page
    path = history / "presence/confirmed.json"
    return module.copy_raster_measured_inputs(
        page, config(), history / output, run_history=history, presence_path=path, presence_sha256=sha(path)
    )


def test_guarded_yes_and_three_exact_reference_copies(raster_page, tmp_path):
    page, frame = raster_page
    source = sources(raster_page, tmp_path)
    receipt = select(raster_page, tmp_path, source)
    assert receipt["readback_verified"] and receipt["value"] == "Yes"
    assert receipt["numeric_writes"] == 0 and frame.evaluate("window.fixtureSelections") == 1
    assert all(frame.locator("#" + name).input_value() == "" for name in module.RAW)
    result = copy(raster_page, tmp_path)
    assert result["raw_measurement_transport_verified"] and result["numeric_writes"] == 3
    assert list(result["verified_fields"]) == list(module.RAW)
    for name, expected in result["evidence"]["measurements"].items():
        assert frame.locator("#" + name).input_value() == expected["value"]
        assert result["verified_fields"][name]["unit"] == expected["unit"]
        assert result["verified_fields"][name]["action_source"] == "reference_diagnostic"
    assert all(frame.locator("#" + name).input_value() == "" for name in module.DERIVED)
    assert not page.get_by_role("checkbox").is_checked()
    for key, value in module.FLAGS.items():
        assert result[key] == value
    assert result["evidence"]["uncertainty"]["kind"] == "conditional_pixel_bounds_not_statistical_confidence"
    assert len(list((tmp_path / "planet-raster-presence-reservations").glob("*.json"))) == 1
    assert len(list((tmp_path / "planet-raster-inputs-reservations").glob("*.json"))) == 1
    with pytest.raises(BrowserSafetyStop, match="inputs_already_reserved"):
        copy(raster_page, tmp_path, output="again")


def test_real_dulat_sources_recompute_offline_if_present():
    history = Path(__file__).resolve().parents[1] / "experiments/full-stellar-probe/20260926-005"
    window = history / "planet-window-90/progress-000/report.json"
    spectrum = history / "guarded-spectrum-99.json"
    if not window.exists() or not spectrum.exists():
        pytest.skip("Saved local Dulat sources are not distributed")
    evidence = module._evidence(module._Owned(history), window, sha(window), spectrum, sha(spectrum))
    assert evidence["star"] == "DULAT"
    assert evidence["measurements"]["line_shift"]["value"] == "0.00004788"
    assert evidence["window"]["estimate"]["supported_dip_components"] == 25


@pytest.mark.parametrize("spelling", ["Dulat", "dUlAt"])
def test_visible_case_is_preserved_while_chart_identity_matches(raster_page, tmp_path, spelling):
    _, frame = raster_page
    frame.locator("div").first.evaluate(
        "(e,name)=>{e.textContent=name;e.style.textTransform='uppercase'}", spelling
    )
    source = sources(raster_page, tmp_path)
    receipt = select(raster_page, tmp_path, source)
    assert receipt["star"] == receipt["evidence"]["star"] == "DULAT"
    _, before, _ = module._Owned(tmp_path).capture(tmp_path / "presence/before")
    assert before["star_name"] == spelling
    result = copy(raster_page, tmp_path)
    assert result["raw_measurement_transport_verified"] and result["numeric_writes"] == 3
    assert frame.locator("div").first.inner_text() == "DULAT"
    _, after, _ = module._Owned(tmp_path).capture(tmp_path / "copy/native-copies/copy-03-after")
    assert after["star_name"] == spelling


def test_live_117_initial_titlecase_identity_offline_if_present():
    history = Path(__file__).resolve().parents[1] / "experiments/full-stellar-probe/20260926-005"
    initial = history / "raster-presence-117/read-guard/initial"
    if not initial.exists():
        pytest.skip("Saved local stopped Dulat diagnostic is not distributed")
    before_hash = sha(initial / "observation.json")
    _, mapping, _ = module._Owned(history).capture(initial)
    assert mapping["star_name"] == "Dulat"
    assert mapping["observation"]["values"]["has_planet"] is None
    assert module._same_star(mapping["star_name"], "DULAT")
    assert not module._same_star(mapping["star_name"], "BEGOLLO")
    assert sha(initial / "observation.json") == before_hash
    stopped = read(history / "raster-presence-117/stopped.json")
    assert not stopped["write_may_have_occurred"] and not stopped["reservation_created"]


@pytest.mark.parametrize("name", ["Dul at", "Dulat ", "DULAT-X", "Dulát", None, ""])
def test_case_normalization_does_not_broaden_star_identity(name):
    assert not module._same_star(name, "DULAT")


@pytest.mark.parametrize(
    "change", ["window", "png", "spectrum", "star", "spectrum_result", "failed", "outside", "symlink"]
)
def test_invalid_hashed_sources_do_not_select(raster_page, tmp_path, change):
    _, frame = raster_page
    source = sources(raster_page, tmp_path)
    if change == "window":
        source["window_report"].write_text(source["window_report"].read_text() + " ")
    elif change == "png":
        (source["window_report"].parent / "chart.png").write_bytes(b"bad")
    elif change == "spectrum":
        source["spectrum_sha256"] = "b" * 64
    elif change in {"star", "spectrum_result"}:
        data = read(source["spectrum_path"])
        if change == "star":
            data["events"][0]["payload"]["chart"]["star"] = "OTHER"
        else:
            data["result"]["line_shift_nm"] = "1"
        write(source["spectrum_path"], data)
        source["spectrum_sha256"] = sha(source["spectrum_path"])
    elif change == "failed":
        write(source["window_report"].parent / "stopped.json", {})
    elif change == "symlink":
        target = tmp_path / "renamed.json"
        source["spectrum_path"].rename(target)
        source["spectrum_path"].symlink_to(target)
    else:
        source["spectrum_path"] = tmp_path.parent / "outside.json"
    with pytest.raises(BrowserSafetyStop):
        select(raster_page, tmp_path, source)
    assert frame.evaluate("window.fixtureSelections") == 0


@pytest.mark.parametrize(
    "change",
    [
        "star",
        "duration",
        "raw",
        "presence",
        "axis",
        "no_dips",
        "dip_depth",
        "spectrum",
        "modal",
        "auth",
        "chart_overlay",
    ],
)
def test_stale_current_state_never_uses_saved_measurements(raster_page, tmp_path, change):
    _, frame = raster_page
    source = sources(raster_page, tmp_path)
    if change == "star":
        frame.locator("div").first.evaluate("e=>e.textContent='OTHER'")
    elif change == "duration":
        frame.get_by_role("textbox").first.fill("10000")
    elif change == "raw":
        frame.locator("#period_days").fill("0")
    elif change == "presence":
        frame.get_by_role("combobox").select_option(label="Yes")
    elif change == "axis":
        frame.locator(".time").nth(1).evaluate("e=>e.textContent='2490'")
    elif change in {"no_dips", "dip_depth"}:
        frame.locator(".dip").evaluate_all(
            "(es,y)=>es.forEach(e=>e.setAttribute('y2',y))", "20.5" if change == "no_dips" else "60.5"
        )
    elif change == "spectrum":
        frame.locator("#red-marker").evaluate(
            "e=>e.onmousemove=()=>document.querySelector('#spectrum-tip').textContent='656.301nm'"
        )
    elif change == "chart_overlay":
        frame.locator("#chart").evaluate(
            "e=>{const r=e.getBoundingClientRect(),d=document.createElement('div');d.style=`position:fixed;left:${r.left+30}px;top:${r.top+20}px;width:100px;height:20px;background:red`;document.body.append(d)}"
        )
    else:
        frame.locator("body").evaluate(
            "(e,h)=>e.insertAdjacentHTML('beforeend',h)",
            "<div role=dialog>Stop</div>" if change == "modal" else "<input type=password>",
        )
    before = frame.evaluate("window.fixtureSelections")
    with pytest.raises(BrowserSafetyStop):
        select(raster_page, tmp_path, source)
    assert frame.evaluate("window.fixtureSelections") == before


@pytest.mark.parametrize("change", ["replacement", "source", "dip", "external"])
def test_postreservation_changes_block_dispatch_and_new_output_retry(
    raster_page, tmp_path, monkeypatch, change
):
    page, frame = raster_page
    source = sources(raster_page, tmp_path)
    original = module.persist_json

    def persist(path, value):
        original(path, value)
        if path == tmp_path / "presence/reserved.json":
            if change == "replacement":
                frame.get_by_role("combobox").evaluate("e=>e.replaceWith(e.cloneNode(true))")
            elif change == "source":
                source["spectrum_path"].write_text("{}")
            elif change == "dip":
                frame.locator(".dip").first.evaluate("e=>e.setAttribute('y2','70')")
            else:
                page.get_by_role("checkbox").check()

    monkeypatch.setattr(module, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop):
        select(raster_page, tmp_path, source)
    assert frame.evaluate("window.fixtureSelections") == 0
    assert read(tmp_path / "presence/stopped.json")["reservation_created"]
    assert len(list((tmp_path / "planet-raster-presence-reservations").glob("*.json"))) == 1
    with pytest.raises(BrowserSafetyStop):
        module.select_raster_detected_planet(
            page, config(), tmp_path / "retry", run_history=tmp_path, **source
        )


@pytest.mark.parametrize("change", ["answer", "replace", "dialog", "overlay"])
def test_uncertain_yes_is_never_retried(raster_page, tmp_path, change):
    _, frame = raster_page
    source = sources(raster_page, tmp_path)
    code = {
        "answer": "document.querySelector('#period_days').value='10'",
        "replace": "e.replaceWith(e.cloneNode(true))",
        "dialog": "confirm('private detail')",
        "overlay": "const r=e.getBoundingClientRect(),d=document.createElement('div');d.style=`position:fixed;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;background:black;z-index:999`;document.body.append(d)",
    }[change]
    frame.get_by_role("combobox").evaluate(
        "(e,code)=>{const old=e.onchange;e.onchange=()=>{old();new Function('e',code)(e)}}", code
    )
    with pytest.raises(BrowserSafetyStop):
        select(raster_page, tmp_path, source)
    assert frame.evaluate("window.fixtureSelections") == 1
    raw = (tmp_path / "presence/stopped.json").read_text()
    assert read(tmp_path / "presence/stopped.json")["write_may_have_occurred"] and "private detail" not in raw
    assert not (tmp_path / "presence/confirmed.json").exists()


@pytest.mark.parametrize("change", ["presence_receipt", "capture", "raw", "class", "derived", "spectrum"])
def test_copy_requires_valid_receipt_and_current_untouched_destinations(raster_page, tmp_path, change):
    _, frame = raster_page
    select(raster_page, tmp_path, sources(raster_page, tmp_path))
    if change == "presence_receipt":
        data = read(tmp_path / "presence/confirmed.json")
        data["readback_verified"] = False
        write(tmp_path / "presence/confirmed.json", data)
    elif change == "capture":
        (tmp_path / "presence/after/observation.json").write_text("{}")
    elif change in {"raw", "derived"}:
        frame.locator("#period_days" if change == "raw" else "#planet_mass").fill("1")
    elif change == "class":
        frame.locator(".choice label").first.evaluate("e=>e.style.borderColor='red'")
    else:
        frame.locator("#red-marker").evaluate(
            "e=>e.onmousemove=()=>document.querySelector('#spectrum-tip').textContent='656.301nm'"
        )
    with pytest.raises(BrowserSafetyStop):
        copy(raster_page, tmp_path)
    assert frame.locator("#line_shift").input_value() == ""


def test_partial_copy_failure_is_durable_and_cannot_resume(raster_page, tmp_path):
    _, frame = raster_page
    select(raster_page, tmp_path, sources(raster_page, tmp_path))
    frame.locator("#line_shift").evaluate("e=>e.onblur=()=>document.querySelector('#period_days').value='7'")
    with pytest.raises(BrowserSafetyStop):
        copy(raster_page, tmp_path)
    assert frame.locator("#line_shift").input_value() != ""
    assert frame.locator("#brightness_drop").input_value() == ""
    assert read(tmp_path / "copy/stopped.json")["write_may_have_occurred"]
    with pytest.raises(BrowserSafetyStop, match="inputs_already_reserved"):
        copy(raster_page, tmp_path, output="retry")


@pytest.mark.parametrize("clears", [False, True])
def test_explicit_inherited_paint_can_preserve_or_clear_without_classifying(raster_page, tmp_path, clears):
    _, frame = raster_page
    frame.locator("body").evaluate(
        """(e,clears)=>{
      e.insertAdjacentHTML('beforeend','<style>.choice.selected label{border-color:white}.choice.selected label::after{opacity:1;background:white}</style>');
      document.querySelectorAll('.choice')[1].classList.add('selected');
      if(clears){const c=document.querySelector('select'),old=c.onchange;
        c.onchange=()=>{old();document.querySelectorAll('.choice').forEach(e=>e.classList.remove('selected'))};}
    }""",
        clears,
    )
    result = select(raster_page, tmp_path, sources(raster_page, tmp_path), preserve_painted_class="ice_giant")
    assert result["inherited_paint_cleared"] is clears
    assert result["painted_class_after"] == (None if clears else "ice_giant")
    assert not result["class_selection_verified"] and result["class_writes"] == 0
    assert copy(raster_page, tmp_path)["raw_measurement_transport_verified"]


def test_mutated_source_at_native_field_reservation_blocks_the_fill(raster_page, tmp_path, monkeypatch):
    import habfly.browser_planet_numeric as numeric

    _, frame = raster_page
    source = sources(raster_page, tmp_path)
    select(raster_page, tmp_path, source)
    original = numeric.persist_json

    def persist(path, value):
        original(path, value)
        if path.name == "copy-01-reserved.json":
            source["spectrum_path"].write_text("{}")

    monkeypatch.setattr(numeric, "persist_json", persist)
    with pytest.raises(BrowserSafetyStop):
        copy(raster_page, tmp_path)
    assert all(frame.locator("#" + name).input_value() == "" for name in module.RAW)
    assert len(list((tmp_path / "planet-raster-inputs-reservations").glob("*.json"))) == 1


@pytest.mark.parametrize("budget", [True, 0, 29, 901, float("nan"), float("inf")])
def test_invalid_explicit_budgets_reject_before_browser_or_output(tmp_path, budget):
    with pytest.raises(BrowserSafetyStop, match="invalid_time_budget"):
        module.select_raster_detected_planet(
            None,
            None,
            tmp_path / "run",
            run_history=tmp_path,
            window_report="unused",
            window_report_sha256="unused",
            spectrum_path="unused",
            spectrum_sha256="unused",
            max_seconds=budget,
        )
    with pytest.raises(BrowserSafetyStop, match="invalid_time_budget"):
        module.copy_raster_measured_inputs(
            None,
            None,
            tmp_path / "copy",
            run_history=tmp_path,
            presence_path="unused",
            presence_sha256="unused",
            max_seconds=budget,
        )
    assert not (tmp_path / "run").exists() and not (tmp_path / "copy").exists()


def tooltip_sources(history):
    """Synthetic sensor artifacts; actual offline validator, no browser seam."""
    from test_planet_tooltip_reference import fixture

    diagnostics = [fixture(history, f"diagnostic-{i}", 1000 * (i + 1)) for i in range(3)]
    directory = history / "settled-overview"
    directory.mkdir()
    for name in ("report.json", "chart.png"):
        (directory / name).write_bytes((history / "diagnostic-0-overview" / name).read_bytes())
    events = [
        {"kind": "observation", "payload": {"chart": {"star": "EXAMPLE", "source": "visible_tooltips"}}}
    ]
    readings = {"blue": "656.29995212nm", "red": "656.30004788nm"}
    for sequence, marker in enumerate(("blue", "red"), 1):
        events.extend(
            [
                {
                    "kind": "action_proposed",
                    "payload": {
                        "kind": "HOVER",
                        "surface": "spectrum",
                        "marker": marker,
                        "sequence": sequence,
                    },
                },
                {
                    "kind": "action_result",
                    "payload": {
                        "sequence": sequence,
                        "task_completed": False,
                        "spectrum_sample": {
                            "marker": marker,
                            "source": "visible_spectrum_tooltip",
                            "wavelength_text": readings[marker],
                        },
                    },
                },
            ]
        )
    spectrum = history / "spectrum.json"
    write(
        spectrum,
        {
            "visibility": "full_glyph_and_occlusion_checked",
            "events": events,
            "result": spectrum_excursion("656.3nm", readings["blue"], readings["red"]),
        },
    )
    return {
        "window_report": directory / "report.json",
        "window_report_sha256": sha(directory / "report.json"),
        "spectrum_path": spectrum,
        "spectrum_sha256": sha(spectrum),
        "tooltip_reference": {"diagnostics": diagnostics, "expected_star": "Example"},
    }


def test_tooltip_transport_evidence_keeps_sample_semantics_and_all_source_pins(tmp_path, monkeypatch):
    from habfly.browser_no_planet_workflow import _Evidence
    from habfly.browser_positive_planet_workflow import _RasterEvidence

    source = tooltip_sources(tmp_path)
    monkeypatch.setattr(
        module, "load_planet_window_measurements", lambda *_a, **_kw: pytest.fail("raster fallback")
    )
    book = _Evidence(tmp_path)
    owner = _RasterEvidence(book)
    evidence = module._evidence(owner, **source)
    assert evidence["mode"] == module.TOOLTIP_MODE
    assert evidence["measurements"]["brightness_drop"] == {
        "value": "0.762",
        "unit": "%",
        "source": module.TOOLTIP_MODE,
        "physical_bounds": None,
    }
    period = evidence["measurements"]["period_days"]
    assert period["value"] == "1000" and period["compatibility_interval"] == {
        "lower": "999",
        "upper": "1001",
        "endpoints": "open",
    }
    assert not ({"lower", "upper"} & period.keys())
    assert evidence["tooltip_measurements"]["answer_authorized"] is False
    assert set(evidence["tooltip_measurements"]["source_sha256"]) <= book.hashes.keys()
    assert {
        tmp_path / p for p in evidence["tooltip_measurements"]["validated_directories"]
    } <= book.clean_directories
    assert module._flags_for(evidence)["provenance"] == module.TOOLTIP_MODE
    assert module._action_source(evidence) == "explicit_approximate_reference_visible_tooltips_v1_not_learned"
    module._reload(owner, evidence)
    book.unchanged()


@pytest.mark.parametrize("change", ["png", "axes", "star", "source", "mode", "descriptor"])
def test_tooltip_sources_and_exact_settled_anchor_cannot_change(tmp_path, change):
    source = tooltip_sources(tmp_path)
    owner = module._Owned(tmp_path)
    evidence = module._evidence(owner, **source)
    if change in {"png", "axes", "star"}:
        path = source["window_report"]
        value = read(path)
        if change == "png":
            import io

            from PIL import Image

            png = path.parent / "chart.png"
            picture = Image.open(io.BytesIO(png.read_bytes())).convert("RGB")
            picture.putpixel((40, 30), (1, 2, 3))
            buffer = io.BytesIO()
            picture.save(buffer, format="PNG")
            png.write_bytes(buffer.getvalue())
            value["chart_sha256"] = sha(png)
        elif change == "axes":
            value["flux_axis_labels"][0]["center_y"] += 1
        else:
            value["star"] = "OTHER"
        write(path, value)
        source["window_report_sha256"] = sha(path)
        with pytest.raises(BrowserSafetyStop):
            module._evidence(owner, **source)
    else:
        if change == "source":
            (tmp_path / "diagnostic-0/readable.json").write_text("{}")
        elif change == "mode":
            evidence["mode"] = "unknown_reference"
        else:
            evidence.pop("tooltip_reference")
        with pytest.raises((BrowserSafetyStop, KeyError)):
            module._reload(owner, evidence)


def test_tooltip_freshness_replays_exact_overview_and_spectrum(tmp_path):
    source = tooltip_sources(tmp_path)
    owner = module._Owned(tmp_path)
    evidence = module._evidence(owner, **source)
    directory = tmp_path / "fresh"
    (directory / "progress").mkdir(parents=True)
    for name in ("report.json", "chart.png"):
        (directory / "progress" / name).write_bytes((source["window_report"].parent / name).read_bytes())
    (directory / "spectrum.json").write_bytes(source["spectrum_path"].read_bytes())
    path = directory / "progress/report.json"
    recorded = {
        "window": module._overview_window(owner, path, sha(path)),
        "spectrum_sha256": sha(directory / "spectrum.json"),
    }
    module._fresh_evidence(owner, directory, recorded, evidence)
    changed = deepcopy(recorded)
    changed["window"]["chart_sha256"] = "0" * 64
    with pytest.raises(BrowserSafetyStop, match="fresh_window_evidence_changed"):
        module._fresh_evidence(owner, directory, changed, evidence)
    spectrum = read(directory / "spectrum.json")
    spectrum["events"][2]["payload"]["spectrum_sample"]["wavelength_text"] = "656.299951nm"
    spectrum["events"][4]["payload"]["spectrum_sample"]["wavelength_text"] = "656.300049nm"
    spectrum["result"] = spectrum_excursion("656.3nm", "656.299951nm", "656.300049nm")
    write(directory / "spectrum.json", spectrum)
    recorded["spectrum_sha256"] = sha(directory / "spectrum.json")
    with pytest.raises(BrowserSafetyStop, match="fresh_spectrum_evidence_changed"):
        module._fresh_evidence(owner, directory, recorded, evidence)


def test_tooltip_and_raster_share_per_star_reservation_not_mode_retry(tmp_path):
    owner = module._Owned(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for stage in ("presence", "inputs"):
        output = first / stage
        output.mkdir()
        module._reserve(owner, output, "Example", stage, {"mode": module.TOOLTIP_MODE})
        retry = second / stage
        retry.mkdir()
        with pytest.raises(BrowserSafetyStop, match=stage + "_already_reserved"):
            module._reserve(owner, retry, "EXAMPLE", stage, {"mode": module.MODE})
