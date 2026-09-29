"""Offline contracts for explicit geometric-marker spectrum evidence.

Saved event fixtures are synthetic, not native capture/scientific acceptance.
The spectrum normalizer and owned source/measurement readers run unchanged.
Only the live freshness test injects browser leaf operations; it proves dispatch
and artifact compatibility, not native event delivery.
"""

import hashlib
import json
from copy import deepcopy
from decimal import getcontext, localcontext
from types import SimpleNamespace

import pytest

import habfly.browser_planet_spectrum as sensor
import habfly.browser_raster_planet_evidence as module
from habfly.browser import BrowserSafetyStop
from habfly.planet_charts import spectrum_excursion

MODE = "visible_geometric_spectrum_pair_v1"
BLUE = "656.29995212nm"
RED = "656.30004788nm"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def artifact(*, geometric=True, swapped=False):
    markers = ("left", "right") if geometric else ("blue", "red")
    readings = (RED, BLUE) if swapped else (BLUE, RED)
    events = [
        {"kind": "observation", "payload": {"chart": {"star": "EXAMPLE", "source": "visible_tooltips"}}}
    ]
    for index, (marker, text) in enumerate(zip(markers, readings, strict=True), 1):
        events.extend(
            [
                {
                    "kind": "action_proposed",
                    "payload": {"kind": "HOVER", "surface": "spectrum", "sequence": index, "marker": marker},
                },
                {
                    "kind": "action_result",
                    "payload": {
                        "sequence": index,
                        "task_completed": False,
                        "spectrum_sample": {
                            "marker": marker,
                            "wavelength_text": text,
                            "source": "visible_spectrum_tooltip",
                        },
                    },
                },
            ]
        )
    value = {
        "events": events,
        "result": spectrum_excursion("656.3nm", BLUE, RED),
        "visibility": "full_glyph_and_occlusion_checked",
    }
    if geometric:
        value["marker_mode"] = MODE
    return value


@pytest.mark.parametrize("swapped", [False, True])
def test_geometric_raw_identity_is_preserved_and_labels_are_independently_normalized(swapped):
    value = artifact(swapped=swapped)
    before = deepcopy(value)
    result = module._spectrum(value)
    assert result == {
        "star": "EXAMPLE",
        "readings": {"blue": BLUE, "red": RED},
        "result": spectrum_excursion("656.3nm", BLUE, RED),
        "marker_mode": MODE,
    }
    assert value == before
    assert [value["events"][n]["payload"]["spectrum_sample"]["marker"] for n in (2, 4)] == ["left", "right"]
    assert result["result"]["assumptions_verified"] is False


def test_legacy_normalized_shape_and_values_are_unchanged():
    value = artifact(geometric=False)
    assert module._spectrum(value) == {
        "star": "EXAMPLE",
        "readings": {"blue": BLUE, "red": RED},
        "result": value["result"],
    }
    assert "marker_mode" not in value


@pytest.mark.parametrize("precision", [6, 28])
def test_high_precision_symmetric_pair_is_exact_and_isolates_decimal_context(precision):
    left = "655.123456789012345678901234567890nm"
    right = "657.476543210987654321098765432110nm"
    expected = {
        "rest_nm": "656.3",
        "blue_nm": left.removesuffix("nm"),
        "red_nm": right.removesuffix("nm"),
        "line_shift_nm": "1.176543210987654321098765432110",
        "peak_to_peak_nm": "2.353086421975308642197530864220",
        "source": "explicit_visible_excursion_markers",
        "assumptions_verified": False,
    }
    value = artifact()
    value["events"][2]["payload"]["spectrum_sample"]["wavelength_text"] = left
    value["events"][4]["payload"]["spectrum_sample"]["wavelength_text"] = right
    value["result"] = expected
    with localcontext() as context:
        context.prec = precision
        assert sensor.normalize_geometric_spectrum_pair(left, right)["result"] == expected
        assert module._spectrum(value)["result"] == expected
        assert getcontext().prec == precision


@pytest.mark.parametrize("precision", [6, 28])
def test_high_precision_asymmetry_cannot_round_into_an_accepted_pair(precision):
    left = "655.123456789012345678901234567890nm"
    right = "657.476543210987654321098765432119nm"
    value = artifact()
    value["events"][2]["payload"]["spectrum_sample"]["wavelength_text"] = left
    value["events"][4]["payload"]["spectrum_sample"]["wavelength_text"] = right
    with localcontext() as context:
        context.prec = precision
        with pytest.raises(ValueError, match="^asymmetric_visible_spectrum_excursion$"):
            sensor.normalize_geometric_spectrum_pair(left, right)
        with pytest.raises(BrowserSafetyStop, match="invalid_geometric_spectrum_pair"):
            module._spectrum(value)
        assert getcontext().prec == precision


@pytest.mark.parametrize("mode", [None, False, True, 0, "", "unknown", "visible_geometric_spectrum_pair_v2"])
def test_explicit_unknown_marker_mode_never_falls_back(mode):
    value = artifact(geometric=False)
    value["marker_mode"] = mode
    with pytest.raises(BrowserSafetyStop, match="unsupported_spectrum_mode"):
        module._spectrum(value)


@pytest.mark.parametrize(
    "left,right",
    [
        (RED, RED),
        (BLUE, BLUE),
        ("656.3nm", RED),
        ("0nm", RED),
        ("-1nm", RED),
        (BLUE, "656.30004789nm"),
        ("NaNnm", RED),
        ("Infinitynm", RED),
        ("6.56e2nm", RED),
        (BLUE.replace("nm", "m"), RED),
        (None, RED),
        (True, RED),
    ],
)
def test_geometric_invalid_pair_never_substitutes_or_averages_endpoints(left, right):
    value = artifact()
    value["events"][2]["payload"]["spectrum_sample"]["wavelength_text"] = left
    value["events"][4]["payload"]["spectrum_sample"]["wavelength_text"] = right
    with pytest.raises(BrowserSafetyStop, match="invalid_geometric_spectrum_pair"):
        module._spectrum(value)


@pytest.mark.parametrize(
    "change",
    [
        "missing_mode",
        "legacy_markers",
        "wrong_sample_marker",
        "extra_hover",
        "event_order",
        "proposed_bool",
        "result_bool",
        "completed",
        "not_visible",
        "changed_result",
        "false_alias",
    ],
)
def test_new_mode_event_and_result_mutations_reject(change):
    value = artifact()
    if change == "missing_mode":
        value.pop("marker_mode")
    elif change == "legacy_markers":
        value["events"] = artifact(geometric=False)["events"]
    elif change == "wrong_sample_marker":
        value["events"][2]["payload"]["spectrum_sample"]["marker"] = "right"
    elif change == "extra_hover":
        value["events"].append(deepcopy(value["events"][1]))
    elif change == "event_order":
        value["events"][1], value["events"][3] = value["events"][3], value["events"][1]
    elif change == "proposed_bool":
        value["events"][1]["payload"]["sequence"] = True
    elif change == "result_bool":
        value["events"][2]["payload"]["sequence"] = True
    elif change == "completed":
        value["events"][2]["payload"]["task_completed"] = True
    elif change == "not_visible":
        value["visibility"] = "unchecked"
    elif change == "changed_result":
        value["result"]["line_shift_nm"] = "1"
    else:
        value["result"]["assumptions_verified"] = 0
    with pytest.raises(BrowserSafetyStop):
        module._spectrum(value)


def sources(root, *, geometric=True):
    from test_browser_raster_planet_evidence import tooltip_sources

    source = tooltip_sources(root)
    write(source["spectrum_path"], artifact(geometric=geometric, swapped=geometric))
    source["spectrum_sha256"] = sha(source["spectrum_path"])
    return source


def fresh_files(root, source, owner):
    directory = root / "fresh"
    (directory / "progress").mkdir(parents=True)
    for name in ("report.json", "chart.png"):
        (directory / "progress" / name).write_bytes((source["window_report"].parent / name).read_bytes())
    (directory / "spectrum.json").write_bytes(source["spectrum_path"].read_bytes())
    path = directory / "progress/report.json"
    return directory, {
        "window": module._overview_window(owner, path, sha(path)),
        "spectrum_sha256": sha(directory / "spectrum.json"),
    }


def test_owned_geometric_source_reload_and_fresh_reader_retain_mode_and_hashes(tmp_path):
    from habfly.browser_no_planet_workflow import _Evidence
    from habfly.browser_positive_planet_workflow import _RasterEvidence

    source = sources(tmp_path)
    owner = _RasterEvidence(_Evidence(tmp_path))
    evidence = module._evidence(owner, **source)
    assert evidence["spectrum"]["marker_mode"] == MODE
    assert evidence["spectrum"]["sha256"] == source["spectrum_sha256"]
    assert evidence["measurements"]["line_shift"]["value"] == "0.00004788"
    module._reload(owner, evidence)
    directory, recorded = fresh_files(tmp_path, source, owner)
    module._fresh_evidence(owner, directory, recorded, evidence)
    source["spectrum_path"].write_bytes(source["spectrum_path"].read_bytes() + b"\n")
    with pytest.raises(BrowserSafetyStop):
        module._reload(owner, evidence)


@pytest.mark.parametrize("original_geometric", [False, True])
def test_fresh_source_cannot_switch_marker_mode_despite_identical_numeric_pair(tmp_path, original_geometric):
    source = sources(tmp_path, geometric=original_geometric)
    owner = module._Owned(tmp_path)
    evidence = module._evidence(owner, **source)
    directory, recorded = fresh_files(tmp_path, source, owner)
    write(directory / "spectrum.json", artifact(geometric=not original_geometric))
    recorded["spectrum_sha256"] = sha(directory / "spectrum.json")
    with pytest.raises(BrowserSafetyStop, match="fresh_spectrum_evidence_changed"):
        module._fresh_evidence(owner, directory, recorded, evidence)


@pytest.mark.parametrize("geometric", [False, True])
def test_live_freshness_dispatch_uses_explicit_recipe_and_same_two_action_budget(
    tmp_path, monkeypatch, geometric
):
    source = sources(tmp_path, geometric=geometric)
    owner = module._Owned(tmp_path)
    evidence = module._evidence(owner, **source)
    calls, sessions = [], []

    def session(_page, _config, emit, **limits):
        assert limits == {"max_actions": 2, "max_seconds": 120}
        s = SimpleNamespace(star="EXAMPLE", emit=emit, closed=False)
        s.close = lambda: setattr(s, "closed", True)
        sessions.append(s)
        emit("observation", artifact()["events"][0]["payload"])
        return s

    def pair(s):
        calls.append("pair")
        for event in artifact(swapped=True)["events"][1:]:
            s.emit(event["kind"], event["payload"])
        return sensor.normalize_geometric_spectrum_pair(RED, BLUE)

    def hover(s, side):
        calls.append(side)
        n = 1 if side == "blue" else 3
        events = artifact(geometric=False)["events"]
        for event in events[n : n + 2]:
            s.emit(event["kind"], event["payload"])
        return events[n + 1]["payload"]["spectrum_sample"]

    def capture(_page, _config, directory, **limits):
        assert limits == {"requested_days": 5000, "max_seconds": 120}
        directory.mkdir()
        for name in ("report.json", "chart.png"):
            (directory / name).write_bytes((source["window_report"].parent / name).read_bytes())

    monkeypatch.setattr(module, "FluxChartSession", session)
    monkeypatch.setattr(module, "hover_spectrum_marker", hover)
    monkeypatch.setattr(sensor, "read_geometric_spectrum_pair", pair)
    monkeypatch.setattr(module, "capture_observation_progress", capture)
    directory = tmp_path / "fresh"
    recorded = module._fresh(None, None, owner, directory, evidence)
    assert calls == (["pair"] if geometric else ["blue", "red"])
    assert len(sessions) == 1 and sessions[0].closed
    module._fresh_evidence(owner, directory, recorded, evidence)
    saved = json.loads((directory / "spectrum.json").read_bytes())
    assert len(saved["events"]) == 5
    assert saved.get("marker_mode") == (MODE if geometric else None)
