"""Synthetic owned artifacts, not native execution or planetary ground truth."""

import hashlib
import io
import json
import socket
from decimal import Decimal

import pytest
from PIL import Image, ImageDraw

from habfly.browser import BrowserSafetyStop
from habfly.browser_observation_progress import trace_progress
from habfly.browser_shallow_transit_probe import (
    FIRST_THREE_HINT_POLICY,
    FIRST_THREE_LOW_INTENSITY_HINT_POLICY,
    overview_hint_metadata,
)
from habfly.contracts import RuntimeEvent
from habfly.planet_tooltip_reference import (
    CAPTURES,
    DIAGNOSTIC_FILES,
    MODE,
    SENSOR,
    SENSOR_FLAGS,
    load_tooltip_reference,
    measurement_manifest,
)


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return checksum(raw)


def image(baseline):
    picture = Image.new("RGB", (280, 196), "black")
    draw = ImageDraw.Draw(picture)
    draw.line((30, baseline, 260, baseline), fill=(80, 132, 154))
    draw.line((100, baseline, 100, baseline + 4), fill=(80, 132, 154))
    output = io.BytesIO()
    picture.save(output, format="PNG")
    return output.getvalue()


def fixture(
    history,
    name,
    center,
    *,
    low="99.238",
    star="EXAMPLE",
    pan=False,
    linked=False,
    values=None,
    recipe="daily_focus_v2",
    overview_hint_index=None,
    overview_hint_policy=FIRST_THREE_HINT_POLICY,
):
    """Declared synthetic native-event seam; real reader/validator always runs."""
    original = history / (name + "-overview")
    original.mkdir()
    overview, current = image(20), image(80)
    if overview_hint_index is not None:
        picture = Image.new("RGB", (280, 196), "black")
        draw = ImageDraw.Draw(picture)
        draw.line((30, 20, 260, 20), fill=(80, 132, 154))
        for day in range(100, 4900, 100):
            picture.putpixel((int(30.5 + day * 0.046), 21), (55, 83, 96))
        # An unselected clipped final hint does not repair a selected hint.
        draw.line((258, 21, 259, 21), fill=(55, 83, 96))
        raw = io.BytesIO()
        picture.save(raw, format="PNG")
        overview = raw.getvalue()
    times = [{"value": str(day), "center_x": 30.5 + day * 0.046} for day in (0, 2500, 5000)]
    flux = [{"value": str(value), "center_y": 20.5 + (100 - value) * 1.3} for value in (0, 50, 100)]
    origin = {
        **trace_progress(overview, times, requested_days=5000),
        "star": star,
        "time_axis_labels": times,
        "flux_axis_labels": flux,
        "chart_sha256": checksum(overview),
        "browser_actions": 0,
        "answer_writes": 0,
    }
    (original / "chart.png").write_bytes(overview)
    origin_hash = save(original / "report.json", origin)
    directory = history / name
    directory.mkdir()
    source_hashes = {"report.json": origin_hash, "chart.png": checksum(overview)}
    hint_interval = [center - 20, center + 20] if linked else [10, 20]
    if overview_hint_index is not None:
        column = (35, 39, 44)[overview_hint_index]
        hint_interval = [(column - 0.5 - 30.5) * (5000 / 230), (column + 1.5 - 30.5) * (5000 / 230)]
    scope = {
        "mode": SENSOR,
        "star": star,
        "max_actions": 16,
        "max_seconds": 180,
        "requested_days": 5000,
        "source_dir": original.name,
        "source_hashes": source_hashes,
        "source_owner_recovery": False,
        "candidate_index": 0,
        "candidate_columns": [40, 41],
        "source_hint_day_interval": hint_interval,
        **SENSOR_FLAGS,
    }
    if recipe:
        scope["sensor_recipe"] = recipe
    if overview_hint_index is not None:
        scope.update(
            **overview_hint_metadata(overview_hint_policy),
            candidate_index=overview_hint_index,
            candidate_columns=[column],
        )
    save(directory / "scope.json", scope)
    captures = CAPTURES if recipe else tuple(name for name in CAPTURES if name not in {"zoom-5", "zoom-6"})
    scale = 2.08 if recipe else 1
    for capture_name in captures:
        png = overview if capture_name == "before" else current
        axes = (
            times
            if capture_name == "before"
            else [{"value": str(center + delta), "center_x": 100.5 + delta * scale} for delta in (-10, 0, 10)]
        )
        fluxes = (
            flux
            if capture_name == "before"
            else [{"value": str(value), "center_y": 80 + (100 - value) * 2} for value in (90, 95, 100)]
        )
        (directory / (capture_name + ".png")).write_bytes(png)
        save(
            directory / (capture_name + ".json"),
            {
                "time_axis_labels": axes,
                "flux_axis_labels": fluxes,
                "chart_sha256": checksum(png),
                **SENSOR_FLAGS,
            },
        )
    events = []

    def emit(kind, payload):
        events.append(
            RuntimeEvent(
                event=kind, sequence=len(events), run_id=directory.name, payload=payload
            ).model_dump()
        )

    emit("observation", {"chart": {"source": "visible_tooltips", "star": star}})
    actions = 0

    def action(kind, payload, result):
        nonlocal actions
        actions += 1
        emit("action_proposed", {"kind": kind, "surface": "chart", "sequence": actions, **payload})
        emit("action_result", {"sequence": actions, "task_completed": False, **result})

    zooms = [-500] * 5 + [-250] if recipe else [-500, -500, -500, 250]
    for delta in zooms:
        action(
            "SCROLL",
            {"x_fraction": 0.5, "y_fraction": 0.1, "delta_y": delta},
            {"chart_accessibility": "Normalized Flux Days Observed"},
        )
    if pan:
        action(
            "DRAG",
            {"start_fraction": [0.5, 0.35], "end_fraction": [0.5, 0.6]},
            {"chart_accessibility": "Normalized Flux Days Observed"},
        )
    action(
        "HOVER", {"purpose": "remove_pointer_overlay", "outside_chart": True}, {"pointer_outside_chart": True}
    )
    samples = []
    values = values or ["100", low, "100"]
    for index, value in enumerate(values):
        delta = index - len(values) // 2
        sample = {"day": center + delta, "brightness_percent": value, "source": "visible_hover_tooltip"}
        samples.append(sample)
        payload = {"x_fraction": (100.5 + delta * scale) / 280, "y_fraction": 80 / 195}
        if recipe:
            payload["requested_day"] = sample["day"]
        action("HOVER", payload, {"chart_sample": sample})
    report = {
        "mode": SENSOR,
        "star": star,
        "source_hashes": source_hashes,
        "samples": samples,
        "browser_actions": actions,
        "status": "visible_bracketed_dip" if linked else "visible_decline_not_source_linked",
        "sampled_day_interval": [samples[0]["day"], samples[-1]["day"]],
        "minimum_sampled_brightness_percent": str(min(map(Decimal, values))),
        "sampled_decline_verified": True,
        "daily_coverage_verified": True,
        "source_hint_day_interval": hint_interval,
        "source_hint_linked": linked,
        **SENSOR_FLAGS,
    }
    if recipe:
        report["sensor_recipe"] = recipe
    if overview_hint_index is not None:
        report.update(overview_hint_metadata(overview_hint_policy))
    save(directory / "report.json", report)
    emit("episode_summary", report)
    (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    return {
        "directory": directory.name,
        "files": {path.name: checksum(path.read_bytes()) for path in directory.iterdir()},
    }


@pytest.fixture
def bundle(tmp_path):
    return tmp_path, [fixture(tmp_path, f"diagnostic-{i}", 1000 * (i + 1)) for i in range(3)]


def load(bundle):
    return load_tooltip_reference(bundle[0], bundle[1], expected_star="EXAMPLE")


def rewrite(bundle, index, name, value):
    root, sources = bundle
    path = root / sources[index]["directory"] / name
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    path.write_bytes(raw)
    sources[index]["files"][name] = checksum(raw)


def events(bundle, index=0):
    return [
        json.loads(line)
        for line in (bundle[0] / bundle[1][index]["directory"] / "events.jsonl").read_text().splitlines()
    ]


def rewrite_events(bundle, value, index=0):
    rewrite(bundle, index, "events.jsonl", ("".join(json.dumps(e) + "\n" for e in value)).encode())


def test_independent_golden_brackets_fit_and_exact_sampled_decline(bundle, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Network must not be used"))
    result = load(bundle)
    assert result["mode"] == MODE and result["status"] == "approximate_reference_measurements"
    assert result["period_days"]["value"] == "1000"
    assert result["period_days"]["compatibility_interval"] == {
        "lower": "999",
        "upper": "1001",
        "endpoints": "open",
    }
    assert result["period_days"]["fitted_bracket_points"] == ["1000", "2000", "3000"]
    assert result["brightness_drop_percent"]["value"] == "0.762"
    assert result["brightness_drop_percent"]["physical_bounds"] is None
    assert "lower" not in result["brightness_drop_percent"] and "lower" not in result["period_days"]
    assert len(result["source_sha256"]) == 3 * (len(DIAGNOSTIC_FILES) + 2)
    assert all(not feature["source_hint_linked"] for feature in result["features"])
    for flag in (
        "scientific_verified",
        "aliasing_ruled_out",
        "minimum_depth_verified",
        "period_evidence_verified",
        "task_completed",
        "answer_authorized",
        "learned_perception",
        "training_label",
        "daily_coverage_verified",
        "overview_hint_positions_used",
    ):
        assert result[flag] is False
    assert result == load(bundle)


def test_manifest_and_receipt_are_hash_complete_not_mutable_globals(bundle):
    result = load(bundle)
    manifest = measurement_manifest()
    manifest["limitations"].clear()
    assert measurement_manifest()["limitations"]
    declared = result.pop("receipt_sha256")
    assert (
        declared
        == hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


@pytest.fixture
def bounded_hints(tmp_path):
    return tmp_path, [
        fixture(tmp_path, f"bounded-{i}", 100 * (i + 1), linked=True, overview_hint_index=i) for i in range(3)
    ]


def _rewrite_hint_metadata(bundle, index, **changes):
    root, sources = bundle
    directory = root / sources[index]["directory"]
    scope = json.loads((directory / "scope.json").read_bytes())
    report = json.loads((directory / "report.json").read_bytes())
    scope.update(changes)
    report.update(
        {
            k: v
            for k, v in changes.items()
            if k in {"overview_hint_policy", "overview_hint_recipe", "source_hint_day_interval"}
        }
    )
    rewrite(bundle, index, "scope.json", scope)
    rewrite(bundle, index, "report.json", report)
    stream = events(bundle, index)
    stream[-1]["payload"] = report
    rewrite_events(bundle, stream, index)


def test_first_three_dense_hints_revalidated_without_using_raster_spacing_in_fit(bounded_hints):
    result = load(bounded_hints)
    assert result["period_days"]["value"] == "100"
    assert result["period_days"]["compatibility_interval"] == {
        "lower": "99",
        "upper": "101",
        "endpoints": "open",
    }
    assert [feature["overview_candidate_index"] for feature in result["features"]] == [0, 1, 2]
    assert all(feature["source_hint_linked"] for feature in result["features"])
    assert not result["overview_hint_positions_used"]
    assert not result["learned_perception"] and not result["task_completed"]
    assert not result["answer_authorized"] and result["brightness_drop_percent"]["physical_bounds"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"overview_hint_policy": None},
        {"overview_hint_policy": True},
        {"overview_hint_policy": "unrecognized"},
        {"candidate_index": True},
        {"candidate_index": 0.0},
        {"candidate_index": 3},
        {"candidate_columns": [36]},
        {"candidate_columns": [35.0]},
        {"source_hint_day_interval": [80, 120]},
    ],
)
def test_selected_hint_metadata_cannot_be_relabelled(bounded_hints, changes):
    _rewrite_hint_metadata(bounded_hints, 0, **changes)
    with pytest.raises(BrowserSafetyStop):
        load(bounded_hints)


def test_selected_hint_policy_cannot_mix_with_legacy_bundle(bounded_hints):
    root, specs = bounded_hints
    specs[2] = fixture(root, "legacy-third", 300, linked=True)
    with pytest.raises(BrowserSafetyStop, match="mixed_or_nonconsecutive_overview_hints"):
        load(bounded_hints)


def test_selected_hint_policy_requires_actual_dip_link(bounded_hints):
    root, specs = bounded_hints
    directory = root / specs[0]["directory"]
    report = json.loads((directory / "report.json").read_bytes())
    report.update(source_hint_linked=False, status="visible_decline_not_source_linked")
    rewrite(bounded_hints, 0, "report.json", report)
    stream = events(bounded_hints)
    stream[-1]["payload"] = report
    rewrite_events(bounded_hints, stream)
    with pytest.raises(BrowserSafetyStop, match="selected_hint_not_linked"):
        load(bounded_hints)


@pytest.fixture
def faint_hints(tmp_path):
    return tmp_path, [
        fixture(
            tmp_path,
            f"faint-{i}",
            100 * (i + 1),
            linked=True,
            overview_hint_index=i,
            overview_hint_policy=FIRST_THREE_LOW_INTENSITY_HINT_POLICY,
        )
        for i in range(3)
    ]


def test_v2_recipe_is_pinned_without_changing_measurement_authority(faint_hints):
    result = load(faint_hints)
    metadata = overview_hint_metadata(FIRST_THREE_LOW_INTENSITY_HINT_POLICY)
    assert result["period_days"]["value"] == "100"
    for feature in result["features"]:
        assert all(feature[key] == value for key, value in metadata.items())
    assert result["overview_hint_positions_used"] is False
    assert result["answer_authorized"] is False
    assert result["brightness_drop_percent"]["physical_bounds"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("answer_authorized", 0),
        ("learned_perception", 0.0),
        ("maximum_channel", 64.0),
        ("maximum_channel_residual", True),
        ("sha256", "0" * 64),
    ],
)
def test_v2_reader_rejects_rehashed_recipe_type_or_hash_mutations(faint_hints, key, value):
    recipe = overview_hint_metadata(FIRST_THREE_LOW_INTENSITY_HINT_POLICY)["overview_hint_recipe"]
    recipe[key] = value
    _rewrite_hint_metadata(faint_hints, 0, overview_hint_recipe=recipe)
    with pytest.raises(BrowserSafetyStop, match="unsupported_overview_hint_recipe"):
        load(faint_hints)


@pytest.mark.parametrize("legacy", [False, True])
def test_v2_bundle_rejects_mixed_legacy_or_v1(faint_hints, legacy):
    root, specs = faint_hints
    specs[2] = fixture(root, "older-third", 300, linked=True, overview_hint_index=None if legacy else 2)
    with pytest.raises(BrowserSafetyStop, match="mixed_or_nonconsecutive_overview_hints"):
        load(faint_hints)


def test_v1_recipe_metadata_remains_absent_and_orphan_recipe_rejected(bounded_hints):
    result = load(bounded_hints)
    assert all("overview_hint_recipe" not in feature for feature in result["features"])
    assert result == load(bounded_hints)
    _rewrite_hint_metadata(bounded_hints, 0, overview_hint_recipe={})
    with pytest.raises(BrowserSafetyStop, match="unsupported_overview_hint_recipe"):
        load(bounded_hints)


def test_eight_components_and_decimal_precision_are_preserved(tmp_path):
    low = "99.999999999999999999999999999999999999999999999999999999999999"
    specs = [
        fixture(tmp_path, f"d-{i}", 100 + i * 600, low=low, pan=bool(i % 2), linked=bool(i % 2))
        for i in range(8)
    ]
    result = load_tooltip_reference(tmp_path, list(reversed(specs)), expected_star="Example")
    assert result["period_days"]["exact_representative"] == {"numerator": 600, "denominator": 1}
    assert result["brightness_drop_percent"]["value"] == "0." + "0" * 59 + "1"
    assert [feature["bracket_days"][0] for feature in result["features"]] == [99 + i * 600 for i in range(8)]


@pytest.mark.parametrize(
    "values",
    [
        ["100", "99", "100", "98", "100"],
        ["100"] + ["99"] * 7 + ["100"],
    ],
)
def test_multiple_decline_runs_or_nine_samples_are_not_merged(tmp_path, values):
    specs = [fixture(tmp_path, f"d-{i}", 1000 * (i + 1), values=values) for i in range(3)]
    with pytest.raises(BrowserSafetyStop):
        load_tooltip_reference(tmp_path, specs, expected_star="EXAMPLE")


def test_extra_baseline_samples_trim_to_nearest_bracket(tmp_path):
    specs = [
        fixture(tmp_path, f"d-{i}", 1000 * (i + 1), values=["100", "100", "99", "100", "100"])
        for i in range(3)
    ]
    result = load_tooltip_reference(tmp_path, specs, expected_star="EXAMPLE")
    assert result["features"][0]["bracket_days"] == [999, 1001]
    assert result["period_days"]["compatibility_interval"]["lower"] == "999"


def test_eight_daily_samples_with_pointer_clear_and_pan_reach_sixteen_action_limit(tmp_path):
    specs = [
        fixture(
            tmp_path,
            f"d-{i}",
            1000 * (i + 1),
            pan=True,
            values=["100", "100", "99", "98", "98", "99", "100", "100"],
        )
        for i in range(3)
    ]
    result = load_tooltip_reference(tmp_path, specs, expected_star="EXAMPLE")
    assert result["brightness_drop_percent"]["value"] == "2"
    assert len(result["features"][0]["samples"]) == 8
    assert events((tmp_path, specs))[-1]["payload"]["browser_actions"] == 16


def test_legacy_v1_remains_read_only_compatible_but_cannot_mix_with_v2(tmp_path):
    specs = [fixture(tmp_path, f"d-{i}", 1000 * (i + 1), recipe=None) for i in range(3)]
    result = load_tooltip_reference(tmp_path, specs, expected_star="EXAMPLE")
    assert {f["sensor_recipe"] for f in result["features"]} == {"guarded_pointer_clear_v1"}
    assert result["browser_actions"] == 0
    assert result["period_days"]["value"] == "1000"
    specs[2] = fixture(tmp_path, "daily", 3000)
    with pytest.raises(BrowserSafetyStop, match="mixed_sensor_recipes"):
        load_tooltip_reference(tmp_path, specs, expected_star="EXAMPLE")


@pytest.mark.parametrize("requested_day", [None, True, 999.0, 998])
def test_v2_requires_exact_requested_integer_day(bundle, requested_day):
    value = events(bundle)
    value[15]["payload"]["requested_day"] = requested_day
    rewrite_events(bundle, value)
    with pytest.raises(BrowserSafetyStop, match="requested_day_disagrees"):
        load(bundle)


def test_v2_cannot_omit_requested_day(bundle):
    value = events(bundle)
    del value[15]["payload"]["requested_day"]
    rewrite_events(bundle, value)
    with pytest.raises(BrowserSafetyStop, match="unsupported_sample_event"):
        load(bundle)


def test_v2_rejects_skipped_actual_days_even_when_requested_and_axes_agree(bundle):
    value = events(bundle)
    for event in value:
        payload = event["payload"]
        if "requested_day" in payload:
            day = 1000 + (payload["requested_day"] - 1000) * 2
            payload["requested_day"] = day
            payload["x_fraction"] = (100.5 + (day - 1000) * 2.08) / 280
        if "chart_sample" in payload:
            payload["chart_sample"]["day"] = 1000 + (payload["chart_sample"]["day"] - 1000) * 2
    rewrite_events(bundle, value)
    with pytest.raises(BrowserSafetyStop, match="nonconsecutive_requested_days"):
        load(bundle)


def test_v2_requires_matching_recipe_in_report_and_scope(bundle):
    value = events(bundle)
    del value[-1]["payload"]["sensor_recipe"]
    rewrite(bundle, 0, "report.json", value[-1]["payload"])
    rewrite_events(bundle, value)
    with pytest.raises(BrowserSafetyStop, match="mixed_sensor_recipes"):
        load(bundle)


@pytest.mark.parametrize(
    "centers", [(1000, 2000, 3500), (1000, 2000, 4000), (1000, 1000, 3000), (1000, 1002, 3000)]
)
def test_inconsistent_missing_cycle_or_overlapping_features_stop(tmp_path, centers):
    specs = [fixture(tmp_path, f"d-{i}", center) for i, center in enumerate(centers)]
    with pytest.raises(BrowserSafetyStop):
        load_tooltip_reference(tmp_path, specs, expected_star="EXAMPLE")


@pytest.mark.parametrize("filename", sorted(DIAGNOSTIC_FILES))
def test_every_leaf_is_pinned(bundle, filename):
    path = bundle[0] / bundle[1][0]["directory"] / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(BrowserSafetyStop, match="source_hash_mismatch"):
        load(bundle)


@pytest.mark.parametrize("bad_hash", [None, "", True, 0, "a" * 63, "A" * 64, "z" * 64])
def test_explicit_valid_pins_are_required(bundle, bad_hash):
    bundle[1][0]["files"]["readable.png"] = bad_hash
    with pytest.raises(BrowserSafetyStop, match="explicit_hash_required"):
        load(bundle)


@pytest.mark.parametrize("marker", ["stopped.json", "invalidated.json"])
@pytest.mark.parametrize("location", ["diagnostic-0", "diagnostic-0-overview"])
def test_failed_sources_rejected(bundle, marker, location):
    (bundle[0] / location / marker).write_text("{}")
    with pytest.raises(BrowserSafetyStop, match="failed_evidence"):
        load(bundle)


def test_changed_overview_rejected(bundle):
    (bundle[0] / "diagnostic-0-overview/chart.png").write_bytes(image(81))
    with pytest.raises(BrowserSafetyStop, match="source_hash_mismatch"):
        load(bundle)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "duplicate", "too_few", "too_many", "wrong_star", "symlink", "outside"]
)
def test_source_identity_and_owned_path_contract(bundle, tmp_path, change):
    root, specs = bundle
    if change == "missing":
        del specs[0]["files"]["readable.png"]
    elif change == "extra":
        specs[0]["files"]["../extra.json"] = "0" * 64
    elif change == "duplicate":
        specs[1] = specs[0]
    elif change == "too_few":
        specs.pop()
    elif change == "too_many":
        specs.extend([specs[0]] * 6)
    elif change == "wrong_star":
        with pytest.raises(BrowserSafetyStop, match="mixed_method_or_star"):
            load_tooltip_reference(root, specs, expected_star="OTHER")
        return
    elif change == "symlink":
        path = root / "link"
        path.symlink_to(root / specs[0]["directory"], target_is_directory=True)
        specs[0]["directory"] = str(path)
    else:
        specs[0]["directory"] = str(tmp_path.parent)
    with pytest.raises(BrowserSafetyStop):
        load(bundle)


@pytest.mark.parametrize(
    "change",
    [
        "method",
        "writes",
        "bool_writes",
        "budget",
        "recovery",
        "png_hash",
        "axes",
        "sample_value",
        "missing_confirmation",
        "wrong_run",
        "sequence",
        "bad_day",
        "unknown_hover",
        "missing_clear",
        "false_clear",
        "nonvertical_pan",
    ],
)
def test_rehashed_but_semantically_invalid_evidence_rejected(bundle, change):
    root, specs = bundle
    directory = root / specs[0]["directory"]
    if change in {"method", "writes", "bool_writes", "budget", "recovery"}:
        value = json.loads((directory / "scope.json").read_text())
        key, replacement = {
            "method": ("mode", "different_sensor"),
            "writes": ("answer_writes", 1),
            "bool_writes": ("answer_writes", False),
            "budget": ("max_actions", 32),
            "recovery": ("source_owner_recovery", True),
        }[change]
        value[key] = replacement
        rewrite(bundle, 0, "scope.json", value)
    elif change in {"png_hash", "axes"}:
        value = json.loads((directory / "readable.json").read_text())
        if change == "png_hash":
            value["chart_sha256"] = "0" * 64
        else:
            value["time_axis_labels"][1]["value"] = "4000"
        rewrite(bundle, 0, "readable.json", value)
    else:
        value = events(bundle)
        if change == "sample_value":
            value[-4]["payload"]["chart_sample"]["brightness_percent"] = "1"
        elif change == "missing_confirmation":
            value.pop(-2)
        elif change == "wrong_run":
            value[1]["run_id"] = "other"
        elif change == "sequence":
            value[2]["payload"]["sequence"] = 2
        elif change == "bad_day":
            value[-2]["payload"]["chart_sample"]["day"] = 5001
        elif change == "unknown_hover":
            value[13]["payload"]["purpose"] = "something_else"
        elif change == "missing_clear":
            del value[13:15]
            for index, row in enumerate(value):
                row["sequence"] = index
        elif change == "false_clear":
            value[14]["payload"]["pointer_outside_chart"] = False
        elif change == "nonvertical_pan":
            value[1]["payload"] = {
                "kind": "DRAG",
                "surface": "chart",
                "sequence": 1,
                "start_fraction": [0.3, 0.3],
                "end_fraction": [0.5, 0.5],
            }
        rewrite_events(bundle, value)
    with pytest.raises(BrowserSafetyStop):
        load(bundle)


@pytest.mark.parametrize(
    "samples",
    [
        [(999, "99"), (1000, "98"), (1001, "100")],
        [(999, "100"), (1000, "100"), (1001, "100")],
        [(1000, "100"), (1000, "99"), (1001, "100")],
        [(999, "100"), (1000, "NaN"), (1001, "100")],
        [(999, "100"), (1000, "99"), (5001, "100")],
    ],
)
def test_invalid_brackets_never_become_partial_measurements(bundle, samples):
    value = events(bundle)
    new = [
        {"day": day, "brightness_percent": brightness, "source": "visible_hover_tooltip"}
        for day, brightness in samples
    ]
    for row, sample in zip([e for e in value if "chart_sample" in e["payload"]], new, strict=True):
        row["payload"]["chart_sample"] = sample
    report = value[-1]["payload"]
    report["samples"] = new
    rewrite(bundle, 0, "report.json", report)
    rewrite_events(bundle, value)
    with pytest.raises(BrowserSafetyStop):
        load(bundle)


def test_source_mutation_at_final_check_prevents_receipt(bundle, monkeypatch):
    from habfly.planet_tooltip_reference import _Sources

    original = _Sources.unchanged

    def change_then_check(book):
        (bundle[0] / "diagnostic-0/readable.png").write_bytes(b"changed")
        original(book)

    monkeypatch.setattr(_Sources, "unchanged", change_then_check)
    with pytest.raises(BrowserSafetyStop, match="source_hash_mismatch"):
        load(bundle)
