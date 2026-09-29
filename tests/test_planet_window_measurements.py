"""Synthetic raster geometry, not ground-truth planetary or training examples."""

import hashlib
import io
import json
import socket

import pytest
from PIL import Image, ImageDraw

from habfly.planet_window_measurements import (
    estimate_planet_window_measurements,
    load_planet_window_measurements,
    measurement_manifest,
)

BLUE = (80, 132, 154)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def axes(maximum=5000, minimum_flux=0):
    times = [
        {"value": str(day), "center_x": 30.5 + day * 230 / maximum}
        for day in range(0, maximum + 1, maximum // 10)
    ]
    flux = [
        {"value": str(minimum_flux + (100 - minimum_flux) * step / 10), "center_y": 150.5 - step * 13}
        for step in range(1, 11)
    ]
    return times, flux


def png(*, centers=(60, 106, 152, 198, 244), bottoms=33, end=260, widths=1, edit=None):
    image = Image.new("RGB", (280, 195))
    draw = ImageDraw.Draw(image)
    for x in range(30, 261, 23):
        draw.line((x, 20, x, 150), fill=(60, 60, 60))
    for y in range(20, 151, 13):
        draw.line((30, y, 260, y), fill=(60, 60, 60))
    draw.line((30, 20, end, 20), fill=BLUE)
    depths = [bottoms] * len(centers) if isinstance(bottoms, int) else bottoms
    for x, bottom in zip(centers, depths, strict=True):
        draw.rectangle((x, 20, x + widths - 1, bottom), fill=BLUE)
    if edit:
        edit(draw)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def estimate(image=None, *, maximum=5000, minimum_flux=0, **overrides):
    image = png() if image is None else image
    times, flux = axes(maximum, minimum_flux)
    options = {
        "expected_chart_sha256": hashlib.sha256(image).hexdigest(),
        "expected_axis_sha256": digest({"time": times, "flux": flux}),
    }
    options.update(overrides)
    return estimate_planet_window_measurements(image, times, flux, **options)


def assert_error(result, code=None):
    assert result["status"] == "measurement_error" and result["error"]
    if code:
        assert result["error"]["code"] == code
    assert result["period_days"] is None and result["brightness_drop_percent"] is None
    assert result["planet_decision"] is None and not result["absence_proven"]
    assert not result["scientific_verified"] and not result["training_label"]
    assert result["answer_writes"] == result["browser_actions"] == 0


def test_independent_linear_pixel_golden_has_explicit_uncertainty_not_science():
    result = estimate()
    assert result["status"] == "approximate_reference_measurements"
    period, depth = result["period_days"], result["brightness_drop_percent"]
    # 46px * 5000days/230px = 1000days. Five observed centers, each +/-1px,
    # constrain the four-spacing span to 184+/-2px, hence 46+/-0.5px.
    assert period["value"] == pytest.approx(1000)
    assert period["lower"] == pytest.approx(45.5 * 5000 / 230)
    assert period["upper"] == pytest.approx(46.5 * 5000 / 230)
    # Pixel center33.5 minus exposed100% baseline20.5 =13px. The visible
    # linear scale is1.3px per percentage point, not an invented flux value.
    assert depth["value"] == pytest.approx(10)
    assert depth["lower"] == pytest.approx(12 / 1.3)
    assert depth["upper"] == pytest.approx(14 / 1.3)
    assert depth["pixel_uncertainty_percent"] == pytest.approx(1 / 1.3)
    assert period["center_uncertainty_pixels"] == depth["deepest_trace_uncertainty_pixels"] == 1
    assert result["supported_dip_components"] == 5 and result["raster_endpoint_verified"]
    for field in (
        "scientific_verified",
        "learned_perception",
        "training_label",
        "task_completed",
        "observation_completed",
        "daily_coverage_verified",
        "missing_events_ruled_out",
        "aliasing_ruled_out",
        "absence_proven",
    ):
        assert result[field] is False
    assert result["planet_decision"] is None
    assert result["method"]["limitations"] and result["approximate"]
    assert result["positive_evidence_sha256"] == digest(result["positive_evidence"])


def test_zoomed_flux_axis_still_uses_visible_linear_percent_units():
    result = estimate(minimum_flux=99.5)
    assert result["brightness_drop_percent"]["value"] == pytest.approx(0.05)
    assert result["brightness_drop_percent"]["pixel_uncertainty_percent"] == pytest.approx(1 / 260)


@pytest.mark.parametrize("centers", [(60, 106), (), (60,)])
def test_requires_at_least_three_observed_components_and_never_inferrs_no(centers):
    assert_error(estimate(png(centers=centers)))


@pytest.mark.parametrize("centers", [(50, 90, 145), (50, 70, 110), (50, 90, 130, 174, 218)])
def test_inconsistent_or_missing_event_like_spacing_is_not_repaired(centers):
    assert_error(estimate(png(centers=centers)), "inconsistent_observed_component_spacing")


def test_consistent_visible_recurrence_does_not_rule_out_aliasing_or_hidden_events():
    result = estimate(png(centers=(50, 90, 130, 170)))
    assert result["period_days"]["value"] == pytest.approx(40 * 5000 / 230)
    assert not result["aliasing_ruled_out"] and not result["missing_events_ruled_out"]
    assert result["supported_dip_components"] == 4


def test_one_pixel_raster_phase_quantization_is_accepted_with_bounded_fit():
    result = estimate(png(centers=(50, 60, 69, 79, 88, 98)))
    assert result["status"] == "approximate_reference_measurements"
    assert result["period_days"]["maximum_center_residual_pixels"] <= 1
    assert result["period_days"]["lower"] <= 9.5 * 5000 / 230 <= result["period_days"]["upper"]


def test_nearby_components_are_not_treated_as_independent_well_separated_dips():
    assert_error(estimate(png(centers=(60, 63, 66))), "dip_components_not_well_separated")


@pytest.mark.parametrize("depths", [(33, 33, 36), (28, 33, 38)])
def test_incompatible_depths_return_no_partial_answers(depths):
    assert_error(
        estimate(png(centers=(60, 110, 160), bottoms=depths)), "inconsistent_observed_component_depths"
    )


def test_depth_variation_within_pixel_uncertainty_uses_deepest_real_support():
    result = estimate(png(centers=(60, 110, 160), bottoms=(32, 33, 34)))
    assert result["brightness_drop_percent"]["value"] == pytest.approx(14 / 1.3)
    assert result["deepest_pixel_y"] == 34.5


def test_explained_antialias_pixel_never_increases_measured_depth():
    def edge(draw):
        draw.line((30, 34, 260, 34), fill=(60, 60, 60))
        draw.point((60, 34), fill=(70, 96, 107))  # half-blue/neutral60, outside trace_mask

    result = estimate(png(edit=edge))
    assert result["status"] == "approximate_reference_measurements"
    assert result["positive_evidence"]["explained_antialias_pixels"] >= 1
    assert result["deepest_pixel_y"] == 33.5
    assert result["brightness_drop_percent"]["value"] == pytest.approx(10)


@pytest.mark.parametrize("kind", ["partial", "clipped", "unknown", "gap", "white", "detached", "broad"])
def test_clipping_palette_occlusion_and_partial_coverage_fail_closed(kind):
    options = {"centers": (60, 106, 152)}
    if kind == "partial":
        options["end"] = 190
    elif kind == "clipped":
        options["bottoms"] = 151
    elif kind == "unknown":
        options["edit"] = lambda draw: draw.point((200, 50), fill=(255, 80, 20))
    elif kind in {"gap", "white"}:
        options["edit"] = lambda draw: draw.rectangle(
            (80, 18, 85, 22), fill="white" if kind == "white" else "black"
        )
    elif kind == "detached":
        options["edit"] = lambda draw: draw.line((200, 40, 200, 48), fill=BLUE)
    else:
        options["widths"] = 5
    assert_error(estimate(png(**options)))


def test_a_full10000_axis_is_not_silently_reinterpreted_as_the_fixed5000_view():
    result = estimate(png(centers=(50, 90, 125)), maximum=10000)
    assert_error(result, "full_0_to_5000_axis_required")


@pytest.mark.parametrize("field", ["expected_chart_sha256", "expected_axis_sha256"])
@pytest.mark.parametrize("bad", [None, "bad", "0" * 64])
def test_immutable_input_hashes_are_required(field, bad):
    assert_error(
        estimate(**{field: bad}), "chart_hash_mismatch" if "chart" in field else "axis_hash_mismatch"
    )


@pytest.mark.parametrize("kind", ["shift", "nonlinear", "missing", "nan", "bool"])
def test_invalid_visible_axis_geometry_is_never_repaired(kind):
    image = png()
    times, flux = axes()
    if kind == "shift":
        times = [{**row, "center_x": row["center_x"] + 5} for row in times]
    elif kind == "nonlinear":
        flux[4]["center_y"] += 4
    elif kind == "missing":
        flux = flux[:-1]
    elif kind == "nan":
        times[4]["value"] = "nan"
    else:
        times[0]["value"] = False
    result = estimate_planet_window_measurements(
        image,
        times,
        flux,
        expected_chart_sha256=hashlib.sha256(image).hexdigest(),
        expected_axis_sha256=digest({"time": times, "flux": flux}),
    )
    assert_error(result)


def capture(tmp_path, image=None):
    image = png() if image is None else image
    times, flux = axes()
    report = {
        "method": "rendered_blue_trace_frontier_v1",
        "source": "chart_crop_pixels_and_visible_axis_labels",
        "requested_days": 5000,
        "star": "Example",
        "browser_actions": 0,
        "answer_writes": 0,
        "chart_sha256": hashlib.sha256(image).hexdigest(),
        "time_axis_labels": times,
        "flux_axis_labels": flux,
        "status": "trace_reaches_requested_end",
        "endpoint_visible": True,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    (tmp_path / "chart.png").write_bytes(image)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_loader_replays_hash_pinned_capture_offline_without_modifying_it(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: pytest.fail("Network forbidden"))
    path, checksum = capture(tmp_path)
    originals = {p: p.read_bytes() for p in tmp_path.iterdir()}
    report = load_planet_window_measurements(path, expected_report_sha256=checksum)
    assert report == load_planet_window_measurements(path, expected_report_sha256=checksum)
    assert report["status"] == "approximate_reference_measurements" and report["star"] == "Example"
    assert report["source_report_sha256"] == checksum
    assert all(p.read_bytes() == value for p, value in originals.items())
    assert set(tmp_path.iterdir()) == set(originals)


@pytest.mark.parametrize(
    "kind", ["report", "image", "identity", "days", "truthy_zero", "missing", "readiness_lie"]
)
def test_loader_rejects_tamper_and_recomputes_visible_readiness(tmp_path, kind):
    path, checksum = capture(
        tmp_path, png(centers=(60, 106, 152), end=190) if kind == "readiness_lie" else None
    )
    if kind == "report":
        path.write_text(path.read_text() + " ")
    elif kind == "image":
        (tmp_path / "chart.png").write_bytes(png(bottoms=34))
    elif kind in {"identity", "days", "truthy_zero", "missing"}:
        report = json.loads(path.read_bytes())
        if kind == "identity":
            report["method"] = "unknown"
        elif kind == "days":
            report["requested_days"] = 10000
        elif kind == "truthy_zero":
            report["answer_writes"] = False
        else:
            del report["flux_axis_labels"]
        path.write_text(json.dumps(report))
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    assert_error(load_planet_window_measurements(path, expected_report_sha256=checksum))


def test_manifest_fingerprints_geometry_uncertainty_and_positive_detector():
    manifest = measurement_manifest()
    checksum = manifest.pop("sha256")
    assert checksum == digest(manifest)
    assert manifest["positive_detector_sha256"]
    manifest["limitations"].clear()
    assert measurement_manifest()["limitations"]
