"""Diagram geometry and abstention, never course-classification accuracy."""

import hashlib
import json
import math
import socket
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import habfly.stellar_reference as module
from habfly.stellar_reference import (
    _coordinate,
    _region,
    classify_hr_reference,
    load_hr_reference,
    validate_hr_reference_source,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Reference geometry and source validation must remain offline")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def replace_pack(monkeypatch, value):
    raw = json.dumps(value).encode() if isinstance(value, dict) else value
    resource = SimpleNamespace(joinpath=lambda _: SimpleNamespace(read_bytes=lambda: raw))
    monkeypatch.setattr(module, "files", lambda _: resource)


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    """Synthetic local PNG tests file verification, never source authenticity."""
    pack, _ = load_hr_reference()
    buffer = BytesIO()
    Image.new("RGB", tuple(pack["source"]["image_size"]), "black").save(buffer, format="PNG")
    raw = buffer.getvalue()
    path = tmp_path / pack["source"]["capture"]
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    pack["source"]["sha256"] = hashlib.sha256(raw).hexdigest()
    replace_pack(monkeypatch, pack)
    return pack, path, tmp_path


def at_pixel(x, y):
    pack, _ = load_hr_reference()
    quantities = {}
    for name, pixel in (("temperature", x), ("luminosity", y)):
        (a, pa), (b, pb) = pack["axes"][name]["ticks"]
        quantities[name] = 10 ** (math.log10(a) + (pixel - pa) * (math.log10(b) - math.log10(a)) / (pb - pa))
    return classify_hr_reference(**quantities)


@pytest.mark.parametrize(
    "x,y,name",
    [
        (396, 290, "main_sequence"),
        (260, 372, "white_dwarf"),
        (440, 214, "red_giant"),
        (440, 116, "supergiant"),
    ],
)
def test_visible_band_interior_anchors(x, y, name):
    result = at_pixel(x, y)
    assert result["status"] == "reference_candidate"
    assert result["selected_class"] == name
    assert result["write_authorized"] is result["training_label"] is result["scientific_verified"] is False
    assert result["runtime_enabled"] is result["source_image_verified"] is False
    assert result["diagnostic_only"] is True and result["boundary_margin_calibrated"] is False


def test_mapping_reproduces_published_tick_positions():
    pack, checksum = load_hr_reference()
    assert len(checksum) == 64
    for spec in (pack["axes"]["temperature"], pack["axes"]["luminosity"]):
        for value, pixel in spec["ticks"]:
            assert _coordinate(value, spec["ticks"]) == pytest.approx(pixel)


@pytest.mark.parametrize(
    "value", [0, -1, True, "5800", None, float("nan"), float("inf"), pytest.param(10**1000, id="overflow")]
)
def test_invalid_inputs_are_not_repaired(value):
    assert (
        classify_hr_reference(luminosity=1, temperature=value)["reason"]
        == "positive_finite_coordinates_required"
    )


def test_unit_mismatch_and_extrapolation_abstain():
    assert (
        classify_hr_reference(luminosity=1, temperature=5800, temperature_unit="C")["reason"]
        == "incompatible_units"
    )
    assert classify_hr_reference(luminosity=1, temperature=100)["reason"] == "outside_reference_diagram"


def test_blank_and_boundary_regions_do_not_default_to_main():
    assert at_pixel(100, 400)["reason"] == "outside_digitized_interiors"
    result = at_pixel(103, 57)
    assert result["status"] == "abstained" and result["selected_class"] is None


def test_polygon_test_and_edge_distance_are_independent():
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert _region([5, 5], polygon) == (True, 5)
    assert _region([12, 5], polygon) == (False, 2)
    assert _region([10, 5], polygon)[1] == 0


def test_near_boundary_abstains_even_inside():
    # One pixel inside the segment from (253,334) to (304,360).
    assert at_pixel(278.5, 348)["status"] == "abstained"


@pytest.mark.parametrize(
    "axis,value,pixel",
    [
        ("temperature", 10000, 313),
        ("temperature", 6000, 386),
        ("luminosity", 1000000, 56.5),
        ("luminosity", 1000, 173),
        ("luminosity", 100, 212),
        ("luminosity", 10, 251),
        ("luminosity", 0.1, 328.5),
        ("luminosity", 0.001, 406.5),
    ],
)
def test_independently_observed_visible_tick_anchors(axis, value, pixel):
    # These positions were read directly from the unscaled 566x575 image,
    # independently of the endpoint transform. They are approximate line
    # centers, not exact astronomical or course-classification boundaries.
    pack, _ = load_hr_reference()
    assert value not in [tick[0] for tick in pack["axes"][axis]["ticks"]]
    assert [value, pixel] in pack["digitization"]["axis_checks"][axis]
    assert _coordinate(value, pack["axes"][axis]["ticks"]) == pytest.approx(pixel, abs=2.5)


def test_reviewed_polygon_coordinates_and_margin_have_not_been_loosened():
    pack, _ = load_hr_reference()
    raw = json.dumps(pack["regions"], sort_keys=True, separators=(",", ":")).encode()
    assert (
        hashlib.sha256(raw).hexdigest() == "26dd6c69a504e336157a73370b07b21afca9bc384b76cedb66e0154b2235870e"
    )
    assert pack["boundary_abstention_pixels"] == 2.0
    assert pack["digitization"]["boundary_margin_calibrated"] is False
    assert pack["digitization"]["visible_label_aliases"]["GIANTS"] == "red_giant"
    assert "GIANTS" in pack["digitization"]["alias_note"]


@pytest.mark.parametrize(
    "path,bad",
    [
        (("version",), True),
        (("version",), 2),
        (("mode",), "learned_prediction"),
        (("browser_actions",), False),
        (("browser_actions",), 1),
        (("classification_learned",), True),
        (("scientific_verified",), True),
        (("training_label",), True),
        (("runtime_enabled",), True),
        (("write_authorized",), True),
        (("diagnostic_only",), 1),
        (("boundary_abstention_pixels",), float("nan")),
        (("boundary_abstention_pixels",), float("inf")),
        (("boundary_abstention_pixels",), -1),
        (("boundary_abstention_pixels",), False),
        (("boundary_abstention_pixels",), 1000),
        (("source", "sha256"), "wrong"),
        (("source", "surface"), "hidden_state"),
        (("source", "coordinate_frame"), "resized_image"),
        (("source", "capture_date"), "2026-02-30"),
        (("source", "image_size"), [True, 575]),
        (("source", "image_size"), [100, 100]),
        (("source", "capture"), "../reference.png"),
        (("source", "capture"), "/reference.png"),
        (("source", "capture"), "https://example.test/reference.png"),
        (("source", "capture"), "foo\\reference.png"),
        (("source", "capture"), "foo\u0000reference.png"),
        (("axes", "temperature", "unit"), "C"),
        (("axes", "temperature", "transform"), "linear"),
        (("axes", "temperature", "ticks"), [[30000, 155], [30000, 486]]),
        (("axes", "temperature", "ticks"), [[30000, 155], [3000, 155]]),
        (("axes", "temperature", "ticks"), [[30000, 486], [3000, 155]]),
        (("axes", "temperature", "ticks"), [[30000, 155], [0, 486]]),
        (("axes", "luminosity", "ticks"), [[1, 290], [float("nan"), 486]]),
        (("axes", "plot_bounds"), [543, 37, 81, 487]),
        (("axes", "plot_bounds"), [81, 37, 543, float("inf")]),
        (("regions", "main_sequence"), []),
        (("regions", "main_sequence"), [[100, 100], [120, 100]]),
        (("regions", "main_sequence"), [[100, 100], [120, 100], [100, 100]]),
        (("regions", "main_sequence"), [[100, 100], [120, 100], [140, 100]]),
        (("regions", "main_sequence"), [[100, 100], [150, 140], [100, 140], [145, 100]]),
        (("regions", "main_sequence"), [[100, 100], [float("nan"), 120], [120, 100]]),
        (("regions", "main_sequence"), [[0, 0], [120, 100], [140, 120]]),
        (("digitization", "boundary_margin_calibrated"), True),
        (("digitization", "boundary_authority"), "course_truth"),
        (("digitization", "visible_label_aliases", "GIANTS"), "supergiant"),
        (("digitization", "axis_check_tolerance_pixels"), 100),
        (("digitization", "axis_check_tolerance_pixels"), float("nan")),
        (("digitization", "axis_checks", "temperature"), [[30000, 155], [3000, 486]]),
        (("digitization", "axis_checks", "temperature"), [[10000, 313], [10000, 313]]),
        (("digitization", "axis_checks", "temperature"), [[10000, 100], [6000, 386]]),
    ],
)
def test_malformed_pack_cannot_become_a_reference_candidate(monkeypatch, path, bad):
    pack, _ = load_hr_reference()
    parent = pack
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = bad
    replace_pack(monkeypatch, pack)
    with pytest.raises(ValueError, match="hr_reference_"):
        load_hr_reference()
    with pytest.raises(ValueError, match="hr_reference_"):
        classify_hr_reference(luminosity=1, temperature=5800)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "duplicate", "invalid_json", "array", "oversized", "nested"]
)
def test_schema_and_json_fail_closed(monkeypatch, change):
    pack, _ = load_hr_reference()
    if change == "missing":
        pack.pop("source")
    elif change == "extra":
        pack["automatic_class_selection"] = True
    elif change == "duplicate":
        pack = json.dumps(pack).replace('"version": 1', '"version": 1, "version": 1').encode()
    elif change == "invalid_json":
        pack = b"private invalid JSON"
    elif change == "array":
        pack = b"[]"
    elif change == "nested":
        pack = b"[" * 2000 + b"0" + b"]" * 2000
    else:
        pack = b" " * 100_001
    replace_pack(monkeypatch, pack)
    with pytest.raises(ValueError, match="hr_reference_") as error:
        load_hr_reference()
    assert "private" not in str(error.value)


def test_source_validation_is_explicit_and_does_not_grant_action_authority(source_fixture):
    _pack, path, root = source_fixture
    before = path.read_bytes()
    verified = validate_hr_reference_source(root)
    assert verified["source_image_verified"] is True
    assert verified["browser_actions"] == 0
    for key in (
        "classification_learned",
        "scientific_verified",
        "training_label",
        "runtime_enabled",
        "write_authorized",
    ):
        assert verified[key] is False
    plain = classify_hr_reference(luminosity=1, temperature=5800)
    sourced = classify_hr_reference(luminosity=1, temperature=5800, source_root=root)
    assert plain["source_image_verified"] is False and sourced["source_image_verified"] is True
    assert {k: v for k, v in sourced.items() if k != "source_image_verified"} == {
        k: v for k, v in plain.items() if k != "source_image_verified"
    }
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "change", ["missing", "changed", "symlink", "dimensions", "invalid_png", "oversized", "directory"]
)
def test_invalid_source_never_yields_a_candidate(source_fixture, monkeypatch, change):
    pack, path, root = source_fixture
    if change == "missing":
        path.unlink()
    elif change == "changed":
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "symlink":
        copy = path.with_name("copy.png")
        path.rename(copy)
        path.symlink_to(copy)
    elif change == "dimensions":
        pack["source"]["image_size"] = [567, 575]
        replace_pack(monkeypatch, pack)
    elif change == "invalid_png":
        path.write_bytes(b"not an image")
        pack["source"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        replace_pack(monkeypatch, pack)
    elif change == "directory":
        path.unlink()
        path.mkdir()
    else:
        path.write_bytes(b"a" * 5_000_001)
    with pytest.raises(ValueError, match="hr_reference_"):
        validate_hr_reference_source(root)
    with pytest.raises(ValueError, match="hr_reference_"):
        classify_hr_reference(luminosity=1, temperature=5800, source_root=root)


@pytest.mark.parametrize("value", [5e-324, 1e308])
@pytest.mark.parametrize("key", ["luminosity", "temperature"])
def test_finite_extremes_abstain_without_crashing(value, key):
    values = {"luminosity": 1, "temperature": 5800, key: value}
    result = classify_hr_reference(**values)
    assert result["status"] == "abstained" and result["reason"] == "outside_reference_diagram"
    assert all(math.isfinite(v) for v in result["diagram_pixel_coordinates"])


def test_original_visible_source_when_available_is_verified_read_only():
    root = Path(__file__).resolve().parents[1]
    pack, _ = load_hr_reference()
    path = root / pack["source"]["capture"]
    if not path.is_file():
        pytest.skip("Original visible reference capture is a local experiment artifact")
    before = path.read_bytes()
    assert pack["source"]["sha256"] == "b1850640a9faf6ccdbd55af6bf5243d79faa6d2238a4fc130c6c458f94acb122"
    assert pack["source"]["image_size"] == [566, 575]
    assert validate_hr_reference_source(root)["source_image_verified"] is True
    assert path.read_bytes() == before
