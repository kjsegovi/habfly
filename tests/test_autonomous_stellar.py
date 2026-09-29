"""Offline public-capture fixtures, not classification accuracy or native proof."""

import hashlib
import json
import math
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_star_preflight import save_capture, stellar, write
from test_stellar_reference import source_fixture  # noqa: F401 - pytest fixture

import habfly.autonomous_stellar as module
from habfly.autonomous_stellar import AutonomousStellarError, decide_stellar_reference
from habfly.knowledge import LocalCalculator
from habfly.stellar_reference import classify_hr_reference, load_hr_reference


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantities(x, y):
    pack, _ = load_hr_reference()
    values = {}
    for name, pixel in (("temperature", x), ("luminosity", y)):
        (a, pa), (b, pb) = pack["axes"][name]["ticks"]
        values[name] = 10 ** (math.log10(a) + (pixel - pa) * (math.log10(b) - math.log10(a)) / (pb - pa))
    return values


def public_capture(x, y):
    """Invert the documented worksheet equations solely to create test inputs."""
    values = quantities(x, y)
    parallax = 0.1
    distance = 3.26 / parallax
    flux = values["luminosity"] * 3.827e26 / (4 * math.pi * (distance * 9460500000000000) ** 2)
    wavelength = 2897768.5 / values["temperature"]
    report = stellar()
    frame = report["frames"][0]
    frame["accessibility"] = (
        frame["accessibility"]
        .replace("0.045", str(parallax))
        .replace("370", str(wavelength))
        .replace("5.68E-10", str(flux))
    )
    return report


@pytest.fixture(autouse=True)
def no_network_or_private_answers(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Decision must use only local public reference calculations")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(LocalCalculator, "reference_answers", forbidden)


@pytest.fixture
def case(tmp_path, source_fixture):  # noqa: F811 - imported pytest fixture
    _, reference, root = source_fixture
    count = 0

    def make(x=396, y=290, *, painted=None):
        nonlocal count
        count += 1
        history = tmp_path / f"history-{count}"
        fresh = history / "fresh"
        capture_sha = save_capture(fresh / "stellar", public_capture(x, y))
        write(
            fresh / "confirmed.json",
            {
                "star": "Althinagon",
                "fresh_blank_numeric_answers_verified": True,
                "class_selection_verified": False,
                "answer_writes": 0,
                "action_source": "deterministic_navigation",
                "painted_stellar_class": painted,
                "task_completed": False,
            },
        )
        return SimpleNamespace(
            history=history,
            fresh=fresh,
            reference=reference,
            source_root=root,
            receipt_sha=sha(fresh / "confirmed.json"),
            capture_sha=capture_sha,
        )

    return make


def decide(item, **options):
    defaults = {
        "fresh_star_sha256": item.receipt_sha,
        "capture_sha256": item.capture_sha,
        "source_root": item.source_root,
        "expected_star": "ALTHINAGON",
    }
    return decide_stellar_reference(item.history, item.fresh, **(defaults | options))


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (396, 290, "main_sequence"),
        (260, 372, "white_dwarf"),
        (440, 214, "red_giant"),
        (440, 116, "supergiant"),
    ],
)
def test_unique_interiors_with_exact_payload_and_no_authority(case, x, y, expected):
    item = case(x, y)
    before = {path: sha(path) for path in item.history.rglob("*") if path.is_file()}
    result = decide(item)
    assert result["decision_kind"] == "stellar_class"
    assert result["status"] == "reference_decision"
    assert set(result["payload"]) == {"selected_class", "reference_rationale", "lifetime_prefix"}
    assert result["payload"]["selected_class"] == expected
    rationale = result["payload"]["reference_rationale"]
    assert 40 <= len(rationale) <= 2000 and len(rationale.split()) >= 8
    assert not any(ord(c) < 32 for c in rationale)
    assert (result["payload"]["lifetime_prefix"] is None) is (expected != "main_sequence")
    assert set(result["calculations"]) == (
        {"distance", "luminosity", "temperature", "mass", "lifetime"}
        if expected == "main_sequence"
        else {"distance", "luminosity", "temperature"}
    )
    assert result["geometry"]["fallback_applied"] is False
    assert result["geometry"]["outside_plot"] is False
    assert result["geometry"]["extrapolated_axis_coordinates"] is False
    assert result["geometry"]["outside_plot_distance_pixels"] == 0
    assert result["geometry"]["diagram_pixel_coordinates"] == pytest.approx([x, y])
    assert result["geometry"]["candidate_interiors"] == [expected]
    for field in (
        "classification_learned",
        "learned_perception",
        "scientific_verified",
        "training_label",
        "course_correctness_verified",
        "write_authorized",
        "current_browser_state_verified",
        "task_completed",
        "project_completed",
    ):
        assert result[field] is False
    assert result["browser_actions"] == 0
    assert result["approximate"] is result["reference_assisted"] is True
    assert result["provenance"]["arithmetic_golden_validation"]["cases"] == 9
    assert result["provenance"]["hr_source_image_verified"] is True
    assert len(result["source_sha256"]) == 3
    assert result["capture_expected_hash_supplied"] is True
    assert result["fresh_star_sha256"] == item.receipt_sha
    for relative, checksum in result["source_sha256"].items():
        assert sha(item.history / relative) == checksum
    assert before == {path: sha(path) for path in item.history.rglob("*") if path.is_file()}
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert decide(item) == result


@pytest.mark.parametrize("x,y", [(278.5, 348), (100, 400), (103, 57)])
def test_gap_or_boundary_has_explicit_approximate_nearest_policy(case, x, y):
    item = case(x, y)
    pack_before = load_hr_reference()
    original = classify_hr_reference(**quantities(x, y), source_root=item.source_root)
    assert original["status"] == "abstained"
    result = decide(item)
    assert result["geometry"]["fallback_applied"] is True
    assert result["geometry"]["nearest_region"] == result["payload"]["selected_class"]
    assert result["geometry"]["distance_gap_pixels"] > 0
    assert result["geometry"]["physical_uncertainty"] is None
    assert result["policy"]["id"] == "nearest_diagram_region_v2"
    assert result["policy"]["version"] == 2
    assert result["mode"] == "approximate_reference_assisted_stellar_v2"
    assert result["policy"]["distance_calibrated"] is False
    assert result["policy"]["maximum_fallback_distance_pixels"] is None
    assert "uncalibrated nearest" in result["payload"]["reference_rationale"]
    assert classify_hr_reference(**quantities(x, y), source_root=item.source_root) == original
    assert load_hr_reference() == pack_before


def test_policy_hash_reconstructible_and_source_hashes_distinct(case):
    item = case()
    result = decide(item)
    raw = json.dumps(result["policy"], sort_keys=True, separators=(",", ":")).encode()
    assert result["policy_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["provenance"]["hr_pack_sha256"] == load_hr_reference()[1]
    assert result["provenance"]["hr_source"]["sha256"] == sha(item.reference)
    assert result["capture_sha256"] == item.capture_sha


@pytest.mark.parametrize("painted", [None, "main_sequence", "white_dwarf", "red_giant", "supergiant"])
def test_inherited_paint_is_never_used_as_class_input(case, painted):
    assert decide(case(440, 214, painted=painted))["payload"]["selected_class"] == "red_giant"


@pytest.mark.parametrize(
    "x,y,expected,distance",
    [
        (80, 290, "white_dwarf", 1),
        (544, 290, "red_giant", 1),
        (300, 36, "supergiant", 1),
        (300, 488, "white_dwarf", 1),
        (80, 36, "main_sequence", math.sqrt(2)),
        (544, 488, "main_sequence", math.sqrt(2)),
    ],
)
def test_v2_outside_projection_is_not_clamped_or_a_default_class(case, x, y, expected, distance):
    item = case(x, y)
    pack_before = load_hr_reference()
    original = classify_hr_reference(**quantities(x, y), source_root=item.source_root)
    assert original["status"] == "abstained" and original["reason"] == "outside_reference_diagram"
    result = decide(item)
    geometry = result["geometry"]
    assert result["payload"]["selected_class"] == expected
    assert geometry["diagram_pixel_coordinates"] == pytest.approx([x, y])
    assert geometry["outside_plot"] is geometry["extrapolated_axis_coordinates"] is True
    assert geometry["outside_plot_distance_pixels"] == pytest.approx(distance)
    assert geometry["fallback_applied"] is True and geometry["candidate_interiors"] == []
    assert geometry["diagnostic_reason"] == "outside_reference_diagram"
    assert geometry["distance_calibrated"] is geometry["extrapolation_calibrated"] is False
    assert geometry["physical_uncertainty"] is None
    assert "extrapolated log-axis coordinates outside the plot" in result["payload"]["reference_rationale"]
    assert result["policy"]["outside_plot_rule"].endswith("without_clamping_or_extending_polygons")
    assert result["policy"]["extrapolation_calibrated"] is False
    assert any("unsupported by visible diagram coverage" in s for s in result["policy"]["limitations"])
    assert classify_hr_reference(**quantities(x, y), source_root=item.source_root) == original
    assert load_hr_reference() == pack_before


def test_egrithori_visible_measurements_have_explicit_extrapolated_reference_choice(case):
    """Recorded public inputs reproduced independently; no course answer label."""
    item = case()
    report = stellar()
    frame = report["frames"][0]
    frame["accessibility"] = (
        frame["accessibility"]
        .replace("0.045", "0.036")
        .replace("370", "1463")
        .replace("5.68E-10", "1.83E-14")
    )
    item.capture_sha = save_capture(item.fresh / "stellar", report)
    before = {p: sha(p) for p in item.history.rglob("*") if p.is_file()}
    result = decide(item)
    assert result["calculations"]["temperature"]["value"] == pytest.approx(1980.703007518797)
    assert result["calculations"]["luminosity"]["value"] == pytest.approx(0.0004410226253553728)
    geometry = result["geometry"]
    assert geometry["diagram_pixel_coordinates"] == pytest.approx([545.679926815259, 421.5371338870886])
    assert geometry["outside_plot_distance_pixels"] == pytest.approx(2.679926815259)
    assert geometry["outside_axis_distance_pixels"] == pytest.approx(
        {"temperature": 2.679926815259, "luminosity": 0}
    )
    assert geometry["nearest_distance_pixels"] == pytest.approx(26.318788420164534)
    assert geometry["outside_plot"] is geometry["extrapolated_axis_coordinates"] is True
    assert result["payload"]["selected_class"] == "main_sequence"
    assert result["payload"]["lifetime_prefix"] == "Ta"
    assert result["scientific_verified"] is result["training_label"] is result["task_completed"] is False
    assert before == {p: sha(p) for p in item.history.rglob("*") if p.is_file()}


@pytest.mark.parametrize("value", [True, None, "123", math.nan, math.inf, -math.inf, 10**400])
def test_invalid_or_nonrepresentable_projection_never_extrapolates(value):
    pack, _ = load_hr_reference()
    diagnostic = {
        "diagram_pixel_coordinates": [value, 200],
        "status": "abstained",
        "reason": "outside_reference_diagram",
    }
    with pytest.raises(AutonomousStellarError, match="invalid_diagram_coordinates"):
        module._geometry(pack, diagnostic)


@pytest.mark.parametrize("value", [True, 0, -1, math.nan, math.inf, -math.inf, 10**400])
def test_invalid_or_nonrepresentable_calculation_never_becomes_geometry(value):
    calculator = SimpleNamespace(
        execute_unclassified_common=lambda *_: SimpleNamespace(ok=True, value=value, unit="K")
    )
    with pytest.raises(AutonomousStellarError, match="calculation_failed_temperature"):
        module._calculate(calculator, "temperature", {})


@pytest.mark.parametrize(
    "reason", ["positive_finite_coordinates_required", "incompatible_units", "outside_digitized_interiors"]
)
def test_extrapolation_cannot_override_other_diagnostic_errors(reason):
    pack, _ = load_hr_reference()
    with pytest.raises(AutonomousStellarError, match="reference_diagram_status_mismatch"):
        module._geometry(
            pack, {"diagram_pixel_coordinates": [544, 290], "status": "abstained", "reason": reason}
        )


def test_nonfinite_distances_never_rank_as_fallback(monkeypatch):
    pack, _ = load_hr_reference()
    diagnostic = {
        "diagram_pixel_coordinates": [544, 290],
        "status": "abstained",
        "reason": "outside_reference_diagram",
    }
    monkeypatch.setattr(module, "_region", lambda *_: (False, math.inf))
    with pytest.raises(AutonomousStellarError, match="invalid_region_distance"):
        module._geometry(pack, diagnostic)


@pytest.mark.parametrize(
    "old,new",
    [("0.045", "1e-320"), ("370", "1e-320"), ("370", "0"), ("370", "-1"), ("5.68E-10", "1e309")],
)
def test_public_numeric_domain_or_overflow_failure_is_not_extrapolated(case, old, new):
    item = case()
    report = stellar()
    frame = report["frames"][0]
    frame["accessibility"] = frame["accessibility"].replace(old, new)
    checksum = save_capture(item.fresh / "stellar", report)
    with pytest.raises(AutonomousStellarError):
        decide(item, capture_sha256=checksum)


def test_saved_egrithori_capture_optional_read_only_compatibility():
    root = Path(__file__).resolve().parents[1]
    history = root / "experiments/browser-project-autonomous-three-star/270d4ed0e0914060a7e91a68c89d58fa"
    fresh = history / "initial-star"
    if not (fresh / "confirmed.json").is_file():
        pytest.skip("Local public capture is not shipped in source-only checkouts")
    files = [fresh / "confirmed.json", fresh / "stellar/observation.json", fresh / "stellar/manifest.json"]
    before = {path: sha(path) for path in files}
    result = decide_stellar_reference(
        history,
        fresh,
        fresh_star_sha256=before[files[0]],
        capture_sha256=before[files[1]],
        source_root=root,
        expected_star="Egrithori",
    )
    assert result["star"] == "Egrithori"
    assert result["mode"] == "approximate_reference_assisted_stellar_v2"
    assert result["payload"]["selected_class"] == "main_sequence"
    assert result["geometry"]["outside_plot_distance_pixels"] == pytest.approx(2.679926815259)
    assert result["scientific_verified"] is result["write_authorized"] is False
    assert result["task_completed"] is result["project_completed"] is False
    assert result["browser_actions"] == 0
    assert before == {path: sha(path) for path in files}


def test_tied_regions_and_overlapping_interiors_reject():
    left = [[0, 0], [8, 0], [8, 10], [0, 10]]
    right = [[12, 0], [20, 0], [20, 10], [12, 10]]
    pack = {"axes": {"plot_bounds": [0, 0, 20, 20]}, "regions": {"white_dwarf": left, "main_sequence": right}}
    diagnostic = {
        "diagram_pixel_coordinates": [10, 5],
        "status": "abstained",
        "reason": "outside_digitized_interiors",
    }
    with pytest.raises(AutonomousStellarError, match="ambiguous_nearest_regions"):
        module._geometry(pack, diagnostic)
    diagnostic.update(diagram_pixel_coordinates=[10, -1], reason="outside_reference_diagram")
    with pytest.raises(AutonomousStellarError, match="ambiguous_nearest_regions"):
        module._geometry(pack, diagnostic)
    pack["regions"]["main_sequence"] = left
    diagnostic.update(diagram_pixel_coordinates=[5, 5], reason="outside_digitized_interiors")
    with pytest.raises(AutonomousStellarError, match="overlapping_reference_regions"):
        module._geometry(pack, diagnostic)


@pytest.mark.parametrize(
    "years,prefix,quantity",
    [
        (0.5, "ka", "0.0005"),
        (999, "ka", "0.999"),
        (1000, "ka", "1"),
        (999999, "ka", "999.999"),
        (1000000, "Ma", "1"),
        (1000000000, "Ga", "1"),
        (1000000000000, "Ta", "1"),
        (1e16, "Ta", "10000"),
    ],
)
def test_supported_prefix_selection_preserves_exact_conversion(years, prefix, quantity):
    assert module._prefix(years) == (prefix, quantity)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fresh_blank_numeric_answers_verified", 1),
        ("class_selection_verified", 0),
        ("answer_writes", False),
        ("answer_writes", 1),
        ("task_completed", 0),
        ("action_source", "learned"),
        ("star", "Otherstar"),
        ("painted_stellar_class", "unknown"),
    ],
)
def test_bad_or_cross_star_receipt_rejects_even_if_rehashed(case, field, value):
    item = case()
    path = item.fresh / "confirmed.json"
    receipt = json.loads(path.read_bytes())
    receipt[field] = value
    write(path, receipt)
    with pytest.raises(AutonomousStellarError):
        decide(item, fresh_star_sha256=sha(path))


@pytest.mark.parametrize(
    "change", ["populated", "disabled", "wrong_unit", "bad_measurement", "unknown_frame", "conditional"]
)
def test_fresh_capture_is_strictly_mapped_and_not_repaired(case, change):
    item = case()
    path = item.fresh / "stellar"
    report = json.loads((path / "observation.json").read_bytes())
    frame = report["frames"][0]
    if change == "populated":
        frame["controls"][0]["value"] = "1"
    elif change == "disabled":
        frame["controls"][0]["enabled"] = False
    elif change == "wrong_unit":
        frame["accessibility"] = frame["accessibility"].replace("wavelength (nm)", "wavelength (m)")
    elif change == "bad_measurement":
        frame["accessibility"] = frame["accessibility"].replace('parallax (") 0.1', 'parallax (") 0')
    elif change == "unknown_frame":
        report["ignored_frame_urls"] = ["https://unapproved.invalid"]
    else:
        report = stellar(conditional=True)
    checksum = save_capture(path, report)
    with pytest.raises(AutonomousStellarError):
        decide(item, capture_sha256=checksum)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fresh_star_sha256", "a" * 64),
        ("capture_sha256", "a" * 64),
        ("fresh_star_sha256", None),
        ("capture_sha256", True),
        ("expected_star", "Otherstar"),
        ("expected_star", "Bad\nStar"),
    ],
)
def test_wrong_or_invalid_explicit_pins_reject(case, field, value):
    with pytest.raises(AutonomousStellarError):
        decide(case(), **{field: value})


def test_legacy_receipt_only_pin_is_explicitly_disclosed(case):
    result = decide(case(), capture_sha256=None)
    assert result["capture_expected_hash_supplied"] is False
    assert len(result["source_sha256"]) == 3


@pytest.mark.parametrize(
    "marker",
    ["stopped.json", "invalidated.json", "class-selection-reserved.json", "lifetime-prefix-reserved.json"],
)
def test_failed_or_already_reserved_source_cannot_authorize_new_setup(case, marker):
    item = case()
    write(item.fresh / marker, {})
    with pytest.raises(AutonomousStellarError):
        decide(item)


@pytest.mark.parametrize("kind", ["receipt", "manifest", "capture", "source_image"])
def test_symlink_sources_rejected(case, tmp_path, kind):
    item = case()
    path = {
        "receipt": item.fresh / "confirmed.json",
        "manifest": item.fresh / "stellar/manifest.json",
        "capture": item.fresh / "stellar/observation.json",
        "source_image": item.reference,
    }[kind]
    target = tmp_path / (kind + "-target")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(AutonomousStellarError):
        decide(item)


def test_outside_history_and_missing_or_corrupted_image_fail_closed(case, tmp_path):
    item = case()
    with pytest.raises(AutonomousStellarError):
        decide_stellar_reference(
            tmp_path / "missing", item.fresh, fresh_star_sha256=item.receipt_sha, source_root=item.source_root
        )
    other = tmp_path / "other-history"
    other.mkdir()
    with pytest.raises(AutonomousStellarError):
        decide_stellar_reference(
            other, item.fresh, fresh_star_sha256=item.receipt_sha, source_root=item.source_root
        )
    item.reference.write_bytes(b"not the pinned image")
    with pytest.raises(AutonomousStellarError):
        decide(item)


@pytest.mark.parametrize("change", ["capture", "receipt", "manifest", "claim", "image", "failed"])
def test_sources_rechecked_after_calculation_before_return(case, monkeypatch, change):
    item = case()
    original = module._geometry

    def mutate(*args):
        result = original(*args)
        if change == "capture":
            path = item.fresh / "stellar/observation.json"
            path.write_bytes(path.read_bytes() + b" ")
        elif change in {"claim", "failed"}:
            write(item.fresh / ("class-selection-reserved.json" if change == "claim" else "stopped.json"), {})
        elif change == "image":
            item.reference.write_bytes(b"changed")
        else:
            path = item.fresh / ("confirmed.json" if change == "receipt" else "stellar/manifest.json")
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(module, "_geometry", mutate)
    with pytest.raises(AutonomousStellarError):
        decide(item)


def test_duplicate_json_and_manifest_disagreement_rejected(case):
    item = case()
    path = item.fresh / "confirmed.json"
    path.write_bytes(path.read_bytes()[:-1] + b',"answer_writes":0}')
    with pytest.raises(AutonomousStellarError, match="duplicate_source_key"):
        decide(item, fresh_star_sha256=sha(path))
    item = case()
    write(item.fresh / "stellar/manifest.json", {"observation_sha256": "a" * 64})
    with pytest.raises(AutonomousStellarError, match="capture_manifest_hash_mismatch"):
        decide(item)


def test_changed_knowledge_pack_not_accepted(case, monkeypatch):
    item = case()
    original = module.load_knowledge_pack
    calls = 0

    def changed():
        nonlocal calls
        calls += 1
        pack = original()
        if calls > 1:
            pack.description += " mutated"
        return pack

    monkeypatch.setattr(module, "load_knowledge_pack", changed)
    with pytest.raises(AutonomousStellarError, match="knowledge_pack_changed"):
        decide(item)


def test_no_arbitrary_exception_text_leaks(case, monkeypatch):
    def fail():
        raise ValueError("private secret URL and arbitrary source values")

    monkeypatch.setattr(module, "load_knowledge_pack", fail)
    with pytest.raises(AutonomousStellarError) as caught:
        decide(case())
    assert str(caught.value) == "autonomous_stellar_invalid_reference_or_source"


def test_actual_reference_source_hash_optional_read_only_gate(tmp_path):
    """The real pinned student-visible PNG is optional in source-only checkouts."""
    root = Path(__file__).resolve().parents[1]
    pack, _ = load_hr_reference()
    if not (root / pack["source"]["capture"]).is_file():
        pytest.skip("Student-visible local source image is not shipped in this checkout")
    history = tmp_path / "real-reference"
    fresh = history / "fresh"
    checksum = save_capture(fresh / "stellar", public_capture(260, 372))
    write(
        fresh / "confirmed.json",
        {
            "star": "Althinagon",
            "fresh_blank_numeric_answers_verified": True,
            "class_selection_verified": False,
            "answer_writes": 0,
            "action_source": "deterministic_navigation",
            "painted_stellar_class": "main_sequence",
        },
    )
    result = decide_stellar_reference(
        history,
        fresh,
        fresh_star_sha256=sha(fresh / "confirmed.json"),
        capture_sha256=checksum,
        source_root=root,
    )
    assert result["payload"]["selected_class"] == "white_dwarf"
    assert (
        result["provenance"]["hr_source"]["sha256"]
        == "b1850640a9faf6ccdbd55af6bf5243d79faa6d2238a4fc130c6c458f94acb122"
    )
