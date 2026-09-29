"""Coarse visible H-R reference interpretation, separate from trained policies.

Inputs are explicit luminosity/temperature values in known units. This helper
does not select measurements, evaluate arithmetic, click controls or repair a
classification. It never extrapolates or converts ambiguous regions into a
default Main Sequence answer. Its pack cannot become a grading oracle.
"""

import hashlib
import json
import math
import re
from datetime import date
from importlib.resources import files
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image

CLASSES = {"main_sequence", "white_dwarf", "red_giant", "supergiant"}
LABEL_ALIASES = {
    "MAIN SEQUENCE": "main_sequence",
    "WHITE DWARFS": "white_dwarf",
    "GIANTS": "red_giant",
    "SUPERGIANTS": "supergiant",
}
_DISABLED = (
    "classification_learned",
    "scientific_verified",
    "training_label",
    "runtime_enabled",
    "write_authorized",
)


def _require(condition, reason):
    if not condition:
        raise ValueError("hr_reference_" + reason)


def _keys(value, names):
    _require(isinstance(value, dict) and set(value) == set(names.split()), "invalid_pack_schema")


def _finite(value):
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 2000


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    crosses = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if crosses[0] * crosses[1] < 0 and crosses[2] * crosses[3] < 0:
        return True
    return any(
        turn == 0
        and min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
        and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
        for turn, p, q, r in (
            (crosses[0], a, b, c),
            (crosses[1], a, b, d),
            (crosses[2], c, d, a),
            (crosses[3], c, d, b),
        )
    )


def _polygon(polygon, bounds):
    left, top, right, bottom = bounds
    _require(isinstance(polygon, list) and 3 <= len(polygon) <= 100, "invalid_polygon")
    for point in polygon:
        _require(
            isinstance(point, list)
            and len(point) == 2
            and all(_finite(v) for v in point)
            and left <= point[0] <= right
            and top <= point[1] <= bottom,
            "invalid_polygon_coordinates",
        )
    _require(len({tuple(p) for p in polygon}) == len(polygon), "duplicate_polygon_vertex")
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)
    _require(abs(area) > 1e-9, "degenerate_polygon")
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if j > i + 1 and (i, j) != (0, len(edges) - 1):
                _require(not _intersects(a, b, c, d), "self_intersecting_polygon")


def _validate_pack(pack):
    _keys(
        pack,
        "version id mode description source axes boundary_abstention_pixels digitization regions limitations classification_learned scientific_verified training_label browser_actions diagnostic_only runtime_enabled write_authorized",
    )
    _require(type(pack["version"]) is int and pack["version"] == 1, "unsupported_version")
    _require(
        pack["id"] == "visible-hr-diagram-coarse-reference-v1" and pack["mode"] == "reference_prediction",
        "unsupported_scope",
    )
    _require(
        all(pack[key] is False for key in _DISABLED)
        and pack["diagnostic_only"] is True
        and type(pack["browser_actions"]) is int
        and pack["browser_actions"] == 0,
        "unsupported_authority",
    )
    _require(
        _text(pack["description"])
        and isinstance(pack["limitations"], list)
        and 1 <= len(pack["limitations"]) <= 20
        and all(_text(v) for v in pack["limitations"]),
        "invalid_description",
    )
    source = pack["source"]
    _keys(source, "title capture sha256 capture_date surface simulation_version image_size coordinate_frame")
    _require(
        _text(source["title"])
        and _text(source["simulation_version"])
        and source["surface"] == "student_visible_reference_image"
        and source["coordinate_frame"] == "unscaled_capture_pixels_top_left_origin"
        and isinstance(source["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", source["sha256"]),
        "invalid_source_metadata",
    )
    _require(
        isinstance(source["capture_date"], str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", source["capture_date"]),
        "invalid_source_date",
    )
    try:
        date.fromisoformat(source["capture_date"])
    except ValueError:
        raise ValueError("hr_reference_invalid_source_date") from None
    capture = source["capture"]
    _require(
        isinstance(capture, str)
        and 1 <= len(capture) <= 500
        and "\\" not in capture
        and all(ord(character) >= 32 for character in capture),
        "invalid_source_path",
    )
    path = PurePosixPath(capture)
    _require(
        not path.is_absolute() and ".." not in path.parts and path.suffix == ".png" and str(path) == capture,
        "invalid_source_path",
    )
    size = source["image_size"]
    _require(
        isinstance(size, list) and len(size) == 2 and all(type(v) is int and 1 < v <= 4096 for v in size),
        "invalid_source_dimensions",
    )
    axes = pack["axes"]
    _keys(axes, "temperature luminosity plot_bounds")
    bounds = axes["plot_bounds"]
    _require(
        isinstance(bounds, list) and len(bounds) == 4 and all(_finite(v) for v in bounds),
        "invalid_plot_bounds",
    )
    left, top, right, bottom = bounds
    _require(0 <= left < right < size[0] and 0 <= top < bottom < size[1], "invalid_plot_bounds")
    for name, unit, low, high in (("temperature", "K", left, right), ("luminosity", "Lsun", top, bottom)):
        axis = axes[name]
        _keys(axis, "unit ticks transform")
        _require(axis["unit"] == unit and axis["transform"] == "log10_decreasing", "invalid_axis_metadata")
        ticks = axis["ticks"]
        _require(
            isinstance(ticks, list)
            and len(ticks) == 2
            and all(isinstance(t, list) and len(t) == 2 and all(_finite(v) for v in t) for t in ticks),
            "invalid_axis_ticks",
        )
        (a, pa), (b, pb) = ticks
        _require(a > b > 0 and low <= pa < pb <= high and math.log10(a) > math.log10(b), "invalid_axis_ticks")
    margin = pack["boundary_abstention_pixels"]
    _require(_finite(margin) and 0 <= margin <= min(right - left, bottom - top), "invalid_boundary_margin")
    _keys(pack["regions"], "main_sequence white_dwarf red_giant supergiant")
    for polygon in pack["regions"].values():
        _polygon(polygon, bounds)
    digitization = pack["digitization"]
    _keys(
        digitization,
        "method boundary_authority boundary_margin_calibrated visible_label_aliases alias_note axis_check_source axis_check_tolerance_pixels axis_checks",
    )
    _require(
        digitization["method"] == "manual_visible_band_interiors"
        and digitization["boundary_authority"] == "analyst_chosen_not_course_boundaries"
        and digitization["boundary_margin_calibrated"] is False
        and digitization["visible_label_aliases"] == LABEL_ALIASES
        and _text(digitization["alias_note"])
        and digitization["axis_check_source"] == "independent_manual_reading_of_visible_gridlines",
        "unsupported_digitization_claims",
    )
    tolerance = digitization["axis_check_tolerance_pixels"]
    _require(_finite(tolerance) and 0 < tolerance <= 3, "invalid_axis_check_tolerance")
    _keys(digitization["axis_checks"], "temperature luminosity")
    for name, checks in digitization["axis_checks"].items():
        _require(isinstance(checks, list) and 2 <= len(checks) <= 20, "invalid_axis_checks")
        for check in checks:
            _require(
                isinstance(check, list)
                and len(check) == 2
                and all(_finite(v) for v in check)
                and check[0] > 0,
                "invalid_axis_checks",
            )
            _require(
                check[0] not in [t[0] for t in axes[name]["ticks"]]
                and abs(_coordinate(check[0], axes[name]["ticks"]) - check[1]) <= tolerance,
                "axis_check_mismatch",
            )
        _require(len({c[0] for c in checks}) == len(checks), "duplicate_axis_check")


def _distance(point, first, second):
    x, y = point
    ax, ay = first
    bx, by = second
    length = (bx - ax) ** 2 + (by - ay) ** 2
    along = max(0, min(1, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / length)) if length else 0
    return math.hypot(x - ax - along * (bx - ax), y - ay - along * (by - ay))


def _region(point, polygon):
    inside, distance = False, math.inf
    x, y = point
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        ax, ay = first
        bx, by = second
        distance = min(distance, _distance(point, first, second))
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside, distance


def _coordinate(value, ticks):
    (a, pa), (b, pb) = ticks
    return pa + (math.log10(value) - math.log10(a)) * (pb - pa) / (math.log10(b) - math.log10(a))


def load_hr_reference():
    """Validate pack geometry/metadata; source PNG bytes are a separate check."""
    raw = files("habfly.packs").joinpath("stellar_hr_reference.json").read_bytes()
    _require(len(raw) <= 100_000, "oversized_pack")

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "duplicate_pack_key")
            value[key] = item
        return value

    try:
        pack = json.loads(raw, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("hr_reference_invalid_pack_json") from None
    _validate_pack(pack)
    return pack, hashlib.sha256(raw).hexdigest()


def _verify_source(pack, source_root):
    """Read one explicit local PNG; never search, download, or alter it."""
    source = pack["source"]
    try:
        root = Path(source_root).resolve(strict=True)
        path = root / source["capture"]
        _require(root.is_dir() and path.resolve().is_relative_to(root), "source_outside_root")
        _require(
            not any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root) and p != root),
            "source_symlink",
        )
        _require(path.is_file(), "source_unavailable")
        with path.open("rb") as stream:
            raw = stream.read(5_000_001)
        _require(len(raw) <= 5_000_000, "source_oversized")
    except (OSError, RuntimeError):
        raise ValueError("hr_reference_source_unavailable") from None
    _require(hashlib.sha256(raw).hexdigest() == source["sha256"], "source_hash_mismatch")
    try:
        with Image.open(BytesIO(raw)) as image:
            _require(
                image.format == "PNG" and list(image.size) == source["image_size"],
                "source_dimensions_or_format_mismatch",
            )
            image.verify()
    except (OSError, SyntaxError, Image.DecompressionBombError):
        raise ValueError("hr_reference_invalid_source_image") from None


def validate_hr_reference_source(source_root):
    """Explicit offline source-byte/size validation, not classification accuracy."""
    pack, checksum = load_hr_reference()
    _verify_source(pack, source_root)
    return {
        "mode": "reference_source_validation",
        "pack_sha256": checksum,
        "source": pack["source"],
        "source_image_verified": True,
        "diagnostic_only": True,
        "browser_actions": 0,
        **dict.fromkeys(_DISABLED, False),
    }


def _positive_finite(value):
    return _finite(value) and value > 0


def classify_hr_reference(
    *, luminosity, temperature, luminosity_unit="Lsun", temperature_unit="K", source_root=None
):
    """Return a diagnostic candidate or abstention, never action authority.

    An explicit source_root also requires the recorded PNG's byte/size check.
    Without it, source_image_verified remains false; no file search is implied.
    """
    pack, checksum = load_hr_reference()
    if source_root is not None:
        _verify_source(pack, source_root)
    result = {
        "mode": pack["mode"],
        "status": "abstained",
        "selected_class": None,
        "classification_learned": False,
        "scientific_verified": False,
        "training_label": False,
        "browser_actions": 0,
        "write_authorized": False,
        "runtime_enabled": False,
        "diagnostic_only": True,
        "source_image_verified": source_root is not None,
        "boundary_margin_calibrated": False,
        "visible_label_aliases": pack["digitization"]["visible_label_aliases"],
        "pack_sha256": checksum,
        "source": pack["source"],
        "limitations": pack["limitations"],
    }
    if luminosity_unit != "Lsun" or temperature_unit != "K":
        return {**result, "reason": "incompatible_units"}
    if not all(_positive_finite(value) for value in (luminosity, temperature)):
        return {**result, "reason": "positive_finite_coordinates_required"}
    point = [
        _coordinate(temperature, pack["axes"]["temperature"]["ticks"]),
        _coordinate(luminosity, pack["axes"]["luminosity"]["ticks"]),
    ]
    result["diagram_pixel_coordinates"] = point
    left, top, right, bottom = pack["axes"]["plot_bounds"]
    if not (left <= point[0] <= right and top <= point[1] <= bottom):
        return {**result, "reason": "outside_reference_diagram"}
    regions = {name: _region(point, polygon) for name, polygon in pack["regions"].items()}
    candidates = [name for name, (inside, _) in regions.items() if inside]
    result["candidate_regions"] = candidates
    if len(candidates) != 1:
        return {
            **result,
            "reason": "overlapping_reference_regions" if candidates else "outside_digitized_interiors",
        }
    if any(distance <= pack["boundary_abstention_pixels"] for _, distance in regions.values()):
        return {**result, "reason": "near_approximate_region_boundary"}
    return {
        **result,
        "status": "reference_candidate",
        "selected_class": candidates[0],
        "reason": "unique_digitized_diagram_interior_not_course_boundary",
    }
