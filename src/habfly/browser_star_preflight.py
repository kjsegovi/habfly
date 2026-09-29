"""Read-only provenance gate before a classified star's learned native writes.

Explicit fresh-class setup supports the four native stellar choices, including
legacy Main Sequence receipts. Class is supplied, not inferred or scientifically
verified. The saved class paint hash must still be compared with the current visible paint by the caller;
an accessibility capture does not itself record the painted circle's CSS.
"""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_stellar import CLASSES, StellarMappingError, map_stellar_capture

_BASE = {"distance", "luminosity", "temperature"}
_ALL = _BASE | {"mass", "radius", "lifetime"}
_HASH = re.compile(r"[a-f0-9]{64}")


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("star_class_preflight_" + reason)


def _zero(value):
    return type(value) is int and value == 0


def _same_star(value, star):
    return isinstance(value, str) and value.casefold() == star.casefold()


class _Evidence:
    def __init__(self, history):
        self.history = Path(history).resolve(strict=True)
        _require(self.history.is_dir(), "missing_history")
        self.hashes, self.directories = {}, set()

    def path(self, path):
        path = Path(path).absolute()
        _require(path.resolve().is_relative_to(self.history), "evidence_outside_history")
        _require(
            not any(p.is_symlink() for p in (path, *path.parents) if p != self.history),
            "symlink_evidence",
        )
        return path.resolve()

    def clean(self, path):
        directory = self.path(path)
        _require(directory.is_dir() and directory != self.history, "missing_evidence_directory")
        _require(
            not any((directory / name).exists() for name in ("stopped.json", "invalidated.json")),
            "failed_evidence",
        )
        self.directories.add(directory)
        return directory

    def read(self, path):
        path = self.path(path)
        _require(path.is_file() and path.stat().st_size <= 32_000_000, "missing_or_oversized_evidence")
        raw = path.read_bytes()
        name, checksum = str(path.relative_to(self.history)), hashlib.sha256(raw).hexdigest()
        _require(name not in self.hashes or self.hashes[name] == checksum, "evidence_changed")
        self.hashes[name] = checksum
        return raw

    def json(self, path):
        value = json.loads(self.read(path))
        _require(isinstance(value, dict), "invalid_evidence_object")
        return value

    def capture(self, directory):
        directory = self.clean(directory)
        manifest = self.json(directory / "manifest.json")
        raw = self.read(directory / "observation.json")
        checksum = hashlib.sha256(raw).hexdigest()
        _require(manifest.get("observation_sha256") == checksum, "capture_hash_mismatch")
        report = json.loads(raw)
        _require(isinstance(report, dict) and report.get("ignored_frame_urls") == [], "unknown_capture_frame")
        return map_stellar_capture(
            report,
            capture_sha256=checksum,
            allow_color_selection=True,
            allow_main_sequence_fields=True,
        )

    def unchanged(self):
        for directory in list(self.directories):
            self.clean(directory)
        for name in list(self.hashes):
            self.read(self.history / name)


def _blank_mapping(mapping, star, measurements, *, conditional):
    _require(_same_star(mapping.get("star_name"), star), "star_mismatch")
    values = mapping["observation"]["values"]
    fields = values["browser_field_map"]
    _require(values["measurements"] == measurements, "class_measurements_changed")
    _require(
        values["conditional_fields_visible"] is conditional
        and set(fields) == (_ALL if conditional else _BASE)
        and values["color"]["selected"] is None
        and all(v["value_known"] is True and v["current_value"] == "" for v in fields.values())
        and (not conditional or values["lifetime_prefix"]["selected"] is None),
        "class_source_not_fresh_blank",
    )


def _validate(evidence, class_dir, star, selected_class):
    directory = evidence.clean(class_dir)
    classification = evidence.json(directory / "confirmed.json")
    scope = evidence.json(directory / "scope.json")
    clicks = classification.get("class_clicks")
    _require(
        _same_star(classification.get("star"), star)
        and classification.get("selected_class") == selected_class
        and classification.get("readback_verified") is True
        and _zero(classification.get("numeric_writes"))
        and classification.get("learned_classification") is False
        and classification.get("correctness_verified") is False
        and _same_star(scope.get("star"), star)
        and scope.get("intended_class") == selected_class,
        "unsupported_class_receipt",
    )
    # Old Main Sequence receipts predate these explicit disclaimers. New class
    # variants require them; neither a missing flag nor a truthy value is proof.
    _require(
        all(
            record.get(key, False if selected_class == "main_sequence" else None) is False
            for record in (classification, scope)
            for key in ("scientific_verified", "training_label")
        ),
        "unsupported_class_claims",
    )
    _require(
        classification.get("action_source") == "explicit_fresh_star_class_setup"
        and classification.get("task_completed") is False
        and classification.get("intermediate_is_training_label") is False
        and type(clicks) is int
        and clicks in {1, 2}
        and type(scope.get("max_class_clicks")) is int
        and scope["max_class_clicks"] == clicks
        and scope.get("clears_inherited_paint") is (clicks == 2)
        and _zero(scope.get("numeric_writes"))
        and all(
            scope.get(k) is False
            for k in ("learned_classification", "automatic_retry", "intermediate_is_training_label")
        ),
        "unsupported_class_scope",
    )
    fresh_dir = evidence.clean(scope["fresh_star"])
    _require(fresh_dir != directory, "class_and_fresh_evidence_overlap")
    fresh_receipt = evidence.json(fresh_dir / "confirmed.json")
    painted = fresh_receipt.get("painted_stellar_class")
    _require(
        fresh_receipt.get("fresh_blank_numeric_answers_verified") is True
        and fresh_receipt.get("class_selection_verified") is False
        and _zero(fresh_receipt.get("answer_writes"))
        and fresh_receipt.get("action_source") == "deterministic_navigation"
        and _same_star(fresh_receipt.get("star"), star)
        and painted in {None, "main_sequence", "red_giant", "supergiant", "white_dwarf"}
        and (painted == selected_class) == (clicks == 2),
        "unsupported_fresh_star_receipt",
    )
    fresh = evidence.capture(fresh_dir / "stellar")
    source_hash = evidence.hashes[str((fresh_dir / "stellar/observation.json").relative_to(evidence.history))]
    _require(scope.get("source_capture_sha256") == source_hash, "class_source_hash_mismatch")
    measurements = fresh["observation"]["values"]["measurements"]
    for mapping in (fresh, evidence.capture(directory / "before")):
        _blank_mapping(mapping, star, measurements, conditional=False)
    if clicks == 2:
        _blank_mapping(evidence.capture(directory / "intermediate"), star, measurements, conditional=False)
    else:
        _require(not (directory / "intermediate").exists(), "unexpected_intermediate_capture")
    after = evidence.capture(directory / "after")
    _blank_mapping(after, star, measurements, conditional=selected_class == "main_sequence")
    expected_paths = [directory / f"event-{i}.json" for i in range(clicks * 2)]
    _require(sorted(directory.glob("event-*.json")) == expected_paths, "missing_class_readback_event")
    events = [evidence.json(path) for path in expected_paths]
    _require(
        [e.get("event") for e in events] == ["action_proposed", "action_result"] * clicks,
        "missing_class_readback_event",
    )
    intermediate = "red_giant" if selected_class == "white_dwarf" else "white_dwarf"
    first = intermediate if clicks == 2 else selected_class
    reservation = evidence.json(fresh_dir / "class-selection-reserved.json")
    _require(
        _same_star(reservation.get("star"), star)
        and reservation.get("source_capture_sha256") == source_hash
        and reservation.get("previous_unconfirmed_paint") == painted
        and reservation.get("selected_class") == first
        and reservation.get("action_source") == "reference_diagnostic"
        and type(reservation.get("max_clicks")) is int
        and reservation["max_clicks"] == 1
        and reservation.get("automatic_retry") is False,
        "class_reservation_mismatch",
    )
    for index, expected in enumerate([first, selected_class] if clicks == 2 else [selected_class]):
        proposed, result = (events[index * 2 + i]["payload"] for i in (0, 1))
        _require(
            proposed.get("kind") == "SELECT"
            and proposed.get("target") == "stellar_class"
            and proposed.get("value") == expected
            and proposed.get("action_source") == "reference_diagnostic"
            and proposed.get("correctness_verified") is False
            and result.get("selected_class") == expected
            and result.get("readback_verified") is True
            and result.get("selection_source") == "reference_diagnostic"
            and result.get("correctness_verified") is False
            and result.get("task_completed") is False
            and isinstance(result.get("rendering_sha256"), str)
            and _HASH.fullmatch(result["rendering_sha256"]),
            "class_action_event_mismatch",
        )
        if index == 0:
            _require(
                result.get("fresh_star_capture_sha256") == source_hash
                and result.get("previous_unconfirmed_paint") == painted,
                "class_initial_readback_mismatch",
            )
        else:
            _require(
                proposed.get("previous") == result.get("previous") == intermediate
                and isinstance(proposed.get("revision_reason"), str)
                and bool(proposed["revision_reason"].strip())
                and proposed["revision_reason"] == result.get("revision_reason"),
                "class_revision_readback_mismatch",
            )
    evidence.unchanged()
    _require(sorted(directory.glob("event-*.json")) == expected_paths, "class_events_changed")
    return {
        "schema_version": 1,
        "mode": "explicit_class_source_preflight",
        "star": after["star_name"],
        "selected_class": selected_class,
        "measurements": deepcopy(measurements),
        "class_rendering_sha256": events[-1]["payload"]["rendering_sha256"],
        "source_capture_sha256": source_hash,
        "class_dir": str(directory.relative_to(evidence.history)),
        "fresh_star": str(fresh_dir.relative_to(evidence.history)),
        "source_hashes": dict(evidence.hashes),
        "class_clicks": clicks,
        "reference_prediction": True,
        "classification_learned": False,
        "correctness_verified": False,
        "scientific_verified": False,
        "training_label": False,
        "current_class_paint_verified": False,
        "browser_actions": 0,
        "task_completed": False,
    }


def validate_star_class_source(run_history, class_dir, star, selected_class):
    """Read/hash an owned setup chain; reject unsupported or changed evidence.

    Call again and compare source_hashes if the caller pauses before dispatch.
    Nothing is persisted, inferred, selected, repaired, or fetched remotely.
    """
    _require(isinstance(star, str) and bool(star.strip()), "invalid_star")
    _require(isinstance(selected_class, str) and selected_class in CLASSES, "unsupported_class")
    try:
        return _validate(_Evidence(run_history), class_dir, star, selected_class)
    except BrowserSafetyStop:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, StellarMappingError):
        raise BrowserSafetyStop("star_class_preflight_invalid_evidence") from None


def matches_mapping(source, mapping, *, class_rendering_sha256=None):
    """Compare the class-aware session's pinned visible values without mutation.

    Allows existing numeric values for the subsequent color component. Native
    numeric sessions independently require blank write destinations. Pass a
    fresh visible class-paint digest to check it; None makes no paint claim.
    """
    try:
        values = mapping["observation"]["values"]
        selected = source["selected_class"]
        main = selected == "main_sequence"
        return bool(
            source["mode"] == "explicit_class_source_preflight"
            and selected in CLASSES
            and _same_star(mapping["star_name"], source["star"])
            and _same_star(values["star_name"], source["star"])
            and values["measurements"] == source["measurements"]
            and values["selected_browser_class"] == source["selected_class"]
            and values["conditional_fields_visible"] is main
            and set(values["browser_field_map"]) == (_ALL if main else _BASE)
            and (class_rendering_sha256 is None or class_rendering_sha256 == source["class_rendering_sha256"])
        )
    except (TypeError, KeyError, AttributeError):
        return False
