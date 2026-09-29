"""Fresh readback of an explicit No under the user's autosave assumption.

No Save control is bound, clicked, or awaited. The receipt is not persistence
proof, and a historical failed Save directory cannot become a readback receipt.
"""

import hashlib
import time
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_autosave import autosave_flags, validate_autosave_receipt
from .browser_no_planet_save import _blank_no, _load_choice
from .browser_numeric import screen_identity
from .browser_planet_numeric import PlanetNumericSession, planet_projection
from .browser_probe import save_probe


def _require(value, reason):
    if not value:
        raise BrowserSafetyStop("no_planet_autosave_" + reason)


def _choice_sources(book, choice_path, checksum):
    path = book.path(choice_path)
    book.clean(path.parent)
    _require(hashlib.sha256(book.read(path)).hexdigest() == checksum, "choice_hash_changed")
    receipt, capture, mapping = _load_choice(path, checksum, book.history)
    key = hashlib.sha256(receipt["star"].casefold().encode()).hexdigest()
    for item in (
        path.parent / "reserved.json",
        path.parent / "preselect.json",
        book.history / "planet-window-choice-reservations" / f"{key}.json",
    ):
        book.read(item)
    _require(book.capture(path.parent / "after") == capture, "choice_capture_changed")
    for label, source in (
        ("saved_evidence", book.history / receipt["source"]),
        ("fresh_evidence", path.parent / "fresh-progress/report.json"),
        ("preselect_evidence", path.parent / "preselect-progress/report.json"),
    ):
        book.clean(source.parent)
        _require(
            hashlib.sha256(book.read(source)).hexdigest() == receipt[label]["report_sha256"],
            "choice_report_changed",
        )
        _require(
            hashlib.sha256(book.read(source.parent / "chart.png")).hexdigest()
            == receipt[label]["chart_sha256"],
            "choice_chart_changed",
        )
        if "policy_sha256" in receipt[label]:
            _require(
                hashlib.sha256(book.read(source.parent / "policy.json")).hexdigest()
                == receipt[label]["policy_sha256"],
                "choice_policy_changed",
            )
    return receipt, capture, mapping


def _same(report, mapping, choice, original, original_mapping):
    _blank_no(mapping)
    _require(mapping["star_name"].casefold() == choice["star"].casefold(), "star_changed")
    _require(
        planet_projection(report, mapping) == planet_projection(original, original_mapping),
        "visible_answers_changed",
    )


def load_no_planet_readback(book, directory, *, choice_dir, expected_star):
    """Rebuild all source and capture evidence without a browser or old Save."""
    from .browser_no_planet_workflow import _Evidence, _planet

    directory, choice_dir = book.clean(directory), book.clean(choice_dir)
    receipt = book.json(directory / "confirmed.json")
    validate_autosave_receipt(
        receipt,
        branch="no_planet",
        star=expected_star,
        output=str(directory.relative_to(book.history)),
        extra_keys={"choice_path", "choice_sha256"},
    )
    path = choice_dir / "confirmed.json"
    _require(receipt["choice_path"] == str(path.relative_to(book.history)), "choice_path_changed")
    source_book = _Evidence(book.history)
    choice, original, original_mapping = _choice_sources(source_book, path, receipt["choice_sha256"])
    key = hashlib.sha256(choice["star"].casefold().encode()).hexdigest()
    _require(
        not (book.history / "no-planet-save-reservations" / f"{key}.json").exists(),
        "prior_explicit_save_attempt",
    )
    _require(source_book.hashes == receipt["source_sha256"], "source_hashes_changed")
    for name, digest in source_book.hashes.items():
        _require(hashlib.sha256(book.read(book.history / name)).hexdigest() == digest, "source_changed")
    for source in source_book.clean_directories:
        book.clean(source)
    before, after = book.capture(directory / "before"), book.capture(directory / "after")
    initial = book.capture(directory / "read-guard/initial")
    book.read(directory / "read-guard/scope.json")
    for capture in (initial, before, after):
        _same(capture, _planet(capture), choice, original, original_mapping)
    _require(
        receipt["before_sha256"] == screen_identity(before)
        and receipt["after_sha256"] == screen_identity(after),
        "readback_hash_changed",
    )
    expected = {
        "confirmed.json",
        "read-guard/scope.json",
        *(
            f"{part}/{leaf}.json"
            for part in ("before", "after", "read-guard/initial")
            for leaf in ("observation", "manifest")
        ),
    }
    tree = book.tree(directory)
    expected_dirs = {str(p) for name in expected for p in Path(name).parents if p != Path(".")}
    _require(
        {name for name, kind in tree if kind == "file"} == expected
        and {name for name, kind in tree if kind == "directory"} == expected_dirs,
        "unexpected_readback_files",
    )
    book.closed_trees[directory] = tree
    book.unchanged()
    return {"choice": choice, "save_capture": after, "save_mapping": _planet(after)}


def readback_no_planet_work(
    page,
    config,
    output,
    *,
    run_history,
    choice_path,
    choice_sha256,
    timeout_seconds=60,
    cancelled=lambda: False,
):
    from .browser_no_planet_workflow import _Evidence

    if type(timeout_seconds) not in {int, float} or not 1 <= timeout_seconds <= 60 or not callable(cancelled):
        raise ValueError("A bounded read-only autosave check is required")
    book = _Evidence(run_history)
    directory = book.path(output)
    choice, original, original_mapping = _choice_sources(book, Path(choice_path), choice_sha256)
    key = hashlib.sha256(choice["star"].casefold().encode()).hexdigest()
    _require(
        not (book.history / "no-planet-save-reservations" / f"{key}.json").exists(),
        "prior_explicit_save_attempt",
    )
    directory.mkdir(parents=True, exist_ok=False)
    deadline, session = time.monotonic() + timeout_seconds, None

    def guard():
        _require(not cancelled(), "cancelled")
        _require(time.monotonic() < deadline, "time_limit")
        book.unchanged()

    try:
        guard()
        session = PlanetNumericSession(
            page, config, directory / "read-guard", max_seconds=timeout_seconds, _allow_no_planet=True
        )
        captures = []
        for label in ("before", "after"):
            guard()
            report, mapping, choices, _ = session.current()
            _same(report, mapping, choice, original, original_mapping)
            _require(choices["selected"] == choice["painted_class_after"], "class_paint_changed")
            save_probe(report, directory / label)
            captures.append(report)
        guard()
        receipt = {
            "schema_version": 1,
            **autosave_flags(),
            "branch": "no_planet",
            "star": choice["star"],
            "output": str(directory.relative_to(book.history)),
            "choice_path": str(book.path(choice_path).relative_to(book.history)),
            "choice_sha256": choice_sha256,
            "source_sha256": dict(book.hashes),
            "before_sha256": screen_identity(captures[0]),
            "after_sha256": screen_identity(captures[1]),
        }
        validate_autosave_receipt(
            receipt,
            branch="no_planet",
            star=choice["star"],
            output=receipt["output"],
            extra_keys={"choice_path", "choice_sha256"},
        )
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        persist_json(
            directory / "stopped.json",
            {
                "mode": autosave_flags()["mode"],
                "reason": str(exc)
                if isinstance(exc, BrowserSafetyStop)
                else "no_planet_autosave_readback_failed",
                "browser_actions": 0,
                "save_click_delivered": False,
                "persistence_verified": False,
                "task_completed": False,
                "automatic_retry": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("no_planet_autosave_readback_failed") from None
    finally:
        if session is not None:
            session.close()
