"""Explicit one-use continuation of a proven pre-Play observation timeout.

Never called automatically by the normal scheduler. The duration is not written
again; the original star reservation and failure remain immutable. Historical
classification paint was not captured, so only fresh paint preservation is
verified. A successful click is not proof of animation, completion or a planet.
"""

import hashlib
import time

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_no_planet_save import _sync_directory
from .browser_no_planet_workflow import _Evidence
from .browser_numeric import screen_identity
from .browser_planet import map_planet_capture
from .browser_planet_numeric import PlanetNumericSession
from .browser_planet_observation import MODE as START_MODE
from .browser_planet_observation import _duration, _same_handle, _transport, observation_projection
from .browser_probe import save_probe

MODE = "explicit_predispatch_observation_continuation"
CONTINUATION_SECONDS = 60
RAW_UNITS = {"line_shift": "nm", "brightness_drop": "%", "period_days": "day"}


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("observation_continuation_" + reason)


def _mapping(report):
    return map_planet_capture(report, capture_sha256=screen_identity(report))


def _fields(mapping, duration):
    values = mapping["observation"]["values"]
    fields = values["browser_field_map"]
    _require(
        values["has_planet"] is None and set(fields) == {*RAW_UNITS, "observation_days"},
        "unset_presence_and_raw_fields_required",
    )
    _require(
        fields["observation_days"]["current_value"] == duration
        and fields["observation_days"]["unit"] == "day",
        "duration_mismatch",
    )
    _require(
        all(
            fields[name]["current_value"] == "" and fields[name]["unit"] == unit
            for name, unit in RAW_UNITS.items()
        ),
        "raw_answers_changed",
    )


def validate_observation_continuation_source(failed_output, run_history):
    """Validate the exact known predispatch timeout, without browser access.

    This intentionally allows a later continuation child for offline receipt
    verification, but never an original dispatch/after/confirmation artifact.
    The caller must separately enforce the exclusive continuation claim.
    """
    book = _Evidence(run_history)
    source = book.path(failed_output)
    _require(source.is_dir(), "missing_source")
    stopped = book.json(source / "stopped.json")
    _require(
        stopped
        == {
            "mode": START_MODE,
            "reason": "planet_copy_time_limit",
            "duration_write_may_have_occurred": True,
            "play_click_may_have_occurred": False,
            "automatic_retry": False,
            "observation_completed": False,
            "planet_presence": None,
            "task_completed": False,
        }
        and stopped["duration_write_may_have_occurred"] is True
        and stopped["play_click_may_have_occurred"] is False
        and all(
            stopped[name] is False for name in ("automatic_retry", "observation_completed", "task_completed")
        ),
        "source_not_proven_predispatch_timeout",
    )
    _require(
        not any(
            (source / name).exists()
            for name in ("dispatched.json", "dispatch.json", "confirmed.json", "after", "invalidated.json")
        ),
        "original_dispatch_or_disposition_present",
    )
    intent = book.json(source / "reserved.json")
    star = intent.get("star")
    _require(isinstance(star, str) and star.strip() == star and bool(star), "invalid_star")
    expected = {
        "schema_version": 1,
        "mode": START_MODE,
        "star": star,
        "days": 5000,
        "output": str(source.relative_to(book.history)),
        "max_duration_writes": 1,
        "max_play_clicks": 1,
        "answer_writes": 0,
        "automatic_retry": False,
        "observation_completed": False,
        "planet_presence": None,
        "task_completed": False,
    }
    _require(
        intent == expected
        and all(
            type(intent[name]) is int
            for name in ("schema_version", "days", "max_duration_writes", "max_play_clicks", "answer_writes")
        )
        and all(
            intent[name] is False for name in ("automatic_retry", "observation_completed", "task_completed")
        ),
        "invalid_original_intent",
    )
    claim = (
        book.history
        / "observation-start-reservations"
        / (hashlib.sha256(star.casefold().encode()).hexdigest() + ".json")
    )
    _require(book.json(claim) == intent, "original_reservation_mismatch")
    play = book.json(source / "play-reserved.json")
    _require(
        play == {**intent, "duration_committed": True} and play["duration_committed"] is True,
        "unconfirmed_duration_commit",
    )
    scope = book.json(source / "read-guard/scope.json")
    _require(
        scope
        == {
            "scope": "planet_exact_copy_transport",
            "max_writes": 7,
            "max_seconds": scope.get("max_seconds"),
            "star": star,
            "task_completed": False,
            "scientific_choices": "caller_supplied",
            "retries": False,
        }
        and type(scope["max_seconds"]) in {int, float}
        and 5 <= scope["max_seconds"] <= 120
        and scope["task_completed"] is False
        and scope["retries"] is False,
        "invalid_original_scope",
    )
    captures = {name: book.capture(source / name) for name in ("read-guard/initial", "before", "committed")}
    mapped = {name: _mapping(report) for name, report in captures.items()}
    baseline = observation_projection(captures["before"], mapped["before"])
    for name, mapping in mapped.items():
        _require(mapping["star_name"] == star, "source_star_mismatch")
        _fields(mapping, "5000" if name == "committed" else "")
        _require(observation_projection(captures[name], mapping) == baseline, "source_public_context_changed")
    # Unrelated stars are not part of this receipt's provenance. Recheck the
    # legacy inventory for conflicts each time, without making later collection
    # of another star change this star's immutable source hash set.
    legacy_book = _Evidence(book.history)
    for legacy in book.history.glob("reference-observation-*/reserved.json"):
        record = legacy_book.json(legacy)
        _require(isinstance(record.get("star"), str) and record["star"].strip(), "unreadable_legacy_history")
        _require(record["star"].casefold() != star.casefold(), "competing_legacy_observation")
    legacy_book.unchanged()
    book.unchanged()
    return book, {
        "source": source,
        "intent": intent,
        "capture": captures["committed"],
        "mapping": mapped["committed"],
        "star": star,
    }


def continue_planet_observation(page, config, failed_output, *, run_history, cancelled=lambda: False):
    """One explicitly requested remaining Play action; fixed fresh 60s budget.

    Output is fixed under the failed source, preventing a new output name from
    authorizing another continuation. Cancellation consumes any created child;
    no cleanup, duration rewrite, reset, retry, or budget extension is offered.
    """
    if not callable(cancelled):
        raise TypeError("A cancellation callback is required")
    deadline = time.monotonic() + CONTINUATION_SECONDS
    book, source = validate_observation_continuation_source(failed_output, run_history)
    directory = source["source"] / "play-continuation"
    claim = source["source"] / "continuation-reserved.json"
    book.path(directory)
    book.path(claim)
    _require(not directory.exists() and not claim.exists(), "already_consumed")
    directory.mkdir(exist_ok=False)
    guard, last_report, reserved, attempted = None, None, False, False
    popups = []

    def popup(_):
        popups.append(True)

    page.on("popup", popup)

    def check():
        try:
            value = cancelled()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:  # noqa: BLE001 - caller errors may contain private text
            raise BrowserSafetyStop("observation_continuation_cancellation_failed") from None
        _require(type(value) is bool, "cancellation_failed")
        _require(not value, "cancelled")
        _require(not popups, "unexpected_popup")
        _require(time.monotonic() < deadline, "time_limit")
        book.unchanged()
        current_sources, _ = validate_observation_continuation_source(source["source"], book.history)
        _require(current_sources.hashes == book.hashes, "source_history_changed")
        # New original disposition files also invalidate the source; they are
        # not byte changes to any previously hashed successful artifact.
        _require(
            not any(
                (source["source"] / name).exists()
                for name in (
                    "dispatched.json",
                    "dispatch.json",
                    "confirmed.json",
                    "after",
                    "invalidated.json",
                )
            ),
            "original_disposition_changed",
        )

    try:
        check()
        guard = PlanetNumericSession(
            page, config, directory / "read-guard", max_seconds=CONTINUATION_SECONDS, _allow_unset_planet=True
        )
        guard.deadline = min(guard.deadline, deadline)
        initial, mapping, paint, answers = guard.report, guard.mapping, guard.choices, guard.handles
        last_report = initial
        _fields(mapping, "5000")
        _require(
            mapping["star_name"] == source["star"]
            and observation_projection(initial, mapping)
            == observation_projection(source["capture"], source["mapping"]),
            "source_current_public_state_mismatch",
        )
        duration = _duration(guard.frame, mapping)
        play = _transport(guard.frame, idle=True)
        _require(not duration.evaluate("e=>e===document.activeElement"), "duration_still_focused")
        check()

        def current(*, after_click=False):
            nonlocal last_report
            check()
            report, newer, current_paint, handles = guard.read()
            last_report = report
            _fields(newer, "5000")
            current_duration = _duration(guard.frame, newer)
            transport = _transport(guard.frame, idle=not after_click)
            _require(
                newer["star_name"] == source["star"]
                and current_paint == paint
                and _same_handle(duration, current_duration)
                and not current_duration.evaluate("e=>e===document.activeElement")
                and set(handles) == set(answers)
                and all(_same_handle(handle, handles[name]) for name, handle in answers.items())
                and observation_projection(initial, mapping, allow_transport_change=after_click)
                == observation_projection(report, newer, allow_transport_change=after_click),
                "current_fields_or_context_changed",
            )
            if not after_click:
                _require(_same_handle(play, transport), "play_replaced")
            check()
            return report

        # The constructor's full public capture was just validated against the
        # immutable source and bound to fresh paint/duration/answer/Play handles.
        # Keep it as the baseline; the post-claim full guard still checks all of
        # those facts immediately before dispatch. A second pre-claim full read
        # would consume the fixed budget without another action to validate.
        before = initial
        save_probe(before, directory / "before")
        intent = {
            "schema_version": 1,
            "mode": MODE,
            "star": source["star"],
            "days": 5000,
            "source_output": str(source["source"].relative_to(book.history)),
            "output": str(directory.relative_to(book.history)),
            "source_sha256": dict(book.hashes),
            "before_sha256": screen_identity(before),
            "fixed_budget_seconds": CONTINUATION_SECONDS,
            "original_reservation_retained": True,
            "source_play_click_dispatched": False,
            "source_duration_committed": True,
            "max_duration_writes": 0,
            "max_play_clicks": 1,
            "max_total_play_clicks": 1,
            "answer_writes": 0,
            "chart_navigation": False,
            "original_class_paint_available": False,
            "historical_class_paint_equivalence_verified": False,
            "current_class_paint": paint,
            "scientific_verified": False,
            "automatic_retry": False,
            "observation_completed": False,
            "planet_presence": None,
            "task_completed": False,
        }
        check()
        persist_json(claim, intent)
        reserved = True
        _sync_directory(source["source"])
        persist_json(directory / "reserved.json", intent)
        before = current()
        save_probe(before, directory / "pre-dispatch-observation")
        check()
        persist_json(directory / "dispatch.json", {"kind": "CLICK", "visible_label": "Play", "max_clicks": 1})
        _sync_directory(directory)
        check()
        attempted = True
        play.click(timeout=3000)
        persist_json(directory / "dispatched.json", {"play_click_returned": True})
        after = current(after_click=True)
        save_probe(after, directory / "after")
        check()
        receipt = {
            **intent,
            "after_sha256": screen_identity(after),
            "duration_readback": "5000",
            "duration_committed": True,
            "play_click_dispatched_once": True,
            "readback_verified": True,
            "answers_unchanged": True,
            "current_class_paint_unchanged": True,
            "correctness_verified": False,
            "observation_started_verified": False,
        }
        persist_json(directory / "confirmed.json", receipt)
        return receipt
    except BaseException as exc:
        if last_report is not None:
            save_probe(last_report, directory / "last-verified-observation")
        persist_json(
            directory / "stopped.json",
            {
                "mode": MODE,
                "reason": str(exc)
                if isinstance(exc, BrowserSafetyStop)
                else "observation_continuation_failed",
                "continuation_reserved": reserved,
                "duration_write_may_have_occurred": False,
                "play_click_may_have_occurred": attempted,
                "automatic_retry": False,
                "observation_completed": False,
                "planet_presence": None,
                "task_completed": False,
            },
        )
        if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
            raise
        raise BrowserSafetyStop("observation_continuation_failed") from None
    finally:
        page.remove_listener("popup", popup)
        if guard is not None:
            guard.close()
