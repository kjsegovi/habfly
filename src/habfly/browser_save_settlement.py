"""Mapper-neutral, opt-in settling for one fresh, already-reserved Save.

Callbacks retain each caller's native/source/scientific guards. This module
never reserves or dispatches a click and cannot reopen a failed attempt.
"""

import hashlib
import json
import math
import time

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_numeric import screen_identity
from .browser_probe import save_probe

POLICY = "bounded_read_only_pre_dispatch_settle_v1"
OPTION_KEYS = {"settle_reserved_notice", "reserved_notice_policy", "reserved_phase_timeout_seconds"}


def _require(value, reason):
    if not value:
        raise BrowserSafetyStop("reserved_save_" + reason)


def typed_equal(value, expected):
    """JSON equality without bool/int aliases in immutable action records."""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(typed_equal(value[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(typed_equal(a, b) for a, b in zip(value, expected))
    return value == expected


def intent_options(enabled, seconds):
    return (
        {
            "settle_reserved_notice": True,
            "reserved_notice_policy": POLICY,
            "reserved_phase_timeout_seconds": seconds,
        }
        if enabled
        else {}
    )


class ReservedSavePhase:
    def __init__(self, directory, *, seconds, claims, check, quick_check):
        self.directory, self.started = directory, time.monotonic()
        self.deadline = self.started + seconds
        self.claims = {}
        for path, expected in claims.items():
            _require(not path.is_symlink(), "reservation_changed")
            raw = path.read_bytes()
            _require(typed_equal(json.loads(raw), expected), "reservation_changed")
            self.claims[path] = hashlib.sha256(raw).hexdigest()
        self.full_check, self.quick_check = check, quick_check
        self.longest_current_seconds = 0.0
        self.records, self.captures = [], []
        self.root = directory / "reserved-settlement"
        self.root.mkdir()

    def check(self, *, full=False):
        (self.full_check if full else self.quick_check)()
        _require(time.monotonic() < self.deadline, "phase_timeout")
        _require(
            all(
                not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest() == h
                for p, h in self.claims.items()
            ),
            "reservation_changed",
        )
        _require(
            not any(
                (self.directory / n).exists()
                for n in ("stopped.json", "confirmed.json", "dispatch-budget-rejected.json")
            ),
            "attempt_closed",
        )
        _require(time.monotonic() < self.deadline, "phase_timeout")

    def _probe(self, notice, kind, capture):
        self.check(full=True)
        present = notice()
        _require(type(present) is bool and len(self.records) < 601, "invalid_notice_probe")
        self.check(full=True)
        record = {
            "index": len(self.records),
            "kind": kind,
            "capture": capture,
            "notice_present": present,
            "monotonic_seconds": time.monotonic(),
        }
        persist_json(self.root / f"probe-{len(self.records):03d}.json", record)
        self.records.append(record)
        self.check()
        return present

    def settle(self, *, current, button, notice, wait):
        while True:
            self.check(full=True)
            report = self._current(current)
            self.check(full=True)
            capture = f"capture-{len(self.captures):03d}"
            save_probe(report, self.root / capture)
            self.captures.append(capture)
            control = button()
            present = self._probe(notice, "full_guard", capture)
            if not present:
                _require(control.is_enabled(), "control_unavailable")
                files = {}
                for name in self.captures:
                    for leaf in ("manifest.json", "observation.json"):
                        path = self.root / name / leaf
                        files[str(path.relative_to(self.directory))] = hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                for index in range(len(self.records)):
                    path = self.root / f"probe-{index:03d}.json"
                    files[str(path.relative_to(self.directory))] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                persist_json(
                    self.directory / "reserved-notice-settled.json",
                    {
                        "schema_version": 1,
                        "policy": POLICY,
                        "started_monotonic_seconds": self.started,
                        "deadline_monotonic_seconds": self.deadline,
                        "probes": self.records,
                        "captures": self.captures,
                        "source_sha256": files,
                        "compared_capture_sha256": screen_identity(report),
                        "notice_present": False,
                        "reservation_retained": True,
                        "save_click_dispatched": False,
                        "fresh_save_acknowledgement_verified": False,
                        "later_notice_excluded": False,
                        "automatic_retry": False,
                        "task_completed": False,
                    },
                )
                self.check(full=True)
                return report
            while present:
                self.check()
                wait(100)
                self.check()
                button()
                present = self._probe(notice, "wait", capture)

    def _current(self, current):
        started = time.monotonic()
        try:
            return current()
        finally:
            self.longest_current_seconds = max(self.longest_current_seconds, time.monotonic() - started)

    def admit_dispatch(self):
        """Minimum observed-readback headroom, not a completion guarantee.

        The existing three-second click allowance plus the slowest complete
        guarded current callback must fit in the original remaining deadline.
        Acknowledgement/persistence and future latency remain unbounded by this
        estimate; all existing post-click checks still use the same deadline.
        """
        self.check(full=True)
        self.check()
        self._check_dispatch_headroom()

    def _check_dispatch_headroom(self):
        # No callbacks after reading the clock: neither admission nor the
        # final timeout calculation may approve stale pre-callback headroom.
        longest = self.longest_current_seconds
        _require(
            type(longest) in {int, float} and math.isfinite(longest) and longest > 0,
            "invalid_readback_timing",
        )
        remaining = self.deadline - time.monotonic()
        required = 3.0 + longest
        if not remaining > required:
            persist_json(
                self.directory / "dispatch-budget-rejected.json",
                {
                    "schema_version": 1,
                    "policy": "observed_readback_dispatch_admission_v1",
                    "longest_guarded_read_seconds": longest,
                    "remaining_seconds": remaining,
                    "required_seconds": required,
                    "click_allowance_seconds": 3.0,
                    "readback_multiplier": 1,
                    "deadline_monotonic_seconds": self.deadline,
                    "estimate_not_guarantee": True,
                    "deadline_extended": False,
                    "save_click_dispatched": False,
                    "automatic_retry": False,
                    "task_completed": False,
                },
            )
            raise BrowserSafetyStop("reserved_save_insufficient_readback_budget")

    def click_timeout(self):
        self.check(full=True)
        self._check_dispatch_headroom()
        remaining = int((self.deadline - time.monotonic()) * 1000)
        _require(remaining > 0, "phase_timeout")
        return min(3000, remaining)

    def acknowledge(self, *, current, button, notice, wait):
        while True:
            self.check()
            control = button()
            present = notice()
            self.check()
            if present and control.is_enabled():
                persist_json(
                    self.directory / "acknowledgement.json",
                    {
                        "visible_text": "Data saved",
                        "source": "fully_exposed_footer_text",
                        "notice_was_already_present": False,
                    },
                )
                self.check(full=True)
                report = self._current(current)
                self.check(full=True)
                return report
            wait(100)

    def finish(self, after):
        self.check(full=True)
        persist_json(
            self.directory / "reserved-phase-confirmed.json",
            {
                "schema_version": 1,
                "policy": POLICY,
                "started_monotonic_seconds": self.started,
                "deadline_monotonic_seconds": self.deadline,
                "finished_monotonic_seconds": time.monotonic(),
                "settlement_sha256": hashlib.sha256(
                    (self.directory / "reserved-notice-settled.json").read_bytes()
                ).hexdigest(),
                "acknowledgement_sha256": hashlib.sha256(
                    (self.directory / "acknowledgement.json").read_bytes()
                ).hexdigest(),
                "after_sha256": screen_identity(after),
                "shared_deadline_verified": True,
                "maximum_save_dispatches": 1,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        self.check(full=True)


def validate_reserved_phase(book, directory, intent, *, after, same_view, predispatched=None):
    """Adopt every declared diagnostic hash; legacy opt-out bytes stay untouched."""
    _require(
        not (directory / "dispatch-budget-rejected.json").exists()
        and not (directory / "dispatch-budget-rejected.json").is_symlink(),
        "dispatch_budget_rejected",
    )
    present = OPTION_KEYS & set(intent)
    if not present:
        _require(
            not (directory / "reserved-notice-settled.json").exists()
            and not (directory / "reserved-phase-confirmed.json").exists()
            and not (directory / "reserved-settlement").exists(),
            "undeclared_policy",
        )
        return
    seconds = intent.get("reserved_phase_timeout_seconds")
    _require(
        present == OPTION_KEYS
        and intent.get("settle_reserved_notice") is True
        and intent.get("reserved_notice_policy") == POLICY
        and type(seconds) in {int, float}
        and math.isfinite(seconds)
        and 0.1 <= seconds <= 30,
        "invalid_policy",
    )
    settled = book.json(directory / "reserved-notice-settled.json")
    complete = book.json(directory / "reserved-phase-confirmed.json")
    _require(
        typed_equal(
            book.json(directory / "dispatch.json"),
            {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1},
        )
        and typed_equal(
            book.json(directory / "acknowledgement.json"),
            {
                "visible_text": "Data saved",
                "source": "fully_exposed_footer_text",
                "notice_was_already_present": False,
            },
        ),
        "dispatch_acknowledgement_mismatch",
    )
    start, deadline = settled.get("started_monotonic_seconds"), settled.get("deadline_monotonic_seconds")
    end = complete.get("finished_monotonic_seconds")
    _require(
        all(type(v) in {int, float} and math.isfinite(v) for v in (start, deadline, end))
        and start < end < deadline
        and math.isclose(deadline - start, seconds, abs_tol=1e-7),
        "invalid_timing",
    )
    probes, captures = settled.get("probes"), settled.get("captures")
    _require(
        isinstance(probes, list)
        and 1 <= len(probes) <= 601
        and isinstance(captures, list)
        and 1 <= len(captures) <= len(probes),
        "invalid_records",
    )
    expected, last_time, capture_index, previous = {}, start, -1, None
    for index, probe in enumerate(probes):
        _require(
            isinstance(probe, dict)
            and set(probe) == {"index", "kind", "capture", "notice_present", "monotonic_seconds"}
            and type(probe["index"]) is int
            and probe["index"] == index
            and type(probe["notice_present"]) is bool
            and type(probe["monotonic_seconds"]) in {int, float}
            and last_time <= probe["monotonic_seconds"] < end,
            "invalid_probe",
        )
        if probe["kind"] == "full_guard":
            _require(
                previous is None or previous["kind"] == "wait" and previous["notice_present"] is False,
                "guard_sequence",
            )
            capture_index += 1
        else:
            _require(
                probe["kind"] == "wait" and previous is not None and previous["notice_present"] is True,
                "wait_sequence",
            )
        _require(probe["capture"] == f"capture-{capture_index:03d}", "capture_sequence")
        name = f"reserved-settlement/probe-{index:03d}.json"
        _require(book.json(directory / name) == probe, "probe_changed")
        expected[name] = hashlib.sha256(book.read(directory / name)).hexdigest()
        previous, last_time = probe, probe["monotonic_seconds"]
    _require(
        previous["kind"] == "full_guard"
        and previous["notice_present"] is False
        and captures == [f"capture-{i:03d}" for i in range(capture_index + 1)],
        "settlement_not_verified",
    )
    last_capture = None
    for name in captures:
        last_capture = book.capture(directory / "reserved-settlement" / name)
        _require(same_view(last_capture), "scientific_state_changed")
        for leaf in ("manifest.json", "observation.json"):
            path = f"reserved-settlement/{name}/{leaf}"
            expected[path] = hashlib.sha256(book.read(directory / path)).hexdigest()
    _require(
        predispatched is not None and screen_identity(last_capture) == screen_identity(predispatched),
        "final_capture_mismatch",
    )
    _require(
        settled
        == {
            "schema_version": 1,
            "policy": POLICY,
            "started_monotonic_seconds": start,
            "deadline_monotonic_seconds": deadline,
            "probes": probes,
            "captures": captures,
            "source_sha256": expected,
            "compared_capture_sha256": screen_identity(last_capture),
            "notice_present": False,
            "reservation_retained": True,
            "save_click_dispatched": False,
            "fresh_save_acknowledgement_verified": False,
            "later_notice_excluded": False,
            "automatic_retry": False,
            "task_completed": False,
        },
        "settlement_mismatch",
    )
    _require(
        complete
        == {
            "schema_version": 1,
            "policy": POLICY,
            "started_monotonic_seconds": start,
            "deadline_monotonic_seconds": deadline,
            "finished_monotonic_seconds": end,
            "settlement_sha256": hashlib.sha256(
                book.read(directory / "reserved-notice-settled.json")
            ).hexdigest(),
            "acknowledgement_sha256": hashlib.sha256(
                book.read(directory / "acknowledgement.json")
            ).hexdigest(),
            "after_sha256": screen_identity(after),
            "shared_deadline_verified": True,
            "maximum_save_dispatches": 1,
            "automatic_retry": False,
            "task_completed": False,
        },
        "completion_mismatch",
    )
    for value in (settled, complete):
        _require(type(value["schema_version"]) is int, "invalid_schema")
        for key in (
            "notice_present",
            "reservation_retained",
            "save_click_dispatched",
            "fresh_save_acknowledgement_verified",
            "later_notice_excluded",
            "automatic_retry",
            "task_completed",
            "shared_deadline_verified",
        ):
            if key in value:
                _require(type(value[key]) is bool, "invalid_flag")
    _require(type(complete["maximum_save_dispatches"]) is int, "invalid_dispatch_limit")
    root = book.clean(directory / "reserved-settlement")
    tree = book.tree(root)
    _require(
        {name for name, kind in tree if kind == "file"}
        == {name.removeprefix("reserved-settlement/") for name in expected}
        and {name for name, kind in tree if kind == "directory"} == set(captures),
        "diagnostic_tree_changed",
    )
    book.closed_trees[root] = tree
    book.unchanged()
