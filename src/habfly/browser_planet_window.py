"""Cooperative 5,000-day observation scheduler for an owned project attempt.

Each advance performs one bounded component or read-only poll. It never repeats
Play, pans/zooms, extends a window, writes numerical answers, saves or assesses.
The caller controls scheduling/abort and owns the browser on the same thread.
"""

import hashlib
import json
import math
import time
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_observation_progress import capture_observation_progress, trace_progress
from .browser_planet_observation import start_planet_observation
from .browser_planet_window_choice import select_no_planet_from_window
from .planet_window_dip import analyze_window_dip
from .planet_window_policy import analyze_planet_window, policy_manifest

SCHEDULING_RULE = "complete_rendered_window_before_positive_handoff_v1"


def _partial_palette_wait(analysis, progress):
    """A fresh poll may resolve a partial trace; it can never authorize No.

    The fixed raster policy remains unchanged. Require an established blue
    frontier short of the endpoint, not merely an unknown/empty painted chart.
    All captures and their unresolved analyses remain in the episode evidence.
    """
    day = progress.get("approximate_rendered_day")
    columns = progress.get("trace_columns")
    return (
        analysis.get("status") == "insufficient_visual_evidence"
        and analysis.get("reason") == "unknown_plot_palette"
        and progress.get("method") == "rendered_blue_trace_frontier_v1"
        and type(progress.get("requested_days")) is int
        and progress.get("requested_days") == 5000
        and progress.get("status") == "trace_partial_or_unresolved"
        and progress.get("endpoint_visible") is False
        and progress.get("observation_completed") is False
        and type(columns) is int
        and columns >= 8
        and type(day) in {int, float}
        and math.isfinite(day)
        and 0 < day < 5000
    )


class PlanetWindowSession:
    """New blank-duration stars only; legacy observations use explicit replay."""

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        select_no=False,
        allow_shallow_reference=False,
        allow_baseline_edge_reference=False,
        allow_baseline_band_reference=False,
        allow_single_event_reference=False,
        max_seconds=600,
        poll_interval=5,
        max_polls=60,
        emit=lambda *_: None,
    ):
        if (
            type(select_no) is not bool
            or type(allow_shallow_reference) is not bool
            or type(allow_baseline_edge_reference) is not bool
            or type(allow_baseline_band_reference) is not bool
            or type(allow_single_event_reference) is not bool
            or allow_single_event_reference
            and not allow_baseline_band_reference
            or allow_baseline_band_reference
            and not allow_baseline_edge_reference
            or type(max_seconds) not in {int, float}
            or not 30 <= max_seconds <= 900
            or type(poll_interval) not in {int, float}
            or not 1 <= poll_interval <= 30
            or type(max_polls) is not int
            or not 1 <= max_polls <= 120
        ):
            raise ValueError("Window scheduling needs explicit bounded budgets")
        self.output, self.history = Path(output).resolve(), Path(run_history).resolve()
        if not self.history.is_dir() or not self.output.resolve().is_relative_to(self.history.resolve()):
            raise ValueError("Window output must belong to the existing owned attempt")
        self.output.mkdir(parents=True, exist_ok=False)
        self.page, self.config, self.emit = page, config, emit
        self.select_no = select_no
        self.allow_shallow_reference = allow_shallow_reference
        self._shallow_enabled = allow_shallow_reference
        self.allow_baseline_edge_reference = self._baseline_edge_enabled = allow_baseline_edge_reference
        self.allow_baseline_band_reference = self._baseline_band_enabled = allow_baseline_band_reference
        self.allow_single_event_reference = self._single_event_enabled = allow_single_event_reference
        selected_policy = policy_manifest()
        self._baseline_source = None
        self._baseline_band_source = None
        self._single_event_source = None
        if allow_baseline_edge_reference:
            from . import planet_window_baseline_edge

            selected_policy = planet_window_baseline_edge.policy_manifest()
            self._baseline_source = Path(planet_window_baseline_edge.__file__)
        if allow_baseline_band_reference:
            from . import planet_window_baseline_band

            selected_policy = planet_window_baseline_band.policy_manifest()
            self._baseline_band_source = Path(planet_window_baseline_band.__file__)
        if allow_single_event_reference:
            from . import planet_window_single_event

            selected_policy = planet_window_single_event.policy_manifest()
            self._single_event_source = Path(planet_window_single_event.__file__)
        self._policy_json = json.dumps(selected_policy, sort_keys=True, allow_nan=False)
        self.deadline = time.monotonic() + max_seconds
        self.interval, self.max_polls = poll_interval, max_polls
        self._scheduling_options = (select_no, poll_interval, max_polls, self.deadline)
        self.next_poll, self.polls = 0.0, 0
        self.phase, self.star, self.failure = "ready_to_start", None, None
        self.report = None
        self._advancing = False
        self.progress_path, self.analysis, self.receipt = None, None, None
        self._progress_sha = None
        self.waiting_for_clean_partial_chart = False
        self._endpoint_ready = False
        self._prior_dip_directory = None
        self._prior_dip_pins = ()
        self.scope = {
            "mode": "bounded_planet_window",
            "observation_limit_days": 5000,
            "policy": selected_policy,
            "select_assumed_no": select_no,
            "max_seconds": max_seconds,
            "poll_interval": poll_interval,
            "max_polls": max_polls,
            "reference_prediction": True,
            "learned_perception": False,
            "automatic_restart": False,
            "chart_navigation": False,
            "partial_palette_wait": "fresh read-only polls within original deadline and poll limit",
            "scheduling_rule": SCHEDULING_RULE,
            "optimizer_updates": 0,
            "task_completed": False,
        }
        if allow_shallow_reference:
            self.scope["shallow_reference_enabled"] = True
        if allow_baseline_edge_reference:
            self.scope["baseline_edge_reference_enabled"] = True
            self._baseline_source_sha = hashlib.sha256(self._baseline_source.read_bytes()).hexdigest()
            self.scope["baseline_edge_source_sha256"] = self._baseline_source_sha
        if allow_baseline_band_reference:
            self.scope["baseline_band_reference_enabled"] = True
            self._baseline_band_source_sha = hashlib.sha256(
                self._baseline_band_source.read_bytes()
            ).hexdigest()
            self.scope["baseline_band_source_sha256"] = self._baseline_band_source_sha
        if allow_single_event_reference:
            self.scope["single_event_reference_enabled"] = True
            self._single_event_source_sha = hashlib.sha256(self._single_event_source.read_bytes()).hexdigest()
            self.scope["single_event_source_sha256"] = self._single_event_source_sha
        persist_json(self.output / "scope.json", self.scope)
        self._scope_sha = hashlib.sha256((self.output / "scope.json").read_bytes()).hexdigest()
        self._scheduling_scope = json.dumps(self.scope, sort_keys=True, allow_nan=False)

    def _remember_dip(self, directory):
        """Retain first observed evidence; it never supplies numeric answers."""
        if self._prior_dip_directory is None:
            self._prior_dip_directory = directory
            self._prior_dip_pins = tuple(
                (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in sorted(directory.iterdir())
                if path.is_file()
            )
        self._check_prior_dip()

    def _check_prior_dip(self):
        if self._prior_dip_directory is None:
            return
        directory = self._prior_dip_directory
        try:
            if (
                not directory.resolve().is_relative_to(self.output)
                or any(path.is_symlink() for path in (directory, *directory.parents))
                or {path.name for path in directory.iterdir()} != {name for name, _ in self._prior_dip_pins}
                or not self._prior_dip_pins
                or any(
                    (directory / name).is_symlink()
                    or not (directory / name).is_file()
                    or hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest
                    for name, digest in self._prior_dip_pins
                )
            ):
                raise BrowserSafetyStop("planet_window_prior_dip_source_changed")
        except OSError:
            raise BrowserSafetyStop("planet_window_prior_dip_source_changed") from None

    def _check_policy(self):
        if (
            self.allow_single_event_reference is not self._single_event_enabled
            or self.scope.get("single_event_reference_enabled", False) is not self._single_event_enabled
            or self._single_event_enabled
            and (
                self.allow_baseline_band_reference is not True
                or self.scope.get("baseline_band_reference_enabled") is not True
                or self.scope.get("single_event_source_sha256") != self._single_event_source_sha
                or hashlib.sha256(self._single_event_source.read_bytes()).hexdigest()
                != self._single_event_source_sha
                or json.dumps(self.scope.get("policy"), sort_keys=True, allow_nan=False) != self._policy_json
                or hashlib.sha256((self.output / "scope.json").read_bytes()).hexdigest() != self._scope_sha
            )
        ):
            raise BrowserSafetyStop("planet_window_single_event_permission_changed")
        if (
            self.allow_baseline_band_reference is not self._baseline_band_enabled
            or self.scope.get("baseline_band_reference_enabled", False) is not self._baseline_band_enabled
            or self._baseline_band_enabled
            and (
                not self._baseline_edge_enabled
                or self.scope.get("baseline_band_source_sha256") != self._baseline_band_source_sha
                or hashlib.sha256(self._baseline_band_source.read_bytes()).hexdigest()
                != self._baseline_band_source_sha
                or json.dumps(self.scope.get("policy"), sort_keys=True, allow_nan=False) != self._policy_json
                or hashlib.sha256((self.output / "scope.json").read_bytes()).hexdigest() != self._scope_sha
            )
        ):
            raise BrowserSafetyStop("planet_window_baseline_band_permission_changed")
        if (
            self.allow_baseline_edge_reference is not self._baseline_edge_enabled
            or self.scope.get("baseline_edge_reference_enabled", False) is not self._baseline_edge_enabled
            or json.dumps(self.scope.get("policy"), sort_keys=True, allow_nan=False) != self._policy_json
            or self._baseline_edge_enabled
            and (
                self.scope.get("baseline_edge_source_sha256") != self._baseline_source_sha
                or hashlib.sha256(self._baseline_source.read_bytes()).hexdigest() != self._baseline_source_sha
                or hashlib.sha256((self.output / "scope.json").read_bytes()).hexdigest() != self._scope_sha
            )
        ):
            raise BrowserSafetyStop("planet_window_baseline_edge_permission_changed")

    def _check_scheduling(self):
        if time.monotonic() >= self.deadline:
            raise BrowserSafetyStop("planet_window_time_limit")
        if (
            self.scope.get("scheduling_rule") != SCHEDULING_RULE
            or json.dumps(self.scope, sort_keys=True, allow_nan=False) != self._scheduling_scope
            or any(
                type(current) is not type(original) or current != original
                for current, original in zip(
                    (self.select_no, self.interval, self.max_polls, self.deadline),
                    self._scheduling_options,
                    strict=True,
                )
            )
            or hashlib.sha256((self.output / "scope.json").read_bytes()).hexdigest() != self._scope_sha
        ):
            raise BrowserSafetyStop("planet_window_scheduling_rule_changed")

    @property
    def finished(self):
        return self.phase in {
            "dip_observed",
            "assumed_no_ready",
            "no_selected",
            "stopped",
            "aborted",
            "shallow_reference_required",
        }

    def ready_to_advance(self):
        """Read-only scheduling hint; a deadline is due even before the next poll.

        This does not extend either timer, capture the chart, or repeat Play.
        Finished children remain due so an owner can consume their terminal state.
        """
        return self.phase != "observing" or time.monotonic() >= min(self.next_poll, self.deadline)

    def state(self):
        return {
            **self.scope,
            "phase": self.phase,
            "star": self.star,
            "polls": self.polls,
            "failure_reason": self.failure,
            "window_status": (self.analysis or {}).get("status"),
            "waiting_for_clean_partial_chart": self.phase == "observing"
            and self.waiting_for_clean_partial_chart,
            "prior_dip_observed": self._prior_dip_directory is not None,
            "waiting_for_positive_endpoint": self.phase == "observing"
            and self._prior_dip_directory is not None
            and not self._endpoint_ready,
            "decision_readback_verified": bool(self.receipt and self.receipt.get("readback_verified")),
            **(
                {
                    "analysis": {
                        key: self.analysis[key]
                        for key in (
                            "status",
                            "reason",
                            "approximation",
                            "visible_candidate_events",
                            "possible_planet_ignored",
                            "policy",
                        )
                        if key in self.analysis
                    }
                }
                if self._single_event_enabled and self.analysis is not None
                else {}
            ),
        }

    def _finish(self):
        if self.report is not None:
            return self.report
        report = self.state()
        if self._prior_dip_directory is not None:
            report["first_dip_evidence"] = {
                "capture_directory": str(self._prior_dip_directory.relative_to(self.history)),
                "source_sha256": dict(self._prior_dip_pins),
                "measurement_authority": False,
            }
        if self.progress_path:
            report["progress_path"] = str(self.progress_path.relative_to(self.history))
            # Preserve the captured identity even when a source-change stop
            # means the original file was removed or replaced. Never re-pin it.
            report["progress_sha256"] = self._progress_sha
        if self.receipt:
            report["choice_path"] = str((self.output / "choice/confirmed.json").relative_to(self.history))
        persist_json(self.output / "report.json", report)
        self.report = report
        self.emit("episode_summary", {**report, "completed": False, "component_only": True})
        return report

    def abort(self):
        """Stop scheduling, not the site's simulation; no compensating action."""
        if not self.finished:
            self.phase, self.failure = "aborted", "operator_aborted"
            self._finish()
        return self.state()

    def advance(self):
        if self.finished:
            return self.state()
        if self._advancing:
            raise BrowserSafetyStop("planet_window_reentrant_advance")
        self._advancing = True
        try:
            return self._advance()
        finally:
            self._advancing = False

    def _advance(self):
        if time.monotonic() >= self.deadline:
            self.phase, self.failure = "stopped", "planet_window_time_limit"
            return self._finish()
        try:
            self._check_policy()
            self._check_prior_dip()
            self._check_scheduling()
            if (
                self.allow_shallow_reference is not self._shallow_enabled
                or self.scope.get("shallow_reference_enabled", False) is not self._shallow_enabled
            ):
                raise BrowserSafetyStop("planet_window_shallow_permission_changed")
            if self.phase == "ready_to_start":
                self.emit("state", {"stage": "planet_observation_setup", **self.state()})
                if self.finished:
                    return self.state()
                self._check_policy()
                self._check_scheduling()
                receipt = start_planet_observation(
                    self.page, self.config, self.output / "start", run_history=self.history, days=5000
                )
                self.star = receipt["star"]
                self.phase, self.next_poll = "observing", time.monotonic() + self.interval
                self.emit("action_result", {"observation_request": receipt, "task_completed": False})
            elif self.phase == "observing":
                if time.monotonic() < self.next_poll:
                    return self.state()
                if self.polls >= self.max_polls:
                    raise BrowserSafetyStop("planet_window_poll_limit")
                directory = self.output / f"progress-{self.polls:03d}"
                report = capture_observation_progress(self.page, self.config, directory, requested_days=5000)
                self.polls += 1
                if report["star"].casefold() != self.star.casefold():
                    raise BrowserSafetyStop("planet_window_star_changed")
                self.progress_path = directory / "report.json"
                self._progress_sha = hashlib.sha256(self.progress_path.read_bytes()).hexdigest()
                png = (directory / "chart.png").read_bytes()
                readiness = trace_progress(png, report["time_axis_labels"], requested_days=5000)
                if report.get("chart_sha256") != hashlib.sha256(png).hexdigest() or any(
                    type(report.get(key)) is not type(value) or report.get(key) != value
                    for key, value in readiness.items()
                ):
                    raise BrowserSafetyStop("planet_window_readiness_changed")
                self._endpoint_ready = readiness["endpoint_visible"] is True
                analyzer, policy_options = analyze_planet_window, {}
                if self._baseline_edge_enabled:
                    if self._single_event_enabled:
                        from .planet_window_single_event import analyze_recorded_planet_window
                    elif self._baseline_band_enabled:
                        from .planet_window_baseline_band import analyze_recorded_planet_window
                    else:
                        from .planet_window_baseline_edge import analyze_recorded_planet_window

                    persist_json(directory / "policy.json", self.scope["policy"])
                    analyzer, policy_options = (
                        analyze_recorded_planet_window,
                        {"policy": self.scope["policy"]},
                    )
                self.analysis = analyzer(
                    png,
                    report["time_axis_labels"],
                    report["flux_axis_labels"],
                    **policy_options,
                )
                # Keep the pinned negative policy unchanged. The supplemental
                # detector can only hand off visible dips, never authorize No.
                if self.analysis["status"] == "insufficient_visual_evidence":
                    positive = analyze_window_dip(
                        png,
                        report["time_axis_labels"],
                        report["flux_axis_labels"],
                    )
                    persist_json(directory / "positive-dip-analysis.json", positive)
                    if positive["status"] == "dip_observed":
                        self.analysis = {
                            **positive,
                            "negative_policy_status": self.analysis["status"],
                            "negative_policy_reason": self.analysis["reason"],
                            "negative_policy": self.analysis["policy"],
                        }
                persist_json(directory / "analysis.json", self.analysis)
                if self.analysis["status"] == "dip_observed":
                    self._remember_dip(directory)
                self.waiting_for_clean_partial_chart = _partial_palette_wait(self.analysis, report)
                self._check_scheduling()
                self.emit(
                    "observation",
                    {"chart": self.analysis, "progress": {"star": self.star, "task_completed": False}},
                )
                if self.finished:
                    return self.state()
                self._check_prior_dip()
                self._check_scheduling()
                status = self.analysis["status"]
                if self._prior_dip_directory is not None and not self._endpoint_ready:
                    # A dip is not yet a complete measurement window. Keep the
                    # original analyses and use only fresh, bounded read polls.
                    # A later partial clean frame cannot erase the earlier dip.
                    pass
                elif self._prior_dip_directory is not None and status != "dip_observed":
                    raise BrowserSafetyStop("planet_window_prior_dip_conflict")
                elif status == "dip_observed":
                    self.phase = "dip_observed"  # Further positive measurement work is separate.
                elif status == "assume_no_planet":
                    self.phase = "ready_to_select_no" if self.select_no else "assumed_no_ready"
                elif (
                    status == "insufficient_visual_evidence"
                    and self.analysis["reason"] != "missing_trace_start_or_insufficient_trace"
                    and not self.waiting_for_clean_partial_chart
                ):
                    if self.allow_shallow_reference and report.get("endpoint_visible") is True:
                        # Explicit handoff only. Frozen analyses remain intact,
                        # and no No/Yes answer or positive evidence is supplied.
                        self.phase = "shallow_reference_required"
                    else:
                        raise BrowserSafetyStop("planet_window_visual_evidence_unresolved")
                self.next_poll = time.monotonic() + self.interval
            elif self.phase == "ready_to_select_no":
                if self._prior_dip_directory is not None:
                    raise BrowserSafetyStop("planet_window_prior_dip_conflict")
                from .browser_classification import read_planet_class_choices
                from .browser_stellar import SIMULATION_URL

                frames = [frame for frame in self.page.frames if frame.url == SIMULATION_URL]
                if len(frames) != 1:
                    raise BrowserSafetyStop("planet_window_frame_changed")
                choices, _ = read_planet_class_choices(frames[0])
                self.emit(
                    "action_proposed",
                    {
                        "kind": "SELECT",
                        "target": "has_planet",
                        "value": "No",
                        "action_source": "user_approved_window_reference_not_learned",
                        "calibrated": False,
                    },
                )
                if self.finished:
                    return self.state()
                self._check_policy()
                self._check_scheduling()
                self._check_prior_dip()
                if self._prior_dip_directory is not None:
                    raise BrowserSafetyStop("planet_window_prior_dip_conflict")
                self.receipt = select_no_planet_from_window(
                    self.page,
                    self.config,
                    self.output / "choice",
                    run_history=self.history,
                    evidence_path=self.progress_path,
                    evidence_sha256=hashlib.sha256(self.progress_path.read_bytes()).hexdigest(),
                    preserve_painted_class=choices["selected"],
                    **({"policy": self.scope["policy"]} if self._baseline_edge_enabled else {}),
                )
                self.phase = "no_selected"
                self.emit("action_result", {"presence_choice": self.receipt, "task_completed": False})
            else:
                raise BrowserSafetyStop("planet_window_unknown_phase")
        except Exception as exc:  # noqa: BLE001 - never leak driver URLs or credentials
            if self.finished:
                return self.state()
            self.failure = (
                str(exc) if isinstance(exc, BrowserSafetyStop) else "planet_window_component_failed"
            )
            self.phase = "stopped"
            self.emit("error", {"type": "PlanetWindowStop", "message": self.failure})
        if self.finished:
            return self._finish()
        self.emit("state", {"stage": "planet_window", **self.state()})
        return self.state()
