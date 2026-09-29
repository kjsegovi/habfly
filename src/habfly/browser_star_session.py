"""Cooperative continuation of one explicitly classified, fresh stellar detail.

This is not a scientific classifier or a full-project completion claim. Frozen
stellar policies run first, then the bounded visible 5,000-day reference policy.
A dip optionally continues through bounded reference measurements and a frozen
planet policy, stopping before classification or Save. Non-main No verification
uses the same applicable-field Save/readback verifier as Main. The caller owns
browser lifetime and scheduling.
"""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_no_planet_save import MODE as NO_PLANET_SAVE_MODE
from .browser_no_planet_save import save_no_planet_work
from .browser_no_planet_workflow import verify_no_planet_workflow
from .browser_planet_spectrum import GEOMETRIC_PAIR_MODE, capture_spectrum_excursion
from .browser_planet_window import PlanetWindowSession
from .browser_positive_steps import PositivePlanetSteps
from .browser_project_navigation import navigate_project
from .browser_raster_planet_evidence import _spectrum
from .browser_star_preflight import matches_mapping, validate_star_class_source
from .browser_stellar import CLASSES, SIMULATION_URL
from .browser_stellar_steps import FullStellarColorSteps, FullStellarNumericSteps
from .data import load_graph
from .lifetime_prefix import PREFIX_YEARS
from .project_events import ProjectEventRelay


class BrowserStarSession:
    """One bounded step per advance; no login, classification or collection."""

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        star,
        class_dir,
        selected_class,
        lifetime_prefix,
        dataset,
        checkpoint,
        color_experiment,
        graph_path,
        seed=8500000,
        planet_pilot=None,
        planet_final_evaluation=None,
        planet_supplied_evaluation=None,
        planet_seed=15000000,
        planet_preserve_painted_class=None,
        no_planet_save_settle_seconds=20,
        shallow_reference=False,
        two_event_reference=False,
        allow_baseline_edge_reference=False,
        allow_baseline_band_reference=False,
        allow_single_event_reference=False,
        save_strategy="explicit",
        emit=lambda *_: None,
    ):
        self.history, self.output = Path(run_history).resolve(), Path(output).resolve()
        self.class_dir = Path(class_dir).resolve()
        if (
            not self.history.is_dir()
            or not self.output.is_relative_to(self.history)
            or self.output == self.history
            or not self.class_dir.is_relative_to(self.history)
            or not (self.class_dir / "confirmed.json").is_file()
            or not isinstance(star, str)
            or not star.strip()
            or selected_class not in CLASSES
            or (selected_class == "main_sequence" and lifetime_prefix not in PREFIX_YEARS)
            or (selected_class != "main_sequence" and lifetime_prefix is not None)
            or type(seed) is not int
            or type(planet_seed) is not int
            or type(shallow_reference) is not bool
            or type(two_event_reference) is not bool
            or (two_event_reference and not shallow_reference)
            or type(allow_baseline_edge_reference) is not bool
            or type(allow_baseline_band_reference) is not bool
            or type(allow_single_event_reference) is not bool
            or (allow_single_event_reference and not allow_baseline_band_reference)
            or type(save_strategy) is not str
            or save_strategy not in {"explicit", "autosave"}
            or (allow_baseline_band_reference and not allow_baseline_edge_reference)
            or (shallow_reference and planet_pilot is None)
            or (planet_pilot is None) != (planet_final_evaluation is None)
            or (
                planet_supplied_evaluation is not None
                and (
                    planet_pilot is None
                    or not isinstance(planet_supplied_evaluation, (str, Path))
                    or not Path(planet_supplied_evaluation).is_dir()
                )
            )
            or planet_preserve_painted_class not in {None, "gas_giant", "ice_giant", "terrestrial"}
            or type(no_planet_save_settle_seconds) not in {int, float}
            or not 0.1 <= no_planet_save_settle_seconds <= 30
            or not callable(emit)
        ):
            raise ValueError("An owned, explicitly classified fresh star and compatible options are required")
        receipt = json.loads((self.class_dir / "confirmed.json").read_bytes())
        if receipt.get("star", "").casefold() != star.casefold():
            raise ValueError("Class receipt belongs to a different star")
        self.class_source = validate_star_class_source(self.history, self.class_dir, star, selected_class)
        self.page, self.config, self._callback = page, config.model_copy(deep=True), emit
        self._advancing = self._closing = self.event_forwarding_failed = False
        self._relay = ProjectEventRelay(self._relayed)
        self.star, self.component = star, None
        self.phase, self.failure, self.report = "stellar_numeric", None, None
        self.model_options = {
            "selected_class": selected_class,
            "lifetime_prefix": lifetime_prefix,
            "graph_path": Path(graph_path),
        }
        self.numeric_options = {"dataset": Path(dataset), "checkpoint": Path(checkpoint), "seed": seed}
        self.color_experiment = Path(color_experiment)
        self.no_planet_save_settle_seconds = no_planet_save_settle_seconds
        self.save_strategy = self._save_strategy = save_strategy
        self._autosave_sources = (
            {
                Path(__file__).with_name(name): hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in ("browser_autosave.py", "browser_no_planet_readback.py")
            }
            if save_strategy == "autosave"
            else {}
        )
        self.shallow_reference = shallow_reference
        self._shallow_enabled = shallow_reference
        self.two_event_reference = self._two_event_enabled = two_event_reference
        self.allow_baseline_edge_reference = self._baseline_edge_enabled = allow_baseline_edge_reference
        self._baseline_edge_source = Path(__file__).with_name("planet_window_baseline_edge.py")
        self._baseline_edge_sha = (
            hashlib.sha256(self._baseline_edge_source.read_bytes()).hexdigest()
            if self._baseline_edge_enabled
            else None
        )
        self.allow_baseline_band_reference = self._baseline_band_enabled = allow_baseline_band_reference
        self._baseline_band_source = Path(__file__).with_name("planet_window_baseline_band.py")
        self._baseline_band_sha = (
            hashlib.sha256(self._baseline_band_source.read_bytes()).hexdigest()
            if self._baseline_band_enabled
            else None
        )
        self.allow_single_event_reference = self._single_event_enabled = allow_single_event_reference
        self._single_event_source = Path(__file__).with_name("planet_window_single_event.py")
        self._single_event_sha = (
            hashlib.sha256(self._single_event_source.read_bytes()).hexdigest()
            if self._single_event_enabled
            else None
        )
        self.tooltip_reference = self._original_window_evidence = None
        self._shallow_hashes = {}
        self._shallow_adoption = self._shallow_tree = None
        self.planet_options = (
            None
            if planet_pilot is None
            else {
                "pilot": Path(planet_pilot),
                "final_evaluation": Path(planet_final_evaluation),
                "seed": planet_seed,
                "preserve_painted_class": planet_preserve_painted_class,
            }
        )
        self._supplied_evaluation = (
            Path(planet_supplied_evaluation).resolve() if planet_supplied_evaluation is not None else None
        )
        self._supplied_option = (
            str(self._supplied_evaluation) if self._supplied_evaluation is not None else None
        )
        self._supplied_report_hash = (
            hashlib.sha256((self._supplied_evaluation / "report.json").read_bytes()).hexdigest()
            if self._supplied_evaluation is not None
            else None
        )
        self.window_evidence = self.spectrum_evidence = None
        self._save_outcome = None
        self.artifact_paths = {"class_dir": str(self.class_dir.relative_to(self.history))}
        self.output.mkdir(parents=True, exist_ok=False)
        self.scope = {
            "mode": "single_star_reference_window_continuation",
            "star": star,
            "selected_class": selected_class,
            "classification_learned": False,
            "observation_limit_days": 5000,
            "planet_presence_learned": False,
            "positive_planet_enabled": self.planet_options is not None,
            "scientific_verified": False,
            "planet_classification_enabled": False,
            "non_main_no_verification_enabled": True,
            "optimizer_updates": 0,
            "automatic_save_retry": False,
            "no_planet_save_settle_seconds": no_planet_save_settle_seconds,
            "no_planet_save_pre_dispatch_policy": "bounded_read_only_pre_dispatch_settle_v1",
            "automatic_deadline_increase": False,
            "assessment_enabled": False,
            "submission_enabled": False,
            "project_completed": False,
        }
        if shallow_reference:
            self.scope["shallow_reference_enabled"] = True
        if two_event_reference:
            self.scope["two_event_reference_enabled"] = True
        if allow_baseline_edge_reference:
            self.scope["baseline_edge_reference_enabled"] = True
            self.scope["baseline_edge_source_sha256"] = self._baseline_edge_sha
        if allow_baseline_band_reference:
            self.scope["baseline_band_reference_enabled"] = True
            self.scope["baseline_band_source_sha256"] = self._baseline_band_sha
        if allow_single_event_reference:
            self.scope["single_event_reference_enabled"] = True
            self.scope["single_event_source_sha256"] = self._single_event_sha
        if save_strategy == "autosave":
            self.scope.update(
                save_strategy="autosave",
                persistence_verified=False,
                autosave_source_sha256=self._autosave_sources[
                    Path(__file__).with_name("browser_autosave.py")
                ],
                autosave_readback_source_sha256=self._autosave_sources[
                    Path(__file__).with_name("browser_no_planet_readback.py")
                ],
            )
        if self._supplied_evaluation is not None:
            self.scope["supplied_stellar_inputs_enabled"] = True
        persist_json(self.output / "scope.json", self.scope)
        self._autosave_scope_sha = (
            hashlib.sha256((self.output / "scope.json").read_bytes()).hexdigest()
            if self._save_strategy == "autosave"
            else None
        )
        self._two_event_scope_sha = (
            hashlib.sha256((self.output / "scope.json").read_bytes()).hexdigest()
            if self._two_event_enabled
            else None
        )
        self._baseline_band_scope_sha = (
            hashlib.sha256((self.output / "scope.json").read_bytes()).hexdigest()
            if self._baseline_band_enabled
            else None
        )

    @property
    def finished(self):
        return self.phase in {
            "verified_no_planet",
            "planet_measurement_required",
            "planet_classification_required",
            "planet_calculation_unsupported",
            "stopped",
            "aborted",
        }

    def _check_sources(self):
        self._check_save_strategy()
        if (
            (str(self._supplied_evaluation) if self._supplied_evaluation is not None else None)
            != self._supplied_option
            or self.scope.get("supplied_stellar_inputs_enabled", False)
            is not (self._supplied_option is not None)
            or (
                self._supplied_evaluation is not None
                and hashlib.sha256((self._supplied_evaluation / "report.json").read_bytes()).hexdigest()
                != self._supplied_report_hash
            )
        ):
            raise BrowserSafetyStop("star_session_supplied_permission_changed")
        if (
            self.shallow_reference is not self._shallow_enabled
            or self.scope.get("shallow_reference_enabled", False) is not self._shallow_enabled
        ):
            raise BrowserSafetyStop("star_session_shallow_permission_changed")
        self._check_two_event_permission()
        self._check_baseline_band_permission()
        if (
            self.allow_baseline_edge_reference is not self._baseline_edge_enabled
            or self.scope.get("baseline_edge_reference_enabled", False) is not self._baseline_edge_enabled
            or self._baseline_edge_enabled
            and (
                self.scope.get("baseline_edge_source_sha256") != self._baseline_edge_sha
                or hashlib.sha256(self._baseline_edge_source.read_bytes()).hexdigest()
                != self._baseline_edge_sha
            )
        ):
            raise BrowserSafetyStop("star_session_baseline_edge_permission_changed")
        if (
            validate_star_class_source(
                self.history, self.class_dir, self.star, self.model_options["selected_class"]
            )
            != self.class_source
        ):
            raise BrowserSafetyStop("star_session_class_source_changed")
        self._check_planet_sources()

    def ready_to_advance(self):
        """Validate pinned sources even while the window is waiting, without UI calls."""
        self._check_sources()
        if self.phase == "planet_window" and self.component is not None:
            ready = getattr(self.component, "ready_to_advance", None)
            if ready is not None:
                if not callable(ready):
                    raise BrowserSafetyStop("star_session_invalid_readiness")
                result = ready()
                if type(result) is not bool:
                    raise BrowserSafetyStop("star_session_invalid_readiness")
                return result
        return True

    def state(self):
        child = self.component.state() if self.component else None
        return {
            **self.scope,
            "phase": self.phase,
            "stage": self.phase,
            "finished": self.finished,
            "task_completed": self.phase == "verified_no_planet",
            "failure_reason": self.failure,
            "event_forwarding_failed": self.event_forwarding_failed,
            "component": child,
            "artifact_paths": deepcopy(self.artifact_paths),
            "window_evidence": deepcopy(self.window_evidence),
            "spectrum_evidence": deepcopy(self.spectrum_evidence),
            **(
                {
                    "save_outcome_uncertain": self._save_outcome["save_outcome_uncertain"],
                    "save_outcome": deepcopy(self._save_outcome),
                }
                if self._save_outcome is not None
                else {}
            ),
            "reference_measurements": (
                deepcopy(child.get("reference_measurements"))
                if child
                and not self.failure
                and self.phase in {"positive_planet", "planet_classification_required"}
                else None
            ),
        }

    def emit(self, event, payload):
        try:
            self._callback(event, payload)
        except Exception:  # noqa: BLE001 - failure details can contain credentials
            if not self.event_forwarding_failed:
                self.event_forwarding_failed = True
                persist_json(self.output / "event-forwarding-failed.json", {"failed": True})
            if not self._closing:
                raise BrowserSafetyStop("star_session_event_forwarding_failed") from None

    def _relayed(self, event, payload):
        # Child lifecycle is nested by ProjectEventRelay. Only this owner may
        # report a star handoff/completion; keep its current reference atomic.
        if event == "state":
            payload = {
                **payload,
                "stage": self.phase,
                "star": self.star,
                "reference_measurements": self.state()["reference_measurements"],
            }
        self.emit(event, payload)

    def _bind_events(self, name):
        forward = self._relay.bind(name, star=self.star)

        def guarded(*args):
            if self.finished and not self._closing:
                raise BrowserSafetyStop("star_session_already_stopped")
            forward(*args)
            if self.finished and not self._closing:
                raise BrowserSafetyStop("star_session_already_stopped")
            if not self._closing:
                # A UI callback may revoke permission during action_proposed.
                # Recheck before returning to the child's native dispatch.
                self._check_two_event_permission()
                self._check_baseline_band_permission()
                self._check_save_strategy()

        return guarded

    def _check_two_event_permission(self):
        if (
            self.two_event_reference is not self._two_event_enabled
            or self.scope.get("two_event_reference_enabled", False) is not self._two_event_enabled
        ):
            raise BrowserSafetyStop("star_session_two_event_permission_changed")
        if self._two_event_enabled:
            self._owned_bytes(self.output / "scope.json", self._two_event_scope_sha)

    def _check_save_strategy(self):
        if (
            type(self.save_strategy) is not str
            or self.save_strategy != self._save_strategy
            or self.scope.get("save_strategy", "explicit") != self._save_strategy
            or self._save_strategy == "autosave"
            and (
                self.scope.get("persistence_verified") is not False
                or any(
                    hashlib.sha256(path.read_bytes()).hexdigest() != sha
                    for path, sha in self._autosave_sources.items()
                )
                or self.scope.get("autosave_source_sha256")
                != self._autosave_sources[Path(__file__).with_name("browser_autosave.py")]
                or self.scope.get("autosave_readback_source_sha256")
                != self._autosave_sources[Path(__file__).with_name("browser_no_planet_readback.py")]
            )
        ):
            raise BrowserSafetyStop("star_session_save_strategy_changed")
        if self._save_strategy == "autosave":
            self._owned_bytes(self.output / "scope.json", self._autosave_scope_sha)

    def _check_baseline_band_permission(self):
        self._check_single_event_permission()
        if (
            self.allow_baseline_band_reference is not self._baseline_band_enabled
            or self.scope.get("baseline_band_reference_enabled", False) is not self._baseline_band_enabled
            or self._baseline_band_enabled
            and (
                self.allow_baseline_edge_reference is not True
                or self.scope.get("baseline_edge_reference_enabled") is not True
                or self.scope.get("baseline_edge_source_sha256") != self._baseline_edge_sha
                or hashlib.sha256(self._baseline_edge_source.read_bytes()).hexdigest()
                != self._baseline_edge_sha
                or self.scope.get("baseline_band_source_sha256") != self._baseline_band_sha
                or hashlib.sha256(self._baseline_band_source.read_bytes()).hexdigest()
                != self._baseline_band_sha
            )
        ):
            raise BrowserSafetyStop("star_session_baseline_band_permission_changed")
        if self._baseline_band_enabled:
            self._owned_bytes(self.output / "scope.json", self._baseline_band_scope_sha)

    def _check_single_event_permission(self):
        if (
            self.allow_single_event_reference is not self._single_event_enabled
            or self.scope.get("single_event_reference_enabled", False) is not self._single_event_enabled
            or self._single_event_enabled
            and (
                self.allow_baseline_band_reference is not True
                or self.scope.get("baseline_band_reference_enabled") is not True
                or self.scope.get("single_event_source_sha256") != self._single_event_sha
                or hashlib.sha256(self._single_event_source.read_bytes()).hexdigest()
                != self._single_event_sha
            )
        ):
            raise BrowserSafetyStop("star_session_single_event_permission_changed")

    def _owned_bytes(self, path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = self.history / path
        if (
            not path.resolve().is_relative_to(self.history)
            or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file()
            or path.stat().st_size > 2_000_000
            or any((path.parent / name).exists() for name in ("stopped.json", "invalidated.json"))
        ):
            raise BrowserSafetyStop("star_session_invalid_owned_evidence")
        raw = path.read_bytes()
        if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
            raise BrowserSafetyStop("star_session_source_hash_changed")
        return raw

    def _record_failed_save(self):
        """Describe bounded owned failure records, never authorize or retry Save.

        A dispatch marker precedes the native click, so it is not proof that a
        click returned. A painted acknowledgement is not a verified readback.
        Missing/invalid records remain unknown, not a claim that nothing saved.
        """
        directory = self.output / "save"
        summary = {
            "scope": "noncanonical_child_save_diagnostic_v1",
            "star": self.star,
            "save_dir": str(directory.relative_to(self.history)),
            "evidence_status": "unavailable_or_invalid",
            "save_outcome_uncertain": None,
            "reservation_retained": None,
            "dispatch_recorded": None,
            "acknowledgement_recorded": None,
            "final_readback_verified": False,
            "canonical_receipt": False,
            "task_completed": False,
            "project_completed": False,
            "automatic_retry": False,
            "source_sha256": {},
        }
        self._save_outcome = summary
        hashes = {}

        def reject(*_):
            raise ValueError("invalid_save_diagnostic")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    reject()
                result[key] = value
            return result

        def record(path):
            if not path.resolve().is_relative_to(self.history) or any(
                item.is_symlink() for item in (path, *path.parents)
            ):
                reject()
            if not path.exists():
                return None
            if not path.is_file() or not 0 < path.stat().st_size <= 64_000:
                reject()
            raw = path.read_bytes()
            if len(raw) > 64_000:
                reject()
            value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
            if not isinstance(value, dict):
                reject()
            hashes[str(path.relative_to(self.history))] = hashlib.sha256(raw).hexdigest()
            return value

        try:
            if (directory / "invalidated.json").exists():
                reject()
            stopped = record(directory / "stopped.json")
            if (
                not isinstance(stopped, dict)
                or stopped.get("mode") != NO_PLANET_SAVE_MODE
                or type(stopped.get("reservation_created")) is not bool
                or type(stopped.get("save_may_have_occurred")) is not bool
                or stopped.get("automatic_retry") is not False
                or stopped.get("task_completed") is not False
            ):
                reject()
            intent = record(directory / "reserved.json")
            key = hashlib.sha256(self.star.casefold().encode()).hexdigest()
            reservation = record(self.history / "no-planet-save-reservations" / f"{key}.json")
            dispatch = record(directory / "dispatch.json")
            acknowledgement = record(directory / "acknowledgement.json")
            if record(directory / "confirmed.json") is not None:
                reject()  # A success receipt needs the existing full workflow verifier.
            if stopped["reservation_created"]:
                source = self.output / "window/choice/confirmed.json"
                if record(source) is None:
                    reject()
                if (
                    not isinstance(intent, dict)
                    or not isinstance(reservation, dict)
                    or json.dumps(intent, sort_keys=True) != json.dumps(reservation, sort_keys=True)
                    or type(intent.get("schema_version")) is not int
                    or intent["schema_version"] != 1
                    or intent.get("mode") != NO_PLANET_SAVE_MODE
                    or not isinstance(intent.get("star"), str)
                    or intent["star"].casefold() != self.star.casefold()
                    or intent.get("output") != summary["save_dir"]
                    or intent.get("choice_path") != str(source.relative_to(self.history))
                    or intent.get("choice_sha256") != hashes.get(str(source.relative_to(self.history)))
                    or intent.get("kind") != "CLICK"
                    or intent.get("visible_label") != "Save"
                    or type(intent.get("max_save_clicks")) is not int
                    or intent["max_save_clicks"] != 1
                    or intent.get("automatic_retry") is not False
                    or intent.get("task_completed") is not False
                ):
                    reject()
            elif intent is not None or reservation is not None or stopped["save_may_have_occurred"]:
                reject()
            if dispatch is not None and (
                dispatch != {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1}
                or type(dispatch.get("max_clicks")) is not int
            ):
                reject()
            if acknowledgement is not None and (
                acknowledgement
                != {
                    "visible_text": "Data saved",
                    "source": "fully_exposed_footer_text",
                    "notice_was_already_present": False,
                }
                or acknowledgement.get("notice_was_already_present") is not False
                or dispatch is None
            ):
                reject()
            if not stopped["save_may_have_occurred"] and (
                dispatch is not None or acknowledgement is not None
            ):
                reject()
            summary.update(
                evidence_status="recorded_stop",
                save_outcome_uncertain=stopped["save_may_have_occurred"],
                reservation_retained=stopped["reservation_created"],
                dispatch_recorded=dispatch is not None,
                acknowledgement_recorded=acknowledgement is not None,
                source_sha256=hashes,
            )
        except Exception:  # noqa: BLE001,S110 - keep first failure; never echo diagnostic file data
            pass

    def _pin_window(self):
        report = self.component.report
        if not isinstance(report, dict) or not isinstance(report.get("progress_path"), str):
            raise BrowserSafetyStop("star_session_missing_window_evidence")
        path = self.history / report["progress_path"]
        if (
            not path.resolve().is_relative_to(self.output / "window")
            or not isinstance(report.get("progress_sha256"), str)
            or len(report["progress_sha256"]) != 64
        ):
            raise BrowserSafetyStop("star_session_invalid_window_evidence")
        captured = json.loads(self._owned_bytes(path, report["progress_sha256"]))
        if (
            captured.get("star", "").casefold() != self.star.casefold()
            or captured.get("requested_days") != 5000
        ):
            raise BrowserSafetyStop("star_session_window_source_changed")
        chart = path.parent / "chart.png"
        chart_hash = captured.get("chart_sha256")
        if not isinstance(chart_hash, str) or len(chart_hash) != 64:
            raise BrowserSafetyStop("star_session_missing_window_chart_hash")
        self._owned_bytes(chart, chart_hash)
        self.window_evidence = {
            "progress_path": str(path.relative_to(self.history)),
            "progress_sha256": report["progress_sha256"],
            "chart_path": str(chart.relative_to(self.history)),
            "chart_sha256": chart_hash,
        }
        persist_json(self.output / "window-handoff.json", self.window_evidence)

    def _check_planet_sources(self):
        if self._shallow_adoption is not None:
            if {
                "window_evidence": self.window_evidence,
                "tooltip_reference": self.tooltip_reference,
            } != self._shallow_adoption:
                raise BrowserSafetyStop("star_session_shallow_cached_source_changed")
            paths = sorted(
                str(path.relative_to(self.history)) for path in (self.output / "shallow").rglob("*")
            )
            if paths != self._shallow_tree:
                raise BrowserSafetyStop("star_session_shallow_tree_changed")
        for relative, checksum in self._shallow_hashes.items():
            self._owned_bytes(relative, checksum)
        if self._original_window_evidence:
            for path, checksum in (("progress_path", "progress_sha256"), ("chart_path", "chart_sha256")):
                self._owned_bytes(
                    self._original_window_evidence[path], self._original_window_evidence[checksum]
                )
        if self.window_evidence:
            self._owned_bytes(self.window_evidence["progress_path"], self.window_evidence["progress_sha256"])
            self._owned_bytes(self.window_evidence["chart_path"], self.window_evidence["chart_sha256"])
        if self.spectrum_evidence:
            self._owned_bytes(self.spectrum_evidence["path"], self.spectrum_evidence["sha256"])

    def _adopt_shallow(self):
        from .browser_raster_planet_evidence import TOOLTIP_MODE, _overview_window, _Owned
        from .browser_shallow_transit_probe import FLAGS
        from .browser_shallow_transit_steps import MODE
        from .planet_tooltip_reference import TWO_MODE, load_tooltip_reference

        report = self.component.report
        two_events = isinstance(report, dict) and report.get("measurement_mode") == TWO_MODE
        if (
            not isinstance(report, dict)
            or report.get("mode") != MODE
            or report.get("phase") != "measurements_ready"
            or report.get("finished") is not True
            or report.get("failure_reason") is not None
            or report.get("task_completed") is not False
            or report.get("project_completed") is not False
            or report.get("star", "").casefold() != self.star.casefold()
            or report.get("measurement_mode") not in {TOOLTIP_MODE, TWO_MODE}
            or two_events
            and (
                not self._two_event_enabled
                or report.get("two_event_reference_enabled") is not True
                or type(report.get("probes")) is not int
                or report["probes"] != 2
                or type(report.get("restorations")) is not int
                or report["restorations"] != 2
                or type(report.get("max_native_actions")) is not int
                or report["max_native_actions"] != 38
            )
            or any(
                type(report.get(key)) is not type(value) or report[key] != value
                for key, value in FLAGS.items()
            )
            or any(
                report.get(key) is not False
                for key in ("native_action_outcome_uncertain", "event_forwarding_failed", "cleanup_failed")
            )
        ):
            raise BrowserSafetyStop("star_session_shallow_measurements_unverified")
        directory = self.output / "shallow"
        if json.loads(self._owned_bytes(directory / "report.json")) != report:
            raise BrowserSafetyStop("star_session_shallow_report_changed")
        window, descriptor = report["window_evidence"], report["tooltip_reference"]
        if (
            not isinstance(window, dict)
            or set(window) != {"progress_path", "progress_sha256", "chart_path", "chart_sha256"}
            or not isinstance(descriptor, dict)
            or set(descriptor)
            != ({"expected_star", "diagnostics", "mode"} if two_events else {"expected_star", "diagnostics"})
            or two_events
            and descriptor.get("mode") != TWO_MODE
            or descriptor["expected_star"].casefold() != self.star.casefold()
            or any(
                not isinstance(window[key], str) or not re.fullmatch(r"[0-9a-f]{64}", window[key])
                for key in ("progress_sha256", "chart_sha256")
            )
        ):
            raise BrowserSafetyStop("star_session_invalid_shallow_handoff")
        scoped = [
            window["progress_path"],
            window["chart_path"],
            *(spec["directory"] for spec in descriptor["diagnostics"]),
        ]
        for name in scoped:
            path = self.history / name
            if not path.resolve().is_relative_to(directory) or any(
                p.is_symlink() for p in (path, *path.parents)
            ):
                raise BrowserSafetyStop("star_session_shallow_source_outside_child")
        anchor = _overview_window(
            _Owned(self.history), self.history / window["progress_path"], window["progress_sha256"]
        )
        if (
            anchor["star"].casefold() != self.star.casefold()
            or anchor["chart_sha256"] != window["chart_sha256"]
            or self.history / window["chart_path"]
            != (self.history / window["progress_path"]).parent / "chart.png"
        ):
            raise BrowserSafetyStop("star_session_shallow_final_source_changed")
        measured = load_tooltip_reference(
            self.history,
            descriptor["diagnostics"],
            expected_star=self.star,
            **({"mode": TWO_MODE} if two_events else {}),
        )
        if measured != report["measurements"]:
            raise BrowserSafetyStop("star_session_shallow_measurements_changed")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise BrowserSafetyStop("star_session_invalid_owned_evidence")
            if path.is_file():
                self._shallow_hashes[str(path.relative_to(self.history))] = hashlib.sha256(
                    self._owned_bytes(path)
                ).hexdigest()
        self._shallow_tree = sorted(str(path.relative_to(self.history)) for path in directory.rglob("*"))
        self.window_evidence, self.tooltip_reference = deepcopy(window), deepcopy(descriptor)
        self._shallow_adoption = deepcopy({"window_evidence": window, "tooltip_reference": descriptor})
        self._check_planet_sources()
        self.artifact_paths["shallow_dir"] = str(directory.relative_to(self.history))

    def _finish(self):
        if self.report is None:
            report = self.state()
            try:
                self.emit("episode_summary", {**report, "completed": report["task_completed"]})
            except BrowserSafetyStop as exc:
                self.phase, self.failure = "stopped", str(exc)
                report = self.state()
            self.report = report
            persist_json(self.output / "report.json", report)
            self._relay.retire()
        return self.state()

    def abort(self):
        if not self.finished:
            self.phase, self.failure = "aborted", "operator_aborted"
            self._closing = True
            try:
                if self.component:
                    self.component.abort()
                self._finish()
            finally:
                self._closing = False
        return self.state()

    def close(self):
        """Stop only this scheduler; the caller retains the browser and artifacts."""
        return self.abort()

    def advance(self):
        if self.finished:
            return self.state()
        if self._advancing:
            self.phase, self.failure = "stopped", "star_session_reentrant_advance"
            self._closing = True
            try:
                if self.component:
                    self.component.abort()
                self._finish()
            finally:
                self._closing = False
            raise BrowserSafetyStop("star_session_reentrant_advance")
        self._advancing = True
        try:
            return self._advance()
        finally:
            self._advancing = False

    def _advance(self):
        try:
            self._check_sources()
            if self.phase in {"stellar_numeric", "stellar_color"}:
                numeric = self.phase == "stellar_numeric"
                if self.component is None:
                    factory = FullStellarNumericSteps if numeric else FullStellarColorSteps
                    options = self.numeric_options if numeric else {"experiment": self.color_experiment}
                    self.component = factory(
                        self.page,
                        self.config,
                        self.output / ("numeric" if numeric else "color"),
                        **self.model_options,
                        **options,
                        emit=self._bind_events("stellar.numeric" if numeric else "stellar.color"),
                    )
                    self.artifact_paths["numeric_dir" if numeric else "color_dir"] = str(
                        (self.output / ("numeric" if numeric else "color")).relative_to(self.history)
                    )
                    if not self.component.finished:
                        pinned = self.component.session.mapping
                        if not matches_mapping(self.class_source, pinned):
                            raise BrowserSafetyStop("star_session_stellar_star_changed")
                else:
                    self.component.advance()
                if self.finished:
                    if self.component and not self.component.finished:
                        self.component.abort()
                    return self.state()
                if self.component.finished:
                    if not self.component.state()["transport_verified"]:
                        raise BrowserSafetyStop("star_session_stellar_transport_failed")
                    self.component = None
                    self._relay.retire()
                    self.phase = "stellar_color" if numeric else "navigate_planet"
            elif self.phase == "navigate_planet":
                result = navigate_project(
                    self.page, self.config, self.output / "to-planet", "planet", expected_star=self.star
                )
                if (
                    result.get("same_star_verified") is not True
                    or result.get("destination_verified") is not True
                ):
                    raise BrowserSafetyStop("star_session_planet_navigation_unverified")
                for rule in self.config.frames:
                    if rule.url == SIMULATION_URL:
                        rule.required_text = result["suggested_required_text"]
                self.phase = "planet_window"
            elif self.phase == "planet_window":
                if self.component is None:
                    self.component = PlanetWindowSession(
                        self.page,
                        self.config,
                        self.output / "window",
                        run_history=self.history,
                        select_no=True,
                        **({"allow_shallow_reference": True} if self.shallow_reference else {}),
                        **({"allow_baseline_edge_reference": True} if self._baseline_edge_enabled else {}),
                        **({"allow_baseline_band_reference": True} if self._baseline_band_enabled else {}),
                        **({"allow_single_event_reference": True} if self._single_event_enabled else {}),
                        emit=self._bind_events("planet.window"),
                    )
                    self.artifact_paths["window_dir"] = str(
                        (self.output / "window").relative_to(self.history)
                    )
                else:
                    self.component.advance()
                if self.finished:
                    if self.component and not self.component.finished:
                        self.component.abort()
                    return self.state()
                if self.component.finished:
                    state = self.component.state()
                    if state.get("phase") in {"stopped", "aborted"}:
                        raise BrowserSafetyStop(
                            state.get("failure_reason") or "star_session_window_unverified"
                        )
                    if (
                        not isinstance(state.get("star"), str)
                        or state["star"].casefold() != self.star.casefold()
                    ):
                        raise BrowserSafetyStop("star_session_window_star_changed")
                    if state["phase"] in {"dip_observed", "shallow_reference_required"}:
                        self._pin_window()
                        self._relay.retire()
                        if (
                            self.model_options["selected_class"] != "main_sequence"
                            and self._supplied_evaluation is None
                        ):
                            self.phase, self.failure = (
                                "planet_calculation_unsupported",
                                "planet_requires_supplied_main_sequence",
                            )
                        elif self.planet_options is None:
                            self.phase = "planet_measurement_required"
                        elif state["phase"] == "shallow_reference_required":
                            if not self.shallow_reference:
                                raise BrowserSafetyStop("star_session_shallow_reference_not_enabled")
                            self._original_window_evidence = deepcopy(self.window_evidence)
                            self.phase, self.component = "shallow_reference", None
                        else:
                            self.phase, self.component = "capture_spectrum", None
                    elif state["phase"] == "no_selected" and state["decision_readback_verified"]:
                        self._relay.retire()
                        self.artifact_paths["choice_dir"] = str(
                            (self.output / "window/choice").relative_to(self.history)
                        )
                        self.phase, self.component = "save_no_planet", None
                    else:
                        raise BrowserSafetyStop(
                            state.get("failure_reason") or "star_session_window_unverified"
                        )
            elif self.phase == "shallow_reference":
                from .browser_shallow_transit_steps import ShallowTransitSteps

                if self.component is None:
                    self.component = ShallowTransitSteps(
                        self.page,
                        self.config,
                        self.output / "shallow",
                        run_history=self.history,
                        star=self.star,
                        source_report=self.history / self.window_evidence["progress_path"],
                        source_report_sha256=self.window_evidence["progress_sha256"],
                        emit=self._bind_events("planet.shallow"),
                        **({"allow_two_events": True} if self._two_event_enabled else {}),
                    )
                else:
                    self.component.advance()
                if self.finished:
                    self.component.abort()
                    return self.state()
                if self.component.finished:
                    self._adopt_shallow()
                    self._relay.retire()
                    self.phase, self.component = "capture_spectrum", None
            elif self.phase == "capture_spectrum":
                result = capture_spectrum_excursion(
                    self.page,
                    self.config,
                    self.output / "spectrum",
                    expected_star=self.star,
                    emit=self._bind_events("planet.spectrum"),
                    marker_mode=GEOMETRIC_PAIR_MODE,
                )
                if self.finished:
                    return self.state()
                path = self.output / "spectrum/spectrum.json"
                raw = self._owned_bytes(path)
                saved = json.loads(raw)
                if saved != result or _spectrum(saved)["star"].casefold() != self.star.casefold():
                    raise BrowserSafetyStop("star_session_spectrum_source_changed")
                self.spectrum_evidence = {
                    "path": str(path.relative_to(self.history)),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                self.artifact_paths["spectrum_dir"] = str(path.parent.relative_to(self.history))
                self._check_planet_sources()
                persist_json(self.output / "spectrum-handoff.json", self.spectrum_evidence)
                self._relay.retire()
                self.phase = "positive_planet"
            elif self.phase == "positive_planet":
                if self.component is None:
                    self.component = PositivePlanetSteps(
                        self.page,
                        self.config,
                        self.output / "positive",
                        run_history=self.history,
                        star=self.star,
                        window_report=self.history / self.window_evidence["progress_path"],
                        window_report_sha256=self.window_evidence["progress_sha256"],
                        spectrum_path=self.history / self.spectrum_evidence["path"],
                        spectrum_sha256=self.spectrum_evidence["sha256"],
                        graph=load_graph(self.model_options["graph_path"]),
                        supplied_star_class=self.model_options["selected_class"],
                        **(
                            {"tooltip_reference": self.tooltip_reference}
                            if self.tooltip_reference is not None
                            else {}
                        ),
                        **self.planet_options,
                        **(
                            {
                                "supplied_evaluation": self._supplied_evaluation,
                                "supplied_stellar_sources": {
                                    "numeric_dir": self.output / "numeric",
                                    "color_dir": self.output / "color",
                                    "class_dir": self.class_dir,
                                    "navigation_dir": self.output / "to-planet",
                                },
                            }
                            if self._supplied_evaluation is not None
                            and self.model_options["selected_class"] != "main_sequence"
                            else {}
                        ),
                        emit=self._bind_events("planet.positive"),
                    )
                    self.artifact_paths.update(
                        {
                            key: str((self.output / "positive" / child).relative_to(self.history))
                            for key, child in (
                                ("positive_dir", "."),
                                ("presence_dir", "presence"),
                                ("raw_dir", "raw"),
                                ("derived_dir", "derived"),
                            )
                        }
                    )
                else:
                    self.component.advance()
                if self.finished:
                    self.component.close()
                    return self.state()
                child = self.component.state()
                if child.get("star", "").casefold() != self.star.casefold():
                    raise BrowserSafetyStop("star_session_positive_star_changed")
                self._check_planet_sources()
                if self.component.finished:
                    if (
                        child.get("phase") != "planet_classification_required"
                        or child.get("derived_transport_verified") is not True
                        or child.get("task_completed") is not False
                    ):
                        raise BrowserSafetyStop(
                            child.get("failure_reason") or "star_session_positive_transport_failed"
                        )
                    self.component.close()
                    self._relay.retire()
                    self.phase = "planet_classification_required"
            elif self.phase == "save_no_planet":
                source = self.output / "window/choice/confirmed.json"
                self.artifact_paths["save_dir"] = str((self.output / "save").relative_to(self.history))
                try:
                    if self._save_strategy == "autosave":
                        from .browser_no_planet_readback import readback_no_planet_work

                        readback_no_planet_work(
                            self.page,
                            self.config,
                            self.output / "save",
                            run_history=self.history,
                            choice_path=source,
                            choice_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            cancelled=lambda: self.finished,
                        )
                    else:
                        save_no_planet_work(
                            self.page,
                            self.config,
                            self.output / "save",
                            run_history=self.history,
                            choice_path=source,
                            choice_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            settle_timeout_seconds=self.no_planet_save_settle_seconds,
                            settle_reserved_notice=True,
                            cancelled=lambda: self.finished,
                        )
                except BaseException:
                    try:
                        if self._save_strategy == "explicit":
                            self._record_failed_save()
                    except Exception:  # noqa: BLE001,S110 - keep the adapter's original failure
                        pass
                    raise
                if self.finished:
                    return self.state()
                self.phase = "verify_no_planet"
            elif self.phase == "verify_no_planet":
                receipt = verify_no_planet_workflow(
                    self.page,
                    self.config,
                    self.output / "workflow",
                    run_history=self.history,
                    numeric_dir=self.output / "numeric",
                    color_dir=self.output / "color",
                    class_dir=self.class_dir,
                    choice_dir=self.output / "window/choice",
                    save_dir=self.output / "save",
                )
                if (
                    receipt.get("task_completed") is not True
                    or receipt.get("star", "").casefold() != self.star.casefold()
                ):
                    raise BrowserSafetyStop("star_session_workflow_unverified")
                self.phase = "verified_no_planet"
                self.artifact_paths["workflow_dir"] = str(
                    (self.output / "workflow").relative_to(self.history)
                )
            else:
                raise BrowserSafetyStop("star_session_unknown_phase")
        except Exception as exc:  # noqa: BLE001 - never emit driver URLs or credentials
            if self.finished:
                return self.state()
            self.failure = str(exc) if isinstance(exc, BrowserSafetyStop) else "star_session_component_failed"
            self.phase = "stopped"
            self._closing = True
            try:
                if self.component and not self.component.finished:
                    self.component.abort()
                self.emit("error", {"type": "BrowserStarStop", "message": self.failure})
            finally:
                self._closing = False
        if self.finished:
            return self._finish()
        try:
            self.emit("state", self.state())
        except BrowserSafetyStop as exc:
            self.phase, self.failure = "stopped", str(exc)
            if self.component and not self.component.finished:
                self.component.abort()
            return self._finish()
        return self.state()
