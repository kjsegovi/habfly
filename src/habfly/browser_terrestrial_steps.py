"""Cooperative terrestrial glue around existing guarded, reference-assisted tools.

Construction and decision handoffs are offline. Gas identity and habitability
remain explicit supplied judgments, never inferred from a crop, temperature,
course score, or liquid-water indicator. The caller owns browser lifetime.
"""

import hashlib
import json
import math
import re
import time
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_autosave import autosave_workflow_flags, validate_autosave_workflow_flags
from .browser_gas_controls import compare_visible_gas_candidates, select_reference_gases
from .browser_habitability import GASES
from .browser_habitability_actions import HabitabilityMenuSession
from .browser_habitability_choice import confirmed_phase_reference, select_habitability_reference
from .browser_habitability_save import _complete, _mapping, save_habitability_work
from .browser_habitability_steps import HabitabilityTemperatureSteps
from .browser_no_planet_save import _sync_directory
from .browser_no_planet_workflow import _Evidence, _planet
from .browser_numeric import committed_display
from .browser_positive_finalize import _capture_autosave_readback, _known_save_attempts
from .browser_positive_planet_workflow import (
    _identity_matches,
    _match_stellar_planet_sources,
    _planet_sources,
    _read_planet,
    _stellar_sources,
)
from .browser_probe import inspect_page, save_probe
from .browser_project_navigation import navigate_project
from .browser_save_settlement import intent_options
from .browser_stellar import SIMULATION_URL
from .browser_terrestrial_workflow import (
    _clean,
    _final_save,
    _gas_sources,
    _gas_writes,
    _greenhouse_source,
    _phase_choice_sources,
    _same_habitat,
    _temperature_source,
    _unchanged,
    verify_terrestrial_workflow,
)
from .browser_water_chamber import WaterChamberSession
from .contracts import RuntimeEvent
from .habitability_knowledge import HabitabilityCalculator
from .model import graph_fingerprint
from .project_events import ProjectEventRelay
from .supplied_browser_modes import source_options, terrestrial_mode
from .training.planet_sequence import file_hash

MODE = "cooperative_terrestrial_reference_workflow"
HANDOFFS = {"awaiting_gases", "awaiting_habitability"}


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("terrestrial_steps_" + reason)


def _terrestrial_autosave(
    page, config, output, *, history, phase_dir, choice_dir, star, check, timeout_seconds
):
    source_book, source = _phase_choice_sources(history, phase_dir, choice_dir)

    def validate(report, mapping, choices, handles):
        _complete(mapping)
        _require(
            _identity_matches(mapping["star_name"], star)
            and choices["selected"] == source["choice"]["choice"]
            and _same_habitat(report, source["capture"]),
            "autosave_current_habitability_changed",
        )

    return _capture_autosave_readback(
        page,
        config,
        output,
        source_book=source_book,
        star=star,
        branch="terrestrial",
        session_factory=HabitabilityMenuSession,
        validate=validate,
        check=check,
        timeout_seconds=timeout_seconds,
    )


def _sources(book, directories, class_sha, *, supplied_inputs=False):
    options = source_options(supplied_inputs)
    _require(isinstance(class_sha, str) and re.fullmatch(r"[a-f0-9]{64}", class_sha), "class_hash_required")
    raw = book.read(directories["planet_class_dir"] / "confirmed.json")
    _require(hashlib.sha256(raw).hexdigest() == class_sha, "supplied_class_changed")
    stellar = _stellar_sources(
        book, *(directories[k] for k in ("numeric_dir", "color_dir", "class_dir")), **options
    )
    planet = _planet_sources(
        book, *(directories[k] for k in ("raw_dir", "derived_dir", "planet_class_dir")), **options
    )
    _require(_identity_matches(stellar["star"], planet["star"]), "cross_star_sources")
    _require(planet["planet_class"] == "terrestrial", "explicit_terrestrial_class_required")
    if supplied_inputs:
        _match_stellar_planet_sources(stellar, planet, planet["class_capture"], **options)
    else:
        values = _planet(planet["class_capture"])["observation"]["values"]
        for p, s in (("stellar_mass", "mass"), ("stellar_radius", "radius")):
            committed_display(
                stellar["readbacks"][s]["display_value"], values["stellar_inputs"][p]["display_text"]
            )
    _unchanged(book)
    return {**stellar, **planet, "star": stellar["star"], "save_capture": planet["class_capture"]}


def _claim_path(history, star):
    return (
        history
        / "terrestrial-workflow-reservations"
        / (hashlib.sha256(star.casefold().encode()).hexdigest() + ".json")
    )


class TerrestrialSteps:
    """Each advance schedules one existing bounded stage or one learned step.

    The navigation stage includes a read-only Planet preflight. The existing
    synchronous gas comparison, chamber query, Save and final verifier retain
    their bounded internal calls; pause takes effect between those calls.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        numeric_dir,
        color_dir,
        class_dir,
        raw_dir,
        derived_dir,
        planet_class_dir,
        planet_class_sha256,
        candidates,
        pilot,
        final_evaluation,
        graph,
        seed=20000000,
        max_seconds=1800,
        timeout_seconds=20,
        settle_timeout_seconds=20,
        settle_reserved_notice=False,
        supplied_inputs=False,
        save_strategy="explicit",
        emit=lambda _: None,
    ):
        options = source_options(supplied_inputs)
        _require(
            isinstance(candidates, (list, tuple))
            and 1 <= len(candidates) <= len(GASES)
            and all(isinstance(v, str) for v in candidates)
            and len(set(candidates)) == len(candidates)
            and set(candidates) <= set(GASES),
            "explicit_gas_candidates_required",
        )
        _require(
            callable(emit)
            and type(save_strategy) is str
            and save_strategy in {"explicit", "autosave"}
            and type(settle_reserved_notice) is bool
            and type(seed) is int
            and type(max_seconds) in {int, float}
            and math.isfinite(max_seconds)
            and 0 < max_seconds <= 1800
            and all(
                type(v) in {int, float} and math.isfinite(v) and 0.1 <= v <= 30
                for v in (timeout_seconds, settle_timeout_seconds)
            ),
            "invalid_budgets_or_callback",
        )
        self.book = _Evidence(run_history)
        self.output = self.book.path(output)
        self.directories = {
            name + "_dir": self.book.path(value)
            for name, value in (
                ("numeric", numeric_dir),
                ("color", color_dir),
                ("class", class_dir),
                ("raw", raw_dir),
                ("derived", derived_dir),
                ("planet_class", planet_class_dir),
            )
        }
        _require(
            self.output != self.book.history
            and all(
                not self.output.is_relative_to(p) and not p.is_relative_to(self.output)
                for p in self.directories.values()
            ),
            "output_overlaps_source",
        )
        self.bundle = _sources(self.book, self.directories, planet_class_sha256, **options)
        self.supplied_inputs = supplied_inputs
        if save_strategy == "autosave":
            _known_save_attempts(self.book, self.bundle["star"])
            prior_claim = (
                self.book.history
                / "habitability-save-reservations"
                / (hashlib.sha256(self.bundle["star"].casefold().encode()).hexdigest() + ".json")
            )
            _require(not prior_claim.exists(), "prior_save_intent")
        self.claim = self.book.path(_claim_path(self.book.history, self.bundle["star"]))
        _require(not self.claim.exists(), "already_reserved")
        self.page, self.config, self.graph = page, config.model_copy(deep=True), graph
        rules = [r for r in self.config.frames if r.url == SIMULATION_URL]
        _require(len(rules) == 1, "unsupported_simulation_boundary")
        self._config = self.config.model_dump(mode="json")
        self.pilot, self.final_evaluation = Path(pilot).resolve(), Path(final_evaluation).resolve()
        # Freeze identities now; the existing child performs complete checkpoint
        # and held-out-gate validation before inference, not another evaluation.
        checkpoint = self.pilot / "training/checkpoint.pt"
        self._model_paths = {
            "checkpoint": checkpoint,
            "metadata": checkpoint.with_suffix(".pt.json"),
            "pilot_report": self.pilot / "report.json",
            "final_gate": self.final_evaluation / "report.json",
        }
        self._model_hashes = {key: file_hash(path) for key, path in self._model_paths.items()}
        self._graph_hash = graph_fingerprint(graph)
        self._pack_hash = self._current_pack_hash()
        self.candidates, self.class_sha, self.seed = tuple(candidates), planet_class_sha256, seed
        self.max_seconds, self.timeout, self.settle_timeout = (
            max_seconds,
            timeout_seconds,
            settle_timeout_seconds,
        )
        self.settle_reserved_notice = settle_reserved_notice
        self.save_strategy = save_strategy
        self._fixed_budget_options = tuple(
            (type(value), value)
            for value in (
                self.max_seconds,
                self.timeout,
                self.settle_timeout,
                self.settle_reserved_notice,
                self.save_strategy,
            )
        )
        self._started, self._callback, self._sequence = time.monotonic(), emit, 0
        self._busy = self._finalizing = self._cancelled = self._forward_failed = self._verified = False
        self.finished, self.report, self.failure = False, None, None
        self.phase, self.advances, self.component = "navigate_habitability", 0, None
        self._relay = ProjectEventRelay(self._receive)
        self._expected = self._gas_after = self._temperature = self._greenhouse_after = None
        self.gas_decision = self.habitability_decision = self.phase_reference = None
        self.save_receipt = self.workflow_receipt = self._workflow_sha = None
        self.calls, self.claimed = [], False
        self.cleanup_failed = False
        self.output.mkdir(parents=True, exist_ok=False)
        self._stream = (self.output / "events.jsonl").open("x", encoding="utf-8")
        try:
            self._emit(
                "hello",
                {
                    "protocol_version": 1,
                    "component": MODE,
                    "constructor_browser_actions": 0,
                    "source_sha256": dict(self.book.hashes),
                    "model_source_sha256": self._model_hashes,
                    "graph_hash": self._graph_hash,
                    "knowledge_pack_hash": self._pack_hash,
                    **(autosave_workflow_flags() if self.save_strategy == "autosave" else {}),
                },
            )
            self._check()
            self._emit("state", self.state())
            self._check()
        except BaseException as exc:
            self._stop(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise

    def _emit(self, kind, payload):
        event = RuntimeEvent(event=kind, sequence=self._sequence, run_id=self.output.name, payload=payload)
        line = event.model_dump_json()
        self._stream.write(line + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            try:
                self._callback(json.loads(line))
            except Exception:  # noqa: BLE001 - never expose callback/private diagnostics
                self._forward_failed = True

    def _receive(self, kind, payload):
        self._check()
        self._emit(kind, payload)
        self._check()

    def _check(self):
        _require(not self._cancelled, "operator_aborted")
        _require(not self._forward_failed, "event_forwarding_failed")
        _require(
            tuple(
                (type(value), value)
                for value in (
                    self.max_seconds,
                    self.timeout,
                    self.settle_timeout,
                    self.settle_reserved_notice,
                    self.save_strategy,
                )
            )
            == self._fixed_budget_options,
            "fixed_options_changed",
        )
        _require(time.monotonic() - self._started < self.max_seconds, "deadline_exhausted")
        _require(self.config.model_dump(mode="json") == self._config, "boundary_changed")
        _unchanged(self.book)
        _require(
            all(file_hash(p) == self._model_hashes[k] for k, p in self._model_paths.items())
            and graph_fingerprint(self.graph) == self._graph_hash
            and self._current_pack_hash() == self._pack_hash,
            "frozen_source_changed",
        )

    def _current_pack_hash(self):
        if self.supplied_inputs:
            from .habitability_supplied_inputs import SuppliedTemperatureCalculator

            return SuppliedTemperatureCalculator().pack.checksum
        return HabitabilityCalculator().pack.checksum

    def _pin(self, directory):
        directory = _clean(self.book, directory)
        files = sorted(p for p in directory.rglob("*") if p.is_file() or p.is_symlink())
        _require(1 <= len(files) <= 3000, "stage_artifact_limit")
        for path in files:
            self.book.read(path)
        self._check()

    def _persisted(self, directory, filename, returned):
        self._pin(directory)
        saved = self.book.json(directory / filename)
        _require(saved == returned, "stage_receipt_mismatch")
        return saved

    def _current(self):
        self._check()
        report = inspect_page(self.page, self.config)
        _require(not report["ignored_frame_urls"], "unknown_visible_frame")
        mapping = _mapping(report)
        _require(_identity_matches(mapping["star_name"], self.bundle["star"]), "current_star_changed")
        values = mapping["observation"]["values"]
        for value, expected in (
            (
                values["measurements"]["stellar_luminosity"]["display_text"],
                self.bundle["readbacks"]["luminosity"]["display_value"],
            ),
            (
                values["measurements"]["orbital_radius"]["display_text"],
                self.bundle["planet_readbacks"]["orbital_radius"]["display"]["display_value"],
            ),
        ):
            committed_display(expected, value)
        _require(self._expected is None or _same_habitat(self._expected, report), "current_fields_changed")
        self._check()
        return report, mapping

    def _reserve_owner(self):
        if self.claimed:
            return
        self.claim.parent.mkdir(exist_ok=True)
        persist_json(
            self.claim,
            {
                "schema_version": 1,
                "mode": MODE,
                "star": self.bundle["star"],
                "output": str(self.output.relative_to(self.book.history)),
                "planet_class_sha256": self.class_sha,
                "source_sha256": dict(self.book.hashes),
                "automatic_retry": False,
                "task_completed": False,
                "meaning": "single_owner_stage_budget_not_native_write_receipt",
            },
        )
        _sync_directory(self.claim.parent)
        _sync_directory(self.book.history)
        self.book.read(self.claim)
        self.claimed = True

    def _call(self, stage):
        self._check()
        _require(stage not in self.calls, "stage_already_attempted")
        persist_json(
            self.output / (stage + "-call-reserved.json"),
            {
                "stage": stage,
                "star": self.bundle["star"],
                "adapter_call_reserved": True,
                "native_write_dispatched": False
                if stage == "save" and self.save_strategy == "autosave"
                else "unknown_until_adapter_evidence",
                "automatic_retry": False,
            },
        )
        self.calls.append(stage)
        self._emit("state", {**self.state(), "scheduled_call": stage})
        self._check()

    def state(self):
        decisions = {
            "planet_class": {
                "directory": str(self.directories["planet_class_dir"].relative_to(self.book.history)),
                "confirmed_sha256": self.class_sha,
            }
        }
        for key, directory, filename, hash_key in (
            ("gas_comparison", "gas-comparison", "report.json", "report_sha256"),
            ("phase", "phase", "confirmed.json", "confirmed_sha256"),
        ):
            path = self.output / directory
            digest = self.book.hashes.get(str((path / filename).relative_to(self.book.history)))
            if digest is not None:
                decisions[key] = {"directory": str(path.relative_to(self.book.history)), hash_key: digest}
        return {
            **intent_options(self.settle_reserved_notice, self.timeout),
            **source_options(self.supplied_inputs),
            "component": MODE,
            "star": self.bundle["star"],
            "planet_class": "terrestrial",
            "phase": self.phase,
            "status": "finished" if self.finished else "paused" if self.phase in HANDOFFS else "ready",
            "finished": self.finished,
            "task_completed": self._verified,
            "failure_reason": self.failure,
            "advances": self.advances,
            "max_advances": 160,
            "max_seconds": self.max_seconds,
            "pause_counts_toward_deadline": True,
            "cancellation": "between_bounded_adapter_calls",
            "gas_candidates": list(self.candidates),
            "gas_decision": deepcopy(self.gas_decision),
            "phase_reference": deepcopy(self.phase_reference),
            "decision_sources": decisions,
            "habitability_decision": deepcopy(self.habitability_decision),
            "component_state": self.component.state() if self.component else None,
            **(autosave_workflow_flags() if self.save_strategy == "autosave" else {}),
            "save_acknowledgement_verified": self.save_strategy == "explicit"
            and self.save_receipt is not None,
            **(
                {"visible_readback_verified": self.save_receipt is not None}
                if self.save_strategy == "autosave"
                else {}
            ),
            "save_dispatch_attempts_recorded": int((self.output / "save/dispatch.json").is_file()),
            "workflow_dir": str((self.output / "workflow").relative_to(self.book.history))
            if self._verified
            else None,
            "workflow_sha256": self._workflow_sha if self._verified else None,
            "automatic_retry": False,
            "optimizer_updates": 0,
            "scientific_verified": False,
            "gas_identification_learned": False,
            "habitability_decision_learned": False,
            "assessment_enabled": False,
            "submission_enabled": False,
            "journal_writes": 0,
            "project_completed": False,
            "cleanup_failed": self.cleanup_failed,
        }

    def provide_gases(self, *, gases, rationale, supplied_greenhouse_increment):
        _require(
            not self.finished and not self._busy and self.phase == "awaiting_gases", "gas_handoff_not_ready"
        )
        _require(
            isinstance(gases, (list, tuple))
            and gases
            and all(isinstance(g, str) for g in gases)
            and len(set(gases)) == len(gases)
            and set(gases) <= set(self.candidates)
            and isinstance(rationale, str)
            and 20 <= len(rationale.strip()) <= 2000
            and type(supplied_greenhouse_increment) in {int, float}
            and supplied_greenhouse_increment in {0, 10, 30, 100},
            "invalid_supplied_gases",
        )
        return self._decision(
            "gas",
            {
                "gases": list(gases),
                "rationale": rationale.strip(),
                "supplied_greenhouse_increment": supplied_greenhouse_increment,
            },
            "select_gases",
        )

    def provide_habitability(self, *, choice, rationale):
        _require(
            not self.finished and not self._busy and self.phase == "awaiting_habitability",
            "habitability_handoff_not_ready",
        )
        _require(
            choice in {"habitable", "not_habitable"}
            and isinstance(rationale, str)
            and 40 <= len(rationale.strip()) <= 2000,
            "invalid_supplied_habitability",
        )
        # No automatic replacement for an invalid supplied choice. The existing
        # choice adapter owns the liquid-water applicability check at execution.
        return self._decision(
            "habitability", {"choice": choice, "rationale": rationale.strip()}, "select_habitability"
        )

    def provide_autonomous_gases(self, *, gases, rationale):
        """Opt in explicitly to reference gases then actual absorption readback.

        The immutable initial decision does not contain a guessed increment.
        A separate, pinned greenhouse-derived.json is created after the existing
        gas transport confirms the combined selection. Legacy provide_gases
        retains its required supplied increment and comparison check.
        """
        _require(
            not self.finished and not self._busy and self.phase == "awaiting_gases", "gas_handoff_not_ready"
        )
        _require(
            isinstance(gases, (list, tuple))
            and gases
            and set(self.candidates) == set(GASES)
            and all(isinstance(g, str) for g in gases)
            and len(set(gases)) == len(gases)
            and set(gases) <= set(self.candidates)
            and isinstance(rationale, str)
            and 20 <= len(rationale.strip()) <= 2000,
            "invalid_autonomous_gases",
        )
        return self._decision(
            "gas",
            {
                "gases": list(gases),
                "rationale": rationale.strip(),
                "greenhouse_mode": "derive_after_selection",
            },
            "select_gases",
        )

    def _decision(self, kind, decision, phase):
        self._busy = True
        try:
            self._check()
            path = self.output / (kind + "-decision.json")
            persist_json(
                path,
                {
                    **decision,
                    "star": self.bundle["star"],
                    "provenance": "reference_prediction",
                    "source_sha256": dict(self.book.hashes),
                    "task_completed": False,
                },
            )
            self.book.read(path)
            setattr(self, kind + "_decision", deepcopy(decision))
            self.phase = phase
            self._emit("state", self.state())
            self._check()
        except BaseException as exc:
            self._stop(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            self._busy = False
        return self.state()

    def advance(self):
        if self.finished:
            return self.state()
        if self._busy:
            self._cancelled = True
            raise BrowserSafetyStop("terrestrial_steps_reentrant_advance")
        self._busy = True
        try:
            self._check()
            if self.phase in HANDOFFS:
                return self.state()
            _require(self.advances < 160, "advance_limit")
            self.advances += 1
            self._reserve_owner()
            if self.phase != "temperature":
                self._call(self.phase)
            self._advance_stage()
            if not self.finished:
                self._check()
                self._emit("state", self.state())
                self._check()
        except BaseException as exc:
            self._stop(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            self._busy = False
        return self.state()

    def _advance_stage(self):
        phase = self.phase
        if phase == "navigate_habitability":
            _read_planet(self.page, self.config, self.output / "planet-readback", self.bundle)
            self._check()
            nav = navigate_project(
                self.page,
                self.config,
                self.output / "to-habitability",
                "habitability",
                expected_star=self.bundle["star"],
            )
            self._persisted(self.output / "to-habitability", "confirmed.json", nav)
            _require(
                nav.get("same_star_verified") is True and nav.get("destination_verified") is True,
                "navigation_unverified",
            )
            for rule in self.config.frames:
                if rule.url == SIMULATION_URL:
                    rule.required_text = nav["suggested_required_text"]
            self._config = self.config.model_dump(mode="json")
            self._expected, _ = self._current()
            save_probe(self._expected, self.output / "habitability-initial")
            self._pin(self.output / "habitability-initial")
            self.phase = "compare_gases"
        elif phase == "compare_gases":
            self._current()
            directory = self.output / "gas-comparison"
            result = compare_visible_gas_candidates(
                self.page, self.config, directory, candidates=list(self.candidates)
            )
            saved = self._persisted(directory, "report.json", result)
            _require(
                saved.get("baseline_restored") is True
                and saved.get("task_completed") is False
                and saved.get("candidates") == list(self.candidates),
                "comparison_unverified",
            )
            before, after = _gas_writes(
                self.book, directory, [(g, v) for g in self.candidates for v in (True, False)], saved["star"]
            )
            _require(
                _identity_matches(saved["star"], self.bundle["star"])
                and _same_habitat(self._expected, before)
                and _same_habitat(before, after),
                "comparison_changed_conditions",
            )
            self._expected, self.phase = after, "awaiting_gases"
        elif phase == "select_gases":
            self._current()
            directory = self.output / "gas-selection"
            result = select_reference_gases(
                self.page,
                self.config,
                directory,
                comparison=self.output / "gas-comparison",
                gases=self.gas_decision["gases"],
                rationale=self.gas_decision["rationale"],
            )
            self._persisted(directory, "report.json", result)
            _, self._gas_after = _gas_sources(
                self.book, self.output / "gas-comparison", directory, self.bundle["star"]
            )
            if self.gas_decision.get("greenhouse_mode") == "derive_after_selection":
                from .autonomous_planet import decide_greenhouse

                comparison = self.output / "gas-comparison"
                reference = decide_greenhouse(
                    self.book.history,
                    comparison,
                    self.book.hashes[str((comparison / "report.json").relative_to(self.book.history))],
                    directory,
                    self.book.hashes[str((directory / "report.json").relative_to(self.book.history))],
                    expected_star=self.bundle["star"],
                )
                _require(reference.get("status") == "decided", "autonomous_greenhouse_unresolved")
                self._check()
                path = self.output / "greenhouse-derived.json"
                persist_json(path, reference)
                self.book.read(path)
                for name, digest in reference["source_sha256"].items():
                    _require(
                        file_hash(self.book.history / name) == digest, "autonomous_greenhouse_source_changed"
                    )
                    self.book.read(self.book.history / name)
                # Keep gas-decision.json immutable: the owner pins it at the
                # supplied-decision boundary before this later readback stage.
                self.gas_decision = {**self.gas_decision, **reference["payload"]}
            self._expected, self.phase = self._gas_after, "temperature_initializing"
        elif phase == "temperature_initializing":
            self._current()
            factory, options = HabitabilityTemperatureSteps, {}
            if self.supplied_inputs:
                from .browser_supplied_steps import SuppliedHabitabilityTemperatureSteps

                link = self.bundle["derived"]["provenance"]["supplied_stellar_inputs"]
                factory = SuppliedHabitabilityTemperatureSteps
                options = {
                    "run_history": self.book.history,
                    "expected_star": self.bundle["star"],
                    "supplied_star_class": self.bundle["class"],
                    "supplied_stellar_inputs_path": self.book.history / link["path"],
                    "supplied_stellar_inputs_sha256": link["sha256"],
                }
            self.component = factory(
                self.page,
                self.config,
                self.output / "temperature",
                pilot=self.pilot,
                final_evaluation=self.final_evaluation,
                graph=self.graph,
                seed=self.seed,
                supplied_greenhouse_increment=self.gas_decision["supplied_greenhouse_increment"],
                emit=self._relay.bind("terrestrial.temperature", star=self.bundle["star"]),
                **options,
            )
            if not self.component.finished:
                _require(
                    _same_habitat(self._expected, self.component.session.report),
                    "temperature_initial_fields_changed",
                )
            self.phase = "temperature"
            self._temperature_done()
        elif phase == "temperature":
            self.component.advance()
            self._temperature_done()
        elif phase == "greenhouse":
            _, mapping = self._current()
            reference = HabitabilityCalculator().greenhouse_reference(
                mapping["observation"]["values"]["readouts"]["absorption"]["display_text"]
            )
            _require(
                reference.get("ok") is True
                and reference.get("increment_kelvin") == self.gas_decision["supplied_greenhouse_increment"],
                "supplied_greenhouse_disagrees",
            )
            session = HabitabilityMenuSession(self.page, self.config, self.output / "greenhouse")
            try:
                self._check()
                _require(_same_habitat(self._expected, session.report), "greenhouse_initial_fields_changed")
                result = session.greenhouse_reference()
            finally:
                session.close()
            self._persisted(self.output / "greenhouse", "confirmed.json", result)
            _, self._greenhouse_after = _greenhouse_source(
                self.book, self.output / "greenhouse", self._temperature, self._expected
            )
            self._expected, self.phase = self._greenhouse_after, "chamber"
        elif phase == "chamber":
            _, mapping = self._current()
            values = mapping["observation"]["values"]
            chamber = WaterChamberSession(
                self.page,
                self.config,
                self.output / "chamber",
                emit=self._relay.bind("terrestrial.chamber", star=self.bundle["star"]),
            )
            result = chamber.query(
                values["measurements"]["pressure"]["display_text"],
                values["readouts"]["surface_temp"]["display_text"],
                source="reference_diagnostic",
            )
            self._persisted(self.output / "chamber", "confirmed.json", result)
            self._relay.retire()
            self._current()
            self.phase = "water_phase"
        elif phase == "water_phase":
            self._current()
            session = HabitabilityMenuSession(self.page, self.config, self.output / "phase")
            try:
                self._check()
                _require(_same_habitat(self._expected, session.report), "phase_initial_fields_changed")
                result = session.water_phase_from_chamber(self.output / "chamber")
            finally:
                session.close()
            self._persisted(self.output / "phase", "confirmed.json", result)
            self._expected = self.book.capture(self.output / "phase/after")
            self.phase_reference = confirmed_phase_reference(self.output / "phase", _mapping(self._expected))
            self.phase = "awaiting_habitability"
        elif phase == "select_habitability":
            self._current()
            result = select_habitability_reference(
                self.page,
                self.config,
                self.output / "choice",
                phase_record=self.output / "phase",
                name=self.habitability_decision["choice"],
                rationale=self.habitability_decision["rationale"],
            )
            self._persisted(self.output / "choice", "confirmed.json", result)
            self._expected = self.book.capture(self.output / "choice/after")
            self.phase = "save"
        elif phase == "save":
            self._current()
            receipt = (
                self._autosave_readback()
                if self.save_strategy == "autosave"
                else save_habitability_work(
                    self.page,
                    self.config,
                    self.output / "save",
                    run_history=self.book.history,
                    phase_dir=self.output / "phase",
                    choice_dir=self.output / "choice",
                    timeout_seconds=self.timeout,
                    settle_timeout_seconds=self.settle_timeout,
                    settle_reserved_notice=self.settle_reserved_notice,
                    cancelled=self._save_cancelled,
                )
            )
            self._persisted(self.output / "save", "confirmed.json", receipt)
            _, saved, self._expected = _final_save(
                self.book,
                self.output / "save",
                self.output / "phase",
                self.output / "choice",
                self._greenhouse_after,
            )
            _require(saved == receipt, "save_receipt_mismatch")
            self.save_receipt, self.phase = receipt, "verify"
        elif phase == "verify":
            self._current()
            sources = {
                **self.directories,
                **{
                    name + "_dir": self.output / path
                    for name, path in (
                        ("gas_comparison", "gas-comparison"),
                        ("gas_selection", "gas-selection"),
                        ("temperature", "temperature"),
                        ("greenhouse", "greenhouse"),
                        ("phase", "phase"),
                        ("choice", "choice"),
                        ("save", "save"),
                    )
                },
            }
            result = verify_terrestrial_workflow(
                self.page,
                self.config,
                self.output / "workflow",
                run_history=self.book.history,
                **sources,
                **source_options(self.supplied_inputs),
            )
            self.workflow_receipt = self._persisted(self.output / "workflow", "confirmed.json", result)
            validate_autosave_workflow_flags(result, enabled=self.save_strategy == "autosave")
            _require(
                result.get("mode") == terrestrial_mode(self.supplied_inputs)
                and result.get("task_completed") is True
                and result.get("authority") == "visible_workflow_readback"
                and _identity_matches(result.get("star"), self.bundle["star"]),
                "workflow_not_verified",
            )
            self._workflow_sha = hashlib.sha256(
                self.book.read(self.output / "workflow/confirmed.json")
            ).hexdigest()
            self._check()
            self._finish("verified_terrestrial", True)
        else:
            raise BrowserSafetyStop("terrestrial_steps_unknown_phase")

    def _autosave_readback(self):
        return _terrestrial_autosave(
            self.page,
            self.config,
            self.output / "save",
            history=self.book.history,
            phase_dir=self.output / "phase",
            choice_dir=self.output / "choice",
            star=self.bundle["star"],
            check=self._check,
            timeout_seconds=self.timeout,
        )

    def _temperature_done(self):
        self._check()
        if self.component.finished:
            child = self.component.state()
            _require(
                child.get("equilibrium_transport_verified") is True
                and child.get("task_completed") is False
                and child.get("sources_unchanged") is True
                and child.get("checkpoint_unchanged") is True
                and type(child.get("optimizer_updates")) is int
                and child["optimizer_updates"] == 0,
                "temperature_not_verified",
            )
            self._pin(self.output / "temperature")
            self._temperature, self._expected = _temperature_source(
                self.book,
                self.output / "temperature",
                self._gas_after,
                **(
                    {"supplied_inputs": True, "stellar": self.bundle, "planet": self.bundle}
                    if self.supplied_inputs
                    else {}
                ),
            )
            self._relay.retire()
            self.phase = "greenhouse"

    def _save_cancelled(self):
        self._check()
        return False

    def _stop(self, exc):
        if self.finished:
            return
        self.failure = (
            str(exc)
            if isinstance(exc, BrowserSafetyStop) and re.fullmatch(r"[a-z][a-z0-9_:.\-]{0,180}", str(exc))
            else "terrestrial_steps_operation_failed"
        )
        try:
            if self.component and not self.component.finished:
                self._relay.retire()
                self.component.abort()
        except Exception:  # noqa: BLE001 - cleanup cannot turn failure into completion
            self.cleanup_failed = True
        try:
            self._emit("error", {"reason": self.failure})
        finally:
            self._finish(self.failure, False)

    def _finish(self, outcome, verified):
        if self.finished:
            return
        self.finished, self._finalizing, self._verified, self.phase = True, True, verified, "finished"
        try:
            self._emit("episode_summary", {**self.state(), "outcome": outcome})
            if verified:
                try:
                    self._check()
                except Exception:  # noqa: BLE001 - late source/callback changes invalidate owner success
                    self._verified, self.failure = False, "terrestrial_steps_finalization_invalidated"
                    outcome = self.failure
                    self._emit("error", {"reason": outcome})
                    self._emit("episode_summary", {**self.state(), "outcome": outcome})
            self._relay.retire()
            self._stream.close()
            self.report = {
                "schema_version": 1,
                "mode": MODE,
                **self.state(),
                "outcome": outcome,
                "source_sha256": dict(self.book.hashes),
                "adapter_calls_reserved": list(self.calls),
                "events_sha256": file_hash(self.output / "events.jsonl"),
            }
            persist_json(self.output / "report.json", self.report)
            if not self._verified:
                persist_json(
                    self.output / "stopped.json",
                    {"reason": outcome, "automatic_retry": False, "task_completed": False},
                )
        except BaseException:
            self._verified, self.failure = False, "terrestrial_steps_artifact_finalization_failed"
            self.report = {**self.state(), "artifact_finalization_failed": True}
            self._stream.close()
            try:
                persist_json(self.output / "finalization_failed.json", self.report)
            except OSError:
                pass
            raise
        finally:
            self._finalizing = False

    def abort(self):
        if not self.finished or self._finalizing:
            self._cancelled = True
            if not self._busy and not self._finalizing:
                self._stop(BrowserSafetyStop("terrestrial_steps_operator_aborted"))
        return self.state()

    def close(self):
        return self.abort()
