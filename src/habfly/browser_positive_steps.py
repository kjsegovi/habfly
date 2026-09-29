"""Bounded positive-planet handoff, ending before classification or Save.

Presence selection is one bounded synchronous advance; cancellation queued
during it takes effect when it returns. Raw and learned children retain their
native pre-fill cancellation boundaries. The caller owns browser lifetime.
"""

import json
from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_planet_numeric import PlanetNumericSession
from .browser_planet_policy import require_final_gate
from .browser_planet_steps import PlanetDerivedSteps
from .browser_raster_planet_evidence import (
    MODE,
    RAW,
    _blank_current,
    _evidence,
    _Owned,
    _presence,
    _reload,
    _same_star,
    _sha,
    select_raster_detected_planet,
)
from .browser_raster_steps import RasterInputSteps
from .contracts import RuntimeEvent
from .environments.planet_calculations import FIELDS, SCOPE
from .model import graph_fingerprint
from .planet_knowledge import PlanetCalculator
from .training.checkpoints import load_checkpoint
from .training.planet_sequence import file_hash, read

SUPPLIED_OWNER_MODE = "positive_planet_supplied_inputs_to_frozen_derived_v1"


class _CancelledAdvance(Exception):
    pass


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("positive_planet_" + reason)


def _frozen_options(pilot, final_evaluation, graph):
    checkpoint = Path(pilot) / "training/checkpoint.pt"
    paths = {
        "checkpoint": checkpoint,
        "metadata": checkpoint.with_suffix(".pt.json"),
        "gate": Path(final_evaluation) / "report.json",
    }
    hashes = {name: file_hash(path) for name, path in paths.items()}
    require_final_gate(read(paths["gate"]), hashes["checkpoint"])
    content = read(paths["metadata"])["provenance"]["planet_calculations"]
    pack_hash = PlanetCalculator().pack.checksum
    _require(content["scope"] == SCOPE and content["knowledge_pack_hash"] == pack_hash, "checkpoint_scope")
    policy, manifest = load_checkpoint(checkpoint, graph, content_pack=content)
    _require(
        len(graph.body_ids) == 2000
        and policy.hidden_size == 16
        and policy.observation_encoding == "structured_planet_tool_v1",
        "checkpoint_contract",
    )
    _require(graph_fingerprint(graph) == manifest["graph_hash"], "checkpoint_graph")
    _require(all(file_hash(path) == hashes[name] for name, path in paths.items()), "model_source_changed")
    return paths, {**hashes, "graph": manifest["graph_hash"], "knowledge_pack": pack_hash}


class PositivePlanetSteps:
    """Presence → three reference copies → frozen four-derived-field policy.

    No recovery, classification, Save, habitability, assessment, or submission.
    ``emit(kind, payload)`` matches the outer project scheduler. Child envelopes
    are retained byte-equivalently as objects under ``component_event``; their
    summaries become outer state events, never outer episode completion.
    """

    def __init__(
        self,
        page,
        config,
        output,
        *,
        run_history,
        star,
        window_report,
        window_report_sha256,
        spectrum_path,
        spectrum_sha256,
        pilot,
        final_evaluation,
        graph,
        supplied_star_class,
        supplied_evaluation=None,
        supplied_stellar_sources=None,
        seed=15000000,
        preserve_painted_class=None,
        presence_max_seconds=600,
        inputs_max_seconds=900,
        tooltip_reference=None,
        emit=lambda *_: None,
    ):
        _require(callable(emit) and type(seed) is int, "invalid_runtime_options")
        self._supplied_inputs = supplied_evaluation is not None
        if self._supplied_inputs:
            from .supplied_browser_modes import NON_MAIN_CLASSES

            _require(supplied_star_class in NON_MAIN_CLASSES, "unsupported_supplied_class")
            _require(
                type(supplied_stellar_sources) is dict
                and set(supplied_stellar_sources)
                == {"numeric_dir", "color_dir", "class_dir", "navigation_dir"},
                "supplied_stellar_sources_required",
            )
        else:
            _require(
                supplied_star_class == "main_sequence" and supplied_stellar_sources is None,
                "unsupported_supplied_class",
            )
        _require(preserve_painted_class in {None, "gas_giant", "ice_giant", "terrestrial"}, "invalid_paint")
        for limit in (presence_max_seconds, inputs_max_seconds):
            _require(type(limit) in {int, float} and 30 <= limit <= 900, "invalid_time_budget")
        self.owner = _Owned(run_history)
        try:
            evidence = _evidence(
                self.owner,
                window_report,
                window_report_sha256,
                spectrum_path,
                spectrum_sha256,
                **({"tooltip_reference": tooltip_reference} if tooltip_reference is not None else {}),
            )
            _require(_same_star(star, evidence["star"]), "source_star_changed")
            if self._supplied_inputs:
                self._prepare_supplied(
                    pilot, supplied_evaluation, graph, supplied_stellar_sources, star, supplied_star_class
                )
            else:
                self._model_paths, self._model_hashes = _frozen_options(pilot, final_evaluation, graph)
        except BrowserSafetyStop:
            raise
        except Exception:  # noqa: BLE001 - loader/source details may contain private paths
            raise BrowserSafetyStop("positive_planet_preflight_failed") from None
        self.page, self.config = page, config.model_copy(deep=True)
        self.output = self.owner.output(output)
        self.star, self.evidence, self.graph = star, evidence, graph
        self._callback, self._sequence = emit, 0
        self._run_id = "positive-planet-" + _sha(self.owner.relative(self.output).encode())[:16]
        self._stream = (self.output / "events.jsonl").open("x")
        self.phase, self.failure, self.component = "presence", None, None
        self.finished, self.report = False, None
        self._advancing = self._stopping = self._forward_failed = self._presence_entered = False
        self._component_kind = None
        self._child_hashes, self._completed = {}, {}
        self._presence_sha256 = None
        self._presence_options = {
            "window_report": self.owner.path(window_report),
            "window_report_sha256": window_report_sha256,
            "spectrum_path": self.owner.path(spectrum_path),
            "spectrum_sha256": spectrum_sha256,
            "preserve_painted_class": preserve_painted_class,
            "max_seconds": presence_max_seconds,
        }
        if tooltip_reference is not None:
            self._presence_options["tooltip_reference"] = deepcopy(tooltip_reference)
        self._inputs_max_seconds = inputs_max_seconds
        self._derived_options = {
            "pilot": Path(pilot),
            "final_evaluation": Path(supplied_evaluation if self._supplied_inputs else final_evaluation),
            "graph": graph,
            "supplied_star_class": supplied_star_class,
            "seed": seed,
        }
        self.scope = {
            "mode": SUPPLIED_OWNER_MODE
            if self._supplied_inputs
            else "positive_planet_reference_to_frozen_derived",
            "star": star,
            "measurement_mode": self.evidence.get("mode", MODE),
            "learned_perception": False,
            "scientific_verified": False,
            "training_label": False,
            "task_completed": False,
            "project_completed": False,
            "classification_source": "supplied_not_learned",
            "supplied_star_class": supplied_star_class,
            "class_writes": 0,
            "save_clicks": 0,
            "habitability_writes": 0,
            "assessment_clicks": 0,
            "submission_clicks": 0,
            "optimizer_updates": 0,
            "automatic_retry": False,
            "cancellation": "between_bounded_presence_stage_and_child_decisions",
            "model_source_hashes": self._model_hashes,
        }
        try:
            persist_json(self.output / "scope.json", self.scope)
            self._emit("hello", {"protocol_version": 1, **self.scope})
            self._check()
            session = PlanetNumericSession(
                page,
                self.config,
                self.output / "preflight",
                max_seconds=presence_max_seconds,
                _allow_unset_planet=True,
            )
            try:
                self._pinned(session.mapping)
                _blank_current(
                    session.mapping,
                    session.choices,
                    self.evidence,
                    presence=None,
                    painted=preserve_painted_class,
                )
            finally:
                session.close()
            self._check()
            self._emit("state", self.state())
            self._check()
        except (KeyboardInterrupt, SystemExit):
            self._stop("operator_aborted", aborted=True)
            raise
        except Exception as exc:  # noqa: BLE001 - durable sanitized startup failure
            self._stop(self._reason(exc))

    @staticmethod
    def _reason(exc):
        return str(exc) if isinstance(exc, BrowserSafetyStop) else "positive_planet_component_failed"

    def _prepare_supplied(self, pilot, evaluation, graph, sources, star, actual_class):
        from .browser_no_planet_workflow import _Evidence
        from .browser_stellar_sources import load_stellar_sources
        from .browser_supplied_provenance import (
            archive_supplied_input_transfer_gate,
            load_archived_supplied_input_transfer_gate,
        )
        from .browser_supplied_steps import load_supplied_frozen_policy
        from .planet_supplied_inputs import adapter_manifest, load_supplied_planet_pack
        from .planet_supplied_stellar_source import _navigation

        _, manifest, gate = load_supplied_frozen_policy("planet", pilot, evaluation, graph)
        self._supplied_book = _Evidence(self.owner.history)
        self._stellar_source_dirs = {key: self.owner.path(path) for key, path in sources.items()}
        stellar = load_stellar_sources(
            self._supplied_book,
            **{key: path for key, path in self._stellar_source_dirs.items() if key != "navigation_dir"},
        )
        _require(
            stellar["bundle"]["class"] == actual_class and _same_star(stellar["bundle"]["star"], star),
            "supplied_stellar_identity_changed",
        )
        self._supplied_initial_inputs = _navigation(
            self._supplied_book, self._stellar_source_dirs["navigation_dir"], stellar["bundle"]
        )
        self._supplied_gate_link = archive_supplied_input_transfer_gate(
            self.owner.history, gate, task="planet"
        )
        load_archived_supplied_input_transfer_gate(
            self._supplied_book,
            self._supplied_gate_link,
            task="planet",
            checkpoint_sha256=gate["identity"]["checkpoint_sha256"],
            graph_hash=manifest["graph_hash"],
            pack_hash=load_supplied_planet_pack().checksum,
            adapter_sha256=adapter_manifest()["sha256"],
        )
        checkpoint = Path(pilot) / "training/checkpoint.pt"
        self._model_paths = {
            "checkpoint": checkpoint,
            "metadata": checkpoint.with_suffix(".pt.json"),
            "gate": Path(evaluation) / "report.json",
        }
        self._model_hashes = {
            **{name: file_hash(path) for name, path in self._model_paths.items()},
            "graph": manifest["graph_hash"],
            "knowledge_pack": load_supplied_planet_pack().checksum,
        }
        self._supplied_pins = {**gate["source_sha256"], **gate["artifact_sha256"]}
        self._supplied_receipt_link = None

    def _prepare_derived_supplied_inputs(self):
        from .planet_supplied_inputs import adapter_manifest, load_supplied_planet_pack
        from .planet_supplied_stellar_source import build_supplied_planet_stellar_inputs

        _require(self._supplied_receipt_link is None, "supplied_receipt_already_created")
        capture = self.output / "raw/native-copies/copy-03-after"
        receipt = build_supplied_planet_stellar_inputs(
            self.owner.history,
            **self._stellar_source_dirs,
            raw_dir=self.output / "raw",
            current_capture_dir=capture,
            current_capture_sha256=_sha(self.owner.read(capture / "observation.json")),
            expected_star=self.star,
            selected_class=self._derived_options["supplied_star_class"],
            expected_pack_hash=load_supplied_planet_pack().checksum,
            expected_adapter_sha256=adapter_manifest()["sha256"],
        )
        # This sibling is outside every closed raw/navigation/stellar source tree.
        directory = self.output / "supplied-stellar-inputs"
        directory.mkdir(exist_ok=False)
        path = directory / "receipt.json"
        persist_json(path, receipt)
        self._supplied_receipt_link = {
            "path": self.owner.relative(path),
            "sha256": _sha(self.owner.read(path)),
        }
        return {
            "run_history": self.owner.history,
            "expected_star": self.star,
            "supplied_stellar_inputs_path": path,
            "supplied_stellar_inputs_sha256": self._supplied_receipt_link["sha256"],
        }

    def _reference(self):
        values = self.evidence["measurements"]
        mode = self.evidence.get("mode", MODE)
        if mode != MODE:
            from .browser_raster_planet_evidence import TOOLTIP_MODES, TWO_TOOLTIP_MODE, _two_tooltip_metadata

            _require(mode in TOOLTIP_MODES, "unsupported_measurement_mode")
            two = mode == TWO_TOOLTIP_MODE
            return {
                "mode": mode,
                "star": self.star,
                "observation_limit_days": 5000,
                "period_days": {
                    "value": float(values["period_days"]["value"]),
                    "unit": "days",
                    "compatibility_interval": {
                        "lower": float(values["period_days"]["compatibility_interval"]["lower"]),
                        "upper": float(values["period_days"]["compatibility_interval"]["upper"]),
                        "endpoints": "open",
                    },
                    "interpretation": (
                        "one assumed-consecutive spacing; no confirmed recurrence or physical uncertainty"
                        if two
                        else "observed recurrence; bracket compatibility, not physical uncertainty"
                    ),
                    **({"estimate_kind": "single_spacing"} if two else {}),
                },
                "brightness_drop_percent": {
                    "value": float(values["brightness_drop"]["value"]),
                    "unit": "percent",
                    "physical_bounds": None,
                    "interpretation": "maximum sampled decline, not resolved transit minimum",
                },
                "line_shift": {"value": float(values["line_shift"]["value"]), "unit": "nm"},
                "learned_perception": False,
                "scientific_verified": False,
                "training_label": False,
                "period_evidence_verified": False,
                "minimum_depth_verified": False,
                **(_two_tooltip_metadata(self.evidence["tooltip_measurements"]) if two else {}),
            }
        return {
            "mode": MODE,
            "star": self.star,
            "observation_limit_days": 5000,
            "period_days": {
                **{key: float(values["period_days"][key]) for key in ("value", "lower", "upper")},
                "unit": "days",
            },
            "brightness_drop_percent": {
                **{key: float(values["brightness_drop"][key]) for key in ("value", "lower", "upper")},
                "unit": "percent",
            },
            "line_shift": {"value": float(values["line_shift"]["value"]), "unit": "nm"},
            "learned_perception": False,
            "scientific_verified": False,
            "training_label": False,
        }

    def state(self):
        return {
            **deepcopy(self.scope),
            "phase": self.phase,
            "stage": self.phase,
            "finished": self.finished,
            "failure_reason": self.failure,
            "event_forwarding_failed": self._forward_failed,
            "component": self.component.state() if self.component else None,
            "reference_measurements": None if self.failure else self._reference(),
            "presence_stage_entered": self._presence_entered,
            "native_actions_may_have_occurred": self._presence_entered,
            "derived_transport_verified": self.phase == "planet_classification_required",
            "child_artifacts": deepcopy(self._completed),
        }

    def _tree(self, directory):
        paths = sorted(directory.rglob("*"))
        hashes, total = {}, 0
        for path in paths:
            self.owner.path(path)  # rejects symlinks even for a directory
            if path.is_file():
                raw = self.owner.read(path, limit=16 * 1024 * 1024)
                total += len(raw)
                _require(total <= 64 * 1024 * 1024, "child_artifacts_oversized")
                hashes[self.owner.relative(path)] = _sha(raw)
        return hashes

    def _check(self):
        if self.finished:
            raise _CancelledAdvance
        _require(not self._forward_failed, "event_forwarding_failed")
        _reload(self.owner, self.evidence)
        _require(_same_star(self.star, self.evidence["star"]), "source_star_changed")
        if self._supplied_inputs:
            from .planet_supplied_inputs import load_supplied_planet_pack

            pack_hash = load_supplied_planet_pack().checksum
            self._supplied_book.unchanged()
            _require(
                all(file_hash(path) == digest for path, digest in self._supplied_pins.items()),
                "supplied_sources_changed",
            )
            if self._supplied_receipt_link is not None:
                _require(
                    _sha(self.owner.read(self.owner.history / self._supplied_receipt_link["path"]))
                    == self._supplied_receipt_link["sha256"],
                    "supplied_receipt_changed",
                )
        else:
            pack_hash = PlanetCalculator().pack.checksum
        _require(
            all(file_hash(path) == self._model_hashes[name] for name, path in self._model_paths.items())
            and graph_fingerprint(self.graph) == self._model_hashes["graph"]
            and pack_hash == self._model_hashes["knowledge_pack"],
            "model_source_changed",
        )
        for kind, expected in self._child_hashes.items():
            _require(self._tree(self.output / kind) == expected, "completed_child_changed")
        if self.component is not None and getattr(self.component, "session", None) is not None:
            self._pinned(self.component.session.mapping)

    def _pinned(self, mapping):
        _require(_same_star(mapping.get("star_name"), self.star), "current_star_changed")
        _require(_same_star(mapping.get("star_name"), self.evidence["star"]), "current_source_changed")
        if self._supplied_inputs:
            from .planet_supplied_stellar_source import _inputs

            _require(
                _inputs(mapping, self.star) == self._supplied_initial_inputs,
                "current_supplied_stellar_inputs_changed",
            )

    def _presence_dispatch_guard(self, mapping):
        self._check()
        self._pinned(mapping)
        self._check()

    def _emit(self, event, payload):
        line = RuntimeEvent(
            event=event, payload=payload, run_id=self._run_id, sequence=self._sequence
        ).model_dump_json()
        self._stream.write(line + "\n")
        self._stream.flush()
        self._sequence += 1
        if not self._forward_failed:
            try:
                self._callback(event, json.loads(line)["payload"])
            except Exception:  # noqa: BLE001 - callback details can include credentials
                self._forward_failed = True
                if not self._stopping:
                    raise BrowserSafetyStop("positive_planet_event_forwarding_failed") from None

    def _forward(self, item):
        if self.finished and not self._stopping:
            raise BrowserSafetyStop("positive_planet_parent_stopped")
        if not self._stopping:
            self._check()
        original = deepcopy(item)
        payload = {
            **deepcopy(item["payload"]),
            "component_identity": self._component_kind,
            "component_event": original,
        }
        event = item["event"]
        if event in {"hello", "state", "episode_summary"}:
            payload = {
                "stage": self.phase,
                "star": self.star,
                "component_identity": self._component_kind,
                "component_event": original,
                "component_summary": deepcopy(item["payload"]),
                "reference_measurements": None if self.failure else self._reference(),
            }
            event = "state"
        self._emit(event, payload)
        if self.finished and not self._stopping:
            raise BrowserSafetyStop("positive_planet_parent_stopped")
        if not self._stopping:
            self._check()

    def _record(self, kind, report):
        path = self.output / kind / ("confirmed.json" if kind == "presence" else "report.json")
        _require(self.owner.json(path) == report, "child_report_changed")
        tree = self._tree(self.output / kind)
        self._child_hashes[kind] = tree
        self._completed[kind] = {
            "path": self.owner.relative(path),
            "sha256": _sha(self.owner.read(path)),
            "files": len(tree),
        }
        persist_json(self.output / f"{kind}-completed.json", {"report": self._completed[kind], "files": tree})

    def advance(self):
        if self.finished:
            return self.state()
        if self._advancing:
            self._stop("positive_planet_reentrant_advance")
            raise BrowserSafetyStop("positive_planet_reentrant_advance")
        self._advancing = True
        try:
            self._check()
            if self.phase == "presence":
                self._emit("state", {**self.state(), "stage": "selecting_reference_presence"})
                self._check()
                self._presence_entered = True
                receipt = select_raster_detected_planet(
                    self.page,
                    self.config,
                    self.output / "presence",
                    run_history=self.owner.history,
                    **self._presence_options,
                    **({"before_dispatch": self._presence_dispatch_guard} if self._supplied_inputs else {}),
                )
                self._check()
                path = self.output / "presence/confirmed.json"
                self._presence_sha256 = _sha(self.owner.read(path))
                verified, _, mapping = _presence(self.owner, path, self._presence_sha256)
                self._pinned(mapping)
                _require(receipt == verified, "presence_receipt_changed")
                self._record("presence", receipt)
                self.phase = "raw"
            elif self.phase in {"raw", "derived"}:
                kind = self.phase
                if self.component is None:
                    self._component_kind = kind
                    if kind == "raw":
                        child = RasterInputSteps(
                            self.page,
                            self.config,
                            self.output / "raw",
                            run_history=self.owner.history,
                            presence_path=self.output / "presence/confirmed.json",
                            presence_sha256=self._presence_sha256,
                            max_seconds=self._inputs_max_seconds,
                            emit=self._forward,
                        )
                    else:
                        factory, additional = PlanetDerivedSteps, {}
                        if self._supplied_inputs:
                            from .browser_supplied_steps import SuppliedPlanetDerivedSteps

                            factory = SuppliedPlanetDerivedSteps
                            additional = self._prepare_derived_supplied_inputs()
                        child = factory(
                            self.page,
                            self.config,
                            self.output / "derived",
                            **self._derived_options,
                            **additional,
                            emit=self._forward,
                        )
                    self.component = child
                    if self.finished:
                        child.abort()
                        return self.state()
                    self._check()
                    if child.session is not None:
                        self._pinned(child.session.mapping)
                    if kind == "derived" and not child.finished:
                        fields = child.session.mapping["observation"]["values"]["browser_field_map"]
                        _require(
                            all(
                                fields[name]["current_value"] == self.evidence["measurements"][name]["value"]
                                and fields[name]["unit"] == self.evidence["measurements"][name]["unit"]
                                for name in RAW
                            ),
                            "raw_current_source_changed",
                        )
                else:
                    self.component.advance()
                    if self.finished:
                        self.component.abort()
                        return self.state()
                self._check()
                if self.component.finished:
                    report = self.component.report
                    flag = (
                        "raw_measurement_transport_verified" if kind == "raw" else "planet_transport_verified"
                    )
                    _require(
                        report and report.get(flag) is True and report.get("task_completed") is False,
                        kind + "_transport_failed",
                    )
                    _require(
                        set(report.get("verified_fields", {})) == set(RAW if kind == "raw" else FIELDS),
                        kind + "_field_set_changed",
                    )
                    self._record(kind, report)
                    self.component.close()
                    self.component = None
                    self.phase = "derived" if kind == "raw" else "planet_classification_required"
            else:
                raise BrowserSafetyStop("positive_planet_unknown_phase")
            self._check()
            if self.phase == "planet_classification_required":
                self._finish_handoff()
            else:
                self._emit("state", self.state())
                self._check()
        except _CancelledAdvance:
            pass
        except (KeyboardInterrupt, SystemExit):
            self._stop("operator_aborted", aborted=True)
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize child/driver failures
            if not self.finished:
                self._stop(self._reason(exc))
        finally:
            self._advancing = False
        return self.state()

    def _finish_handoff(self):
        report = self.state()
        report["finished"] = True
        self._emit("episode_summary", report)
        self._check()
        persist_json(self.output / "report.json", report)
        self._stream.close()
        self.report, self.finished = report, True

    def _stop(self, reason, *, aborted=False):
        if self.finished:
            return self.state()
        self.finished, self._stopping = True, True
        self.phase, self.failure = "aborted" if aborted else "stopped", reason
        try:
            if self.component is not None:
                self.component.abort()
            self.report = self.state()
            persist_json(self.output / "stopped.json", self.report)
            self._emit("error", {"reason": reason})
            self._emit("episode_summary", self.report)
            persist_json(self.output / "report.json", self.report)
        except Exception:  # noqa: BLE001 - no artifact/callback failure may resume dispatch
            self.report = {**self.state(), "artifact_finalization_failed": True}
        finally:
            self._stream.close()
            self._stopping = False
        return self.state()

    def abort(self):
        return self._stop("operator_aborted", aborted=True)

    def close(self):
        return self.abort()
