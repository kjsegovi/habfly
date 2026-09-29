"""Explicit supplied-class transfer, with original weights and owned evidence.

These controllers never promote a legacy main-sequence gate. The new frozen
transfer gate and the visible stellar-input receipt are required independently.
Only the model view omits the class feature; tools and journals keep the actual
class. The existing native primitives retain their cancellation/write budgets.
"""

from pathlib import Path

import torch

from . import habitability_supplied_inputs as temperature
from . import planet_supplied_inputs as planet
from .browser import BrowserSafetyStop
from .browser_habitability_steps import HabitabilityTemperatureSteps
from .browser_no_planet_workflow import _Evidence
from .browser_planet_steps import PlanetDerivedSteps
from .browser_supplied_provenance import (
    _adopt_receipt,
    archive_supplied_input_transfer_gate,
    load_archived_supplied_input_transfer_gate,
)
from .browser_supplied_tool_envs import SuppliedPlanetBrowserToolEnv, SuppliedTemperatureBrowserToolEnv
from .model import graph_fingerprint
from .planet_supplied_stellar_source import matches_mapping
from .supplied_browser_modes import DERIVED_SUPPLIED_MODE, NON_MAIN_CLASSES, TEMPERATURE_SUPPLIED_MODE
from .training.checkpoints import load_checkpoint, source_hash
from .training.planet_sequence import file_hash, read
from .training.supplied_input_transfer import require_supplied_input_transfer_gate


def _require(condition, reason):
    if not condition:
        raise BrowserSafetyStop("supplied_transfer_" + reason)


def _module(task):
    _require(task in {"planet", "temperature"}, "unsupported_task")
    return planet if task == "planet" else temperature


def _pack(task):
    return (
        planet.load_supplied_planet_pack()
        if task == "planet"
        else temperature.load_supplied_temperature_pack()
    )


def load_supplied_frozen_policy(task, pilot, final_evaluation, graph):
    """Validate the new recorded gate, then load only its original frozen weights.

    No gate cases are regenerated, no policy/expert runs, and no browser starts.
    Gate paths are explicit local configuration, not policy observations.
    """
    module = _module(task)
    checkpoint = Path(pilot) / "training/checkpoint.pt"
    metadata = checkpoint.with_suffix(".pt.json")
    checkpoint_hash, metadata_hash = file_hash(checkpoint), file_hash(metadata)
    content = read(metadata)["provenance"][
        "planet_calculations" if task == "planet" else "habitability_calculations"
    ]
    _require(
        content["scope"] == module.LEGACY_SCOPE and content["knowledge_pack_hash"] == module.BASE_PACK_HASH,
        "original_content_changed",
    )
    pack, adapter = _pack(task), module.adapter_manifest()
    gate = require_supplied_input_transfer_gate(
        final_evaluation,
        task=task,
        checkpoint_sha256=checkpoint_hash,
        graph_hash=graph_fingerprint(graph),
        pack_hash=pack.checksum,
        adapter_sha256=adapter["sha256"],
    )
    identity = gate["identity"]
    _require(
        gate["current_sources_verified"] is True
        and gate["transfer_gate_passed"] is True
        and identity["parent_metadata_sha256"] == metadata_hash
        and identity["parent_content_hash"] == source_hash(content)
        and identity["parent_scope"] == module.LEGACY_SCOPE
        and identity["parent_pack_hash"] == module.BASE_PACK_HASH,
        "gate_original_identity_changed",
    )
    torch.set_num_threads(1)
    policy, manifest = load_checkpoint(checkpoint, graph, content_pack=content)
    _require(
        len(graph.body_ids) == 2000
        and policy.hidden_size == 16
        and policy.selection_mode == "measurement_result_v3"
        and policy.observation_encoding
        == ("structured_planet_tool_v1" if task == "planet" else "structured_habitability_tool_v1")
        and manifest["graph_hash"] == identity["graph_hash"]
        and manifest["content_pack_hash"] == identity["parent_content_hash"]
        and file_hash(checkpoint) == checkpoint_hash
        and file_hash(metadata) == metadata_hash,
        "model_contract_changed",
    )
    policy.requires_grad_(False)
    policy.eval()
    policy.action_temperature = policy.target_temperature = 1.0
    policy.calibration = {"status": "uncalibrated", "scope": "supplied_input_browser_transfer"}
    return policy, manifest, gate


class _SuppliedSources:
    def _configure_supplied(
        self,
        *,
        run_history,
        expected_star,
        supplied_star_class,
        supplied_stellar_inputs_path,
        supplied_stellar_inputs_sha256,
    ):
        _require(
            type(supplied_star_class) is str and supplied_star_class in NON_MAIN_CLASSES,
            "explicit_non_main_class_required",
        )
        _require(type(expected_star) is str and bool(expected_star.strip()), "expected_star_required")
        self._supplied_book = _Evidence(run_history)
        path = Path(supplied_stellar_inputs_path)
        path = self._supplied_book.path(path if path.is_absolute() else self._supplied_book.history / path)
        self._supplied_link = {
            "path": str(path.relative_to(self._supplied_book.history)),
            "sha256": supplied_stellar_inputs_sha256,
        }
        self._actual_class, self._expected_star = supplied_star_class, expected_star
        self._supplied_source_pins = {}

    def _load_supplied(self, pilot, graph):
        self.policy, manifest, gate = load_supplied_frozen_policy(
            self._task, pilot, self._source_paths["final_gate_report_sha256"].parent, graph
        )
        self._graph_hash, self._pack_hash = manifest["graph_hash"], _pack(self._task).checksum
        self._adapter_hash = _module(self._task).adapter_manifest()["sha256"]
        self._supplied_receipt = _adopt_receipt(self._supplied_book, self._supplied_link, self._actual_class)
        _require(
            self._supplied_receipt["star"].casefold() == self._expected_star.casefold(),
            "receipt_star_changed",
        )
        self._supplied_gate_link = archive_supplied_input_transfer_gate(
            self._supplied_book.history, gate, task=self._task
        )
        load_archived_supplied_input_transfer_gate(
            self._supplied_book,
            self._supplied_gate_link,
            task=self._task,
            checkpoint_sha256=self._source_hashes["checkpoint_sha256"],
            graph_hash=self._graph_hash,
            pack_hash=self._pack_hash,
            adapter_sha256=self._adapter_hash,
        )
        self._supplied_identity = gate["identity"]
        # Keep private case/trace paths out of runtime event provenance. Current
        # trusted source bytes and owned proof bytes are rechecked at boundaries.
        self._supplied_source_pins = {**gate["source_sha256"], **gate["artifact_sha256"]}
        package = Path(__file__).parent
        for name in (
            "browser_supplied_steps.py",
            "browser_supplied_tool_envs.py",
            "browser_supplied_provenance.py",
            "planet_supplied_stellar_source.py",
            "browser_stellar_sources.py",
            "supplied_browser_modes.py",
            "browser_planet_steps.py",
            "browser_planet_policy.py",
            "browser_planet_numeric.py",
            "browser_habitability_steps.py",
            "browser_habitability_policy.py",
            "browser_habitability_numeric.py",
            "browser.py",
            "browser_probe.py",
            "browser_raster_planet_evidence.py",
            "browser_raster_steps.py",
        ):
            path = package / name
            self._supplied_source_pins[str(path.resolve())] = file_hash(path)

    def _check_sources(self):
        super()._check_sources()
        try:
            self._supplied_book.unchanged()
            _require(
                _module(self._task).adapter_manifest()["sha256"] == self._adapter_hash
                and all(file_hash(path) == digest for path, digest in self._supplied_source_pins.items()),
                "source_changed",
            )
        except Exception:  # noqa: BLE001 - no source path or private data in failure logs
            self._sources_unchanged = False
            raise BrowserSafetyStop("supplied_transfer_source_changed") from None

    def _current_pack_hash(self):
        return _pack(self._task).checksum

    def _additional_provenance(self):
        identity = self._supplied_identity
        return {
            "supplied_stellar_inputs": dict(self._supplied_link),
            "supplied_input_transfer_gate": dict(self._supplied_gate_link),
            "supplied_star_class": self._actual_class,
            "classification_source": "supplied_not_learned",
            "supplied_input_adapter_sha256": self._adapter_hash,
            "original_checkpoint_content_hash": identity["parent_content_hash"],
            "original_knowledge_pack_hash": identity["parent_pack_hash"],
            "original_scope": identity["parent_scope"],
        }

    def _inference_observation(self, observation):
        return _module(self._task).inference_view(
            observation, expected_adapter_sha256=self._adapter_hash, expected_pack_hash=self._pack_hash
        )


class SuppliedPlanetDerivedSteps(_SuppliedSources, PlanetDerivedSteps):
    _task = "planet"
    _report_scope = DERIVED_SUPPLIED_MODE

    def __init__(
        self,
        *args,
        run_history,
        expected_star,
        supplied_star_class,
        supplied_stellar_inputs_path,
        supplied_stellar_inputs_sha256,
        **kwargs,
    ):
        self._configure_supplied(
            run_history=run_history,
            expected_star=expected_star,
            supplied_star_class=supplied_star_class,
            supplied_stellar_inputs_path=supplied_stellar_inputs_path,
            supplied_stellar_inputs_sha256=supplied_stellar_inputs_sha256,
        )
        super().__init__(*args, supplied_star_class=supplied_star_class, **kwargs)

    def _load_model(self, graph):
        self._load_supplied(self.checkpoint.parent.parent, graph)

    def _environment(self, *, supplied_star_class, seed):
        _require(
            supplied_star_class == self._actual_class
            and matches_mapping(self._supplied_receipt, self.session.mapping),
            "current_stellar_inputs_changed",
        )
        return SuppliedPlanetBrowserToolEnv(self.session, supplied_star_class=self._actual_class, seed=seed)


class SuppliedHabitabilityTemperatureSteps(_SuppliedSources, HabitabilityTemperatureSteps):
    _task = "temperature"
    _report_scope = TEMPERATURE_SUPPLIED_MODE

    def __init__(
        self,
        *args,
        run_history,
        expected_star,
        supplied_star_class,
        supplied_stellar_inputs_path,
        supplied_stellar_inputs_sha256,
        **kwargs,
    ):
        self._configure_supplied(
            run_history=run_history,
            expected_star=expected_star,
            supplied_star_class=supplied_star_class,
            supplied_stellar_inputs_path=supplied_stellar_inputs_path,
            supplied_stellar_inputs_sha256=supplied_stellar_inputs_sha256,
        )
        super().__init__(*args, **kwargs)

    def _load_model(self, pilot, graph):
        self._load_supplied(pilot, graph)

    def _environment(self, *, supplied_greenhouse_increment, seed):
        _require(
            self.session.mapping.get("star_name", "").casefold() == self._expected_star.casefold(),
            "current_temperature_star_changed",
        )
        return SuppliedTemperatureBrowserToolEnv(
            self.session.mapping,
            supplied_star_class=self._actual_class,
            supplied_greenhouse_increment=supplied_greenhouse_increment,
            seed=seed,
        )
