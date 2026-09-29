"""Offline atomic ingestion of verified terrestrial visible-workflow evidence.

No browser actions, new live inspection, scientific labels, historical Save
receipts, assessments, or submissions are synthesized. The existing journal
schema and canonical identities are shared with the No and gas/ice importers.
"""

import fcntl
import os

from .browser_autosave import (
    ACKNOWLEDGEMENT_SOURCE as AUTOSAVE_ACKNOWLEDGEMENT_SOURCE,
)
from .browser_autosave import (
    MODE as AUTOSAVE_MODE,
)
from .browser_autosave import (
    validate_autosave_workflow_flags,
)
from .browser_collected_revisit import parse_stellar_row, verify_detail_row
from .browser_habitability_save import MODE as SAVE_MODE
from .browser_habitability_save import _complete, _mapping
from .browser_no_planet_workflow import _Evidence, _planet, _stellar, _values_match
from .browser_numeric import screen_identity
from .browser_planet_numeric import planet_projection
from .browser_positive_planet_workflow import _measurement_mode
from .browser_terrestrial_workflow import _clean, _load_sources, _same_habitat, _unchanged
from .project_evidence import _flags, _hashes, _sha
from .project_inventory_source import load_inventory_source
from .project_positive_evidence import MEASUREMENT_MODES
from .project_positive_evidence import _scopes as _planet_scopes
from .project_progress import (
    Collected,
    Evidence,
    ProgressError,
    StageEvidence,
    StageRecorded,
    TaskReceipt,
    _json,
)
from .supplied_browser_modes import (
    DERIVED_SUPPLIED_MODE,
    TEMPERATURE_SUPPLIED_MODE,
    TERRESTRIAL_SUPPLIED_MODE,
    source_options,
)

MODE = "terrestrial_visible_workflow_readback"


def _require(condition, reason):
    if not condition:
        raise ProgressError("project_terrestrial_evidence_" + reason)


def _source_directories(book, hashes, *, supplied_inputs=False):
    source_options(supplied_inputs)
    _hashes(book, hashes)
    names = (
        "numeric",
        "color",
        "class",
        "raw",
        "derived",
        "planet_class",
        "gas_comparison",
        "gas_selection",
        "temperature",
        "greenhouse",
        "phase",
        "choice",
        "save",
    )
    found = {name: [] for name in names}
    for name in hashes:
        path = book.history / name
        if path.name not in {"manifest.json", "confirmed.json", "report.json"}:
            continue
        value, label = book.json(path), None
        if path.name == "manifest.json":
            label = {
                "full_stellar_numeric_transport_verified": "numeric",
                "color_transport_verified": "color",
            }.get(value.get("outcome"))
        elif path.name == "report.json":
            if value.get("mode") in MEASUREMENT_MODES and value.get("stage") == "inputs":
                label = "raw"
            elif value.get("scope") == (
                DERIVED_SUPPLIED_MODE if supplied_inputs else "four_derived_planet_browser_transport"
            ):
                label = "derived"
            elif value.get("scope") == (
                TEMPERATURE_SUPPLIED_MODE
                if supplied_inputs
                else "one_equilibrium_copy_with_supplied_warming_local_proposal"
            ):
                label = "temperature"
            elif value.get("baseline_restored") is True and "candidates" in value:
                label = "gas_comparison"
            elif value.get("action_source") == "explicit_visual_reference_not_learned":
                label = "gas_selection"
        elif value.get("action_source") == "explicit_fresh_star_class_setup":
            label = "class"
        elif value.get("kind") == "SELECT":
            if value.get("value") in {"terrestrial", "gas_giant", "ice_giant"}:
                label = "planet_class"
            elif value.get("field") == "greenhouse":
                label = "greenhouse"
            elif value.get("field") == "water_phase":
                label = "phase"
            elif value.get("choice") in {"habitable", "not_habitable"}:
                label = "choice"
        elif (
            value.get("mode") == SAVE_MODE
            and value.get("kind") == "CLICK"
            or value.get("mode") == AUTOSAVE_MODE
            and value.get("branch") == "terrestrial"
        ):
            label = "save"
        if label:
            found[label].append(path.parent)
    _require(all(len(paths) == 1 for paths in found.values()), "ambiguous_workflow_sources")
    return {name + "_dir": paths[0] for name, paths in found.items()}


def _workflow(book, directory, rows):
    directory = _clean(book, directory)
    receipt = book.json(directory / "confirmed.json")
    _require(
        type(receipt.get("schema_version")) is int
        and receipt["schema_version"] == 1
        and receipt.get("mode") in {MODE, TERRESTRIAL_SUPPLIED_MODE}
        and receipt.get("authority") == "visible_workflow_readback"
        and receipt.get("classification_provenance") == "reference_prediction"
        and type(receipt.get("navigation_clicks")) is int
        and receipt["navigation_clicks"] == 3,
        "unsupported_workflow",
    )
    autosave = receipt.get("save_strategy") == "autosave"
    validate_autosave_workflow_flags(receipt, enabled=autosave)
    _flags(
        receipt,
        true=("task_completed",)
        if autosave
        else ("task_completed", "save_acknowledgement_verified", "source_save_click_delivered"),
        false=(
            "learned_classification",
            "scientific_verified",
            "correctness_verified",
            "browser_acceptance_passed",
            "course_completion_verified",
            "project_completed",
            "submitted",
            "training_label",
            "cross_session_persistence_verified",
            "hidden_values_inspected",
        )
        + (("save_acknowledgement_verified", "source_save_click_delivered") if autosave else ()),
        zero=(
            "answer_writes",
            "habitability_writes",
            "na_writes",
            "save_clicks",
            "assessment_clicks",
            "score_transfer_clicks",
            "submission_clicks",
        ),
    )
    supplied_inputs = receipt["mode"] == TERRESTRIAL_SUPPLIED_MODE
    options = source_options(supplied_inputs)
    directories = _source_directories(book, receipt.get("source_sha256"), **options)
    source_book = _Evidence(book.history)
    bundle = _load_sources(source_book, **directories, **options)
    _require((bundle.get("save_strategy") == "autosave") == autosave, "save_strategy_mismatch")
    _require(source_book.hashes == receipt["source_sha256"], "workflow_source_chain_mismatch")
    outcome = bundle["choice"]["choice"]
    _require(
        bundle["planet_class"] == "terrestrial" and outcome in {"habitable", "not_habitable"},
        "explicit_terrestrial_outcome_required",
    )
    expected = {
        "star": bundle["star"],
        "stellar_fields": bundle["readbacks"],
        "color": bundle["color"],
        "classification": bundle["class"],
        "numeric_provenance": bundle["numeric_provenance"],
        "color_provenance": bundle["color_provenance"],
        "planet_fields": bundle["planet_readbacks"],
        "raw_measurement_evidence": bundle["raw"]["evidence"],
        "derived_provenance": bundle["derived"]["provenance"],
        "save_acknowledgement_source": AUTOSAVE_ACKNOWLEDGEMENT_SOURCE
        if autosave
        else "explicit_final_habitability_save_fresh_visible_footer",
        "planet": {
            "outcome": "planet",
            "value": "terrestrial",
            "provenance": "reference_prediction",
            "measurement_provenance": _measurement_mode(bundle["raw"]["evidence"]),
            "approximate": True,
            "transport_verified": True,
            "scientific_verified": False,
            "training_label": False,
        },
        "gas_selection": {
            "gases": bundle["gas"]["gases"],
            "provenance": "reference_prediction",
            "transport_verified": True,
            "learned_gas_identification": False,
            "scientific_verified": False,
        },
        "temperature_provenance": bundle["temperature"]["provenance"],
        "equilibrium_transport": bundle["temperature"]["native_receipt"],
        "greenhouse_reference": bundle["greenhouse"]["evidence"],
        "phase_reference": bundle["choice"]["evidence"],
        "habitability": {
            "outcome": outcome,
            "provenance": "reference_prediction",
            "transport_verified": True,
            "scientific_verified": False,
            "learned_habitability_decision": False,
            "source": "explicit_native_choice_with_confirmed_chamber_phase",
        },
    }
    _require(all(receipt.get(key) == value for key, value in expected.items()), "workflow_fact_mismatch")
    _flags(
        receipt["planet"],
        true=("approximate", "transport_verified"),
        false=("scientific_verified", "training_label"),
    )
    _flags(
        receipt["gas_selection"],
        true=("transport_verified",),
        false=("learned_gas_identification", "scientific_verified"),
    )
    _flags(
        receipt["habitability"],
        true=("transport_verified",),
        false=("scientific_verified", "learned_habitability_decision"),
    )
    captures = {
        section: book.capture(directory / f"{section}-readback/verified")
        for section in ("stellar", "planet", "habitability")
    }
    _require(
        receipt.get("current_screen_sha256")
        == {key: screen_identity(value) for key, value in captures.items()},
        "workflow_current_capture_hash_mismatch",
    )
    stellar, planet, habitat = (
        _stellar(captures["stellar"]),
        _planet(captures["planet"]),
        _mapping(captures["habitability"]),
    )
    _values_match(stellar, bundle)
    _complete(habitat)
    _require(
        planet["star_name"].casefold() == bundle["star"].casefold()
        and planet["observation"]["values"]["has_planet"] == "Yes"
        and planet_projection(captures["planet"], planet)
        == planet_projection(bundle["class_capture"], _planet(bundle["class_capture"]))
        and habitat["star_name"].casefold() == bundle["star"].casefold()
        and _same_habitat(captures["habitability"], bundle["habitability_capture"]),
        "workflow_current_branch_mismatch",
    )
    matched = [row for row in rows if row["name"].casefold() == bundle["star"].casefold()]
    _require(len(matched) == 1, "workflow_star_not_in_inventory")
    verify_detail_row(
        stellar, {"selected": bundle["class"]}, parse_stellar_row(matched[0]["text"], matched[0]["name"])
    )
    _unchanged(source_book)
    for source_directory in source_book.clean_directories:
        _clean(book, source_directory)
    book.closed_trees.update(source_book.closed_trees)
    return receipt, directories


def _scopes(book, workflow, directories):
    scopes = _planet_scopes(book, workflow, directories)
    common = {"transport_verified": True, "scientific_verified": False, "training_label": False}

    def source(name, filename="confirmed.json"):
        return _sha(book.read(directories[name + "_dir"] / filename))

    scopes.update(
        {
            "gas_selection": {
                **common,
                "provenance": "reference_prediction",
                "source_sha256": source("gas_selection", "report.json"),
                "gases": workflow["gas_selection"]["gases"],
                "learned_gas_identification": False,
            },
            "temperature_calculations": {
                **common,
                "provenance": "learned_prediction",
                "source_sha256": source("temperature", "report.json"),
                "checkpoint_provenance": workflow["temperature_provenance"],
                "equilibrium_transport": workflow["equilibrium_transport"],
                "surface_proposal_is_not_browser_readback": True,
                **(
                    {"scope": TEMPERATURE_SUPPLIED_MODE, "calibration_verified": False}
                    if workflow.get("mode") == TERRESTRIAL_SUPPLIED_MODE
                    else {}
                ),
            },
            "greenhouse_selection": {
                **common,
                "provenance": "reference_prediction",
                "source_sha256": source("greenhouse"),
                "reference": workflow["greenhouse_reference"],
            },
            "water_phase": {
                **common,
                "provenance": "reference_prediction",
                "source_sha256": source("phase"),
                "reference": workflow["phase_reference"],
                "source": "visible_chamber_indicator",
                "learned_phase_identification": False,
            },
            "habitability": {
                **common,
                "provenance": "reference_prediction",
                "source_sha256": source("choice"),
                "outcome": workflow["habitability"]["outcome"],
                "learned_habitability_decision": False,
            },
        }
    )
    return scopes


def _stage_records(star_id, workflow, workflow_sha, directories, book):
    mode = _measurement_mode(workflow["raw_measurement_evidence"])

    def proof(name, filename, provenance, value=None):
        return Evidence(
            source_sha256=_sha(book.read(directories[name + "_dir"] / filename)),
            provenance=provenance,
            value=value,
            transport_verified=True,
            scientific_verified=False,
        )

    current = Evidence(source_sha256=workflow_sha, provenance="tool_verified", transport_verified=True)
    evidence = (
        StageEvidence(decision=proof("numeric", "manifest.json", "learned_prediction"), confirmation=current),
        StageEvidence(
            decision=proof("color", "manifest.json", "learned_prediction", workflow["color"]),
            confirmation=current,
        ),
        StageEvidence(
            decision=proof("class", "confirmed.json", "reference_prediction", workflow["classification"]),
            confirmation=current,
        ),
        StageEvidence(
            decision=proof("planet_class", "confirmed.json", "reference_prediction", "terrestrial"),
            confirmation=current,
            outcome="planet",
            policy_label=mode,
            observation_limit_days=5000,
        ),
        StageEvidence(
            decision=proof(
                "choice", "confirmed.json", "reference_prediction", workflow["habitability"]["outcome"]
            ),
            confirmation=current,
            outcome=workflow["habitability"]["outcome"],
            policy_label=mode,
            observation_limit_days=5000,
        ),
    )
    return [
        StageRecorded(star_id=star_id, stage=stage, evidence=item)
        for stage, item in zip(
            ("stellar_numeric", "stellar_color", "stellar_classification", "planet", "habitability"),
            evidence,
            strict=True,
        )
    ]


def _plan(progress, inventory, workflow, workflow_sha, inventory_sha, directories, book):
    state = progress.reduce()
    _require(
        not state.pending and state.receipt("submission") is None, "journal_has_uncertain_or_submitted_write"
    )
    names = {row["name"].casefold(): row["name"] for row in inventory["rows"]}
    _require(
        {star.name.casefold() for star in state.stars.values()} <= set(names), "inventory_dropped_prior_star"
    )
    by_name, new_names = {star.name.casefold(): star.id for star in state.stars.values()}, []
    for name in names.values():
        if name.casefold() not in by_name:
            star_id = "star:" + _sha(name.casefold().encode())
            progress = progress.append(Collected(star_id=star_id, name=name, source_sha256=inventory_sha))
            by_name[name.casefold()] = star_id
            new_names.append(name)
    star_id = by_name[workflow["star"].casefold()]
    planned = _stage_records(star_id, workflow, workflow_sha, directories, book)
    original, missing = progress.reduce().stars[star_id], False
    for record in planned:
        current = getattr(original, record.stage)
        _require(current is None or current == record.evidence, "revised_stage_evidence")
        _require(not (missing and current is not None), "nonprefix_partial_stages")
        missing |= current is None
    for record in planned:
        if getattr(progress.reduce().stars[star_id], record.stage) is None:
            progress = progress.append(record)
    star = progress.reduce().stars[star_id]
    task = TaskReceipt(
        star_id=star_id,
        star_revision=star.revision,
        source_sha256=workflow_sha,
        authority="visible_workflow_readback",
        task_completed=True,
    )
    _require(star.completion is None or star.completion == task, "revised_workflow_receipt")
    if star.completion is None:
        progress = progress.append(task)
    return progress, star_id, new_names


def import_verified_terrestrial(journal, run_history, inventory_dir, workflow_dir):
    """Import recorded native proof atomically; an exact reimport is a no-op."""
    book = _Evidence(run_history)
    _require(book.path(journal.path).parent == book.history, "journal_outside_attempt")
    _require(
        {book.path(p) for p in book.history.glob("project-progress-*.jsonl")} == {book.path(journal.path)},
        "ambiguous_canonical_attempt_journal",
    )
    inventory_dir, workflow_dir = book.path(inventory_dir), book.path(workflow_dir)
    inventory_source = load_inventory_source(book, inventory_dir)
    inventory, rows = inventory_source.receipt, inventory_source.rows
    workflow, directories = _workflow(book, workflow_dir, rows)
    scopes = _scopes(book, workflow, directories)
    inventory_sha, workflow_sha = (
        _sha(book.read(p / "confirmed.json")) for p in (inventory_dir, workflow_dir)
    )
    with journal.path.open("r+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        before = journal._read(stream)
        after, star_id, new_names = _plan(
            before, inventory, workflow, workflow_sha, inventory_sha, directories, book
        )
        binding = {
            "schema_version": 1,
            **before.header(),
            **inventory_source.binding_fields(),
            "star_id": star_id,
            "inventory_path": str(inventory_dir.relative_to(book.history)),
            "inventory_sha256": inventory_sha,
            "workflow_path": str(workflow_dir.relative_to(book.history)),
            "workflow_sha256": workflow_sha,
            "evidence_kind": workflow["mode"],
            "evidence_scopes": scopes,
            "validated_artifact_sha256": dict(book.hashes),
        }
        binding_path = book.history / (
            "project-evidence-" + _sha((journal.path.name + star_id).encode()) + ".json"
        )
        if binding_path.exists() or binding_path.is_symlink():
            _require(book.json(binding_path) == binding, "revised_import_binding")
        binding_text = _json(binding) + "\n"
        records = after.records[len(before.records) :]
        records_text = "".join(_json(record.model_dump(mode="json")) + "\n" for record in records)
        _unchanged(book)
        if not binding_path.exists():
            fd = os.open(binding_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as claim:
                claim.write(binding_text)
                claim.flush()
                os.fsync(claim.fileno())
            parent = os.open(book.history, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        if records:
            stream.seek(0, os.SEEK_END)
            stream.write(records_text)
            stream.flush()
            os.fsync(stream.fileno())
        return {
            "star_id": star_id,
            "appended_records": len(records),
            "idempotent": not records,
            "newly_collected": new_names,
            "workflow_sha256": workflow_sha,
            "inventory_sha256": inventory_sha,
            "authority": "recorded_visible_workflow_readback_not_new_live_inspection",
            **inventory_source.binding_fields(),
            "browser_actions": 0,
            "evidence_scopes": scopes,
            "progress": after.reduce().report(),
        }
