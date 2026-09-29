"""Offline import of one verified positive/non-terrestrial workflow.

This imports recorded visible readbacks, not new live verification or historical
browser-write receipts. Explicit approximate references, learned derived calculations,
and reference classification remain separate immutable evidence scopes. The
existing No-planet importer and canonical journal schema are unchanged.
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
from .browser_no_planet_workflow import _Evidence, _planet, _stellar, _values_match
from .browser_numeric import screen_identity
from .browser_planet_numeric import planet_projection
from .browser_positive_planet_workflow import _load_sources, _measurement_mode
from .planet_tooltip_reference import MODE as TOOLTIP_MEASUREMENTS
from .planet_tooltip_reference import TWO_MODE as TWO_TOOLTIP_MEASUREMENTS
from .project_evidence import _flags, _hashes, _sha
from .project_inventory_source import load_inventory_source
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
    POSITIVE_SUPPLIED_MODE,
    TERRESTRIAL_SUPPLIED_MODE,
    source_options,
)

MODE = "positive_planet_visible_workflow_readback"
MEASUREMENTS = "approximate_reference_raster"
MEASUREMENT_MODES = (MEASUREMENTS, TOOLTIP_MEASUREMENTS, TWO_TOOLTIP_MEASUREMENTS)


def _require(condition, reason):
    if not condition:
        raise ProgressError("project_positive_evidence_" + reason)


def _source_directories(book, hashes, *, supplied_inputs=False):
    source_options(supplied_inputs)
    _hashes(book, hashes)
    found = {name: [] for name in ("numeric", "color", "class", "raw", "derived", "planet_class", "save")}
    for name in hashes:
        path = book.history / name
        if path.name not in {"manifest.json", "confirmed.json", "report.json"}:
            continue
        value, label = book.json(path), None
        if path.name == "manifest.json":
            if value.get("outcome") == "full_stellar_numeric_transport_verified":
                label = "numeric"
            elif value.get("outcome") == "color_transport_verified":
                label = "color"
        elif path.name == "report.json":
            if value.get("mode") in MEASUREMENT_MODES and value.get("stage") == "inputs":
                label = "raw"
            elif value.get("scope") == (
                DERIVED_SUPPLIED_MODE if supplied_inputs else "four_derived_planet_browser_transport"
            ):
                label = "derived"
        elif value.get("action_source") == "explicit_fresh_star_class_setup":
            label = "class"
        elif value.get("kind") == "SELECT" and value.get("value") in {
            "gas_giant",
            "ice_giant",
            "terrestrial",
        }:
            label = "planet_class"
        elif (
            value.get("kind") == "CLICK"
            and value.get("visible_label") == "Save"
            or value.get("mode") == AUTOSAVE_MODE
            and value.get("branch") == "positive"
        ):
            label = "save"
        if label:
            found[label].append(path.parent)
    _require(all(len(paths) == 1 for paths in found.values()), "ambiguous_workflow_sources")
    return {name + "_dir": paths[0] for name, paths in found.items()}


def _workflow(book, directory, rows):
    directory = book.clean(directory)
    receipt = book.json(directory / "confirmed.json")
    _require(
        type(receipt.get("schema_version")) is int
        and receipt["schema_version"] == 1
        and receipt.get("mode") in {MODE, POSITIVE_SUPPLIED_MODE}
        and receipt.get("authority") == "visible_workflow_readback"
        and receipt.get("classification_provenance") == "reference_prediction"
        and type(receipt.get("navigation_clicks")) is int
        and receipt["navigation_clicks"] == 2,
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
            "hidden_values_blank_verified",
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
    supplied_inputs = receipt["mode"] == POSITIVE_SUPPLIED_MODE
    options = source_options(supplied_inputs)
    directories = _source_directories(book, receipt.get("source_sha256"), **options)
    source_book = _Evidence(book.history)
    bundle = _load_sources(source_book, **directories, **options)
    _require((bundle.get("save_strategy") == "autosave") == autosave, "save_strategy_mismatch")
    _require(source_book.hashes == receipt["source_sha256"], "workflow_source_chain_mismatch")
    _require(bundle["planet_class"] in {"gas_giant", "ice_giant"}, "unsupported_habitability_branch")
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
        else "explicit_save_fresh_visible_footer",
        "planet": {
            "outcome": "planet",
            "value": bundle["planet_class"],
            "provenance": "reference_prediction",
            "measurement_provenance": _measurement_mode(bundle["raw"]["evidence"]),
            "approximate": True,
            "transport_verified": True,
            "scientific_verified": False,
            "training_label": False,
        },
        "habitability": {
            "outcome": "not_applicable",
            "applicability_reason": "non_terrestrial_planet",
            "provenance": "reference_prediction",
            "branch_applicability_verified": True,
            "transport_verified": False,
            "scientific_verified": False,
            "source": "explicit_branch_applicability_not_field_write",
        },
    }
    _require(all(receipt.get(key) == value for key, value in expected.items()), "workflow_fact_mismatch")
    _flags(
        receipt["planet"],
        true=("approximate", "transport_verified"),
        false=("scientific_verified", "training_label"),
    )
    _flags(
        receipt["habitability"],
        true=("branch_applicability_verified",),
        false=("transport_verified", "scientific_verified"),
    )
    captures = {
        section: book.capture(directory / f"{section}-readback/verified") for section in ("stellar", "planet")
    }
    _require(
        receipt.get("current_screen_sha256")
        == {key: screen_identity(value) for key, value in captures.items()},
        "workflow_current_capture_hash_mismatch",
    )
    stellar, planet = _stellar(captures["stellar"]), _planet(captures["planet"])
    _values_match(stellar, bundle)
    _require(
        planet["star_name"].casefold() == bundle["star"].casefold()
        and planet["observation"]["values"]["has_planet"] == "Yes"
        and planet_projection(captures["planet"], planet)
        == planet_projection(bundle["save_capture"], _planet(bundle["save_capture"])),
        "workflow_current_planet_mismatch",
    )
    matched = [row for row in rows if row["name"].casefold() == bundle["star"].casefold()]
    _require(len(matched) == 1, "workflow_star_not_in_inventory")
    verify_detail_row(
        stellar, {"selected": bundle["class"]}, parse_stellar_row(matched[0]["text"], matched[0]["name"])
    )
    source_book.unchanged()
    # Carry upstream clean-directory guards into the final locked precommit
    # check too: a newly recorded uncertain/invalidated source is not a byte
    # change to one of the already-hashed successful artifacts.
    for source_directory in source_book.clean_directories:
        book.clean(source_directory)
    book.closed_trees.update(source_book.closed_trees)
    return receipt, directories


def _scopes(book, workflow, directories):
    """Keep all three planet evidence origins; never call the whole stage learned."""
    common = {"transport_verified": True, "scientific_verified": False, "training_label": False}
    mode = _measurement_mode(workflow["raw_measurement_evidence"])
    two_metadata = {}
    if mode == TWO_TOOLTIP_MEASUREMENTS:
        from .browser_raster_planet_evidence import _two_tooltip_metadata

        two_metadata = _two_tooltip_metadata(workflow["raw_measurement_evidence"]["tooltip_measurements"])
    supplied_inputs = workflow.get("mode") in {POSITIVE_SUPPLIED_MODE, TERRESTRIAL_SUPPLIED_MODE}
    scopes = {
        "raw_measurements": {
            **common,
            "provenance": mode,
            "source_sha256": _sha(book.read(directories["raw_dir"] / "report.json")),
            "approximate": True,
            "learned_perception": False,
            "measurements": workflow["raw_measurement_evidence"]["measurements"],
            "uncertainty": workflow["raw_measurement_evidence"]["uncertainty"],
            **two_metadata,
        },
        "derived_calculations": {
            **common,
            "provenance": "learned_prediction",
            "source_sha256": _sha(book.read(directories["derived_dir"] / "report.json")),
            "scope": DERIVED_SUPPLIED_MODE if supplied_inputs else "four_derived_planet_browser_transport",
            "input_provenance": mode,
            "checkpoint_provenance": workflow["derived_provenance"],
        },
        "classification": {
            **common,
            "provenance": "reference_prediction",
            "source_sha256": _sha(book.read(directories["planet_class_dir"] / "confirmed.json")),
            "value": workflow["planet"]["value"],
            "learned_classification": False,
        },
        "habitability": {
            "provenance": "reference_prediction",
            "outcome": "not_applicable",
            "applicability_reason": "non_terrestrial_planet",
            "branch_applicability_verified": True,
            "transport_verified": False,
            "scientific_verified": False,
            "training_label": False,
        },
    }
    if supplied_inputs:
        scopes["supplied_stellar_inputs"] = {
            "source": workflow["derived_provenance"]["supplied_stellar_inputs"],
            "actual_class": workflow["classification"],
            "provenance": "reference_prediction",
            "learned_stellar_mass_radius": False,
            "scientific_verified": False,
            "training_label": False,
            "calibration_verified": False,
        }
    if workflow.get("save_strategy") == "autosave":
        scopes["save_readback"] = {
            "mode": AUTOSAVE_MODE,
            "authority": "visible_readback_only",
            "source_sha256": _sha(book.read(directories["save_dir"] / "confirmed.json")),
            "visible_readback_verified": True,
            "save_click_delivered": False,
            "save_acknowledgement_verified": False,
            "persistence_verified": False,
            "cross_session_persistence_verified": False,
            "scientific_verified": False,
        }
    return scopes


def _stage_records(star_id, workflow, workflow_sha, directories, book):
    mode = _measurement_mode(workflow["raw_measurement_evidence"])

    def proof(path, provenance, value=None):
        return Evidence(
            source_sha256=_sha(book.read(path)),
            provenance=provenance,
            value=value,
            transport_verified=True,
            scientific_verified=False,
        )

    current = Evidence(source_sha256=workflow_sha, provenance="tool_verified", transport_verified=True)
    numeric = StageEvidence(
        decision=proof(directories["numeric_dir"] / "manifest.json", "learned_prediction"),
        confirmation=current,
    )
    color = StageEvidence(
        decision=proof(directories["color_dir"] / "manifest.json", "learned_prediction", workflow["color"]),
        confirmation=current,
    )
    classification = StageEvidence(
        decision=proof(
            directories["class_dir"] / "confirmed.json", "reference_prediction", workflow["classification"]
        ),
        confirmation=current,
    )
    planet = StageEvidence(
        decision=proof(
            directories["planet_class_dir"] / "confirmed.json",
            "reference_prediction",
            workflow["planet"]["value"],
        ),
        confirmation=current,
        outcome="planet",
        policy_label=mode,
        observation_limit_days=5000,
    )
    habitat = StageEvidence(
        decision=Evidence(
            source_sha256=workflow_sha,
            provenance="reference_prediction",
            value="not_applicable",
            transport_verified=False,
            scientific_verified=False,
        ),
        outcome="not_applicable",
        applicability_reason="non_terrestrial_planet",
        branch_applicability_verified=True,
        policy_label=mode,
        observation_limit_days=5000,
    )
    return [
        StageRecorded(star_id=star_id, stage=stage, evidence=evidence)
        for stage, evidence in zip(
            ("stellar_numeric", "stellar_color", "stellar_classification", "planet", "habitability"),
            (numeric, color, classification, planet, habitat),
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
    by_name = {star.name.casefold(): star.id for star in state.stars.values()}
    new_names = []
    for name in names.values():
        if name.casefold() not in by_name:
            star_id = "star:" + _sha(name.casefold().encode())
            progress = progress.append(Collected(star_id=star_id, name=name, source_sha256=inventory_sha))
            by_name[name.casefold()] = star_id
            new_names.append(name)
    star_id = by_name[workflow["star"].casefold()]
    planned = _stage_records(star_id, workflow, workflow_sha, directories, book)
    original = progress.reduce().stars[star_id]
    missing = False
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


def import_verified_positive_planet(journal, run_history, inventory_dir, workflow_dir):
    """Append one immutable, idempotent positive batch to the canonical attempt.

    The journal and binding share the No importer's identities and lock. Exact
    interrupted prefixes may finish; uncertain browser writes and revised
    evidence may not. This never creates a WriteReserved or WriteReceipt.
    """
    book = _Evidence(run_history)
    _require(book.path(journal.path).parent == book.history, "journal_outside_attempt")
    _require(
        {book.path(path) for path in book.history.glob("project-progress-*.jsonl")}
        == {book.path(journal.path)},
        "ambiguous_canonical_attempt_journal",
    )
    inventory_dir, workflow_dir = book.path(inventory_dir), book.path(workflow_dir)
    inventory_source = load_inventory_source(book, inventory_dir)
    inventory, rows = inventory_source.receipt, inventory_source.rows
    workflow, directories = _workflow(book, workflow_dir, rows)
    scopes = _scopes(book, workflow, directories)
    inventory_sha, workflow_sha = (
        _sha(book.read(directory / "confirmed.json")) for directory in (inventory_dir, workflow_dir)
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
            # Includes both canonical source-chain artifacts and recorded
            # current readbacks. Reimport cannot rewrite even metadata that is
            # deliberately ignored by semantic screen-identity comparisons.
            "validated_artifact_sha256": dict(book.hashes),
        }
        binding_path = book.history / (
            "project-evidence-" + _sha((journal.path.name + star_id).encode()) + ".json"
        )
        if binding_path.exists() or binding_path.is_symlink():
            _require(book.json(binding_path) == binding, "revised_import_binding")
        # Serialize everything before the first durable mutation. Unsupported
        # metadata (including NaN) must not leave an empty binding behind.
        binding_text = _json(binding) + "\n"
        records = after.records[len(before.records) :]
        records_text = "".join(_json(record.model_dump(mode="json")) + "\n" for record in records)
        book.unchanged()
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
