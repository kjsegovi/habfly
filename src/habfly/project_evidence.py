"""Offline, immutable ingestion of verified collection and No/N/A workflows.

No browser state is read or fabricated here. These are recorded readbacks, not a
new live check, scientific acceptance, or retroactive browser-write receipts.
"""

import fcntl
import hashlib
import math
import os
import re
from pathlib import Path

from .browser_collected_revisit import parse_stellar_row, verify_detail_row
from .browser_no_planet_save import CONDITIONAL_FIELDS, RAW_FIELDS
from .browser_no_planet_workflow import _blank_no, _Evidence, _load_sources, _planet, _stellar, _values_match
from .browser_numeric import screen_identity
from .browser_planet_numeric import planet_projection
from .browser_planet_window_choice import MODE as CHOICE_MODE
from .browser_planet_window_choice import recorded_policy_manifest
from .browser_project_inventory import _capture_inventory
from .browser_stellar import CLASSES
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


def _require(condition, reason):
    if not condition:
        raise ProgressError("project_evidence_" + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _flags(value, *, true=(), false=(), zero=()):
    _require(
        all(value.get(key) is True for key in true)
        and all(value.get(key) is False for key in false)
        and all(type(value.get(key)) is int and value[key] == 0 for key in zero),
        "unsupported_receipt_flags",
    )


def _hashes(book, mapping):
    _require(isinstance(mapping, dict) and 1 <= len(mapping) <= 500, "missing_source_hashes")
    for name, expected in mapping.items():
        _require(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and isinstance(expected, str)
            and re.fullmatch(r"[a-f0-9]{64}", expected),
            "invalid_source_hash",
        )
        _require(_sha(book.read(book.history / name)) == expected, "source_hash_mismatch")


def _inventory(book, directory):
    directory = book.clean(directory)
    receipt = book.json(directory / "confirmed.json")
    _require(
        type(receipt.get("schema_version")) is int
        and receipt["schema_version"] == 1
        and receipt.get("mode") == "read_only_collected_star_inventory"
        and receipt.get("authority") == "visible_collected_list_readback"
        and receipt.get("section") == "stellar",
        "unsupported_inventory",
    )
    _flags(
        receipt,
        true=("collection_count_verified", "complete_visible_list_verified"),
        false=(
            "hidden_catalog_read",
            "learned_perception",
            "scientific_verified",
            "task_completed",
            "project_completed",
            "cross_session_persistence_verified",
            "config_mutated",
            "automatic_retry",
        ),
        zero=(
            "browser_actions",
            "navigation_clicks",
            "scrolls",
            "answer_writes",
            "save_clicks",
            "deletion_clicks",
            "assessment_clicks",
            "submission_clicks",
        ),
    )
    before, after = book.capture(directory / "before"), book.capture(directory / "after")
    counts, parsed = _capture_inventory(after)
    _require(
        screen_identity(before) == screen_identity(after)
        and _capture_inventory(before) == (counts, parsed)
        and receipt.get("viewing") == counts
        and type(receipt.get("total_collected")) is int
        and receipt["total_collected"] == counts["total"]
        and 1 <= counts["total"] <= 30
        and type(receipt.get("visible_row_count")) is int
        and receipt["visible_row_count"] == len(parsed),
        "inventory_readback_mismatch",
    )
    expected_hashes = {
        f"{section}/observation.json": _sha(book.read(directory / section / "observation.json"))
        for section in ("before", "after")
    }
    _require(receipt.get("source_sha256") == expected_hashes, "inventory_capture_hash_mismatch")
    rows = receipt.get("rows")
    _require(isinstance(rows, list) and len(rows) == len(parsed), "inventory_rows_mismatch")
    for row, source in zip(rows, parsed):
        box = row.get("visible_name_box", {})
        _require(
            row.get("name") == source["name"]
            and row.get("source_sha256") == expected_hashes["after/observation.json"]
            and row.get("row_accessibility_sha256") == _sha(source["text"].encode())
            and set(box) == {"x", "y", "width", "height"}
            and all(type(v) in {int, float} and math.isfinite(v) and v >= 0 for v in box.values())
            and box["width"] > 0
            and box["height"] > 0,
            "inventory_named_row_mismatch",
        )
    return receipt, parsed


def _source_directories(book, hashes):
    _hashes(book, hashes)
    found = {name: [] for name in ("numeric", "color", "class", "choice", "save")}
    for name in hashes:
        path = book.history / name
        if path.name not in {"manifest.json", "confirmed.json"}:
            continue
        value = book.json(path)
        label = None
        if path.name == "manifest.json":
            if value.get("outcome") == "full_stellar_numeric_transport_verified":
                label = "numeric"
            elif value.get("outcome") == "color_transport_verified":
                label = "color"
        elif value.get("action_source") == "explicit_fresh_star_class_setup":
            label = "class"
        elif value.get("mode") == CHOICE_MODE:
            label = "choice"
        elif value.get("mode") in {
            "bounded_window_no_planet_save_acknowledgement",
            "no_planet_predispatch_read_only_reconciliation",
        } or (
            value.get("mode") == "user_approved_autosave_visible_readback_v1"
            and value.get("branch") == "no_planet"
        ):
            label = "save"
        if label:
            found[label].append(path.parent)
    _require(all(len(paths) == 1 for paths in found.values()), "ambiguous_workflow_sources")
    return {name + "_dir": paths[0] for name, paths in found.items()}


def _workflow(book, directory, inventory_rows):
    directory = book.clean(directory)
    receipt = book.json(directory / "confirmed.json")
    from .browser_autosave import validate_autosave_workflow_flags

    autosave = receipt.get("save_strategy") == "autosave"
    validate_autosave_workflow_flags(receipt, enabled=autosave)
    _require(
        type(receipt.get("schema_version")) is int
        and receipt["schema_version"] == 1
        and receipt.get("mode") == "no_planet_visible_workflow_readback"
        and receipt.get("authority") == "visible_workflow_readback"
        and receipt.get("classification_provenance") == "reference_prediction"
        and type(receipt.get("navigation_clicks")) is int
        and receipt["navigation_clicks"] == 2,
        "unsupported_workflow",
    )
    _flags(
        receipt,
        true=("task_completed",) + (() if autosave else ("save_acknowledgement_verified",)),
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
        + (("save_acknowledgement_verified",) if autosave else ()),
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
    selected = receipt.get("classification")
    _require(isinstance(selected, str) and selected in CLASSES, "unsupported_stellar_class")
    required_fields = {"distance", "luminosity", "temperature"}
    if selected == "main_sequence":
        required_fields |= {"mass", "radius", "lifetime"}
    fields, provenance = receipt.get("stellar_fields"), receipt.get("numeric_provenance")
    _require(
        isinstance(fields, dict)
        and set(fields) == required_fields
        and isinstance(provenance, dict)
        and provenance.get("selected_class") == selected
        and (
            isinstance(provenance.get("lifetime_prefix"), str)
            and provenance["lifetime_prefix"] in {"ka", "Ma", "Ga", "Ta"}
            if selected == "main_sequence"
            else "lifetime_prefix" in provenance and provenance["lifetime_prefix"] is None
        ),
        "stellar_applicability_mismatch",
    )
    directories = _source_directories(book, receipt.get("source_sha256"))
    source_book = _Evidence(book.history)
    bundle = _load_sources(source_book, **directories)  # Pure offline upstream source-chain verification.
    _require((bundle.get("save_strategy") == "autosave") is autosave, "workflow_save_strategy_mismatch")
    for source_directory in source_book.clean_directories:
        book.clean(source_directory)
    book.closed_trees.update(source_book.closed_trees)
    _require(source_book.hashes == receipt["source_sha256"], "workflow_source_chain_mismatch")
    policy = recorded_policy_manifest(bundle["choice"]["policy"])
    _require(bundle["choice"]["policy"] == policy, "policy_identity_mismatch")
    expected = {
        "star": bundle["star"],
        "stellar_fields": bundle["readbacks"],
        "color": bundle["color"],
        "classification": bundle["class"],
        "numeric_provenance": bundle["numeric_provenance"],
        "color_provenance": bundle["color_provenance"],
        "source_save_click_delivered": bundle["save_click_delivered"],
        "source_save_resumed_same_intent": bundle["save_resumed_same_intent"],
        "source_save_continuation_stopped_predispatch": bundle["save_continuation_stopped_predispatch"],
        "save_acknowledgement_source": bundle["acknowledgement_source"],
        "visible_blank_raw_fields": list(RAW_FIELDS),
        "conditionally_absent_derived_fields": list(CONDITIONAL_FIELDS),
        "planet": {
            "outcome": "no_planet",
            "provenance": "reference_prediction",
            "transport_verified": True,
            "scientific_verified": False,
            "policy_label": policy["version"],
            "observation_limit_days": 5000,
        },
        "habitability": {
            "outcome": "not_applicable",
            "applicability_reason": "no_planet",
            "provenance": "reference_prediction",
            "transport_verified": False,
            "branch_applicability_verified": True,
            "scientific_verified": False,
            "source": "explicit_branch_applicability_not_field_write",
        },
    }
    _require(all(receipt.get(key) == value for key, value in expected.items()), "workflow_fact_mismatch")
    _require(
        all(
            type(receipt.get(key)) is bool
            for key in (
                "source_save_click_delivered",
                "source_save_resumed_same_intent",
                "source_save_continuation_stopped_predispatch",
            )
        ),
        "invalid_save_source_flags",
    )
    # JSON bools cannot be upgraded from truthy integers by structural equality.
    _flags(receipt["planet"], true=("transport_verified",), false=("scientific_verified",))
    _flags(
        receipt["habitability"],
        true=("branch_applicability_verified",),
        false=("transport_verified", "scientific_verified"),
    )
    _require(type(receipt["planet"]["observation_limit_days"]) is int, "invalid_observation_limit")
    captures = {
        section: book.capture(directory / f"{section}-readback/verified") for section in ("stellar", "planet")
    }
    _require(
        receipt.get("current_screen_sha256")
        == {key: screen_identity(value) for key, value in captures.items()},
        "workflow_current_capture_hash_mismatch",
    )
    stellar = _stellar(captures["stellar"])
    _values_match(stellar, bundle)
    planet = _planet(captures["planet"])
    _blank_no(planet)
    _require(
        planet["star_name"].casefold() == bundle["star"].casefold()
        and planet_projection(captures["planet"], planet)
        == planet_projection(bundle["save_capture"], bundle["save_mapping"]),
        "workflow_current_planet_mismatch",
    )
    rows = [row for row in inventory_rows if row["name"].casefold() == bundle["star"].casefold()]
    _require(len(rows) == 1, "workflow_star_not_in_inventory")
    verify_detail_row(
        stellar, {"selected": bundle["class"]}, parse_stellar_row(rows[0]["text"], rows[0]["name"])
    )
    source_book.unchanged()
    return receipt, directories


def _stage_records(star_id, workflow, workflow_sha, directories, book):
    def proof(path, provenance, value=None, *, transport=True):
        return Evidence(
            source_sha256=_sha(book.read(path)),
            provenance=provenance,
            value=value,
            transport_verified=transport,
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
        decision=proof(directories["choice_dir"] / "confirmed.json", "reference_prediction", "No"),
        confirmation=current,
        outcome="no_planet",
        policy_label=workflow["planet"]["policy_label"],
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
        applicability_reason="no_planet",
        branch_applicability_verified=True,
        policy_label=workflow["planet"]["policy_label"],
        observation_limit_days=5000,
    )
    return [
        StageRecorded(star_id=star_id, stage=stage, evidence=evidence)
        for stage, evidence in zip(
            ("stellar_numeric", "stellar_color", "stellar_classification", "planet", "habitability"),
            (numeric, color, classification, planet, habitat),
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
    original_star = progress.reduce().stars[star_id]
    missing = False
    for record in planned:
        current = getattr(original_star, record.stage)
        _require(current is None or current == record.evidence, "revised_stage_evidence")
        _require(not (missing and current is not None), "nonprefix_partial_stages")
        missing |= current is None
    for record in planned:
        current = getattr(progress.reduce().stars[star_id], record.stage)
        if current is None:
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


def import_verified_no_planet(journal, run_history, inventory_dir, workflow_dir):
    """Append an idempotent, prevalidated batch to this attempt's canonical journal.

    An immutable per-star binding rejects revised source pairs. A matching
    interrupted import may append only still-missing exact records. No browser
    write reservation or acknowledgement is synthesized from historical work.
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
        }
        binding_path = book.history / (
            "project-evidence-" + _sha((journal.path.name + star_id).encode()) + ".json"
        )
        if binding_path.exists() or binding_path.is_symlink():
            _require(book.json(binding_path) == binding, "revised_import_binding")
        book.unchanged()
        if not binding_path.exists():
            fd = os.open(binding_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as claim:
                claim.write(_json(binding) + "\n")
                claim.flush()
                os.fsync(claim.fileno())
            parent = os.open(book.history, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        records = after.records[len(before.records) :]
        if records:
            stream.seek(0, os.SEEK_END)
            stream.write("".join(_json(record.model_dump(mode="json")) + "\n" for record in records))
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
            "progress": after.reduce().report(),
        }
