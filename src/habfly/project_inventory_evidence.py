"""Import a complete visible collection without asserting analyzed work.

Read-only browser evidence is validated by the existing inventory contract.
Only previously unseen names add Collected records; existing scientific stages,
write claims, task receipts and source identities are never synthesized/revised.
"""

import fcntl
import os

from .browser_no_planet_workflow import _Evidence
from .project_evidence import _require, _sha
from .project_inventory_source import load_inventory_source
from .project_progress import Collected, _json


def import_verified_inventory(journal, run_history, inventory_dir):
    """Append collection-only records under the canonical attempt's file lock."""
    book = _Evidence(run_history)
    path = book.path(journal.path)
    _require(path.parent == book.history, "journal_outside_attempt")
    _require(
        {book.path(p) for p in book.history.glob("project-progress-*.jsonl")} == {path},
        "ambiguous_canonical_attempt_journal",
    )
    directory = book.path(inventory_dir)
    inventory_source = load_inventory_source(book, directory)
    inventory = inventory_source.receipt
    digest = _sha(book.read(directory / "confirmed.json"))
    with path.open("r+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        before = journal._read(stream)
        state = before.reduce()
        _require(
            not state.pending and state.receipt("submission") is None,
            "journal_has_uncertain_or_submitted_write",
        )
        names = {row["name"].casefold(): row["name"] for row in inventory["rows"]}
        known = {star.name.casefold() for star in state.stars.values()}
        _require(known <= names.keys(), "inventory_dropped_prior_star")
        after, new_names = before, []
        for key, name in names.items():
            if key not in known:
                after = after.append(
                    Collected(star_id="star:" + _sha(key.encode()), name=name, source_sha256=digest)
                )
                new_names.append(name)
        # Recheck all source bytes after planning but before any journal mutation.
        book.unchanged()
        records = after.records[len(before.records) :]
        if records:
            stream.seek(0, os.SEEK_END)
            stream.write("".join(_json(record.model_dump(mode="json")) + "\n" for record in records))
            stream.flush()
            os.fsync(stream.fileno())
        return {
            "mode": "recorded_collection_inventory_import",
            "authority": "recorded_visible_list_not_new_live_inspection",
            "inventory_sha256": digest,
            **inventory_source.binding_fields(),
            "appended_records": len(records),
            "newly_collected": new_names,
            "idempotent": not records,
            "browser_actions": 0,
            "scientific_stages_added": 0,
            "task_receipts_added": 0,
            "write_receipts_added": 0,
            "progress": after.reduce().report(),
        }
