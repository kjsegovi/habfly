"""Read-only reconciliation of a completed numeric run's final footer race."""

import hashlib
import json
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_classification import read_class_choices
from .browser_full_stellar import UNITS, full_stellar_status_projection
from .browser_numeric import screen_identity
from .browser_probe import inspect_page, save_probe
from .browser_stellar import SIMULATION_URL, map_stellar_capture


def reconcile_stellar_footer(page, config, source, output, *, checkpoint):
    source, output = Path(source), Path(output)
    report_raw = (source / "manifest.json").read_bytes()
    manifest = json.loads(report_raw)
    events_raw = (source / "events.jsonl").read_bytes()
    if (
        hashlib.sha256(events_raw).hexdigest() != manifest.get("events_sha256")
        or manifest.get("outcome") != "screen_changed_during_binding"
        or manifest.get("full_stellar_numeric_transport_verified") is not False
        or manifest.get("steps") != 59
        or manifest.get("write_attempts") != 6
        or set(manifest.get("verified_fields", [])) != set(UNITS)
        or manifest.get("optimizer_updates") != 0
        or manifest.get("checkpoint_unchanged") is not True
        or hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        != manifest["provenance"]["checkpoint_sha256"]
    ):
        raise BrowserSafetyStop("unsupported_stellar_footer_reconciliation")
    events = [json.loads(line) for line in events_raw.splitlines()]
    pending = [e["payload"] for e in events if e["event"] == "action_proposed"][-1]
    if (
        pending.get("action_source") != "frozen_lifetime_checkpoint"
        or pending["action"]["target"] != "59:check"
    ):
        raise BrowserSafetyStop("stellar_final_check_not_proposed")
    changes = [e["payload"]["screen_change"] for e in events if "screen_change" in e["payload"]]
    if len(changes) != 1 or changes[0]["path"] != "screen-change-001.json":
        raise BrowserSafetyStop("unsupported_stellar_screen_change")
    diff_raw = (source / "screen-change-001.json").read_bytes()
    if hashlib.sha256(diff_raw).hexdigest() != changes[0]["sha256"]:
        raise BrowserSafetyStop("stellar_change_evidence_tampered")
    diff = json.loads(diff_raw)
    identity = lambda r: screen_identity(full_stellar_status_projection(r))
    if (
        diff["change_count"] != 1
        or diff["truncated"]
        or diff["changes"][0]["path"] != "/frames/0/accessibility"
        or identity(diff["before"]) != identity(diff["after"])
    ):
        raise BrowserSafetyStop("stellar_change_not_exact_footer")
    before = inspect_page(page, config)
    if (
        before["ignored_frame_urls"]
        or len(page.context.pages) != 1
        or identity(before) != identity(diff["after"])
    ):
        raise BrowserSafetyStop("stellar_reconciliation_live_state_changed")
    mapping = map_stellar_capture(
        before,
        capture_sha256=screen_identity(before),
        allow_color_selection=True,
        allow_main_sequence_fields=True,
    )
    values = mapping["observation"]["values"]
    frame = next(f for f in page.frames if f.url == SIMULATION_URL)
    choices, _ = read_class_choices(frame)
    if choices["selected"] != "main_sequence" or not values["conditional_fields_visible"]:
        raise BrowserSafetyStop("stellar_reconciliation_class_changed")
    readbacks = manifest["numeric_readbacks"]
    for name, field in values["browser_field_map"].items():
        if (
            field["current_value"] != readbacks[name]["display_value"]
            or not readbacks[name]["exact_input_verified"]
        ):
            raise BrowserSafetyStop("stellar_reconciliation_readback_changed")
    if set(readbacks) != set(UNITS) or identity(inspect_page(page, config)) != identity(before):
        raise BrowserSafetyStop("stellar_reconciliation_unstable")
    output.mkdir(parents=True, exist_ok=False)
    save_probe(before, output / "capture")
    receipt = {
        "star": mapping["star_name"],
        "full_stellar_numeric_transport_reconciled": True,
        "original_failure_preserved": True,
        "browser_actions": 0,
        "optimizer_updates": 0,
        "original_verified_fields": manifest["verified_fields"],
        "final_check_proposed_by_frozen_policy": True,
        "reconciliation_source": "read_only_exact_footer_validation",
        "task_completed": False,
        "saved": False,
        "submitted": False,
        "classification_correctness_verified": False,
        "source_sha256": {
            "manifest": hashlib.sha256(report_raw).hexdigest(),
            "events": hashlib.sha256(events_raw).hexdigest(),
            "screen_change": hashlib.sha256(diff_raw).hexdigest(),
        },
    }
    persist_json(output / "reconciled.json", receipt)
    return receipt
