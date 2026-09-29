"""Explicit readback reconciliation for the observed mass/orbit display stop.

Never retry a native write. Reconstruct the frozen policy's original action
prefix locally, verifying it exactly, and then continue only untouched fields.
The original failed run remains immutable and is not relabelled as successful.
"""

from copy import deepcopy
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_numeric import committed_display, screen_identity
from .browser_planet import map_planet_capture
from .browser_planet_numeric import planet_projection, validate_orbit_readout_change
from .contracts import Action, RuntimeEvent
from .training.planet_sequence import file_hash, read


def checked_capture(directory):
    path = Path(directory) / "observation.json"
    if file_hash(path) != read(Path(directory) / "manifest.json")["observation_sha256"]:
        raise BrowserSafetyStop("recovery_capture_hash_mismatch")
    report = read(path)
    return report, map_planet_capture(report, capture_sha256=screen_identity(report))


def load_mass_orbit_recovery(directory, checkpoint_hash):
    directory = Path(directory)
    report = read(directory / "report.json")
    if (
        report.get("outcome") != "unexpected_planet_copy_side_effect"
        or report.get("checkpoint_unchanged") is not True
        or report.get("optimizer_updates") != 0
        or report.get("provenance", {}).get("checkpoint_sha256") != checkpoint_hash
        or report.get("planet_transport_verified") is not False
        or set(report.get("verified_fields", {})) != {"orbital_radius"}
        or set(report.get("write_attempts", [])) != {"orbital_radius", "planet_mass"}
        or file_hash(directory / "events.jsonl") != report.get("events_sha256")
    ):
        raise BrowserSafetyStop("unsupported_planet_recovery_source")
    base = directory / "native-copies"
    stopped, reserved = read(base / "copy-02-stopped.json"), read(base / "copy-02-reserved.json")
    if (
        stopped
        != {
            "reason": "unexpected_planet_copy_side_effect",
            "retry_allowed": False,
            "write_may_have_occurred": True,
            "task_completed": False,
        }
        or reserved.get("destination") != "planet_mass"
        or reserved.get("unit") != "MEarth"
        or reserved.get("action_source") != "checkpoint"
    ):
        raise BrowserSafetyStop("unsupported_planet_recovery_reservation")
    _initial, mapping = checked_capture(base / "initial")
    before, older = checked_capture(base / "copy-02-before")
    after, newer = checked_capture(base / "copy-02-after")
    if older["observation"]["values"]["browser_field_map"]["planet_mass"]["current_value"] != "":
        raise BrowserSafetyStop("planet_recovery_mass_was_not_blank")
    validate_orbit_readout_change(before, after, "planet_mass")
    if planet_projection(before, older, "planet_mass") != planet_projection(after, newer, "planet_mass"):
        raise BrowserSafetyStop("planet_recovery_has_other_side_effects")
    receipts = deepcopy(report["verified_fields"])
    for name, item in {**receipts, "planet_mass": reserved}.items():
        field = newer["observation"]["values"]["browser_field_map"][name]
        display = committed_display(item["value"], field["current_value"])
        if field["unit"] != item["unit"]:
            raise BrowserSafetyStop("planet_recovery_unit_changed")
        receipts[name] = {
            **item,
            "display": display,
            "readback_verified": True,
            "correctness_verified": False,
            "native_write_repeated": False,
            "reconciled_from_stopped_run": name == "planet_mass",
        }
    if newer["star_name"] != mapping["star_name"]:
        raise BrowserSafetyStop("planet_recovery_star_changed")
    # Replay observations stay answer-free initially. Only receipts permit the
    # two completed copies to be replayed as local state transitions.
    initial_values = mapping["observation"]["values"]
    current_values = deepcopy(newer["observation"]["values"])
    for name in receipts:
        if initial_values["browser_field_map"][name]["current_value"] != "":
            raise BrowserSafetyStop("planet_recovery_initial_answers_present")
        current_values["browser_field_map"][name]["current_value"] = ""
    if current_values != initial_values:
        raise BrowserSafetyStop("planet_recovery_measurements_changed")
    rows = [
        RuntimeEvent.model_validate_json(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
    ]
    if [e.sequence for e in rows] != list(range(len(rows))) or len({e.run_id for e in rows}) != 1:
        raise BrowserSafetyStop("invalid_planet_recovery_event_sequence")
    actions = [
        Action.model_validate({k: v for k, v in e.payload.items() if k in Action.model_fields})
        for e in rows
        if e.event == "action_proposed" and "observation_revision" in e.payload
    ]
    if (
        len(actions) != report["steps"] + 1
        or not 1 <= len(actions) <= 128
        or actions[-1].kind != "CLICK"
        or not (actions[-1].target or "").endswith(":copy")
    ):
        raise BrowserSafetyStop("invalid_planet_recovery_action_prefix")
    return {
        "initial_mapping": mapping,
        "after_report": after,
        "after_mapping": newer,
        "receipts": receipts,
        "actions": actions,
        "provenance": {
            "failed_run": str(directory),
            "report_sha256": file_hash(directory / "report.json"),
            "events_sha256": report["events_sha256"],
            "policy_prefix_actions": len(actions),
            "retained_native_fields": sorted(receipts),
            "native_write_retries": 0,
        },
    }


class ReconciledPlanetSession:
    def __init__(self, native, recovery, output, emit):
        self.native, self.recovery, self.emit = native, recovery, emit
        self.mapping = recovery["initial_mapping"]
        self.verified = {}
        self.replay_index = -1
        current, mapped, choices, _ = native.current()
        if choices["selected"] is not None or planet_projection(current, mapped) != planet_projection(
            recovery["after_report"], recovery["after_mapping"]
        ):
            raise BrowserSafetyStop("planet_recovery_live_state_changed")
        persist_json(
            Path(output) / "reconciliation.json",
            {
                **recovery["provenance"],
                "review": "known_mass_dependent_orbit_readout_only",
                "receipts": recovery["receipts"],
                "task_completed": False,
                "original_failure_preserved": True,
                "no_browser_writes_during_reconciliation": True,
                "prefix_reconstruction": "offline_then_live_recheck_before_continuation",
            },
        )

    @property
    def attempted(self):
        return self.native.attempted

    def current(self):
        # Original observations/actions reconstruct local recurrent state only.
        # No native action is possible until every retained copy is replayed;
        # recheck the full live screen before the first subsequent decision.
        if self.replay_index < len(self.recovery["actions"]):
            return None
        return self.native.current()

    def close(self):
        self.native.close()

    def require_prefix(self, action, index):
        prefix = self.recovery["actions"]
        if index != self.replay_index + 1:
            raise BrowserSafetyStop("planet_recovery_prefix_out_of_order")
        if index < len(prefix) and action != prefix[index]:
            raise BrowserSafetyStop("frozen_planet_recovery_prefix_diverged")
        if index == len(prefix):
            self.native.current()
        self.replay_index = index

    def copy(self, destination, text, unit, *, source):
        retained = self.recovery["receipts"]
        if destination in retained:
            receipt = retained[destination]
            if destination in self.verified or (text, unit, source) != (
                receipt["value"],
                receipt["unit"],
                "checkpoint",
            ):
                raise BrowserSafetyStop("planet_recovery_copy_diverged")
            self.verified[destination] = receipt
            self.emit(
                "action_result",
                {**receipt, "replayed_native_receipt": True, "browser_action_executed": False},
            )
            return receipt
        if self.replay_index < len(self.recovery["actions"]) or not set(retained).issubset(self.verified):
            raise BrowserSafetyStop("planet_recovery_prefix_incomplete")
        receipt = self.native.copy(destination, text, unit, source=source)
        self.verified[destination] = receipt
        return receipt
