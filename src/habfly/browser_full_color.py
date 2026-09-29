"""Frozen color-policy transfer on an already classified six-field stellar view."""

from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_classification import prefix_transition_projection
from .browser_color import BrowserColorEnv, ColorJournal, ColorSession
from .browser_full_stellar import FullStellarSession, full_stellar_status_projection
from .browser_numeric import VISIBLE_PREFIX, NumericSession
from .browser_stellar import COLOR_OPTIONS


class FullStellarColorSession(ColorSession):
    def __init__(self, page, config, journal, reference, *, selected_class, lifetime_prefix):
        self.selected_class = selected_class
        self.calculation_class = "giant" if selected_class in {"red_giant", "supergiant"} else selected_class
        self.lifetime_prefix = lifetime_prefix
        super().__init__(page, config, journal, reference)

    _map_report = FullStellarSession._map_report
    _field_labels = FullStellarSession._field_labels

    def _require_same_screen(self, before, after, reason):
        return NumericSession._require_same_screen(
            self, full_stellar_status_projection(before), full_stellar_status_projection(after), reason
        )

    def _color_projection(self, report):
        target = self.mapping["observation"]["values"]["color"]["capture_target_id"]
        return prefix_transition_projection(full_stellar_status_projection(report), target)

    @staticmethod
    def _resolve_color(frame, mapping):
        matches = []
        for control in frame.get_by_role("combobox").all():
            if not control.is_visible():
                continue
            prefix = control.evaluate(VISIBLE_PREFIX)
            if prefix is not None and "".join(prefix.casefold().split()) in {
                "peakλcolor",
                "peakwavelengthcolor",
            }:
                matches.append(control)
        if len(matches) != 1 or not matches[0].is_enabled():
            raise BrowserSafetyStop("ambiguous_or_disabled_color_control")
        handle = matches[0].element_handle(timeout=2000)
        if not handle or not handle.evaluate("e=>e.tagName==='SELECT' && e.isConnected && !e.multiple"):
            raise BrowserSafetyStop("unsupported_native_color_control")
        visible = handle.evaluate(
            "e=>Array.from(e.options).filter(o=>!o.hidden&&getComputedStyle(o).display!=='none').map(o=>({label:o.label,selected:o.selected,disabled:o.disabled}))"
        )
        if [o["label"] for o in visible] != COLOR_OPTIONS or any(o["disabled"] for o in visible):
            raise BrowserSafetyStop("color_option_inventory_changed")
        selected = [o["label"] for o in visible if o["selected"]]
        if (
            len(selected) > 1
            or (selected[0] if selected else None) != mapping["observation"]["values"]["color"]["selected"]
        ):
            raise BrowserSafetyStop("color_selection_readback_mismatch")
        return handle


def run_full_stellar_color(page, config, output, *, selected_class, lifetime_prefix, experiment, graph_path):
    import torch

    from .training.color import file_hash, require_browser_color_gate

    policy, reference = require_browser_color_gate(experiment, graph_path)
    policy.calibration = {"status": "uncalibrated", "scope": "browser_transfer_not_calibrated"}
    provenance = {
        "color_gate_passed": True,
        "color_reference_hash": reference.checksum,
        "color_checkpoint_sha256": file_hash(Path(experiment) / "training/checkpoint.pt"),
        "browser_execution": "autonomous",
        "optimizer_updates": 0,
        "classified_view": True,
    }
    journal = ColorJournal(output, provenance)
    session = FullStellarColorSession(
        page, config, journal, reference, selected_class=selected_class, lifetime_prefix=lifetime_prefix
    )
    outcome = "not_started"
    try:
        session.start()
        env = BrowserColorEnv(session)
        state = None
        with torch.no_grad():
            for _ in range(8):
                observation = env.observe()
                journal.emit("observation", observation.model_dump(mode="json"))
                action, state, diagnostics = policy.act(observation, state)
                action.action_confidence = action.target_confidence = None
                action.calibrated = False
                journal.emit(
                    "action_proposed",
                    {"action": action.model_dump(mode="json"), "action_source": "frozen_color_checkpoint"},
                )
                result = env.step(action, confirm=lambda _: True, authorization="autonomous_opt_in")
                journal.emit("action_result", result.model_dump(mode="json"))
                activity = (
                    diagnostics if diagnostics.get("activity_pathway") else policy.neural_activity(state)
                )
                journal.emit("neural_activity", activity)
                if result.terminated or result.truncated:
                    outcome = (
                        "color_transport_verified"
                        if env.completed and journal.receipt and not result.failure_reason
                        else result.failure_reason or "policy_stopped"
                    )
                    break
            else:
                outcome = "color_step_limit"
    except Exception as exc:  # noqa: BLE001 - never print driver URLs/call logs
        outcome = str(exc) if isinstance(exc, BrowserSafetyStop) else "full_stellar_color_failed"
        journal.emit("error", {"reason": outcome, "exception_type": type(exc).__name__})
    finally:
        session.stopped = True
        page.remove_listener("dialog", session._dialog)
        if file_hash(Path(experiment) / "training/checkpoint.pt") != provenance["color_checkpoint_sha256"]:
            outcome = "color_checkpoint_changed"
        report = journal.finish(outcome, session.attempts)
    return report
