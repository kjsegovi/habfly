"""Six-field transport for an explicitly selected, visibly verified star class.

Scientific class selection is a separate stage. This module does not classify,
grade, save, assess or submit, and never receives expected answers.
"""

import hashlib
import json
import time
from copy import deepcopy

from .browser import BrowserSafetyStop
from .browser_classification import read_class_choices
from .browser_numeric import NumericJournal, NumericSession, screen_identity
from .browser_policy import BrowserToolEnv
from .browser_stellar import CONDITIONAL_FIELDS, FIELDS, StellarMappingError, map_stellar_capture
from .contracts import StepResult, validate_action
from .environments.lifetime import TEMPLATES, required_fields
from .environments.local_stellar import LocalStellarEnv
from .knowledge import load_knowledge_pack
from .lifetime_prefix import lifetime_to_prefix

UNITS = {
    "distance": "ly",
    "luminosity": "Lsun",
    "temperature": "K",
    "mass": "Msun",
    "radius": "Rsun",
    "lifetime": "yr",
}


def full_stellar_status_projection(report):
    """Normalize only Save's busy flag and exact three/six-field autosave footers.

    This transport cannot click Save. Raw evidence is retained. No other enabled
    state, feedback, measurement or answer is removed.
    """
    from .browser_stellar import SIMULATION_URL

    value = deepcopy(report)
    for frame in value.get("frames", []):
        if frame.get("url") != SIMULATION_URL:
            continue
        for control in frame["controls"]:
            if control["role"] == "button" and control["accessibility"] in {
                '- button "Save"',
                '- button "Save" [disabled]',
            }:
                control["accessibility"] = '- button "Save"'
                control["enabled"] = True
        frame["accessibility"] = frame["accessibility"].replace(
            '- button "Save" [disabled]', '- button "Save"'
        )
        text_suffix = "\n1 Rs\nData saved\nSave"
        ax_suffixes = (
            '\n- text: main sequence red giant supergiant white dwarf 1 Rs Data saved\n- button "Save"',
            (
                "\n- text: mass, radius and lifetime are only relevant for main sequence stars "
                'main sequence red giant supergiant white dwarf 1 Rs Data saved\n- button "Save"'
            ),
        )
        # Text and accessibility snapshots are separate browser reads. The
        # transient notice can disappear between them; match each exact footer
        # independently. Never suppress the phrase at another screen location.
        if frame["text"].count("Data saved") == 1 and frame["text"].endswith(text_suffix):
            frame["text"] = frame["text"].removesuffix(text_suffix) + "\n1 Rs\nSave"
        if frame["accessibility"].count("Data saved") == 1:
            for ax_suffix in ax_suffixes:
                if frame["accessibility"].endswith(ax_suffix):
                    frame["accessibility"] = frame["accessibility"].removesuffix(
                        ax_suffix
                    ) + ax_suffix.replace(" Data saved", "")
                    break
    return value


class FullStellarSession(NumericSession):
    ALLOW_COLOR_SELECTION = True
    MAX_WRITES = 6
    MAX_CALCULATIONS = 12

    def _require_same_screen(self, before, after, reason):
        return super()._require_same_screen(
            full_stellar_status_projection(before), full_stellar_status_projection(after), reason
        )

    def _validate_transition(self, before_report, before_mapping, report, *args, **kwargs):
        return super()._validate_transition(
            full_stellar_status_projection(before_report),
            before_mapping,
            full_stellar_status_projection(report),
            *args,
            **kwargs,
        )

    def __init__(self, page, config, journal, *, selected_class, lifetime_prefix=None):
        if selected_class not in {"main_sequence", "red_giant", "supergiant", "white_dwarf"}:
            raise ValueError("An explicit visible class selection is required")
        self.selected_class = selected_class
        self.calculation_class = "giant" if selected_class in {"red_giant", "supergiant"} else selected_class
        self.lifetime_prefix = lifetime_prefix
        if selected_class == "main_sequence":
            lifetime_to_prefix(1, lifetime_prefix)  # Validate explicit choice, do not select one.
        elif lifetime_prefix is not None:
            raise ValueError("Lifetime prefix is inapplicable")
        super().__init__(page, config, journal)

    def _map_report(self, report):
        mapping = map_stellar_capture(
            report,
            capture_sha256=screen_identity(report),
            allow_color_selection=True,
            allow_main_sequence_fields=True,
        )
        values = mapping["observation"]["values"]
        from .browser_stellar import SIMULATION_URL

        frames = [f for f in self.page.frames if f.url == SIMULATION_URL]
        if len(frames) != 1:
            raise BrowserSafetyStop("simulation_frame_changed")
        choices, _ = read_class_choices(frames[0])
        if choices["selected"] != self.selected_class:
            raise BrowserSafetyStop("selected_class_changed")
        if values["conditional_fields_visible"] != (self.selected_class == "main_sequence"):
            raise BrowserSafetyStop("class_conditional_fields_disagree")
        if (
            self.selected_class == "main_sequence"
            and values["lifetime_prefix"]["selected"] != self.lifetime_prefix
        ):
            raise BrowserSafetyStop("lifetime_prefix_changed")
        values["star_class"] = self.calculation_class
        values["selected_browser_class"] = self.selected_class
        return mapping

    def _field_labels(self, mapping):
        return {**FIELDS, **CONDITIONAL_FIELDS, "lifetime(years)": ("lifetime", self.lifetime_prefix)}

    def _calculate_result(self, operation, bindings):
        return self.calculator.execute(operation, bindings, star_class=self.calculation_class)

    def _copy_intent(self, result, destination):
        if destination != "lifetime":
            return super()._copy_intent(result, destination)
        if not result.ok or result.unit != "yr" or result.operation_id != "lifetime":
            raise BrowserSafetyStop("incompatible_lifetime_result")
        field = self.mapping["observation"]["values"]["browser_field_map"].get("lifetime")
        if not field or not field["enabled"] or field["unit"] != self.lifetime_prefix:
            raise BrowserSafetyStop("unavailable_lifetime_destination")
        return {"value": lifetime_to_prefix(result.value, self.lifetime_prefix), "unit": self.lifetime_prefix}


class FullStellarToolEnv(BrowserToolEnv):
    def __init__(self, session, seed):
        self.session = session
        self.native_results = {}
        self.copy_approved = False
        self.copy_authorization = "human_confirmation"
        self.transport_verified = False
        values = session.mapping["observation"]["values"]
        case = {
            "seed": seed,
            "split": "test",
            "star_class": session.calculation_class,
            "required": required_fields(session.calculation_class),
            "measurements": values["measurements"],
        }
        LocalStellarEnv.__init__(self, session.calculator, [case], max_steps=128)
        self.reset(seed=seed)

    def observe(self):
        observation = super().observe()
        observation.instruction = TEMPLATES["manual"][0]
        observation.values["selected_browser_class"] = self.session.selected_class
        observation.values["lifetime_prefix"] = self.session.lifetime_prefix
        observation.progress.update(
            task="browser_full_stellar_numeric", required_fields=len(self.case["required"])
        )
        return observation

    def step(self, action):
        control = validate_action(self.observe(), action)
        key = control.id.split(":", 1)[1] if control else ""
        # Menu/binding decisions use the pinned visible measurement snapshot.
        # Native calculate/copy paths retain their full live checks, including
        # the post-approval copy check. Recheck again before claiming completion.
        self.session._guard()
        if key == "check":
            self.session._current()
        _, _, _, _, info = LocalStellarEnv.step(self, action)
        result = StepResult.model_validate(info["result"])
        if key.startswith("unit_") and action.value != UNITS.get(key[5:]):
            result.failure_reason = "selected_unit_does_not_match_visible_field"
        if key == "check":
            required = set(self.case["required"])
            self.transport_verified = set(self.session.verified) == required and self.units == {
                k: UNITS[k] for k in required
            }
            result.terminated = True
            result.failure_reason = (
                None if self.transport_verified else "full_stellar_numeric_transport_incomplete"
            )
            self.feedback = "Numeric transport checked; classification correctness, course score and completion are unverified."
        if self.tool_error or result.failure_reason:
            result.terminated = True
            result.failure_reason = result.failure_reason or self.tool_error
        self.terminated = result.terminated
        result.observation = self.observe()
        result.reward = result.cumulative_reward = 0
        result.reward_components = {}
        return result


def load_full_stellar_policy(dataset, checkpoint, graph_path):
    import torch

    from .data import load_graph
    from .training.checkpoints import load_checkpoint
    from .training.luminosity_session import load_chain_session

    pack = load_knowledge_pack()
    # A fixed already-published manual seed validates the promotion artifacts;
    # its case is never provided to the browser policy or used as an answer.
    identity, _ = load_chain_session(dataset, checkpoint, pack, 8500000, task="lifetime")
    torch.set_num_threads(1)
    graph = load_graph(graph_path)
    policy, manifest = load_checkpoint(checkpoint, graph, content_pack=identity)
    if len(graph.body_ids) != 2000 or manifest["model"]["observation_encoding"] != "structured_tool_v6":
        raise ValueError("Full stellar transfer requires the promoted 2000-node six-calculation policy")
    policy.calibration = {"status": "uncalibrated", "scope": "browser_transfer_not_calibrated"}
    return policy, {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "graph_hash": manifest["graph_hash"],
        "knowledge_pack_hash": pack.checksum,
        "classification_source": "separate_explicit_selection",
        "optimizer_updates": 0,
        "calibration_scope": "browser_transfer_not_calibrated",
    }


def run_full_stellar_numeric(
    page,
    config,
    output,
    *,
    selected_class,
    lifetime_prefix,
    dataset,
    checkpoint,
    graph_path,
    seed=8500000,
    notify=lambda _: None,
):
    """One capped inference-only diagnostic, no setup, classification choice or scoring.

    The explicitly selected class must already be painted on the same visible
    star. A successful result verifies transport, not correctness of that class.
    """
    import torch

    policy, provenance = load_full_stellar_policy(dataset, checkpoint, graph_path)
    provenance.update(
        browser_execution="autonomous",
        scope="six_calculation_browser_transport",
        selected_class=selected_class,
        lifetime_prefix=lifetime_prefix,
        seed=seed,
        browser_validation="full_at_calculation_each_native_copy_and_completion",
        local_tool_observations="pinned_visible_measurements_not_live_browser_state",
    )
    journal = NumericJournal(output, learned_policy=True, provenance=provenance)
    session = FullStellarSession(
        page, config, journal, selected_class=selected_class, lifetime_prefix=lifetime_prefix
    )
    started = time.monotonic()
    steps, outcome, verified = 0, "not_started", False
    try:
        session.start()
        env = FullStellarToolEnv(session, seed)
        state = None
        with torch.no_grad():
            for _ in range(128):
                observation = env.observe()
                journal.emit("observation", observation.model_dump(mode="json"))
                action, state, _diagnostics = policy.act(observation, state)
                # Browser-domain calibration has not been established.
                action.action_confidence = None
                action.target_confidence = None
                action.calibrated = False
                journal.emit(
                    "action_proposed",
                    {"action": action.model_dump(mode="json"), "action_source": "frozen_lifetime_checkpoint"},
                )
                env.copy_approved = True
                env.copy_authorization = "autonomous_opt_in"
                result = env.step(action)
                steps += 1
                journal.emit("action_result", result.model_dump(mode="json"))
                journal.emit("neural_activity", policy.neural_activity(state))
                notify(
                    {
                        "steps": steps,
                        "verified_fields": list(session.verified),
                        "failure": result.failure_reason,
                    }
                )
                if result.terminated or result.truncated:
                    verified = env.transport_verified and result.failure_reason is None
                    outcome = (
                        "full_stellar_numeric_transport_verified"
                        if verified
                        else result.failure_reason or "policy_stopped"
                    )
                    break
            else:
                outcome = "policy_step_limit"
    except Exception as exc:  # noqa: BLE001 - redact Playwright URLs and call logs
        outcome = (
            str(exc)
            if isinstance(exc, (BrowserSafetyStop, StellarMappingError))
            else "browser_full_stellar_operation_failed"
        )
        journal.emit(
            "error",
            {
                "reason": outcome,
                "exception_type": type(exc).__name__,
                "writes_may_have_occurred": session.attempts > 0,
            },
        )
    finally:
        session.stopped = True
        page.remove_listener("dialog", session._dialog)
        report = {
            "schema_version": 1,
            "scope": "six_calculation_browser_transport",
            "outcome": outcome,
            "full_stellar_numeric_transport_verified": verified,
            "learned_policy": True,
            "classification_learned": False,
            "classification_correctness_verified": False,
            "task_completed": False,
            "browser_acceptance_passed": False,
            "saved": False,
            "assessment_performed": False,
            "submitted": False,
            "optimizer_updates": 0,
            "steps": steps,
            "write_attempts": session.attempts,
            "verified_fields": session.verified,
            "numeric_readbacks": journal.numeric_readbacks,
            "provenance": provenance,
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint_unchanged": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            == provenance["checkpoint_sha256"],
        }
        if not report["checkpoint_unchanged"]:
            report["full_stellar_numeric_transport_verified"] = False
            report["outcome"] = "checkpoint_changed"
        journal.emit("episode_summary", report)
        journal.stream.close()
        report["events_sha256"] = hashlib.sha256((output / "events.jsonl").read_bytes()).hexdigest()
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
