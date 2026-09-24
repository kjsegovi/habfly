"""Headless JSONL engine shared by the Rust TUI and recorded demonstrations."""

import json
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field

from .contracts import Contract, Observation, RuntimeCommand, RuntimeEvent, StepResult, task_completed

LOCAL_CHECKPOINT_TASKS = {"distance", "luminosity", "temperature", "mass", "radius", "lifetime"}
CALCULATION_TASKS = LOCAL_CHECKPOINT_TASKS | {"stellar"}


class RunOptions(Contract):
    seed: int = 0
    stars: int = Field(default=1, ge=1, le=30)
    policy: Literal["expert", "untrained", "checkpoint"] = "expert"
    graph: Path | None = None
    checkpoint: Path | None = None
    content_pack: Path | None = None
    artifact_dir: Path = Path("experiments/runs")
    environment: Literal["simulator", "browser"] = "simulator"
    browser_config: Path | None = None
    browser_setup: Literal["manual", "automatic"] = "manual"
    browser_execution: Literal["supervised", "autonomous"] = "supervised"
    paused: bool = False
    interval: float = Field(default=0.2, ge=0, le=10)
    task: Literal[
        "mini_habworlds",
        "stellar",
        "distance",
        "luminosity",
        "temperature",
        "mass",
        "radius",
        "lifetime",
        "browser_numeric",
    ] = "mini_habworlds"
    spreadsheet_config: Path | None = None
    calculation_backend: Literal["local", "google_sheets"] | None = None
    knowledge_pack: Path | None = None
    dataset: Path | None = None

    @property
    def backend(self):
        return self.calculation_backend or ("google_sheets" if self.spreadsheet_config else "local")


class Runtime:
    def __init__(self, output=None):
        self.output = output or sys.stdout
        self.sequence = 0
        self.run_id = None
        self.status = "idle"
        self.options = RunOptions()
        self.env = self.policy = self.neural_state = self.observation = None
        self.trace = None
        self.trace_path = None
        self.replay_events = None
        self.replay_context = None
        self.last_tick = 0.0
        self.spreadsheet_adapter = None
        self.expected_browser_identity = None  # Optional in-process batch constraint, never credentials.

    def emit(self, event, payload):
        item = RuntimeEvent(event=event, sequence=self.sequence, run_id=self.run_id, payload=payload)
        self.sequence += 1
        line = item.model_dump_json()
        self.output.write(line + "\n")
        self.output.flush()
        if self.trace:
            self.trace.write(line + "\n")
            self.trace.flush()
        return item

    def state(self):
        if self.replay_context is not None:
            self.emit(
                "state",
                {
                    **self.replay_context,
                    "status": self.status,
                    "replay": True,
                    "trace_path": str(self.trace_path),
                },
            )
            return
        self.emit(
            "state",
            {
                "status": self.status,
                "seed": self.options.seed,
                "stars": self.options.stars,
                "policy": self.options.policy,
                "stage": self.options.task
                if self.options.task in CALCULATION_TASKS
                else ("synthetic_demo" if self.options.environment == "simulator" else "browser_inference"),
                "graph": str(self.options.graph or "synthetic-32"),
                "checkpoint": str(self.options.checkpoint or "none"),
                "browser_status": "visible" if self.options.environment == "browser" else "not_connected",
                "calculation_backend": self.options.backend
                if self.options.task in CALCULATION_TASKS
                else None,
                "calculation_mode": (
                    "local_tool_assisted" if self.options.backend == "local" else "google_sheets"
                )
                if self.options.task in CALCULATION_TASKS
                else None,
                "trace_path": str(self.trace_path) if self.trace_path else None,
                **(self.env.state() if self.options.task == "browser_numeric" and self.env else {}),
                **(
                    {
                        "stage": f"{self.options.browser_execution}_browser_numeric_transfer",
                        "calculation_backend": "local",
                        "calculation_mode": "local_tool_assisted",
                    }
                    if self.options.task == "browser_numeric"
                    else {}
                ),
            },
        )

    def start(self, payload):
        options = RunOptions.model_validate(payload)
        if options.browser_setup != "manual" and options.task != "browser_numeric":
            raise ValueError("Automatic setup is supported only for the supervised browser diagnostic")
        if options.browser_execution == "autonomous" and (
            options.task != "browser_numeric" or options.browser_setup != "automatic" or options.stars != 1
        ):
            raise ValueError("Autonomous execution requires automatic browser_numeric setup and one star")
        if options.policy == "checkpoint" and not options.checkpoint:
            raise ValueError("checkpoint policy requires checkpoint path")
        if options.environment == "browser" and options.policy != "checkpoint":
            raise ValueError("Browser runs require a trained checkpoint")
        if options.task in CALCULATION_TASKS and options.environment == "browser":
            raise ValueError("Stellar v1 cannot run against the HabWorlds browser")
        if options.task in CALCULATION_TASKS and not options.dataset:
            raise ValueError("Stellar runtime requires dataset")
        if options.task in LOCAL_CHECKPOINT_TASKS and (
            options.policy != "checkpoint"
            or not options.graph
            or options.backend != "local"
            or options.spreadsheet_config
        ):
            raise ValueError(
                "Local calculation demo requires a checkpoint, graph and local-only calculation backend"
            )
        self.close()
        self.options = options
        self.run_id = uuid.uuid4().hex
        options.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = options.artifact_dir / f"{self.run_id}.jsonl"
        self.trace = self.trace_path.open("x")
        if options.task == "browser_numeric":
            from .browser_policy import BrowserPolicyBridge, load_browser_policy

            self.policy, provenance = load_browser_policy(options)
            if self.expected_browser_identity is not None and any(
                provenance.get(key) != value for key, value in self.expected_browser_identity.items()
            ):
                from .browser import BrowserSafetyStop

                raise BrowserSafetyStop("batch_provenance_changed")
            self.env = BrowserPolicyBridge(options, options.artifact_dir / self.run_id, provenance)
            self.neural_state = self.observation = None
            self.status = (
                "running" if options.browser_execution == "autonomous" and not options.paused else "paused"
            )
            self.emit(
                "hello",
                {
                    "protocol_version": 1,
                    "policy": "checkpoint",
                    "synthetic": False,
                    "activity_source": "checkpoint",
                    "provenance": provenance,
                    "scope": f"{options.browser_execution}_browser_numeric_transfer",
                },
            )
            self.state()
            return
        import torch

        from .content import load_content_pack
        from .data import load_graph, make_demo_graph
        from .environments import MiniHabWorlds
        from .model import ConnectomePolicy

        torch.set_num_threads(1)
        torch.manual_seed(options.seed)
        self.pack = load_content_pack(options.content_pack)
        graph = load_graph(options.graph) if options.graph else make_demo_graph()
        identity = self.pack.model_dump(mode="json")
        if options.task == "stellar":
            from .training.stellar import load_dataset, training_content

            stellar_manifest, stellar_data = load_dataset(options.dataset)
            identity = training_content(stellar_manifest)
        elif options.task in LOCAL_CHECKPOINT_TASKS:
            from .knowledge import load_knowledge_pack
            from .training.distance_session import load_distance_session
            from .training.luminosity_session import load_chain_session

            config = load_knowledge_pack(options.knowledge_pack)
            if options.task == "distance":
                identity, distance_case = load_distance_session(
                    options.dataset, options.checkpoint, config, options.seed
                )
            else:
                identity, distance_case = load_chain_session(
                    options.dataset, options.checkpoint, config, options.seed, task=options.task
                )
        if options.policy == "checkpoint":
            from .training.checkpoints import load_checkpoint

            self.policy, _manifest = load_checkpoint(options.checkpoint, graph, content_pack=identity)
        else:
            self.policy = ConnectomePolicy(graph, hidden_size=8)
        self.policy.eval()
        if options.task == "stellar":
            from .config import calculation_config
            from .training.stellar import make_adapter, make_environment

            config = calculation_config(options)
            if config.content_identity() != stellar_manifest["content"]:
                raise ValueError("Calculation backend and dataset mismatch")
            cases = stellar_data["test"]["cases"]
            if options.seed not in {c["seed"] for c in cases}:
                raise ValueError("Choose a recorded test seed, normally starting at 300000")
            self.spreadsheet_adapter = make_adapter(config)
            try:
                self.spreadsheet_adapter.verify()
                self.env = make_environment(self.spreadsheet_adapter, cases)
                observation, _ = self.env.reset(seed=options.seed)
                self.observation = Observation.model_validate(observation)
            except Exception:
                self.close()
                raise
        elif options.task in LOCAL_CHECKPOINT_TASKS:
            from .environments.distance_diagnostic import DistanceDiagnosticEnv
            from .knowledge import LocalCalculator
            from .training.chained_workflow import workflow_spec

            self.spreadsheet_adapter = LocalCalculator(config)
            self.spreadsheet_adapter.verify()
            workflow = workflow_spec(options.task) if options.task != "distance" else None
            environment = workflow.environment if workflow else DistanceDiagnosticEnv
            self.env = environment(
                self.spreadsheet_adapter,
                [distance_case],
                max_steps=workflow.max_steps if workflow else 32,
            )
            observation, _ = self.env.reset(seed=options.seed)
            self.observation = Observation.model_validate(observation)
        elif options.environment == "browser":
            from .browser import BrowserConfig, TorusBrowser

            if not options.browser_config:
                raise ValueError("browser_config is required")
            self.env = TorusBrowser(BrowserConfig.model_validate_json(options.browser_config.read_text()))
            self.observation = self.env.start()
        else:
            self.env = MiniHabWorlds(stars=options.stars, pack=self.pack)
            observation, _ = self.env.reset(seed=options.seed)
            self.observation = Observation.model_validate(observation)
        self.neural_state = None
        self.status = "paused" if options.paused else "running"
        self.emit(
            "hello",
            {
                "protocol_version": 1,
                "policy": options.policy,
                "content_pack": config.content_identity()
                if options.task in CALCULATION_TASKS
                else self.pack.id,
                "synthetic": self.pack.synthetic,
                "activity_source": "untrained_observer" if options.policy == "expert" else options.policy,
            },
        )
        self.state()
        self.emit("observation", self.observation.model_dump(mode="json"))

    def tick(self):
        if self.replay_events is not None:
            try:
                item = next(self.replay_events)
            except StopIteration:
                self.replay_events = None
                self.status = "completed"
                self.state()
                return
            payload = {**item.payload, "replay": True}
            if item.event == "state":
                payload["recorded_status"] = payload.get("status")
                payload["status"] = self.status
            self.emit(item.event, payload)
            return
        if self.env is None or self.status not in {"running", "paused"}:
            raise ValueError("No active run")
        if self.options.task == "browser_numeric":
            automatic = self.options.browser_execution == "autonomous" and self.status == "running"
            payload = {"approve_copy": True} if automatic and self.env.phase == "awaiting_copy" else {}
            self.browser_step(payload, automatic=automatic)
            self.last_tick = time.monotonic()
            return
        import torch

        with torch.no_grad():
            proposed, self.neural_state, _diagnostics = self.policy.act(self.observation, self.neural_state)
        if self.options.policy == "expert":
            if self.options.task == "stellar":
                from .environments.stellar import stellar_expert

                proposed = (
                    self.env.expert_action(self.observation)
                    if hasattr(self.env, "expert_action")
                    else stellar_expert(self.observation)
                )
            else:
                from .environments import expert_action

                proposed = expert_action(self.observation, self.pack)
        proposed.observation_revision = self.observation.revision
        self.emit(
            "action_proposed",
            {
                **proposed.model_dump(mode="json"),
                "action_source": "scripted_expert"
                if self.options.policy == "expert"
                else self.options.policy,
                "calibration_scope": self.policy.calibration.get("scope", "uncalibrated")
                if self.options.policy != "expert"
                else "not_applicable_to_expert",
                "calibration": self.policy.calibration if self.options.policy != "expert" else {},
            },
        )
        activity = self.policy.neural_activity(self.neural_state)
        if "groups" not in activity:
            activity["groups"] = {
                name: {"mean": value} for name, value in activity.get("populations", {}).items()
            }
        activity["activity_source"] = (
            "untrained_observer" if self.options.policy == "expert" else self.options.policy
        )
        self.emit("neural_activity", activity)
        if self.options.environment == "browser":
            result = self.env.step(proposed)
        else:
            _, _, _, _, info = self.env.step(proposed)
            result = StepResult.model_validate(info["result"])
        self.observation = result.observation
        self.emit("action_result", result.model_dump(mode="json"))
        self.emit("observation", self.observation.model_dump(mode="json"))
        if result.terminated or result.truncated:
            completed = task_completed(self.observation)
            self.status = "completed" if completed else "stopped"
            self.emit(
                "episode_summary",
                {
                    "completed": completed,
                    "steps": result.steps,
                    "reward": result.cumulative_reward,
                    "progress": self.observation.progress,
                    "failure_reason": result.failure_reason,
                    "policy": self.options.policy,
                },
            )
            self.state()
            self.env.close()
        self.last_tick = time.monotonic()

    def browser_step(self, payload, *, automatic=False):
        """One decision or copy per tick; explicit autonomous opt-in never changes scope."""
        bridge = self.env
        if automatic and (self.options.browser_execution != "autonomous" or self.status != "running"):
            raise ValueError("Automatic browser steps require a running autonomous run")
        if not bridge or (not automatic and self.status != "paused"):
            raise ValueError("No paused browser diagnostic")
        if bridge.phase == "setting_up":
            raise ValueError("Scripted setup is in progress; wait for ready or press a to abort")
        ready = payload == {"browser_ready": True}
        approve = payload == {"approve_copy": True}
        if payload and not (ready or approve):
            raise ValueError("Unknown browser step payload")
        if ready and bridge.phase != "awaiting_ready":
            raise ValueError("Browser already captured")
        if approve and bridge.phase != "awaiting_copy":
            raise ValueError("No pending copy")
        if not payload and bridge.phase != "ready":
            raise ValueError("Use b to capture the ready browser, or y to confirm its pending copy")
        try:
            if ready:
                self.observation = bridge.ready()
                self.emit("observation", self.observation.model_dump(mode="json"))
                self.state()
                return
            if approve:
                result = bridge.approve(automatic=automatic)
            else:
                import torch

                with torch.no_grad():
                    proposed, self.neural_state, _ = self.policy.act(self.observation, self.neural_state)
                proposed.observation_revision = self.observation.revision
                # Local calibration cannot establish browser probabilities.
                proposed.action_confidence = proposed.target_confidence = None
                proposed.calibrated = False
                self.emit(
                    "action_proposed",
                    {
                        **proposed.model_dump(mode="json"),
                        "action_source": "checkpoint",
                        "calibration_scope": "browser_transfer_not_calibrated",
                    },
                )
                activity = self.policy.neural_activity(self.neural_state)
                activity["activity_source"] = "checkpoint"
                self.emit("neural_activity", activity)
                result = bridge.propose(proposed)
            if result is not None:
                self.observation = result.observation
                self.emit("action_result", result.model_dump(mode="json"))
                self.emit("observation", self.observation.model_dump(mode="json"))
                if result.terminated or result.truncated:
                    self.status = "stopped"  # This is not a completed HabWorlds star.
                    bridge.close()
                    self.emit(
                        "episode_summary",
                        {
                            **bridge.summary,
                            "completed": False,
                            "steps": result.steps,
                            "transport_artifacts": str(bridge.journal.output),
                        },
                    )
            self.state()
        except Exception as exc:  # noqa: BLE001 - fail closed without leaking browser URLs
            # Playwright exceptions may contain session URLs. Never serialize them.
            import re

            from .browser import BrowserSafetyStop
            from .browser_stellar import StellarMappingError

            code = (
                str(exc)
                if isinstance(exc, (BrowserSafetyStop, StellarMappingError))
                else "browser_operation_failed"
            )
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,100}", code):
                code = "browser_operation_failed"
            bridge.outcome = code
            self.status = "stopped"
            bridge.close()
            self.emit("error", {"type": "BrowserDiagnosticStop", "message": code})
            self.emit("episode_summary", {**bridge.summary, "completed": False, "failure_reason": code})
            self.state()

    def setup_tick(self):
        bridge = self.env
        active = self.status == ("running" if self.options.browser_execution == "autonomous" else "paused")
        if not bridge or not active or bridge.phase != "setting_up":
            return
        from .browser_setup import SetupStop

        try:
            prior_stage = bridge.setup.stage
            observation = bridge.advance_setup()
            if observation is not None:
                self.observation = observation
                self.emit("observation", observation.model_dump(mode="json"))
                self.last_tick = time.monotonic()
            if observation is not None or bridge.setup.stage != prior_stage:
                self.state()
        except SetupStop as exc:
            bridge.outcome = str(exc)
            self.status = "stopped"
            bridge.close(keep_browser_open=True)
            self.emit("error", {"type": "BrowserSetupStop", "message": str(exc)})
            self.emit("episode_summary", {**bridge.summary, "completed": False, "failure_reason": str(exc)})
            self.state()

    def advance_if_due(self):
        """Shared scheduler for JSONL/TUI and bounded reliability runs.

        Commands are handled before this method. Copy proposals and actual
        copies occupy separate ticks so pause/abort can cancel a pending write.
        """
        if (
            self.replay_events is None
            and self.options.task == "browser_numeric"
            and self.env
            and self.env.phase == "setting_up"
        ):
            self.setup_tick()
        elif self.status == "running" and time.monotonic() - self.last_tick >= self.options.interval:
            self.tick()

    def command(self, message):
        command = RuntimeCommand.model_validate(message)
        if command.command == "start":
            try:
                self.start(command.payload)
            except Exception:
                self.status = "stopped"
                self.close()
                raise
        elif command.command == "replay":
            source = Path(command.payload["path"])
            # Validate the complete log before showing any replay events.
            items = read_trace(source)
            self.close()
            self.run_id = f"replay-{uuid.uuid4().hex}"
            self.trace_path = source
            self.replay_context = next((item.payload for item in items if item.event == "state"), {})
            self.replay_events = iter(items)
            self.status = "running"
            self.state()
        elif command.command == "save_trace":
            if not self.trace_path:
                raise ValueError("No trace available")
            if self.trace:
                self.trace.flush()
            path = command.payload.get("path")
            if path and Path(path).resolve() != self.trace_path.resolve():
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with Path(path).open("x") as destination:
                    destination.write(self.trace_path.read_text())
            self.emit("state", {"status": self.status, "trace_path": str(path or self.trace_path)})
        elif command.command == "abort":
            if self.status not in {"running", "paused"}:
                # Quitting a completed demo must not replace its truthful success
                # summary with a second, operator-aborted failure.
                self.state()
                self.close()
                return
            self.status = "aborted"
            self.emit("episode_summary", {"completed": False, "failure_reason": "operator_aborted"})
            self.state()
            self.close()
        elif command.command in {"pause", "resume"}:
            if self.status not in {"running", "paused"}:
                raise ValueError("No active run")
            if (
                self.options.task == "browser_numeric"
                and self.options.browser_execution != "autonomous"
                and self.replay_events is None
                and command.command == "resume"
            ):
                raise ValueError("Browser diagnostic is single-step only: n steps; y confirms one copy")
            self.status = "paused" if command.command == "pause" else "running"
            self.state()
        elif command.command == "step":
            if self.status != "paused":
                if (
                    self.options.task == "browser_numeric"
                    and self.env is not None
                    and self.env.phase == "finished"
                    and self.replay_events is None
                ):
                    raise ValueError(self.env.state()["browser_guidance"])
                raise ValueError("Pause before single-stepping")
            if self.options.task == "browser_numeric" and self.replay_events is None:
                self.browser_step(command.payload)
            else:
                self.tick()

    def close(self):
        if self.env:
            self.env.close()
            self.env = None
        if self.spreadsheet_adapter:
            self.spreadsheet_adapter.close()
            self.spreadsheet_adapter = None
        if self.trace:
            self.trace.close()
            self.trace = None
        self.replay_events = None
        self.replay_context = None


def read_trace(path):
    events = []
    previous = -1
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            item = RuntimeEvent.model_validate_json(line)
            if item.sequence <= previous:
                raise ValueError(f"Non-monotonic event sequence at line {line_number}")
            previous = item.sequence
            events.append(item)
    return events


def serve(input_stream=None, output=None):
    input_stream = input_stream or sys.stdin
    runtime = Runtime(output)
    messages = queue.Queue(maxsize=64)

    def read_commands():
        for line in input_stream:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=read_commands, daemon=True).start()
    runtime.emit("hello", {"protocol_version": 1, "status": "idle"})
    try:
        while True:
            try:
                line = messages.get(timeout=0.05)
            except queue.Empty:
                line = ""
            if line is None:
                break
            try:
                if line:
                    runtime.command(json.loads(line))
                else:
                    runtime.advance_if_due()
            except Exception as exc:  # noqa: BLE001 - protocol boundary must survive malformed commands
                runtime.emit("error", {"message": str(exc), "type": type(exc).__name__})
                if runtime.status == "running":
                    runtime.status = "paused"
                    runtime.state()
    finally:
        runtime.close()
