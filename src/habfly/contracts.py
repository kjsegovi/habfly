"""Versioned, JSON-serializable contracts shared by all HabFly front ends."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ActionKind(StrEnum):
    CLICK = "CLICK"
    TYPE = "TYPE"
    SELECT = "SELECT"
    HOVER = "HOVER"
    DRAG = "DRAG"
    SCROLL = "SCROLL"
    KEYPRESS = "KEYPRESS"
    WAIT = "WAIT"
    STOP = "STOP"


class Control(Contract):
    id: str
    label: str
    role: str = "button"
    value: str = ""
    options: list[str] = Field(default_factory=list)
    enabled: bool = True
    actions: list[ActionKind] = Field(default_factory=lambda: [ActionKind.CLICK])
    surface: Literal["task", "spreadsheet", "calculation"] = "task"


class Observation(Contract):
    revision: int = 0
    instruction: str = ""
    controls: list[Control] = Field(default_factory=list)
    values: dict[str, Any] = Field(default_factory=dict)
    feedback: str = ""
    chart: dict[str, Any] = Field(default_factory=dict)
    chart_crop: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    modalities: list[str] = Field(default_factory=lambda: ["text", "controls"])
    spreadsheet: dict[str, Any] = Field(default_factory=dict)
    calculation: dict[str, Any] = Field(default_factory=dict)


def task_completed(observation: Observation) -> bool:
    """Termination, submission, and successful task completion are distinct."""
    return bool(observation.progress.get("task_completed", observation.progress.get("submitted", False)))


class Action(Contract):
    kind: ActionKind
    target: str | None = None
    value: str | None = None
    x: float | None = Field(default=None, ge=0, le=1)
    y: float | None = Field(default=None, ge=0, le=1)
    dx: float | None = Field(default=None, ge=-1, le=1)
    dy: float | None = Field(default=None, ge=-1, le=1)
    observation_revision: int | None = None
    action_confidence: float | None = Field(default=None, ge=0, le=1)
    target_confidence: float | None = Field(default=None, ge=0, le=1)
    calibrated: bool = False

    @model_validator(mode="after")
    def check_arguments(self):
        if self.kind not in {ActionKind.WAIT, ActionKind.STOP} and not self.target:
            raise ValueError(f"{self.kind} requires a target")
        if self.kind in {ActionKind.TYPE, ActionKind.SELECT, ActionKind.KEYPRESS} and self.value is None:
            raise ValueError(f"{self.kind} requires a value")
        return self


class StepResult(Contract):
    observation: Observation
    reward: float = 0
    reward_components: dict[str, float] = Field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    failure_reason: str | None = None
    steps: int = 0
    cumulative_reward: float = 0


class ArrayArtifact(Contract):
    file: str
    shape: list[int]
    dtype: str
    sha256: str


class GraphManifest(Contract):
    schema_version: Literal[1] = 1
    num_nodes: int = Field(gt=0)
    num_edges: int = Field(ge=0)
    graph_hash: str
    arrays: dict[str, ArrayArtifact]
    metadata_sha256: str
    validation: dict[str, Any]
    source_kind: str = "synthetic"
    sources: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    provenance: dict[str, Any] = Field(default_factory=dict)
    eligible_edges: int | None = None
    selection: dict[str, Any] = Field(default_factory=dict)
    node_feature_names: list[str] = Field(default_factory=list)
    polarity_policy: str = ""
    normalization: str = ""


class CheckpointManifest(Contract):
    schema_version: Literal[1] = 1
    model: dict[str, Any]
    tokenizer: dict[str, Any]
    tokenizer_hash: str
    graph_hash: str
    graph_size: int = Field(gt=0)
    content_pack_hash: str | None = None
    training_stage: str
    seed: int
    git_commit: str | None = None
    evaluation_scores: dict[str, Any] = Field(default_factory=dict)
    optimizer_state: bool = False
    calibration: dict[str, Any] = Field(default_factory=dict)
    action_temperature: float = Field(default=1, gt=0)
    target_temperature: float = Field(default=1, gt=0)
    vision_trained: bool = False
    provenance: dict[str, Any] = Field(default_factory=dict)


class RuntimeCommand(Contract):
    version: Literal[1] = 1
    command: Literal["start", "pause", "resume", "step", "abort", "save_trace", "replay"]
    payload: dict[str, Any] = Field(default_factory=dict)


class RuntimeEvent(Contract):
    version: Literal[1] = 1
    event: Literal[
        "hello",
        "state",
        "observation",
        "action_proposed",
        "action_result",
        "neural_activity",
        "episode_summary",
        "error",
    ]
    sequence: int = Field(ge=0)
    run_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def validate_action(observation: Observation, action: Action) -> Control | None:
    """Resolve observation-local targets before any environment side effect."""
    if action.observation_revision is not None and action.observation_revision != observation.revision:
        raise ValueError("stale_observation")
    if action.kind in {ActionKind.STOP, ActionKind.WAIT}:
        return None
    control = next((c for c in observation.controls if c.id == action.target), None)
    if control is None or not control.enabled:
        raise ValueError("unavailable_target")
    if action.kind not in control.actions:
        raise ValueError("action_not_allowed")
    if action.kind == ActionKind.SELECT and action.value not in control.options:
        raise ValueError("invalid_option")
    return control
