"""Stellar tool-use decisions; calculations are explicit, never auto-solved."""

import random
from typing import ClassVar

import gymnasium as gym

from habfly.contracts import Action, Control, Observation, StepResult, validate_action

from .mini_habworlds import ContractSpace
from .stellar_common import LOCAL_TEMPLATES, UNITS, grade_fields


class LocalStellarEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["ansi"]}
    backend = "local"

    def __init__(self, adapter, cases, *, max_steps=128):
        super().__init__()
        self.adapter, self.pack = adapter, adapter.pack
        self.cases = {c["seed"]: c for c in cases}
        self.max_steps = max_steps
        self.action_space = ContractSpace(Action, Action(kind="WAIT"))
        self.observation_space = ContractSpace(Observation, Observation())

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed not in self.cases:
            raise ValueError("No reference case for this seed")
        self.case = self.cases[seed]
        self.adapter.reset()
        self.steps, self.total = 0, 0.0
        self.terminated = self.truncated = self.completed = False
        self.operation = self.parameter = self.source = self.result = self.destination = ""
        self.bindings, self.results, self.answers, self.units = {}, {}, {}, {}
        self.pending_result = None
        self.last_operation = self.tool_error = None
        self.awarded = set()
        self.feedback = "Choose an operation and read its input requirements."
        self.metrics = {
            k: 0
            for k in (
                "calculation_attempts",
                "calculation_correct",
                "input_attempts",
                "input_correct",
                "copy_attempts",
                "copy_correct",
                "unit_attempts",
                "unit_correct",
                "invalid_actions",
                "tool_errors",
                "infrastructure_failures",
                "api_failures",
            )
        }
        return self.observe().model_dump(mode="json"), {}

    def sources(self):
        return {**self.case["measurements"], **{k: v for k, v in self.results.items() if v["valid"]}}

    def observe(self):
        op = self.pack.operation(self.operation)
        controls = []

        def add(key, label, options=None, value="", enabled=True, surface="calculation"):
            controls.append(
                Control(
                    id=f"{self.steps}:{key}",
                    label=label,
                    surface=surface,
                    role="combobox" if options is not None else "button",
                    value=value,
                    options=options or [],
                    enabled=enabled,
                    actions=["SELECT" if options is not None else "CLICK"],
                )
            )

        if not self.terminated and not self.truncated:
            add("operation", "Calculation", [o.id for o in self.pack.operations], self.operation)
            add("parameter", "Required input", list(op.inputs) if op else [], self.parameter, bool(op))
            add("source", "Measurement or result", list(self.sources()), self.source)
            add("bind", "Bind selected input", enabled=bool(op and self.parameter and self.source))
            add("execute", "Execute calculation", enabled=bool(op))
            add(
                "result",
                "Calculated result",
                [k for k, v in self.results.items() if v["valid"]],
                self.result,
                any(v["valid"] for v in self.results.values()),
            )
            add("destination", "Answer destination", self.case["required"], self.destination)
            add("copy", "Copy selected result", enabled=bool(self.result and self.destination))
            for field in self.case["required"]:
                add(
                    f"unit_{field}",
                    f"Unit for {field}",
                    list(UNITS.values()),
                    self.units.get(field, ""),
                    surface="task",
                )
            add("check", "Check stellar analysis", surface="task")
            random.Random(self.case["seed"] + self.steps).shuffle(controls)
        card = (
            {
                "id": op.id,
                "description": op.description,
                "inputs": {k: v.model_dump() for k, v in op.inputs.items()},
                "expression": op.expression,
                "constants": op.constants,
                "output_unit": op.output_unit,
                "applicable_classes": op.applicable_classes,
                "sources": op.sources,
            }
            if op
            else None
        )
        return Observation(
            revision=self.steps,
            instruction=LOCAL_TEMPLATES[self.case["split"]],
            controls=controls,
            values={
                "star_class": self.case["star_class"],
                "measurements": self.case["measurements"],
                "required_fields": self.case["required"],
                "answers": self.answers,
                "units": self.units,
            },
            feedback=self.feedback,
            progress={
                "task": "stellar",
                "task_completed": self.completed,
                "steps": self.steps,
                "completed_fields": len(self.awarded),
                "required_fields": len(self.case["required"]),
            },
            calculation={
                "backend": "local",
                "calculation_mode": "local_tool_assisted",
                "pack_hash": self.pack.checksum,
                "catalog": {o.id: o.description for o in self.pack.operations},
                "reference_card": card,
                "reference_information": self.pack.references,
                "pending_knowledge": self.pack.pending,
                "operation": self.operation,
                "parameter": self.parameter,
                "source": self.source,
                "bindings": dict(self.bindings),
                "results": {k: dict(v) for k, v in self.results.items()},
                "pending_result": self.pending_result,
                "selected_result": self.result,
                "destination": self.destination,
                "tool_error": self.tool_error,
                "last_operation": self.last_operation,
            },
        )

    def invalidate_pending(self):
        if self.pending_result:
            self.results[self.pending_result]["valid"] = False
            # Do not allow chains to consume an invalidated pending result.
            changed = True
            while changed:
                changed = False
                for result in self.results.values():
                    if result["valid"] and any(
                        ref in self.results and not self.results[ref]["valid"]
                        for ref in result["bindings"].values()
                    ):
                        result["valid"] = False
                        changed = True
            if self.result in self.results and not self.results[self.result]["valid"]:
                self.result = ""
        self.pending_result = None

    def fail_tool(self, reason):
        self.tool_error = reason
        self.metrics["tool_errors"] += 1
        self.feedback = f"Calculation tool: {reason}. Correct the selection or inputs."

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("Reset after an episode ends")
        action = Action.model_validate(action)
        failure = None
        components = {"step": -0.01}
        self.last_operation = self.tool_error = None
        try:
            control = validate_action(self.observe(), action)
            key = control.id.split(":", 1)[1] if control else ""
            if action.kind == "STOP":
                self.terminated, failure = True, "policy_stopped"
            elif action.kind == "WAIT":
                self.feedback = "Waiting leaves the local tool unchanged."
            elif key == "operation":
                self.metrics["calculation_attempts"] += 1
                self.metrics["calculation_correct"] += int(action.value in self.case["required"])
                if self.operation != action.value:
                    # Completed history survives switching to another operation.
                    self.operation, self.parameter = action.value, ""
                    self.bindings, self.pending_result = {}, None
            elif key in ("parameter", "source", "result", "destination"):
                setattr(self, key, action.value)
            elif key == "bind":
                if self.source not in self.sources():
                    self.fail_tool("stale_source")
                else:
                    if self.bindings.get(self.parameter) != self.source:
                        self.invalidate_pending()
                    if self.source not in self.sources():
                        self.fail_tool("stale_source")
                        self.source = ""
                    else:
                        self.bind_input()
            elif key == "execute":
                if len(self.results) >= 16:
                    self.fail_tool("result_history_limit")
                elif any(ref not in self.sources() for ref in self.bindings.values()):
                    self.fail_tool("stale_source")
                else:
                    bound = {k: dict(self.sources()[ref]) for k, ref in self.bindings.items()}
                    result = self.adapter.execute(self.operation, bound, self.case["star_class"])
                    self.last_operation = {
                        "kind": "calculate",
                        "operation": self.operation,
                        "inputs": bound,
                        "result": result.model_dump(),
                        "pack_hash": self.pack.checksum,
                    }
                    if not result.ok:
                        self.fail_tool(result.error)
                    else:
                        ident = f"r{len(self.results) + 1}"
                        self.results[ident] = {
                            "kind": self.operation,
                            "value": result.value,
                            "unit": result.unit,
                            "bindings": dict(self.bindings),
                            "valid": True,
                        }
                        self.pending_result = ident
                        self.feedback = f"Result {ident} recorded. Choose a result and destination to copy."
            elif key == "copy":
                result = self.results.get(self.result)
                if not result or not result["valid"]:
                    self.fail_tool("stale_result")
                else:
                    self.answers[self.destination] = result["value"]
                    self.metrics["copy_attempts"] += 1
                    self.metrics["copy_correct"] += int(result["kind"] == self.destination)
                    self.last_operation = {
                        "kind": "copy",
                        "result_id": self.result,
                        "destination": self.destination,
                        "value": result["value"],
                        "unit": result["unit"],
                    }
            elif key.startswith("unit_"):
                field = key[5:]
                self.units[field] = action.value
                self.metrics["unit_attempts"] += 1
                self.metrics["unit_correct"] += int(action.value == UNITS[field])
            elif key == "check":
                correct, _ = grade_fields(self.case, self.answers, self.units)
                components["new_correct_fields"] = len(correct - self.awarded)
                self.awarded.update(correct)
                self.completed = len(correct) == len(self.case["required"])
                self.terminated = self.completed
                self.feedback = (
                    "Stellar task completed."
                    if self.completed
                    else "One or more answers or units need correction."
                )
        except ValueError as exc:
            self.metrics["invalid_actions"] += 1
            components["invalid_action"] = -0.1
            failure = str(exc)
        self.steps += 1
        if self.steps >= self.max_steps and not self.terminated:
            self.truncated, failure = True, failure or "step_limit"
        _, grades = grade_fields(self.case, self.answers, self.units)
        self.metrics.update(grades)
        reward = sum(components.values())
        self.total += reward
        result = StepResult(
            observation=self.observe(),
            reward=reward,
            reward_components=components,
            terminated=self.terminated,
            truncated=self.truncated,
            failure_reason=failure,
            steps=self.steps,
            cumulative_reward=self.total,
        )
        return (
            result.observation.model_dump(mode="json"),
            reward,
            self.terminated,
            self.truncated,
            {"result": result.model_dump(mode="json"), "metrics": dict(self.metrics)},
        )

    def close(self):
        pass

    def bind_input(self):
        self.bindings[self.parameter] = self.source
        source = self.sources()[self.source]
        spec = self.pack.operation(self.operation).inputs[self.parameter]
        self.metrics["input_attempts"] += 1
        self.metrics["input_correct"] += int(
            source.get("kind") == spec.quantity and source.get("source", "current star") == "current star"
        )

    def expert_action(self, observation):
        return local_stellar_expert(observation, self.pack)


def local_stellar_expert(observation, pack):
    """A reference-card follower, not a consumer of hidden grading answers."""
    obs = Observation.model_validate(observation)
    state, values = obs.calculation, obs.values

    def act(key, value=None):
        target = next(c.id for c in obs.controls if c.id.split(":", 1)[1] == key)
        return Action(
            kind="SELECT" if value is not None else "CLICK",
            target=target,
            value=value,
            observation_revision=obs.revision,
        )

    results = {k: v for k, v in state["results"].items() if v["valid"]}
    for field in values["required_fields"]:
        existing = next(((k, v) for k, v in results.items() if v["kind"] == field), None)
        if existing is None:
            if state["operation"] != field:
                return act("operation", field)
            op = pack.operation(field)
            for parameter, spec in op.inputs.items():
                reference = next(
                    (
                        k
                        for k, v in values["measurements"].items()
                        if v["kind"] == spec.quantity and v["source"] == "current star"
                    ),
                    None,
                )
                if reference is None:
                    reference = next(k for k, v in results.items() if v["kind"] == spec.quantity)
                if state["bindings"].get(parameter) != reference:
                    if state["parameter"] != parameter:
                        return act("parameter", parameter)
                    if state["source"] != reference:
                        return act("source", reference)
                    return act("bind")
            return act("execute")
        result_id, result = existing
        if values["answers"].get(field) != result["value"]:
            if state["selected_result"] != result_id:
                return act("result", result_id)
            if state["destination"] != field:
                return act("destination", field)
            return act("copy")
        if values["units"].get(field) != UNITS[field]:
            return act(f"unit_{field}", UNITS[field])
    return act("check")
