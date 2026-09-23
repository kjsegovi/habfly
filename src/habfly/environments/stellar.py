"""Stellar tasks backed by real Sheets outputs, never a local calculator."""

from __future__ import annotations

import random
from typing import ClassVar

import gymnasium as gym

from habfly.contracts import Action, Control, Observation, StepResult, validate_action
from habfly.spreadsheet import INPUTS, OUTPUTS, UNITS, SpreadsheetError

from .mini_habworlds import ContractSpace
from .stellar_common import TEMPLATES, grade_fields, make_case

__all__ = ["StellarAnalysisEnv", "make_case", "stellar_expert"]


class StellarAnalysisEnv(gym.Env):
    """The policy sees measurements and actual sheet values, not expected answers.

    Reference answers are an immutable, separately collected grading record.
    The adapter is shared across sequential episodes and owned by the caller.
    """

    metadata: ClassVar[dict] = {"render_modes": ["ansi"]}

    def __init__(self, adapter, cases, *, max_steps=128):
        super().__init__()
        self.adapter, self.cases, self.max_steps = adapter, {c["seed"]: c for c in cases}, max_steps
        self.action_space = ContractSpace(Action, Action(kind="WAIT"))
        self.observation_space = ContractSpace(Observation, Observation())
        self.case = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed not in self.cases:
            raise ValueError("No collected reference for this seed; collect the split first")
        self.case = self.cases[seed]
        self.adapter.reset()
        self.steps = 0
        self.total = 0.0
        self.terminated = self.truncated = self.completed = False
        self.selected_measurement = self.selected_input = self.selected_output = self.selected_destination = (
            ""
        )
        self.bindings, self.outputs, self.answers, self.units = {}, {}, {}, {}
        self.awarded = set()
        self.feedback = "Ready. Select a measurement and a spreadsheet input cell."
        self.last_operation = None
        self.metrics = {
            k: 0
            for k in (
                "input_attempts",
                "input_correct",
                "copy_attempts",
                "copy_correct",
                "unit_attempts",
                "unit_correct",
                "invalid_actions",
                "api_failures",
            )
        }
        return self.observe().model_dump(mode="json"), {}

    def observe(self):
        controls = []

        def add(key, label, options=None, enabled=True, surface="spreadsheet", value=""):
            controls.append(
                Control(
                    id=f"{self.steps}:{key}",
                    label=label,
                    role="combobox" if options is not None else "button",
                    value=value,
                    options=options or [],
                    enabled=enabled,
                    surface=surface,
                    actions=["SELECT" if options is not None else "CLICK"],
                )
            )

        if not self.terminated and not self.truncated:
            add(
                "measurement",
                "Source measurement",
                list(self.case["measurements"]),
                value=self.selected_measurement,
            )
            add("input", "Spreadsheet input cell", list(INPUTS.values()), value=self.selected_input)
            add(
                "bind",
                "Stage selected input",
                enabled=bool(self.selected_measurement and self.selected_input),
            )
            add("commit", "Write inputs and read sheet", enabled=len(self.bindings) == 3)
            add(
                "output",
                "Spreadsheet result cell",
                list(OUTPUTS.values()),
                enabled=bool(self.outputs),
                value=self.selected_output,
            )
            add("destination", "Answer destination", self.case["required"], value=self.selected_destination)
            add(
                "copy",
                "Copy selected result",
                enabled=bool(self.outputs and self.selected_output and self.selected_destination),
            )
            for field in self.case["required"]:
                add(
                    f"unit_{field}",
                    f"Unit for {field}",
                    list(dict.fromkeys(UNITS.values())),
                    surface="task",
                    value=self.units.get(field, ""),
                )
            add("check", "Check stellar analysis", surface="task")
            random.Random(self.case["seed"] + self.steps).shuffle(controls)
        return Observation(
            revision=self.steps,
            instruction=TEMPLATES[self.case["split"]],
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
            spreadsheet={
                "calculation_mode": "google_sheets",
                "inputs": INPUTS,
                "output_cells": OUTPUTS,
                "output_units": {k: UNITS[k] for k in OUTPUTS},
                "bindings": self.bindings,
                "results": self.outputs,
                "selected_measurement": self.selected_measurement,
                "selected_input": self.selected_input,
                "selected_output": self.selected_output,
                "selected_destination": self.selected_destination,
                "formula_fingerprint": self.adapter.config.expected_formula_fingerprint,
                "generation": self.adapter.generation,
                "last_operation": self.last_operation,
            },
        )

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("Reset after an episode ends")
        action = Action.model_validate(action)
        failure = None
        reward = -0.01
        components = {"step": -0.01}
        self.last_operation = None
        try:
            control = validate_action(self.observe(), action)
            key = control.id.split(":", 1)[1] if control else ""
            if action.kind == "STOP":
                self.terminated = True
                failure = "policy_stopped"
            elif action.kind == "WAIT":
                self.feedback = "Waiting does not change spreadsheet inputs."
            elif key in ("measurement", "input", "output", "destination"):
                setattr(self, f"selected_{key}", action.value)
            elif key == "bind":
                self.bindings[self.selected_input] = self.selected_measurement
                self.outputs = {}
                self.adapter.invalidate()
                kind = next(k for k, c in INPUTS.items() if c == self.selected_input)
                measurement = self.case["measurements"][self.selected_measurement]
                self.metrics["input_attempts"] += 1
                self.metrics["input_correct"] += int(
                    measurement["kind"] == kind and measurement["source"] == "current star"
                )
                self.feedback = "Input staged; commit all three inputs to recalculate."
            elif key == "commit":
                inputs = {k: self.case["measurements"][self.bindings[c]]["value"] for k, c in INPUTS.items()}
                # Do not repair incorrect measurement/cell choices.
                self.outputs = self.adapter.calculate(inputs)
                self.last_operation = {
                    "kind": "sheet_calculation",
                    "inputs": inputs,
                    "input_units": {
                        k: self.case["measurements"][self.bindings[c]]["unit"] for k, c in INPUTS.items()
                    },
                    "results": self.outputs,
                    "units": {k: UNITS[k] for k in OUTPUTS},
                    "formula_fingerprint": self.adapter.config.expected_formula_fingerprint,
                }
                self.feedback = "Spreadsheet results received."
            elif key == "copy":
                source = next(k for k, c in OUTPUTS.items() if c == self.selected_output)
                self.answers[self.selected_destination] = self.outputs[source]
                self.metrics["copy_attempts"] += 1
                self.metrics["copy_correct"] += int(source == self.selected_destination)
                self.last_operation = {
                    "kind": "copy",
                    "source_cell": self.selected_output,
                    "destination": self.selected_destination,
                    "value": self.outputs[source],
                }
                self.feedback = "Selected spreadsheet result copied exactly."
            elif key.startswith("unit_"):
                field = key[5:]
                self.units[field] = action.value
                self.metrics["unit_attempts"] += 1
                self.metrics["unit_correct"] += int(action.value == UNITS[field])
            elif key == "check":
                correct, _ = grade_fields(self.case, self.answers, self.units)
                newly_correct = correct - self.awarded
                self.awarded.update(correct)
                components["new_correct_fields"] = len(newly_correct)
                reward += len(newly_correct)
                self.completed = len(correct) == len(self.case["required"])
                self.terminated = self.completed
                self.feedback = (
                    "Stellar task completed."
                    if self.completed
                    else "One or more answers or units need correction."
                )
        except SpreadsheetError as exc:
            self.metrics["api_failures"] += 1
            self.truncated = True
            self.outputs = {}
            failure = str(exc)
            self.feedback = "Spreadsheet unavailable; episode stopped without substituting answers."
        except ValueError as exc:
            self.metrics["invalid_actions"] += 1
            failure = str(exc)
            reward -= 0.1
            components["invalid_action"] = -0.1
        self.steps += 1
        if self.steps >= self.max_steps and not self.terminated:
            self.truncated = True
            failure = failure or "step_limit"
        self.total += reward
        self.metrics.update(grade_fields(self.case, self.answers, self.units)[1])
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
        pass  # Shared adapter is closed by the runtime/collection/evaluation owner.


def stellar_expert(observation):
    """Scripted demonstrator using only learner-visible observations."""
    observation = Observation.model_validate(observation)
    sheet, values = observation.spreadsheet, observation.values

    def act(key, value=None):
        target = next(c.id for c in observation.controls if c.id.split(":", 1)[1] == key)
        return Action(
            kind="SELECT" if value is not None else "CLICK",
            target=target,
            value=value,
            observation_revision=observation.revision,
        )

    for kind, cell in INPUTS.items():
        source = next(
            k
            for k, m in values["measurements"].items()
            if m["kind"] == kind and m["source"] == "current star"
        )
        if sheet["bindings"].get(cell) != source:
            if sheet["selected_measurement"] != source:
                return act("measurement", source)
            if sheet["selected_input"] != cell:
                return act("input", cell)
            return act("bind")
    if not sheet["results"]:
        return act("commit")
    for field in values["required_fields"]:
        if values["answers"].get(field) != sheet["results"][field]:
            if sheet["selected_output"] != OUTPUTS[field]:
                return act("output", OUTPUTS[field])
            if sheet["selected_destination"] != field:
                return act("destination", field)
            return act("copy")
        if values["units"].get(field) != UNITS[field]:
            return act(f"unit_{field}", UNITS[field])
    return act("check")
