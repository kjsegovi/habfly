"""Gymnasium adapter for recognition, arithmetic, and composed commands."""

from __future__ import annotations

import random
from typing import ClassVar

import gymnasium as gym

from habfly.contracts import Action, ActionKind, Observation, StepResult, validate_action
from habfly.environments.mini_habworlds import ContractSpace

from .curriculum import Example, curriculum_examples


class CurriculumEnvironment(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["ansi"]}

    def __init__(self, stage="recognize", *, heldout=False, count=32, sequence_length=1):
        super().__init__()
        if sequence_length < 1 or count < 4:
            raise ValueError("Invalid curriculum episode dimensions")
        self.stage, self.heldout, self.count = stage, heldout, count
        self.sequence_length = sequence_length
        self.render_mode = "ansi"
        self.action_space = ContractSpace(Action, Action(kind=ActionKind.WAIT))
        self.observation_space = ContractSpace(Observation, Observation())
        self.examples: list[Example] = []
        self.index = 0
        self.cumulative_reward = 0.0
        self.terminated = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        train, test = curriculum_examples(self.stage, seed=seed or 0, count=self.count)
        candidates = test if self.heldout else train
        rng = random.Random(seed)
        self.examples = [rng.choice(candidates) for _ in range(self.sequence_length)]
        self.index, self.cumulative_reward, self.terminated = 0, 0.0, False
        return self.observe().model_dump(mode="json"), {"stage": self.stage, "heldout": self.heldout}

    def observe(self):
        if not self.examples:
            raise RuntimeError("Reset the curriculum before observing")
        if self.terminated:
            return Observation(revision=self.index, instruction="Curriculum complete.",
                               progress={"completed": self.index, "submitted": self.cumulative_reward == self.sequence_length})
        observation = self.examples[self.index].observation.model_copy(deep=True)
        observation.revision = self.index
        if self.sequence_length > 1:
            sequence = " Then ".join(e.observation.instruction for e in self.examples)
            observation.instruction = f"Follow this sequence: {sequence} Current step: {self.index + 1}."
        observation.progress = {"completed": self.index, "required": self.sequence_length, "submitted": False}
        return observation

    def step(self, action):
        if self.terminated:
            raise RuntimeError("Reset after an episode terminates")
        action = Action.model_validate(action)
        expected = self.examples[self.index].action
        failure = None
        try:
            validate_action(self.observe(), action)
            correct = (action.kind, action.target, action.value) == (expected.kind, expected.target, expected.value)
        except ValueError as exc:
            correct, failure = False, str(exc)
        reward = float(correct)
        self.cumulative_reward += reward
        self.index += 1
        self.terminated = self.index == len(self.examples)
        result = StepResult(observation=self.observe(), reward=reward, reward_components={"exact_action": reward},
                            terminated=self.terminated, failure_reason=failure, steps=self.index,
                            cumulative_reward=self.cumulative_reward)
        return result.observation.model_dump(mode="json"), reward, self.terminated, False, {"result": result.model_dump(mode="json")}

    def render(self):
        return self.observe().model_dump_json(indent=2)
