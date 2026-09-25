"""Separate peak-wavelength color checkpoint; no numerical or classification repair."""

import random

import gymnasium as gym

from habfly.color_reference import COLOR_CONTROL, COLOR_LABELS, ColorReferenceError
from habfly.contracts import Action, Control, Observation, StepResult, validate_action

SCOPE = "learned_peak_wavelength_color_v1"
MAX_STEPS = 8
TEMPLATES = {
    "train": (
        "Use the current star's measurements. Select its peak wavelength color.",
        "For the current star, choose its peak wavelength color, not the reference star's.",
    ),
    "calibration": (
        "Use the current star's peak wavelength to select its color. Ignore the reference star.",
    ),
    "development": (
        "Determine the current star's peak wavelength color. The reference star is a distractor.",
    ),
    "test": ("Analyze the current star, not the reference star: select its peak wavelength color.",),
    "manual": ("Choose the peak wavelength color for the current star using its measurements.",),
    "expert": ("Expert validation: use the current star's peak wavelength color.",),
}
SEEDS = {name: 9600000 + i * 100000 for i, name in enumerate(TEMPLATES)}
# Sampling bounds, not policy features or grading thresholds. Cover each known
# band, with narrow bands represented and no training examples on unresolved ties.
SAMPLE_RANGES = (
    (100, 379),
    (381, 449),
    (451, 474),
    (476, 493),
    (496, 569),
    (571, 589),
    (591, 619),
    (621, 743),
    (745, 1800),
)


def color_cases(split, count, reference):
    if split not in SEEDS or not 1 <= count <= 100:
        raise ValueError("Unknown color split or case budget")
    cases = []
    for index in range(count):
        seed = SEEDS[split] + index
        rng = random.Random(seed)
        low, high = SAMPLE_RANGES[index % len(SAMPLE_RANGES)]
        value = rng.uniform(low, high)
        # Distractor is guaranteed a different band; wrong selection cannot pass
        # merely because two sources have the same band in these diagnostic cases.
        other_range = SAMPLE_RANGES[(index + 4) % len(SAMPLE_RANGES)]
        rows = [
            ("wavelength", "nm", "current star", value),
            ("wavelength", "nm", "reference star", rng.uniform(*other_range)),
            ("temperature", "K", "current star", rng.uniform(2500, 25000)),
            ("wavelength", "K", "current star", rng.uniform(100, 1800)),
        ]
        rng.shuffle(rows)
        measurements, expected_id = {}, None
        for kind, unit, source, number in rows:
            ident = f"m-{rng.getrandbits(80):x}"
            measurements[ident] = {"kind": kind, "unit": unit, "source": source, "value": number}
            if (kind, unit, source) == ("wavelength", "nm", "current star"):
                expected_id = ident
        cases.append(
            {
                "seed": seed,
                "case_id": f"color-{seed}",
                "split": split,
                "instruction": TEMPLATES[split][index % len(TEMPLATES[split])],
                "measurements": measurements,
                "expected_source": expected_id,
                "expected_color": reference.private_label(value),
                "reference_hash": reference.checksum,
            }
        )
    return cases


class ColorEnv(gym.Env):
    def __init__(self, reference, cases):
        self.reference = reference
        self.cases = {row["seed"]: row for row in cases}
        if any(row["reference_hash"] != reference.checksum for row in cases):
            raise ValueError("Color case reference mismatch")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.case = self.cases[seed]
        self.source, self.answer = "", ""
        self.steps, self.total = 0, 0.0
        self.terminated = self.truncated = self.completed = False
        self.error = None
        self.metrics = {
            key: 0
            for key in (
                "input_attempts",
                "input_correct",
                "color_attempts",
                "color_correct",
                "invalid_actions",
                "reference_errors",
            )
        }
        return self.observe().model_dump(mode="json"), {}

    def observe(self):
        colors = list(COLOR_LABELS)
        sources = list(self.case["measurements"])
        rng = random.Random(self.case["seed"] + self.steps)
        rng.shuffle(colors)
        rng.shuffle(sources)
        controls = [
            Control(
                id=f"{self.steps}:source",
                label="Measurement or result",
                role="combobox",
                surface="calculation",
                options=sources,
                value=self.source,
                actions=["SELECT"],
            ),
            Control(
                id=f"{self.steps}:color",
                label=COLOR_CONTROL,
                role="combobox",
                options=colors,
                value=self.answer,
                enabled=bool(self.source),
                actions=["SELECT"],
            ),
            Control(
                id=f"{self.steps}:check",
                label="Check color selection",
                enabled=bool(self.answer),
                actions=["CLICK"],
            ),
        ]
        rng.shuffle(controls)
        return Observation(
            revision=self.steps,
            instruction=self.case["instruction"],
            controls=[] if self.terminated or self.truncated else controls,
            values={
                "measurements": self.case["measurements"],
                "required_fields": ["color"],
                "answers": {"color": self.answer} if self.answer else {},
                "star_class": None,
            },
            feedback=self.error or "Select a measurement, then choose its peak-wavelength band.",
            progress={"task": "color", "task_completed": self.completed, "steps": self.steps},
            calculation={
                "calculation_mode": SCOPE,
                "source": self.source,
                "parameter": "wavelength",
                "reference_card": self.reference.card(),
                "pack_hash": self.reference.checksum,
                "tool_error": self.error,
            },
        )

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("Reset after an episode ends")
        self.error = None
        reward = 0.0
        try:
            action = Action.model_validate(action)
            control = validate_action(self.observe(), action)
            key = control.id.split(":", 1)[1] if control else ""
            if action.kind == "STOP":
                self.terminated, self.error = True, "policy_stopped"
            elif key == "source":
                self.source, self.answer = action.value, ""  # Always invalidate previous answer.
                self.metrics["input_attempts"] += 1
                self.metrics["input_correct"] += self.source == self.case["expected_source"]
            elif key == "color":
                row = self.case["measurements"][self.source]
                if row["kind"] != "wavelength":
                    raise ColorReferenceError("incompatible_measurement_kind")
                self.reference.guard(row["value"], row["unit"])
                self.answer = action.value  # Valid but wrong colors are not repaired.
                self.metrics["color_attempts"] += 1
                self.metrics["color_correct"] += self.answer == self.case["expected_color"]
            elif key == "check":
                self.completed = self.completed_selection()
                self.terminated = True
                reward = float(self.completed)
                self.error = None if self.completed else "incorrect_color_selection"
        except ColorReferenceError as exc:
            self.error, self.terminated = str(exc), True
            self.metrics["reference_errors"] += 1
        except ValueError:
            self.error, self.terminated = "invalid_action", True
            self.metrics["invalid_actions"] += 1
        self.steps += 1
        self.total += reward
        self.truncated = self.steps >= MAX_STEPS and not self.terminated
        if self.truncated:
            self.error = "color_step_limit"
        result = StepResult(
            observation=self.observe(),
            reward=reward,
            cumulative_reward=self.total,
            steps=self.steps,
            terminated=self.terminated,
            truncated=self.truncated,
            failure_reason=self.error,
        )
        return (
            result.observation.model_dump(mode="json"),
            reward,
            self.terminated,
            self.truncated,
            {"result": result.model_dump(mode="json")},
        )

    def expert_action(self, observation):
        key, value = (
            ("source", self.case["expected_source"])
            if not self.source
            else ("color", self.case["expected_color"])
            if not self.answer
            else ("check", None)
        )
        target = next(c for c in observation.controls if c.id.endswith(":" + key))
        return Action(
            kind="CLICK" if key == "check" else "SELECT",
            target=target.id,
            value=value,
            observation_revision=observation.revision,
        )

    def completed_selection(self):
        return self.source == self.case["expected_source"] and self.answer == self.case["expected_color"]
