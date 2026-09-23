"""Fast, deterministic star-analysis fixture with the common observation contract."""

import math
from typing import Any, ClassVar

import gymnasium as gym

from ..content import ContentPack, calculate, load_content_pack
from ..contracts import Action, ActionKind, Control, Observation, StepResult, validate_action


class ContractSpace(gym.Space):
    """Gymnasium space for structured, variable-length JSON contracts."""

    def __init__(self, schema, example):
        super().__init__(shape=None, dtype=None)
        self.schema, self.example = schema, example

    def sample(self, mask=None):
        return self.example.model_dump(mode="json")

    def contains(self, value):
        try:
            self.schema.model_validate(value)
            return True
        except (ValueError, TypeError):
            return False


class MiniHabWorlds(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(self, stars=1, pack: ContentPack | None = None, max_steps: int | None = None):
        if not 1 <= stars <= 30:
            raise ValueError("stars must be between 1 and 30")
        self.required_stars = stars
        self.pack = pack or load_content_pack()
        self.max_steps = max_steps or 80 * stars + 10
        self.render_mode = "ansi"
        self.action_space = ContractSpace(Action, Action(kind=ActionKind.WAIT))
        self.observation_space = ContractSpace(Observation, Observation())
        self._reset_done = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.catalog = []
        for index in range(max(6, self.required_stars + 3)):
            self.catalog.append(
                {
                    "name": f"Star {index + 1:03}",
                    "mass": round(float(self.np_random.uniform(0.5, 2)), 3),
                    "stellar_radius": round(float(self.np_random.uniform(0.5, 2)), 3),
                    "temperature": int(self.np_random.integers(230, 430)),
                    "pressure": round(float(self.np_random.uniform(0.005, 2)), 3),
                    "terrestrial": bool(self.np_random.integers(0, 2)),
                    "period": int(self.np_random.integers(10, 101)),
                    "brightness_drop": round(float(self.np_random.uniform(0.001, 0.03)), 4),
                }
            )
        self.collection: dict[int, dict[str, str]] = {}
        self.awarded: set[int] = set()
        self.selected: int | None = None
        self.screen = "explore"
        self.layout = "grid"
        self.days = ""
        self.observed = False
        self.hovered = False
        self.zoom = 0
        self.pan = 0
        self.ready = False
        self.feedback = "Synthetic fixture; not verified HabWorlds lesson content."
        self.steps, self.total_reward, self.revision = 0, 0.0, 0
        self.terminated = self.truncated = False
        self._last_unchanged = None
        self._repeat = 0
        self._reset_done = True
        return self.observe().model_dump(mode="json"), {
            "content_pack": self.pack.id,
            "synthetic": self.pack.synthetic,
        }

    def _control(self, key, label, role="button", value="", options=None, actions=None):
        return Control(
            id=f"{self.revision}:{key}",
            label=label,
            role=role,
            value=value,
            options=options or [],
            actions=actions or [ActionKind.CLICK],
        )

    def _filled(self, index):
        values = self.collection[index]
        required = [f.name for f in self.pack.fields] + ["habitable"]
        required += [f"{f.name}_unit" for f in self.pack.fields if f.units]
        return all(values.get(name, "").strip() for name in required)

    def observe(self) -> Observation:
        if not self._reset_done:
            raise RuntimeError("reset must be called first")
        controls: list[Control] = []
        values: dict[str, Any] = {"screen": self.screen, "layout": self.layout}
        chart: dict[str, Any] = {}
        instruction = "Collect and analyze stars, then submit the project."
        if self.screen in {"explore", "collection"}:
            controls += [
                self._control("explore", "Explore stars"),
                self._control("collection", "Collection"),
                self._control("layout", "List view" if self.layout == "grid" else "Grid view"),
            ]
            indices = range(len(self.catalog)) if self.screen == "explore" else self.collection.keys()
            for i in indices:
                controls.append(self._control(f"star-{i}", self.catalog[i]["name"]))
                if self.screen == "collection":
                    controls.append(self._control(f"delete-{i}", f"Delete {self.catalog[i]['name']}"))
            controls += [
                self._control("ready", "I am ready to submit project", "checkbox", str(self.ready).lower()),
                self._control("submit", "Submit Project"),
            ]
            values["collected_names"] = [self.catalog[i]["name"] for i in self.collection]
        elif self.screen == "star":
            star = self.catalog[self.selected]
            values.update({k: v for k, v in star.items() if k not in {"period", "brightness_drop"}})
            instruction = f"Analyze {star['name']}. Observe the lightcurve and complete the calculations."
            controls.append(self._control("back", "Back to collection"))
            if self.selected not in self.collection:
                controls.append(self._control("collect", "Collect star"))
            else:
                controls.append(
                    self._control(
                        "days",
                        "Days to observe",
                        "textbox",
                        self.days,
                        actions=[ActionKind.TYPE, ActionKind.KEYPRESS],
                    )
                )
                controls.append(self._control("observe", "Observe star"))
                controls.append(
                    self._control(
                        "chart",
                        "Normalized flux lightcurve",
                        "chart",
                        actions=[ActionKind.HOVER, ActionKind.SCROLL, ActionKind.DRAG],
                    )
                )
                chart = {
                    "sampled": self.observed,
                    "zoom": self.zoom,
                    "pan": self.pan,
                    "x_label": "Days observed",
                    "y_label": "Normalized flux",
                }
                if self.observed:
                    chart["sample_days"] = float(self.days)
                    chart["visible_transit_days"] = list(
                        range(star["period"], int(float(self.days)) + 1, star["period"])
                    )[:100]
                if self.hovered:
                    chart["tooltip"] = {"brightness_drop": star["brightness_drop"], "period": star["period"]}
                form = self.collection[self.selected]
                for f in self.pack.fields:
                    controls.append(
                        self._control(
                            f.name, f.label, "textbox", form.get(f.name, ""), actions=[ActionKind.TYPE]
                        )
                    )
                    if f.units:
                        controls.append(
                            self._control(
                                f"{f.name}_unit",
                                f"{f.label} units",
                                "combobox",
                                form.get(f"{f.name}_unit", ""),
                                f.units,
                                [ActionKind.SELECT],
                            )
                        )
                controls.append(
                    self._control(
                        "habitable",
                        "Habitable",
                        "combobox",
                        form.get("habitable", ""),
                        ["yes", "no"],
                        [ActionKind.SELECT],
                    )
                )
                controls.append(self._control("check", "Check calculations"))
                values["form_complete"] = self._filled(self.selected)
        completed = sum(self._filled(i) for i in self.collection)
        return Observation(
            revision=self.revision,
            instruction=instruction,
            controls=controls,
            values=values,
            feedback=self.feedback,
            chart=chart,
            progress={
                "required_stars": self.required_stars,
                "collected": len(self.collection),
                "analyzed": completed,
                "submitted": self.screen == "submitted",
                "score": self.score(),
                "synthetic": self.pack.synthetic,
            },
        )

    def _correct(self, index):
        star, form = self.catalog[index], self.collection[index]
        correct = 0
        for field in self.pack.fields:
            try:
                answer = float(form.get(field.name, ""))
                expected = calculate(field.formula, star)
                valid = math.isfinite(answer) and abs(answer - expected) <= field.tolerance * max(
                    1, abs(expected)
                )
                correct += int(
                    valid and (not field.units or form.get(f"{field.name}_unit") == field.expected_unit)
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
        lo, hi = self.pack.liquid_temperature
        habitable = (
            (star["terrestrial"] or not self.pack.terrestrial_required)
            and lo <= star["temperature"] <= hi
            and star["pressure"] >= self.pack.minimum_pressure
        )
        return correct + int(form.get("habitable") == ("yes" if habitable else "no"))

    def score(self):
        total = len(self.collection) * (len(self.pack.fields) + 1)
        return sum(self._correct(i) for i in self.collection) / total if total else 0.0

    def step(self, action):
        if not self._reset_done or self.terminated or self.truncated:
            raise RuntimeError("reset required before stepping a finished/uninitialized episode")
        action = Action.model_validate(action)
        before = self.observe()
        self.steps += 1
        components = {"step": -0.01}
        reason = None
        try:
            control = validate_action(before, action)
            key = control.id.split(":", 1)[1] if control else ""
            self.feedback = ""
            if action.kind == ActionKind.STOP:
                self.truncated = True
                reason = "agent_stopped"
            elif action.kind == ActionKind.WAIT:
                pass
            elif key in {"explore", "collection", "back"}:
                self.screen = "explore" if key == "explore" else "collection"
            elif key == "layout":
                self.layout = "list" if self.layout == "grid" else "grid"
            elif key.startswith("star-"):
                self.selected = int(key[5:])
                self.screen = "star"
                self.days, self.observed, self.hovered, self.zoom, self.pan = "", False, False, 0, 0
            elif key.startswith("delete-"):
                del self.collection[int(key[7:])]
            elif key == "collect":
                self.collection.setdefault(self.selected, {})
            elif key == "days" and action.kind == ActionKind.TYPE:
                days = float(action.value)
                if not math.isfinite(days) or not 1 <= days <= 10000:
                    raise ValueError("invalid_observation_days")
                self.days = action.value
                self.observed = self.hovered = False
            elif key == "observe" or (
                key == "days" and action.kind == ActionKind.KEYPRESS and action.value == "Enter"
            ):
                if not self.days:
                    raise ValueError("enter_days_first")
                self.observed = True
            elif key == "chart":
                if action.kind == ActionKind.SCROLL:
                    self.zoom = max(0, min(5, self.zoom + (1 if (action.dy or 0) < 0 else -1)))
                elif action.kind == ActionKind.DRAG:
                    self.pan += action.dx or 0
                elif (
                    self.observed
                    and self.zoom >= 1
                    and float(self.days) >= self.catalog[self.selected]["period"]
                ):
                    self.hovered = True
                else:
                    raise ValueError("observe_and_zoom_before_hover")
            elif key == "check":
                if not self._filled(self.selected):
                    raise ValueError("incomplete_calculations")
                self.feedback = (
                    f"{self._correct(self.selected)}/{len(self.pack.fields) + 1} calculations correct."
                )
                if self.selected not in self.awarded:
                    components["analyzed_star"] = 1.0
                    self.awarded.add(self.selected)
            elif key == "ready":
                self.ready = not self.ready
            elif key == "submit":
                if not self.ready or sum(self._filled(i) for i in self.collection) < self.required_stars:
                    raise ValueError("project_not_ready")
                self.screen, self.terminated = "submitted", True
                components["completed_project"] = 10.0
                self.feedback = "Project submitted."
            elif (
                self.screen == "star"
                and self.selected in self.collection
                and action.kind in {ActionKind.TYPE, ActionKind.SELECT}
            ):
                self.collection[self.selected][key] = action.value
            else:
                raise ValueError("unsupported_action")
        except (ValueError, OverflowError) as exc:
            reason = str(exc)
            components["invalid_action"] = -0.25
            self.feedback = reason
        # Repeating a no-progress action cannot farm intermediate rewards.
        after = self.observe()
        unchanged = (
            before.values,
            before.chart,
            before.progress,
            [(c.label, c.value) for c in before.controls],
        ) == (after.values, after.chart, after.progress, [(c.label, c.value) for c in after.controls])
        if unchanged:
            components["no_progress"] = -0.1
        if self.steps >= self.max_steps and not self.terminated:
            self.truncated, reason = True, "step_limit"
        self.revision += 1
        reward = sum(components.values())
        self.total_reward += reward
        result = StepResult(
            observation=self.observe(),
            reward=reward,
            reward_components=components,
            terminated=self.terminated,
            truncated=self.truncated,
            failure_reason=reason,
            steps=self.steps,
            cumulative_reward=self.total_reward,
        )
        return (
            result.observation.model_dump(mode="json"),
            reward,
            self.terminated,
            self.truncated,
            {"result": result.model_dump(mode="json")},
        )

    def render(self):
        return self.observe().model_dump_json(indent=2)


def expert_action(observation: Observation | dict, pack: ContentPack | None = None) -> Action:
    """Deterministic fixture solver, using only the visible observation."""
    obs = Observation.model_validate(observation)
    pack = pack or load_content_pack()

    def choose(label, kind=ActionKind.CLICK, value=None, **kwargs):
        target = next(c for c in obs.controls if c.label == label)
        return Action(kind=kind, target=target.id, value=value, observation_revision=obs.revision, **kwargs)

    labels = {c.label: c for c in obs.controls}
    if obs.progress.get("submitted"):
        return Action(kind=ActionKind.STOP)
    if "Collect star" in labels:
        return choose("Collect star")
    if obs.values.get("screen") == "star":
        if obs.values.get("form_complete") and "calculations correct" in obs.feedback:
            return choose("Back to collection")
        if not labels["Days to observe"].value:
            return choose("Days to observe", ActionKind.TYPE, "300")
        if not obs.chart.get("sampled"):
            return choose("Observe star")
        if not obs.chart.get("zoom"):
            return choose("Normalized flux lightcurve", ActionKind.SCROLL, dy=-1)
        if not obs.chart.get("tooltip"):
            return choose("Normalized flux lightcurve", ActionKind.HOVER, x=0.5, y=0.5)
        values = {**obs.values, **obs.chart["tooltip"]}
        for field in pack.fields:
            if not labels[field.label].value:
                return choose(field.label, ActionKind.TYPE, f"{calculate(field.formula, values):.5g}")
            if field.units and not labels[f"{field.label} units"].value:
                return choose(f"{field.label} units", ActionKind.SELECT, field.expected_unit)
        if not labels["Habitable"].value:
            lo, hi = pack.liquid_temperature
            yes = (
                (values["terrestrial"] or not pack.terrestrial_required)
                and lo <= values["temperature"] <= hi
                and values["pressure"] >= pack.minimum_pressure
            )
            return choose("Habitable", ActionKind.SELECT, "yes" if yes else "no")
        return choose("Check calculations")
    if obs.progress["analyzed"] >= obs.progress["required_stars"]:
        if labels["I am ready to submit project"].value != "true":
            return choose("I am ready to submit project")
        return choose("Submit Project")
    if obs.values["screen"] != "explore":
        return choose("Explore stars")
    collected = obs.values["collected_names"]
    return choose(
        next(c.label for c in obs.controls if c.label.startswith("Star ") and c.label not in collected)
    )


def generate_demonstrations(seeds, stars=1, pack=None):
    episodes = []
    for seed in seeds:
        env = MiniHabWorlds(stars=stars, pack=pack)
        observation, _ = env.reset(seed=int(seed))
        episode = []
        while not env.terminated and not env.truncated:
            action = expert_action(observation, env.pack)
            following, _, _, _, info = env.step(action)
            episode.append(
                {
                    "observation": observation,
                    "action": action.model_dump(mode="json"),
                    "result": info["result"],
                }
            )
            observation = following
        episodes.append(episode)
        env.close()
    return episodes
