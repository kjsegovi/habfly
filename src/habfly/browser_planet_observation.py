"""One bounded duration commit and Play request, never a planet detector.

The observation input must be blank. A durable, exclusive reservation blocks
every later start for the same star in the caller's explicit run history,
including uncertain requests. There is no retry, recovery, pause, or reset.
Only visible native controls, chart labels, and ordinary captures are read.

New requests default to and cannot exceed the user's 5,000-day budget. Legacy
10,000-day reservations remain immutable and still prevent another request.
The axis range does not prove the plot has finished rendering or collecting;
a separate visible-range/readiness check and presence policy must decide that.
"""

import hashlib
import json
import re
import time
from pathlib import Path

from .browser import BrowserSafetyStop
from .browser_assessment_actions import persist_json
from .browser_numeric import VISIBLE_PREFIX
from .browser_planet_numeric import PlanetNumericSession, planet_projection
from .browser_probe import save_probe
from .browser_setup import rendered_control
from .browser_stellar import NUMBER, SIMULATION_URL
from .browser_water_chamber import chamber_control_exposed

MODE = "bounded_planet_observation_start"
MAX_DAYS = 5000


def observation_projection(report, mapping, *, allow_transport_change=False):
    """Ignore only the duration value, numeric chart axes, and known transport.

    Setting the duration itself can rescale the chart. Every raw capture is
    retained; axis normalization is not a scientific measurement. Unknown chart
    words, feedback, other controls, units, answers, and frames still compare.
    """
    result = planet_projection(report, mapping, "observation_days")
    frame = next(f for f in result["frames"] if f["url"] == SIMULATION_URL)
    charts = 0
    atoms = []
    for key, value in frame["accessibility"]:
        if key == "img" and isinstance(value, str) and value.startswith("Normalized Flux Days Observed "):
            if not re.fullmatch(rf"Normalized Flux Days Observed(?: {NUMBER})+", value):
                raise BrowserSafetyStop("unsupported_observation_chart_labels")
            charts += 1
            value = "Normalized Flux Days Observed <numeric axes>"
        if allow_transport_change:
            key = re.sub(r'^button "(?:Play|Pause)"$', 'button "<observation transport>"', key)
            key = re.sub(r'^img "(?:Play|Pause)"$', 'img "<observation transport>"', key)
            if key == 'button "<observation transport>"' and value in [['img "Play"'], ['img "Pause"']]:
                value = ['img "<observation transport>"']
        atoms.append((key, value))
    if charts != 1:
        raise BrowserSafetyStop("ambiguous_observation_chart")
    frame["accessibility"] = atoms
    frame["text"], count = re.subn(
        rf"Normalized\s+Flux\s+Days\s+Observed(?:\s+{NUMBER})+(?=\s|$)",
        "Normalized Flux Days Observed <numeric axes>",
        frame["text"],
        flags=re.IGNORECASE,
    )
    if count != 1:
        raise BrowserSafetyStop("unsupported_observation_chart_text")
    if allow_transport_change:
        frame["text"] = re.sub(r"(?m)^(?:Play|Pause)$", "<observation transport>", frame["text"])
        for control in frame["controls"]:
            if control["role"] == "button" and re.fullmatch(
                r'- button "(Play|Pause)"(?::\n  - img "\1")?', control["accessibility"]
            ):
                control["accessibility"] = '- button "<observation transport>"'
    return result


def _same_handle(old, new):
    return old.evaluate("(a,b)=>a.isConnected&&a===b", new)


def _duration(frame, mapping):
    spec = mapping["observation"]["values"]["browser_field_map"]["observation_days"]
    matches = []
    for role in ("textbox", "spinbutton"):
        for control in frame.get_by_role(role).all():
            if control.is_visible() and re.sub(
                r"\s+", "", control.evaluate(VISIBLE_PREFIX)
            ).casefold().endswith("observefor"):
                matches.append(control)
    if len(matches) != 1:
        raise BrowserSafetyStop("ambiguous_observation_duration")
    control = matches[0]
    if (
        not control.is_enabled()
        or not rendered_control(control)
        or spec["unit"] != "day"
        or control.input_value() != spec["current_value"]
    ):
        raise BrowserSafetyStop("unverified_observation_duration")
    return control.element_handle(timeout=2000)


def _transport(frame, *, idle):
    names = ("Play",) if idle else ("Play", "Pause")
    matches = [
        b
        for name in names
        for b in frame.get_by_role("button", name=name, exact=True).all()
        if b.is_visible()
    ]
    if len(matches) != 1 or not matches[0].is_enabled() or not chamber_control_exposed(matches[0]):
        raise BrowserSafetyStop("unverified_observation_transport")
    return matches[0].element_handle(timeout=2000)


class PlanetObservationStartSession:
    """A one-use observation request; no scientific completion assertion."""

    def __init__(self, page, config, output, *, run_history, max_seconds=60):
        self.output, self.history = Path(output), Path(run_history)
        if not self.history.is_dir() or not self.output.resolve().is_relative_to(self.history.resolve()):
            raise ValueError("Observation output must belong to an existing explicit run history")
        if type(max_seconds) not in {int, float} or not 5 <= max_seconds <= 120:
            raise ValueError("Observation start requires a 5..120 second budget")
        self.output.mkdir(parents=True, exist_ok=False)
        self.guard = PlanetNumericSession(
            page, config, self.output / "read-guard", max_seconds=max_seconds, _allow_unset_planet=True
        )
        self.stopped = self.reserved = self.fill_attempted = self.play_attempted = False
        try:
            self.before, self.mapping, self.choices, self.answers = self.guard.current()
            self.duration = _duration(self.guard.frame, self.mapping)
            self.play = _transport(self.guard.frame, idle=True)
            if self.duration.input_value() != "":
                raise BrowserSafetyStop("observation_duration_already_populated")
            observation_projection(self.before, self.mapping)
            self.star = self.mapping["star_name"]
            self._check_history()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.stopped = True
        self.guard.close()

    def _check_history(self):
        # The legacy helper used precisely these per-run directories. Other
        # action reservations are deliberately not treated as observation starts.
        for path in self.history.glob("reference-observation-*/reserved.json"):
            try:
                record = json.loads(path.read_text())
                star = record["star"]
                if not isinstance(star, str) or not star.strip():
                    raise ValueError
            except (OSError, ValueError, KeyError, TypeError):
                raise BrowserSafetyStop("unreadable_observation_history") from None
            if star.casefold() == self.star.casefold():
                raise BrowserSafetyStop("observation_already_reserved")
        key = hashlib.sha256(self.star.casefold().encode()).hexdigest()
        self.reservation = self.history / "observation-start-reservations" / f"{key}.json"
        if self.reservation.exists():
            raise BrowserSafetyStop("observation_already_reserved")

    def _read(self, *, expected_duration, transport_change=False, capture_name=None):
        report, mapping, choices, answers = self.guard.read()
        if capture_name:
            save_probe(report, self.output / capture_name)
        duration = _duration(self.guard.frame, mapping)
        if (
            mapping["star_name"] != self.star
            or choices != self.choices
            or duration.input_value() != expected_duration
            or not _same_handle(self.duration, duration)
            or set(answers) != set(self.answers)
            or any(not _same_handle(old, answers[name]) for name, old in self.answers.items())
            or observation_projection(self.before, self.mapping, allow_transport_change=transport_change)
            != observation_projection(report, mapping, allow_transport_change=transport_change)
        ):
            raise BrowserSafetyStop("unexpected_observation_side_effect")
        return report, mapping, _transport(self.guard.frame, idle=not transport_change)

    def start(self, days=MAX_DAYS):
        if self.stopped or self.reserved:
            raise BrowserSafetyStop("observation_start_session_stopped")
        # Require a bounded integer, not code, exponent notation, boolean, or
        # a silently rounded/fixed measurement. This is an observation budget.
        if type(days) is not int or not 1 <= days <= MAX_DAYS:
            raise BrowserSafetyStop("invalid_observation_days")
        text = str(days)
        try:
            before, _, play = self._read(expected_duration="")
            if not _same_handle(self.play, play):
                raise BrowserSafetyStop("observation_play_replaced_before_commit")
            self._check_history()
            save_probe(before, self.output / "before")
            intent = {
                "schema_version": 1,
                "mode": MODE,
                "star": self.star,
                "days": days,
                "output": str(self.output.resolve().relative_to(self.history.resolve())),
                "max_duration_writes": 1,
                "max_play_clicks": 1,
                "answer_writes": 0,
                "automatic_retry": False,
                "observation_completed": False,
                "planet_presence": None,
                "task_completed": False,
            }
            self.reservation.parent.mkdir(exist_ok=True)
            try:
                persist_json(self.reservation, intent)
            except FileExistsError:
                raise BrowserSafetyStop("observation_already_reserved") from None
            self.reserved = True
            persist_json(self.output / "reserved.json", intent)
            self._read(expected_duration="")
            if time.monotonic() >= self.guard.deadline:
                raise BrowserSafetyStop("planet_copy_time_limit")
            self.fill_attempted = True
            self.duration.fill(text, timeout=3000)
            if self.duration.input_value() != text:
                raise BrowserSafetyStop("observation_fill_readback_mismatch")
            # Native blur commits the value before binding Play. The course may
            # replace Play during this commit, so never click its old handle.
            self.duration.press("Tab", timeout=3000)
            _, _, committed_play = self._read(expected_duration=text, capture_name="committed")
            if self.duration.evaluate("e=>e===document.activeElement"):
                raise BrowserSafetyStop("observation_duration_not_blurred")
            # The committed capture and the final full guard bracket the
            # durable Play reservation. Another intervening full capture adds
            # latency without another native action to validate.
            persist_json(self.output / "play-reserved.json", {**intent, "duration_committed": True})
            _, _, final_play = self._read(expected_duration=text)
            if not _same_handle(committed_play, final_play):
                raise BrowserSafetyStop("observation_play_replaced_before_dispatch")
            # A full capture may itself outlast the fixed deadline. Never
            # dispatch just because its guard began while time remained.
            if time.monotonic() >= self.guard.deadline:
                raise BrowserSafetyStop("planet_copy_time_limit")
            self.play_attempted = True
            final_play.click(timeout=3000)
            persist_json(self.output / "dispatched.json", {**intent, "play_click_returned": True})
            self._read(expected_duration=text, transport_change=True, capture_name="after")
            receipt = {
                **intent,
                "duration_readback": text,
                "duration_committed": True,
                "play_click_dispatched_once": True,
                "readback_verified": True,
                "answers_unchanged": True,
                "correctness_verified": False,
                "observation_started_verified": False,
            }
            persist_json(self.output / "confirmed.json", receipt)
            self.stopped = True
            return receipt
        except BaseException as exc:
            self.stopped = True
            persist_json(
                self.output / "stopped.json",
                {
                    "mode": MODE,
                    "reason": str(exc) if isinstance(exc, BrowserSafetyStop) else "observation_start_failed",
                    "duration_write_may_have_occurred": self.fill_attempted,
                    "play_click_may_have_occurred": self.play_attempted,
                    "automatic_retry": False,
                    "observation_completed": False,
                    "planet_presence": None,
                    "task_completed": False,
                },
            )
            if isinstance(exc, (BrowserSafetyStop, KeyboardInterrupt, SystemExit)):
                raise
            raise BrowserSafetyStop("observation_start_failed") from None


def start_planet_observation(page, config, output, *, run_history, days=MAX_DAYS, max_seconds=60):
    session = PlanetObservationStartSession(
        page, config, output, run_history=run_history, max_seconds=max_seconds
    )
    try:
        return session.start(days)
    finally:
        session.close()
