"""Presentation-only captions for Sporky; never an action or evidence authority.

Only fixed captions and recognized operation identifiers leave this reducer.
Observation prose, URLs, credentials, numeric answers, confidence and purported
reasoning are neither retained nor rendered. A proposed operation is labelled
as a proposal; a finished component is not a completed project.
"""

import re

_EVENTS = frozenset(
    {
        "hello",
        "state",
        "observation",
        "action_proposed",
        "action_result",
        "neural_activity",
        "episode_summary",
        "error",
    }
)
_OPERATIONS = {
    "distance": "distance",
    "luminosity": "luminosity",
    "temperature": "temperature",
    "mass": "stellar mass",
    "radius": "stellar radius",
    "lifetime": "stellar lifetime",
    "period_years": "orbital period",
    "radial_velocity": "radial velocity",
    "orbital_radius": "orbital radius",
    "planet_mass": "planet mass",
    "planet_radius": "planet radius",
    "planet_density": "planet density",
    "equilibrium_temp": "equilibrium temperature",
    "surface_temp": "surface temperature",
}
_FIELDS = {
    **_OPERATIONS,
    "line_shift": "spectral shift",
    "brightness_drop": "brightness drop",
    "period_days": "transit period",
    "greenhouse_increment": "greenhouse warming",
}
_TOOL_TARGET = re.compile(
    r"[0-9]{1,10}:(operation|parameter|source|bind|execute|result|destination|copy|check|unit_[a-z_]+)"
)
_LIVE_TARGET = re.compile(r"live:[0-9]{1,10}:([a-z_]+)")
_TEMPLATES = {
    "select": "Selecting {} calculation",
    "calculate": "Preparing {} calculation",
    "calculated": "Calculated {}",
    "destination": "Selecting {} destination",
    "copy": "Preparing copy to {}",
    "copied": "Copied to {}",
    "unit": "Selecting {} units",
    "readback": "Checked {} entry",
}
_FIXED = frozenset(
    {
        "Ready",
        "Status unavailable",
        "Paused",
        "Stopped",
        "Run ended",
        "Task completed",
        "Project completed",
        "Step finished",
        "Waiting",
        "Running",
        "Reading observation",
        "Selecting calculation",
        "Selecting input",
        "Selecting measurement",
        "Binding input",
        "Selecting result",
        "Selecting answer destination",
        "Preparing calculation",
        "Preparing result copy",
        "Checking task",
        "Calculation needs attention",
        "Action needs attention",
        "Reading spectrum",
        "Reading light curve",
        "Adjusting chart view",
        "Selecting planet presence",
        "Selecting planet class",
        "Selecting atmospheric gases",
        "Selecting habitability",
        "Checking entry",
        "Ready for assessment",
        "Awaiting confirmation",
        "Awaiting a decision",
        "Replay paused",
        "Replay ended",
    }
)
ALLOWED_CAPTIONS = _FIXED | frozenset(
    template.format(label) for template in _TEMPLATES.values() for label in _FIELDS.values()
)


def _envelope(value):
    if (
        type(value) is not dict
        or type(value.get("event")) is not str
        or value["event"] not in _EVENTS
        or type(value.get("payload")) is not dict
    ):
        return False
    if "version" in value and (type(value["version"]) is not int or value["version"] != 1):
        return False
    if "sequence" in value and (type(value["sequence"]) is not int or value["sequence"] < 0):
        return False
    run = value.get("run_id")
    return run is None or type(run) is str and 0 < len(run) <= 128


def _leaf(value):
    """Accept both full v1 envelopes and the legacy two-key child envelope."""
    context, seen = [], set()
    current = value
    for _ in range(9):
        if not _envelope(current) or id(current) in seen:
            return None
        seen.add(id(current))
        data = current["payload"]
        child = data.get("component_event")
        if "component_event" not in data:
            return current, tuple(context)
        name = data.get("component_identity", data.get("component"))
        star = data.get("component_star", data.get("star"))
        if (
            type(name) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.-]{0,79}", name) is None
            or star is not None
            and (type(star) is not str or len(star) > 100)
            or not _envelope(child)
        ):
            return None
        expected = "state" if child["event"] in {"hello", "state", "episode_summary"} else child["event"]
        if current["event"] != expected:
            return None
        context.append((name, star, child.get("run_id")))
        current = child
    return None


class MascotStatusReducer:
    """Small deterministic reducer; replay a new trace with a fresh/reset instance.

    No file access, timers, network, model inference, or runtime mutations.
    Output contains only caption/activity/source_label. It is not a receipt.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._run = self._sequence = self._context = None
        self._recent = self._star_token = None
        self._clear_calculation()
        return self._set("Ready", "idle", "Runtime status")

    def snapshot(self):
        return dict(self._status)

    def _set(self, caption, activity, source):
        if activity in ("paused", "stopped", "completed", "unknown"):
            self._recent = None
        elif source == "Proposed action":
            self._recent = (caption, "Recent action")
        elif source in ("Recorded tool result", "Recorded readback"):
            self._recent = (caption, "Recent result")
        self._status = {"caption": caption, "activity": activity, "source_label": source}
        return self.snapshot()

    def _passive(self, caption, activity, source):
        if self._recent is not None:
            recent, label = self._recent
            return self._set(recent, "recent", label)
        return self._set(caption, activity, source)

    def _context_changed(self):
        self._clear_calculation()
        self._recent = None
        self._set("Waiting", "idle", "Runtime status")

    def _clear_calculation(self):
        self._operation = self._destination = self._revision = None

    def _unknown(self):
        self._clear_calculation()
        return self._set("Status unavailable", "unknown", "Runtime status")

    def _named(self, template, field, activity, source, fallback):
        label = _FIELDS.get(field) if type(field) is str else None
        caption = _TEMPLATES[template].format(label) if label else fallback
        return self._set(caption, activity, source)

    def _observe(self, observation):
        self._clear_calculation()
        if type(observation) is not dict:
            return
        revision, calculation = observation.get("revision"), observation.get("calculation")
        if type(revision) is int and revision >= 0:
            self._revision = revision
        if type(calculation) is dict:
            operation, destination = calculation.get("operation"), calculation.get("destination")
            if type(operation) is str and operation in _OPERATIONS:
                self._operation = operation
            if type(destination) is str and destination in _FIELDS:
                self._destination = destination

    def _lifecycle(self, kind, payload, *, nested):
        source = "Component status" if nested else "Runtime status"
        status, phase = payload.get("status"), payload.get("phase", payload.get("stage"))
        if status is not None and type(status) is not str or phase is not None and type(phase) is not str:
            return self._unknown()
        if status in ("aborted", "stopped") or payload.get("failure_reason") not in (None, ""):
            self._clear_calculation()
            return self._set("Stopped", "stopped", source)
        if status == "paused":
            return self._set("Replay paused" if payload.get("replay") is True else "Paused", "paused", source)
        if phase == "unknown_pending" or status == "unknown_pending":
            self._recent = None
            return self._set("Awaiting confirmation", "waiting", source)
        if phase == "awaiting_assessment":
            self._recent = None
            return self._set("Ready for assessment", "waiting", source)
        if phase in (
            "awaiting_class_source",
            "awaiting_planet_class",
            "awaiting_gases",
            "awaiting_habitability",
        ):
            self._recent = None
            return self._set("Awaiting a decision", "waiting", source)
        if kind == "episode_summary" or status == "completed":
            self._recent = None
            if nested or payload.get("component_only") is True:
                return self._set("Step finished", "idle", source)
            if payload.get("project_completed") is True:
                return self._set("Project completed", "completed", source)
            if payload.get("task_completed") is True or payload.get("completed") is True:
                return self._set("Task completed", "completed", source)
            return self._set("Run ended", "idle", source)
        if status == "replay_completed":
            self._recent = None
            return self._set("Replay ended", "idle", source)
        if status == "running":
            return self._passive("Running", "observing", source)
        return self._passive("Ready" if kind == "hello" else "Waiting", "idle", source)

    def _action(self, payload):
        action = payload.get("action", payload)
        if type(action) is not dict or type(action.get("kind")) is not str:
            return self._unknown()
        kind, target, value = action["kind"], action.get("target"), action.get("value")
        if value is not None and type(value) is not str:
            return self._unknown()
        source = "Proposed action"
        if kind == "STOP":
            return self._set("Stopped", "stopped", source)
        if kind == "WAIT":
            return self._set("Waiting", "waiting", source)
        if kind == "HOVER" and action.get("surface") in ("spectrum", "chart"):
            return self._set(
                "Reading spectrum" if action["surface"] == "spectrum" else "Reading light curve",
                "observing",
                source,
            )
        if kind in ("SCROLL", "DRAG") and action.get("surface") == "chart":
            return self._set("Adjusting chart view", "observing", source)
        match = _TOOL_TARGET.fullmatch(target) if type(target) is str else None
        if match:
            revision = action.get("observation_revision")
            if self._revision is not None and int(target.partition(":")[0]) != self._revision:
                return self._unknown()
            if revision is not None and (
                type(revision) is not int or self._revision is not None and revision != self._revision
            ):
                return self._unknown()
            key = match.group(1)
            if key == "operation" and kind == "SELECT":
                return self._named(
                    "select",
                    value if type(value) is str and value in _OPERATIONS else None,
                    "selecting",
                    source,
                    "Selecting calculation",
                )
            if key == "execute" and kind == "CLICK":
                return self._named(
                    "calculate", self._operation, "calculating", source, "Preparing calculation"
                )
            if key == "copy" and kind == "CLICK":
                return self._named("copy", self._destination, "copying", source, "Preparing result copy")
            if key == "destination" and kind == "SELECT":
                return self._named("destination", value, "selecting", source, "Selecting answer destination")
            if key.startswith("unit_") and kind == "SELECT":
                return self._named("unit", key[5:], "selecting", source, "Selecting answer destination")
            caption = {
                ("parameter", "SELECT"): "Selecting input",
                ("source", "SELECT"): "Selecting measurement",
                ("bind", "CLICK"): "Binding input",
                ("result", "SELECT"): "Selecting result",
                ("check", "CLICK"): "Checking task",
            }.get((key, kind))
            if caption:
                return self._set(caption, "selecting" if kind == "SELECT" else "verifying", source)
        if kind == "TYPE":
            live = _LIVE_TARGET.fullmatch(target) if type(target) is str else None
            destination = action.get("destination", live.group(1) if live else None)
            if type(destination) is str and destination in _FIELDS:
                return self._named("copy", destination, "copying", source, "Preparing result copy")
        if kind == "SELECT":
            caption = (
                {
                    "has_planet": "Selecting planet presence",
                    "planet_class": "Selecting planet class",
                    "gases": "Selecting atmospheric gases",
                    "habitability": "Selecting habitability",
                }.get(target)
                if type(target) is str
                else None
            )
            if caption:
                return self._set(caption, "selecting", source)
        return self._unknown()

    def _result(self, payload):
        if payload.get("failure_reason") not in (None, ""):
            self._clear_calculation()
            return self._set("Action needs attention", "stopped", "Recorded result")
        observation = payload.get("observation")
        self._observe(observation)
        calculation = observation.get("calculation", {}) if type(observation) is dict else {}
        if type(calculation) is dict:
            operation = calculation.get("last_operation")
            if calculation.get("tool_error") not in (None, ""):
                return self._set("Calculation needs attention", "waiting", "Recorded tool result")
            if type(operation) is dict:
                if operation.get("kind") == "calculate":
                    result = operation.get("result")
                    if type(result) is dict and result.get("ok") is True:
                        return self._named(
                            "calculated",
                            operation.get("operation"),
                            "calculating",
                            "Recorded tool result",
                            "Step finished",
                        )
                    return self._set("Calculation needs attention", "waiting", "Recorded tool result")
                if operation.get("kind") == "copy":
                    return self._named(
                        "copied",
                        operation.get("destination"),
                        "copying",
                        "Recorded tool result",
                        "Step finished",
                    )
        receipt = payload.get("native_copy")
        if type(receipt) is dict and receipt.get("readback_verified") is True:
            return self._named(
                "readback", receipt.get("destination"), "verifying", "Recorded readback", "Checking entry"
            )
        if payload.get("terminated") is True or payload.get("truncated") is True:
            return self._set("Step finished", "idle", "Recorded result")
        return self.snapshot()

    def consume(self, event):
        """Consume one event; malformed/unsupported inputs cannot leak their text."""
        try:
            return self._consume(event)
        except (TypeError, ValueError, RecursionError):
            return self._unknown()

    def _consume(self, event):
        parsed = _leaf(event)
        if parsed is None:
            return self._unknown()
        leaf, ancestry = parsed
        run, sequence = event.get("run_id"), event.get("sequence")
        if run != self._run:
            self.reset()
            self._run = run
        if type(sequence) is int:
            if self._sequence is not None and sequence <= self._sequence:
                return self.snapshot()
            self._sequence = sequence
        # An ancestor heartbeat has no new native context. Keep the deepest
        # active lineage, but clear a prior star/component's semantic caption.
        context = (run, ancestry, leaf.get("run_id"))
        if self._context is None:
            self._context = context
        else:
            old_run, old_ancestry, old_leaf_run = self._context
            shared = min(len(ancestry), len(old_ancestry))
            compatible = old_run == run and ancestry[:shared] == old_ancestry[:shared]
            if len(ancestry) < len(old_ancestry) and leaf["event"] not in (
                "hello",
                "state",
                "episode_summary",
                "neural_activity",
            ):
                compatible = False
            if ancestry == old_ancestry and old_leaf_run != leaf.get("run_id"):
                compatible = False
            if not compatible:
                self._context_changed()
                self._context = context
            elif len(ancestry) >= len(old_ancestry):
                self._context = context
        outer = event["payload"]
        if "current_star" in outer:
            current = outer["current_star"]
            if current is None:
                token = None
            elif type(current) is dict and type(current.get("star")) is str and len(current["star"]) <= 100:
                ordinal = current.get("ordinal")
                if ordinal is not None and type(ordinal) is not int:
                    return self._unknown()
                token = (ordinal, current["star"].casefold())
            else:
                return self._unknown()
            if token != self._star_token:
                self._context_changed()
                self._star_token = token
        kind, payload = leaf["event"], leaf["payload"]
        if (
            ancestry
            and event["event"] == "state"
            and event["payload"].get("status")
            in ("paused", "stopped", "aborted", "unknown_pending", "replay_completed")
        ):
            return self._lifecycle("state", event["payload"], nested=False)
        if kind == "error":
            self._clear_calculation()
            return self._set("Stopped", "stopped", "Component status" if ancestry else "Runtime status")
        if kind in ("hello", "state", "episode_summary"):
            if kind == "hello":
                self._clear_calculation()
                self._recent = None
            return self._lifecycle(kind, payload, nested=bool(ancestry))
        if kind == "observation":
            self._observe(payload)
            return (
                self.snapshot()
                if self._status["activity"] in ("paused", "stopped", "completed")
                else self._passive("Reading observation", "observing", "Recorded observation")
            )
        if kind == "action_proposed":
            return self._action(payload)
        if kind == "action_result":
            return self._result(payload)
        # Neural summaries never invent thoughts/confidence, but retain an
        # explicitly labelled recent operation instead of implying current work.
        return self._passive(self._status["caption"], self._status["activity"], self._status["source_label"])
