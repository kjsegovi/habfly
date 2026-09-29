"""One project lifecycle around separately recorded component event streams.

A component finishing is not a star, assessment, or project completing. The
owner alone emits those claims. Original component envelopes are preserved for
offline comparison; credentials and authentication events must never be supplied.
"""

import copy
import re

from .contracts import RuntimeEvent


class ProjectEventRelay:
    """Bind callbacks to a component generation and fail closed on stale events.

    The receiver is the owner's version-1 ``emit(event, payload)`` function. Event
    ordering within a recorded child stream is checked without reusing its sequence
    numbers in the outer stream. Retiring a callback prevents a previous star's
    asynchronous events from overwriting the current observation.
    """

    def __init__(self, emit):
        if not callable(emit):
            raise TypeError("Project event receiver must be callable")
        self._emit, self._active, self._generation = emit, None, 0
        self.failed = False

    def bind(self, component, *, star=None):
        if self.failed:
            raise ValueError("Project event relay has failed")
        if not isinstance(component, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,79}", component):
            raise ValueError("Invalid component identity")
        if star is not None and (
            not isinstance(star, str) or not star.strip() or len(star) > 100 or star != star.strip()
        ):
            raise ValueError("Invalid visible star identity")
        self._generation += 1
        token = self._generation
        self._active = token
        previous = {}

        def forward(item, payload=None):
            if self.failed or self._active != token:
                raise ValueError("Stale or failed project component callback")
            try:
                # Components with an outer-style callback do not claim a child
                # sequence. Preserve this distinction instead of inventing one.
                if isinstance(item, str):
                    if not isinstance(payload, dict):
                        raise TypeError("Component payload must be an object")
                    event = RuntimeEvent(event=item, sequence=0, payload=payload)
                    original = {"event": event.event, "payload": copy.deepcopy(event.payload)}
                else:
                    if payload is not None or not isinstance(item, dict):
                        raise ValueError("Invalid component event")
                    event = RuntimeEvent.model_validate(item)
                    original = copy.deepcopy(item)
                    key = event.run_id
                    if event.sequence <= previous.get(key, -1):
                        raise ValueError("Non-monotonic component event sequence")
                    previous[key] = event.sequence
                identity = {"component": component, "component_star": star, "component_event": original}
                if event.event in {"hello", "state", "episode_summary"}:
                    kind = "state"
                    field = {
                        "hello": "component_hello",
                        "state": "component_state",
                        "episode_summary": "component_summary",
                    }[event.event]
                    outgoing = {**identity, field: copy.deepcopy(event.payload)}
                else:
                    kind = event.event
                    outgoing = {**copy.deepcopy(event.payload), **identity}
                self._emit(kind, outgoing)
            except Exception:  # noqa: BLE001 - sanitize and stop the outer event stream
                self.failed = True
                self._active = None
                # No driver error, URL, or arbitrary child payload enters the
                # exception text exposed by the outer protocol boundary.
                raise ValueError("Project component event forwarding failed") from None

        return forward

    def retire(self):
        self._active = None
