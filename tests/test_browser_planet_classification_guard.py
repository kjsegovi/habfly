"""Pure dispatch-boundary tests; no Chromium or application access."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

import habfly.browser_classification as paint
import habfly.browser_planet_classification as module
from habfly.browser import BrowserSafetyStop


@pytest.fixture
def rig(tmp_path, monkeypatch):
    item = SimpleNamespace(
        reads=0,
        writes=0,
        closed=False,
        source="same",
        selected=None,
        fail_guard=False,
        replaced=False,
        output=tmp_path / "class",
    )

    def values():
        capture = {"source": item.source}
        mapping = {
            "star_name": "ALPHA",
            "observation": {
                "values": {"browser_field_map": {key: {"current_value": "1"} for key in module.ANSWER_UNITS}}
            },
        }
        choices = {
            "selected": item.selected,
            "rendering": {k: "selected" if k == item.selected else "blank" for k in module.CLASSES},
        }
        return capture, mapping, choices, {}

    class Handle:
        def evaluate(self, script, other):
            return self is other

        def click(self, **kwargs):
            item.writes += 1
            item.selected = "gas_giant"

    handles = {key: Handle() for key in module.CLASSES}

    class Session:
        def __init__(self, *args, **kwargs):
            self.frame = object()

        def current(self):
            item.reads += 1
            if item.fail_guard:
                raise BrowserSafetyStop("planet_copy_context_changed")
            return deepcopy(values())

        def read(self):
            return deepcopy(values())

        def close(self):
            item.closed = True

    def painted(frame):
        bound = {key: Handle() for key in module.CLASSES} if item.replaced else handles
        return deepcopy(values()[2]), bound

    monkeypatch.setattr(module, "PlanetNumericSession", Session)
    monkeypatch.setattr(module, "planet_projection", lambda capture, mapping: capture["source"])
    monkeypatch.setattr(module, "save_probe", lambda *args: None)
    monkeypatch.setattr(paint, "read_planet_class_choices", painted)
    return item


def select(rig, **kwargs):
    return module.select_planet_class(
        None, None, rig.output, "gas_giant", source="reference_diagnostic", **kwargs
    )


def test_legacy_guard_and_receipt_shape_unchanged(rig):
    receipt = select(rig)
    assert rig.reads == 2 and rig.writes == 1 and rig.closed
    assert receipt == {
        "kind": "SELECT",
        "value": "gas_giant",
        "star": "ALPHA",
        "action_source": "reference_diagnostic",
        "max_class_writes": 1,
        "numeric_writes": 0,
        "correctness_verified": False,
        "task_completed": False,
        "readback_verified": True,
    }


def test_optional_callback_gets_visible_current_source_then_fresh_guard(rig):
    seen = []

    def guard(capture, mapping, choices):
        seen.append((capture, mapping, choices))
        assert rig.reads == 2 and rig.writes == 0

    assert select(rig, before_dispatch=guard)["readback_verified"]
    assert len(seen) == 1 and seen[0][0] == {"source": "same"}
    assert seen[0][1]["star_name"] == "ALPHA" and seen[0][2]["selected"] is None
    assert rig.reads == 3 and rig.writes == 1


@pytest.mark.parametrize(
    "mutation", ["source", "paint", "replacement", "page_boundary", "cancel", "private_error"]
)
def test_callback_effects_cannot_authorize_later_click(rig, mutation):
    def guard(capture, mapping, choices):
        if mutation == "source":
            rig.source = "changed"
        elif mutation == "paint":
            rig.selected = "ice_giant"
        elif mutation == "replacement":
            rig.replaced = True
        elif mutation == "page_boundary":
            rig.fail_guard = True
        elif mutation == "cancel":
            raise BrowserSafetyStop("operator_aborted")
        else:
            raise RuntimeError("sensitive-source-not-public")

    with pytest.raises((BrowserSafetyStop, RuntimeError)):
        select(rig, before_dispatch=guard)
    assert rig.writes == 0 and rig.closed
    stopped = json.loads((rig.output / "stopped.json").read_bytes())
    assert stopped["write_may_have_occurred"] is False and not stopped["automatic_retry"]
    assert not (rig.output / "confirmed.json").exists()
    assert "sensitive" not in json.dumps(stopped)


def test_mutating_callback_arguments_cannot_replace_authoritative_capture(rig):
    def guard(capture, mapping, choices):
        rig.source = capture["source"] = "changed"

    with pytest.raises(BrowserSafetyStop, match="source_changed"):
        select(rig, before_dispatch=guard)
    assert rig.writes == 0


def test_invalid_callback_rejected_without_output(rig):
    with pytest.raises(BrowserSafetyStop, match="invalid_planet_class_dispatch_guard"):
        select(rig, before_dispatch=True)
    assert not rig.output.exists() and rig.reads == rig.writes == 0
