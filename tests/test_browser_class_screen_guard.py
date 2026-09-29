"""Strict projected stellar comparisons and original-pair failure evidence."""
# ruff: noqa: F401,F811 -- imported pytest fixtures

import hashlib
import json
from copy import deepcopy

import pytest
from test_browser_class_rendering_diagnostic import offline_only, real_session
from test_browser_class_setup_steps import create, rig
from test_browser_star_preflight import stellar

import habfly.browser_class_setup_steps as steps_module
import habfly.browser_classification as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_numeric import screen_identity
from habfly.browser_stellar import map_stellar_capture


def footer(report, *, notice=False, busy=False, channel="both"):
    report = deepcopy(report)
    frame = report["frames"][0]
    frame["accessibility"] = frame["accessibility"].rstrip()
    if not any(c["role"] == "button" for c in frame["controls"]):
        frame["controls"].append(
            {
                "id": "simulation-0:c2",
                "role": "button",
                "enabled": True,
                "accessibility": '- button "Save"',
                "actions": [],
                "protected": False,
            }
        )
    # Mirror exactly the already-supported live footer line wrapping.
    frame["text"] = frame["text"].rsplit("1 Rs", 1)[0].rstrip() + "\n1 Rs\nSave"
    if notice and channel in {"text", "both"}:
        frame["text"] = frame["text"].removesuffix("\nSave") + "\nData saved\nSave"
    if notice and channel in {"accessibility", "both"}:
        frame["accessibility"] = frame["accessibility"].replace(
            '1 Rs\n- button "Save"', '1 Rs Data saved\n- button "Save"'
        )
    if busy:
        save = next(c for c in frame["controls"] if c["accessibility"] == '- button "Save"')
        save["enabled"], save["accessibility"] = False, '- button "Save" [disabled]'
        frame["accessibility"] = frame["accessibility"].replace(
            '- button "Save"', '- button "Save" [disabled]'
        )
    return report


def paired_read(monkeypatch, change):
    session, state = real_session(monkeypatch)
    before = footer(session.report)
    after = change(deepcopy(before))
    session.report = deepcopy(before)
    calls = []

    def inspect(*_):
        calls.append(True)
        return deepcopy(before if len(calls) % 2 else after)

    monkeypatch.setattr(module, "inspect_page", inspect)
    return session, state, before, after, calls


@pytest.mark.parametrize("channel", ["text", "accessibility", "both"])
@pytest.mark.parametrize("busy", [False, True])
def test_read_and_current_ignore_only_exact_existing_status_projection(monkeypatch, channel, busy):
    session, state, before, after, calls = paired_read(
        monkeypatch,
        lambda value: footer(value, notice=True, busy=busy, channel=channel),
    )
    report, _, _, _ = session.current()
    assert report == before and before != after
    assert len(calls) == 2 and state.clicks == 0


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("channel", ["text", "accessibility", "both"])
def test_transition_projections_keep_exact_status_only_equivalence(conditional, channel):
    before = footer(stellar(conditional))
    after = footer(before, notice=True, busy=True, channel=channel)
    mapping = map_stellar_capture(
        before,
        capture_sha256=screen_identity(before),
        allow_color_selection=True,
        allow_main_sequence_fields=True,
    )
    assert module.class_transition_projection(before, mapping) == module.class_transition_projection(
        after, mapping
    )
    if conditional:
        target = mapping["observation"]["values"]["lifetime_prefix"]["capture_target_id"]
        assert module.prefix_transition_projection(before, target) == module.prefix_transition_projection(
            after, target
        )


@pytest.mark.parametrize("change", ["unknown_footer", "text_elsewhere", "control", "answer", "star", "modal"])
def test_unknown_public_changes_preserve_both_original_captures_without_queries(monkeypatch, change):
    def mutate(value):
        frame = value["frames"][0]
        if change == "unknown_footer":
            frame["text"] += "\nSaving failed"
        elif change == "text_elsewhere":
            frame["text"] = "Data saved elsewhere\n" + frame["text"]
        elif change == "control":
            next(c for c in frame["controls"] if c["role"] == "textbox")["enabled"] = False
        elif change == "answer":
            next(c for c in frame["controls"] if c["role"] == "textbox")["value"] = "27"
        elif change == "star":
            frame["text"] = frame["text"].replace("ALTHINAGON", "OTHER").replace("Althinagon", "Other")
            frame["accessibility"] += "\n- text: Different star"
        else:
            frame["accessibility"] += '\n- dialog "Unexpected"'
        return value

    session, _, before, after, calls = paired_read(monkeypatch, mutate)
    with pytest.raises(module.ClassScreenMismatch) as caught:
        session.read()
    error = caught.value
    assert str(error) == "screen_changed_during_class_read"
    assert error.before == before and error.after == after and len(calls) == 2
    assert error.diagnostic["projected_screen_changes"]["change_count"] > 0
    assert error.diagnostic["additional_dom_queries"] == 0
    assert not error.diagnostic["native_class_click_invoked"]
    assert not error.diagnostic["native_prefix_selection_invoked"]
    after["frames"][0]["text"] = "mutated caller copy"
    assert error.after != after


@pytest.mark.parametrize("failed_io", [None, "capture", "difference", "pin"])
def test_owner_failure_persists_original_pair_and_stops_even_on_diagnostic_io_failure(
    rig,
    monkeypatch,
    failed_io,
):
    owner = create(rig)
    owner.advance()  # class verified; prefix is next
    before = deepcopy(rig.page.report)
    after = deepcopy(before)
    after["frames"][0]["text"] += "\nUnrecognized mutation"

    def fail(self, prefix):
        self.emit(
            "action_proposed",
            {
                "kind": "SELECT",
                "target": "lifetime_prefix",
                "value": prefix,
                "action_source": "explicit_unit_transport",
            },
        )
        rig.page.writes.append(("prefix", prefix))
        raise module.ClassScreenMismatch(
            "screen_changed_during_class_read",
            before,
            after,
            phase="post_prefix_read",
            class_invoked=False,
            class_returned=False,
            prefix_invoked=True,
            prefix_returned=True,
        )

    monkeypatch.setattr(steps_module.StellarSelectionSession, "select_prefix", fail)
    previous_save, previous_persist, previous_pin = (
        steps_module.save_probe,
        steps_module.persist_json,
        owner._pin,
    )

    def save(report, path):
        if failed_io == "capture" and "class-screen-failure" in path.parts:
            raise OSError("private diagnostic failure")
        return previous_save(report, path)

    def persist(path, payload):
        if failed_io == "difference" and path.name == "difference.json":
            raise OSError("private diagnostic failure")
        return previous_persist(path, payload)

    def pin(path):
        if failed_io == "pin" and "class-screen-failure" in path.parts:
            raise OSError("private diagnostic failure")
        return previous_pin(path)

    monkeypatch.setattr(steps_module, "save_probe", save)
    monkeypatch.setattr(steps_module, "persist_json", persist)
    monkeypatch.setattr(owner, "_pin", pin)
    state = owner.advance()
    assert state["status"] == "stopped" and not state["setup_verified"]
    assert state["failure_reason"] == "screen_changed_during_class_read"
    proof = state["class_screen_failure"]
    assert proof["failure_phase"] == "post_prefix_read"
    assert proof["native_class_click_invoked"] is False
    assert proof["native_class_click_returned"] is False
    assert proof["native_prefix_selection_invoked"] is True
    assert proof["native_prefix_selection_returned"] is True
    if failed_io:
        assert proof["artifact_recording_failed"] is True
        assert "private" not in json.dumps(state)
    else:
        directory = owner.output / "class-screen-failure"
        assert json.loads((directory / "before/observation.json").read_bytes()) == before
        assert json.loads((directory / "after/observation.json").read_bytes()) == after
        assert proof["sha256"] == hashlib.sha256((directory / "difference.json").read_bytes()).hexdigest()
        assert sum("class-screen-failure/" in name for name in state["source_hashes"]) == 5
    actions = deepcopy(rig.page.writes)
    owner.advance()
    assert rig.page.writes == actions == [("class", "main_sequence"), ("prefix", "Ga")]
    assert not (owner.output / "prefix/confirmed.json").exists()


def test_delayed_lifetime_initialization_inside_read_is_not_suppressed(monkeypatch):
    before = footer(stellar(True))
    after = deepcopy(before)
    next(c for c in after["frames"][0]["controls"] if c["accessibility"] == '- textbox "0.00000"')[
        "value"
    ] = "0.000"
    error = module.ClassScreenMismatch(
        "screen_changed_during_class_read",
        before,
        after,
        phase="post_prefix_read",
        class_invoked=False,
        class_returned=False,
        prefix_invoked=True,
        prefix_returned=True,
    )
    assert error.diagnostic["before_projected_sha256"] != error.diagnostic["after_projected_sha256"]
    assert error.diagnostic["projected_screen_changes"]["change_count"] == 1
