"""Bounded constructor/read-failure evidence, never raw exceptions or new queries."""
# ruff: noqa: F401,F811 -- imported pytest fixtures

import json
from copy import deepcopy

import pytest
from test_browser_class_rendering_diagnostic import offline_only, real_session
from test_browser_class_setup_steps import create, rig

import habfly.browser_class_setup_steps as steps_module
import habfly.browser_classification as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_stellar import StellarMappingError


@pytest.mark.parametrize(
    "kind,code,detail,captured",
    [
        ("mapping", "missing_or_ambiguous_measurement", "parallax", True),
        ("timeout", None, None, False),
        ("frame_ready", "frame_not_ready", "score_widgets", False),
        ("frame_count", "frame_count_mismatch", "score_widgets", False),
        ("unknown", None, None, False),
    ],
)
def test_failed_read_preserves_safe_type_stage_and_only_existing_capture(
    rig, monkeypatch, kind, code, detail, captured
):
    session, _state = real_session(monkeypatch)
    report = deepcopy(session.report)
    inspect_original = module.inspect_page
    calls = []

    def inspect(*args):
        calls.append(True)
        if kind == "timeout":
            raise TimeoutError("private-driver-url http://local/?password=secret")
        if kind.startswith("frame_"):
            raise BrowserSafetyStop(
                ("frame_not_ready" if kind == "frame_ready" else "frame_count_mismatch") + ":score_widgets"
            )
        if kind == "unknown":
            raise ValueError("private-token-secret")
        return inspect_original(*args)

    def mapping(*args, **kwargs):
        raise StellarMappingError("missing_or_ambiguous_measurement:parallax")

    monkeypatch.setattr(module, "inspect_page", inspect)
    if captured:
        monkeypatch.setattr(module, "map_stellar_capture", mapping)
    owner = create(rig, "white_dwarf")
    monkeypatch.setattr(owner, "_class_step", session.read)
    result = owner.advance()
    assert result["status"] == "stopped" and result["failure_reason"] == "fresh_class_steps_operation_failed"
    proof = result["class_operation_failure"]
    assert proof["error_code"] == code and proof["error_detail"] == detail
    assert proof["read_stage"] == ("map_stellar_capture" if captured else "inspect_page")
    assert proof["public_capture_available"] is captured
    assert proof["additional_dom_queries"] == 0 and len(calls) == 1
    assert len(proof["repo_frames"]) <= 4
    assert all(set(frame) == {"module", "function", "line"} for frame in proof["repo_frames"])
    saved = json.loads((owner.output / "class-operation-failure/diagnostic.json").read_bytes())
    assert "secret" not in json.dumps(saved) and "http" not in json.dumps(saved)
    if captured:
        assert (
            json.loads((owner.output / "class-operation-failure/capture/observation.json").read_bytes())
            == report
        )
    else:
        assert not (owner.output / "class-operation-failure/capture").exists()
    owner.advance()
    assert len(calls) == 1 and not rig.page.writes


def test_unknown_exception_type_context_and_code_are_redacted():
    secret_type = type("PrivateTokenType", (ValueError,), {})
    exc = secret_type("secret /private/driver-path")
    exc._habfly_class_read_context = {
        "read_phase": "private-phase",
        "read_stage": "private-stage",
        "private_extra": "secret",
        "native_class_click_invoked": 1,
        "report": None,
    }
    proof, report = steps_module._operation_diagnostic(exc, "private-operation")
    assert proof["exception_type"] == "OtherException"
    assert proof["read_phase"] is proof["read_stage"] is proof["operation_phase"] is None
    assert "native_class_click_invoked" not in proof
    assert "private" not in json.dumps(proof).lower() and "secret" not in json.dumps(proof)
    assert report is None
    proof, _ = steps_module._operation_diagnostic(
        BrowserSafetyStop("frame_not_ready:private-secret"), "class_pending"
    )
    assert proof["error_code"] is proof["error_detail"] is None


def test_operation_diagnostic_write_failure_remains_terminal(rig, monkeypatch):
    owner = create(rig, "white_dwarf")
    original = steps_module.persist_json

    def fail():
        raise TimeoutError("private exception")

    def persist(path, payload):
        if "class-operation-failure" in path.parts:
            raise OSError("private disk error")
        return original(path, payload)

    monkeypatch.setattr(owner, "_class_step", fail)
    monkeypatch.setattr(steps_module, "persist_json", persist)
    result = owner.advance()
    assert result["status"] == "stopped" and result["failure_reason"] == "fresh_class_steps_operation_failed"
    assert result["class_operation_failure"]["artifact_recording_failed"] is True
    assert not result["setup_verified"] and not rig.page.writes
    owner.advance()
    assert not rig.page.writes


def test_success_has_no_operation_failure_artifact_or_payload(rig):
    owner = create(rig, "white_dwarf")
    result = owner.advance()
    assert result["setup_verified"] and "class_operation_failure" not in result
    assert not (owner.output / "class-operation-failure").exists()


def test_exception_annotation_failure_preserves_original_failure(monkeypatch):
    session, _state = real_session(monkeypatch)

    class AnnotationDenied(ValueError):
        def __setattr__(self, name, value):
            raise OSError("private annotation failure")

    original = AnnotationDenied("original private failure")

    def fail(*args):
        raise original

    monkeypatch.setattr(module, "inspect_page", fail)
    with pytest.raises(AnnotationDenied) as caught:
        session.read()
    assert caught.value is original
