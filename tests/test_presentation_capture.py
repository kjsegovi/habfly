"""The cosmetic overlay is removed, never painted over scientific pixels."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from habfly.presentation_capture import EVIDENCE_SCREENSHOT_STYLE, MASCOT_OVERLAY_ID, evidence_screenshot


@pytest.mark.parametrize("style", [None, "", "body { background: white; }"])
def test_only_adds_temporary_style_preserving_options_bytes_and_target(style):
    calls = []
    raw = b"exact-original-image-bytes"

    def screenshot(**options):
        calls.append(options)
        return raw

    target = SimpleNamespace(screenshot=screenshot)
    options = {
        "path": Path("evidence.png"),
        "timeout": 317,
        "scale": "css",
        "full_page": False,
        "animations": "allow",
        "style": style,
    }
    original = dict(options)
    assert evidence_screenshot(target, **options) is raw
    assert options == original
    assert calls == [{**options, "style": (style + "\n" if style else "") + EVIDENCE_SCREENSHOT_STYLE}]
    assert "mask" not in calls[0] and "mask_color" not in calls[0]


def test_no_options_or_browser_queries_required():
    calls = []
    target = SimpleNamespace(screenshot=lambda **options: calls.append(options) or b"png")
    assert evidence_screenshot(target) == b"png"
    assert calls == [{"style": EVIDENCE_SCREENSHOT_STYLE}]
    assert EVIDENCE_SCREENSHOT_STYLE == (
        'div#habfly-mascot-overlay-v1[data-habfly-presentation="mascot-v1"] '
        "{ visibility: hidden !important; }"
    )
    assert MASCOT_OVERLAY_ID == "habfly-mascot-overlay-v1"


@pytest.mark.parametrize("style", [False, True, 0, 1, b"css", [], {}, object()])
def test_invalid_style_rejected_without_capture(style):
    def screenshot(**_options):
        pytest.fail("Invalid style must not reach Playwright")

    with pytest.raises(ValueError, match="string or None"):
        evidence_screenshot(SimpleNamespace(screenshot=screenshot), style=style)


def test_capture_exception_propagates_once_without_retry_or_replacement():
    calls = []
    failure = RuntimeError("fixture capture failed")

    def screenshot(**options):
        calls.append(options)
        raise failure

    with pytest.raises(RuntimeError) as caught:
        evidence_screenshot(SimpleNamespace(screenshot=screenshot), timeout=90)
    assert caught.value is failure
    assert calls == [{"timeout": 90, "style": EVIDENCE_SCREENSHOT_STYLE}]


def test_existing_production_screenshots_are_all_routed_through_helper():
    root = Path(__file__).resolve().parents[1] / "src/habfly"
    expected = {
        "browser.py": 2,
        "browser_setup.py": 3,
        "browser_planet_absence.py": 1,
        "browser_gas_controls.py": 1,
        "browser_project_submission_steps.py": 1,
        "browser_transit_sampling.py": 1,
        "browser_observation_progress.py": 1,
        "browser_submission_outcome_diagnostic.py": 1,
        "browser_shallow_transit_steps.py": 1,
        "browser_shallow_transit_probe.py": 1,
        "browser_water_chamber.py": 2,
    }
    for name, count in expected.items():
        nodes = list(ast.walk(ast.parse((root / name).read_text())))
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "screenshot"
            for node in nodes
        ), name
        assert (
            sum(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "evidence_screenshot"
                for node in nodes
            )
            == count
        ), name
