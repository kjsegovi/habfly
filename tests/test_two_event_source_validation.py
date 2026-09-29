"""Source preflight only: borrowed fixtures forbid network and model loading."""
# ruff: noqa: F401,F811

from test_autonomous_validation import no_network_or_models, options, source_fixture

from habfly.autonomous_validation import validate_autonomous_decision_sources
from habfly.browser_shallow_transit_probe import EXACT_TWO_HINT_POLICY, overview_hint_metadata
from habfly.planet_tooltip_reference import TWO_MODE, measurement_manifest


def test_explicit_two_event_preflight_pins_method_hint_and_implementations(options):
    options["project_reference_shallow_transits"] = True
    options["project_two_event_reference"] = True
    report = validate_autonomous_decision_sources(options)
    assert report["references"]["two_event_measurement_method"] == measurement_manifest(TWO_MODE)
    assert report["references"]["two_event_overview_hint"] == overview_hint_metadata(EXACT_TWO_HINT_POLICY)
    for name in ("reference", "probe", "scheduler"):
        assert (
            report["references"]["two_event_" + name + "_sha256"]
            == report["source_files"]["reference.two_event_" + name]["sha256"]
        )
    assert report["sources_verified"] is True
    assert (
        report["browser_readiness_verified"] is report["model_loaded"] is report["training_executed"] is False
    )
    assert report["thirty_star_launch_authorized"] is False


def test_default_preflight_does_not_imply_two_event_optin(options):
    assert options.get("project_two_event_reference", False) is False
    report = validate_autonomous_decision_sources(options)
    assert not any("two_event" in key for key in report["references"])
    assert not any("two_event" in key for key in report["source_files"])
