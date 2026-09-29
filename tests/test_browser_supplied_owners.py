"""Opt-in scheduling fixtures, not a learned policy or native acceptance run."""
# ruff: noqa: F811

from types import SimpleNamespace

import pytest
from test_browser_positive_steps import complete as finish_owner
from test_browser_positive_steps import rig as owner_rig  # noqa: F401
from test_browser_star_session import advance_to, create_positive, positive_rig, rig  # noqa: F401

import habfly.browser_positive_steps as positive
import habfly.browser_supplied_steps as supplied
from habfly.browser import BrowserSafetyStop
from habfly.planet_supplied_inputs import load_supplied_planet_pack
from habfly.supplied_browser_modes import NON_MAIN_CLASSES


@pytest.mark.parametrize("actual_class", NON_MAIN_CLASSES)
def test_non_main_positive_gate_failure_precedes_all_native_actions(owner_rig, monkeypatch, actual_class):
    def denied(*args):
        raise BrowserSafetyStop("test_new_gate_refused")

    monkeypatch.setattr(supplied, "load_supplied_frozen_policy", denied)
    with pytest.raises(BrowserSafetyStop, match="new_gate_refused"):
        owner_rig.create(
            supplied_star_class=actual_class,
            supplied_evaluation=owner_rig.root / "new-gate",
            supplied_stellar_sources={
                key: owner_rig.root / key
                for key in ("numeric_dir", "color_dir", "class_dir", "navigation_dir")
            },
        )
    assert not owner_rig.calls and not (owner_rig.root / "run").exists()


@pytest.mark.parametrize("actual_class", NON_MAIN_CLASSES)
def test_supplied_child_receipt_is_outside_raw_tree_and_uses_owned_relative_path(
    owner_rig, monkeypatch, actual_class
):
    rig = owner_rig
    options_seen, built = [], []
    original_child = positive.PlanetDerivedSteps
    original_record = positive.PositivePlanetSteps._record

    def record(self, kind, report):
        if kind == "raw":
            # Add the scripted fixture's capture before its immutable tree is
            # recorded; production raw children already persist these captures.
            capture = self.output / "raw/native-copies/copy-03-after/observation.json"
            capture.parent.mkdir(parents=True)
            capture.write_text("{}")
        return original_record(self, kind, report)

    def prepare(self, pilot, evaluation, graph, sources, star, selected):
        # Explicit fixture seam: strict gate/archive/source proof has separate tests.
        self._model_paths = {"checkpoint": rig.checkpoint}
        self._model_hashes = {
            "checkpoint": positive.file_hash(rig.checkpoint),
            "graph": "graph",
            "knowledge_pack": load_supplied_planet_pack().checksum,
        }
        self._supplied_book = SimpleNamespace(unchanged=lambda: None)
        self._stellar_source_dirs, self._supplied_pins = sources, {}
        self._supplied_receipt_link = None
        self._supplied_initial_inputs = {"fixture": "M/R"}

    def build(history, **kwargs):
        built.append(kwargs)
        assert kwargs["selected_class"] == actual_class
        assert kwargs["current_capture_dir"] == rig.root / "run/raw/native-copies/copy-03-after"
        return {"fixture_source": True, "actual_class": actual_class}

    def child(*args, **kwargs):
        options_seen.append(kwargs)
        return original_child(*args, **kwargs)

    monkeypatch.setattr(positive.PositivePlanetSteps, "_prepare_supplied", prepare)
    monkeypatch.setattr(positive.PositivePlanetSteps, "_record", record)
    monkeypatch.setattr(supplied, "SuppliedPlanetDerivedSteps", child)
    import habfly.planet_supplied_stellar_source as source

    monkeypatch.setattr(source, "build_supplied_planet_stellar_inputs", build)
    monkeypatch.setattr(source, "_inputs", lambda *args: {"fixture": "M/R"})
    component = rig.create(
        supplied_star_class=actual_class,
        supplied_evaluation=rig.root / "new-gate",
        supplied_stellar_sources={
            key: rig.root / key for key in ("numeric_dir", "color_dir", "class_dir", "navigation_dir")
        },
    )
    for _ in range(10):
        component.advance()
        if component.phase == "derived":
            break
    assert component.phase == "derived" and not component.finished
    finish_owner(component)
    assert component.phase == "planet_classification_required" and not component.failure
    assert component.report["mode"] == positive.SUPPLIED_OWNER_MODE
    assert not component.report["task_completed"] and len(built) == len(options_seen) == 1
    assert options_seen[0]["supplied_star_class"] == actual_class
    assert options_seen[0]["final_evaluation"] == rig.root / "new-gate"
    receipt = options_seen[0]["supplied_stellar_inputs_path"]
    assert receipt == rig.root / "run/supplied-stellar-inputs/receipt.json"
    assert not receipt.is_relative_to(rig.root / "run/raw")


@pytest.mark.parametrize("actual_class", NON_MAIN_CLASSES)
def test_star_session_only_continues_non_main_when_new_gate_is_explicit(positive_rig, actual_class):
    rig = positive_rig
    gate = rig.root / "new-gate"
    gate.mkdir()
    (gate / "report.json").write_text("{}")
    session = create_positive(
        rig, selected_class=actual_class, lifetime_prefix=None, planet_supplied_evaluation=gate
    )
    advance_to(session, "capture_spectrum")
    session.advance()
    session.advance()
    # Recorded fixture child ctor options prove class/source forwarding only.
    options = rig.positive_options[-1]
    assert options["supplied_star_class"] == actual_class and options["supplied_evaluation"] == gate
    assert options["supplied_stellar_sources"] == {
        "numeric_dir": rig.root / "run/numeric",
        "color_dir": rig.root / "run/color",
        "class_dir": rig.root / "class",
        "navigation_dir": rig.root / "run/to-planet",
    }
    session.abort()


def test_gate_identity_change_stops_before_even_a_stellar_decision(positive_rig):
    gate = positive_rig.root / "new-gate"
    gate.mkdir()
    (gate / "report.json").write_text("{}")
    session = create_positive(
        positive_rig, selected_class="white_dwarf", lifetime_prefix=None, planet_supplied_evaluation=gate
    )
    (gate / "report.json").write_text('{"changed":true}')
    session.advance()
    assert session.finished and session.failure == "star_session_supplied_permission_changed"
    assert not positive_rig.calls


def test_main_sequence_keeps_original_controller_when_new_gate_configured(positive_rig):
    gate = positive_rig.root / "new-gate"
    gate.mkdir()
    (gate / "report.json").write_text("{}")
    session = create_positive(positive_rig, planet_supplied_evaluation=gate)
    advance_to(session, "capture_spectrum")
    session.advance()
    session.advance()
    options = positive_rig.positive_options[-1]
    assert "supplied_evaluation" not in options and "supplied_stellar_sources" not in options
    session.abort()
