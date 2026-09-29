"""Reference handoff transport and option gates, not native scientific acceptance."""

# ruff: noqa: F811
import pytest
from test_browser_project_runtime import ready
from test_browser_project_runtime import rig as bridge_rig  # noqa: F401
from test_runtime_browser_project import integrated, options, start  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.runtime import RunOptions, parse_run_options


@pytest.fixture
def rig(bridge_rig):
    return bridge_rig


def terrestrial_ready(rig):
    class Steps(rig.steps):
        def __init__(self, *args, **kwargs):
            self.decisions, self.advances = [], 0
            super().__init__(*args, **kwargs)

        def step(self):
            self.advances += 1
            if self.phase == "ready":
                self.phase = "terrestrial_initializing"
            elif self.phase == "terrestrial_initializing":
                self.phase = "terrestrial_active"
            elif not self.decisions:
                self.phase, self.status = "awaiting_gases", "paused"
            elif len(self.decisions) == 1:
                self.phase, self.status = "awaiting_habitability", "paused"
            else:
                self.finished, self.status, self.phase = True, "handoff", "fixture_proof_pending"

        def provide_gases(self, **payload):
            self.decisions.append(("gases", payload))
            self.phase = "terrestrial_active"

        def provide_habitability(self, **payload):
            self.decisions.append(("habitability", payload))
            self.phase = "terrestrial_active"

    bridge = ready(rig, steps=Steps, reference_planet_continuation=True, terrestrial_options={})
    return bridge


def test_terrestrial_options_default_off_and_handoffs_never_infer_choices(rig):
    bridge = ready(rig)
    assert bridge.terrestrial_options is None and not bridge.scope["terrestrial_continuation"]
    with pytest.raises(BrowserSafetyStop, match="terrestrial_handoff_not_ready"):
        bridge.provide_gases(gases=["CO2"], rationale="fixture", supplied_greenhouse_increment=100)
    bridge.close()


def test_gas_and_habitability_choices_are_offline_and_each_stage_is_scheduled(rig):
    bridge = terrestrial_ready(rig)
    for phase in ("terrestrial_initializing", "terrestrial_active", "awaiting_gases"):
        bridge.step()
        assert bridge.phase == phase and not bridge.state()["task_completed"]
    count = bridge.project.advances
    bridge.step()
    bridge.tick()
    assert bridge.project.advances == count
    with pytest.raises(BrowserSafetyStop, match="explicit_handoff_required"):
        bridge.resume()
    bridge.command(
        {
            "command": "step",
            "payload": {
                "gases": {
                    "gases": ["CO2"],
                    "rationale": "Explicit supplied reference",
                    "supplied_greenhouse_increment": 100,
                }
            },
        }
    )
    assert bridge.phase == "terrestrial_active" and bridge.project.advances == count
    bridge.step()
    assert bridge.phase == "awaiting_habitability"
    assert len(bridge.project.decisions) == 1
    bridge.step()
    assert len(bridge.project.decisions) == 1
    bridge.command(
        {
            "command": "step",
            "payload": {
                "habitability": {
                    "choice": "not_habitable",
                    "rationale": "Explicit supplied judgment, not phase alone",
                }
            },
        }
    )
    assert bridge.phase == "terrestrial_active" and len(bridge.project.decisions) == 2
    bridge.step()
    assert bridge.status == "handoff" and not bridge.report["task_completed"]
    bridge.close()


@pytest.mark.parametrize("phase", ["terrestrial_initializing", "terrestrial_active", "awaiting_gases"])
def test_abort_never_dispatches_later_terrestrial_stage(rig, phase):
    bridge = terrestrial_ready(rig)
    while bridge.phase != phase:
        bridge.step()
    count = bridge.project.advances
    bridge.abort()
    bridge.step()
    bridge.tick()
    assert bridge.project.advances == count and not bridge.project.decisions
    bridge.close()


def test_changed_initial_capture_prevents_gas_handoff(rig):
    bridge = terrestrial_ready(rig)
    for _ in range(3):
        bridge.step()
    (bridge.output / "initial-star/stellar/observation.json").write_text("{}")
    bridge.provide_gases(gases=["CO2"], rationale="Explicit reference", supplied_greenhouse_increment=100)
    assert bridge.status == "stopped" and not bridge.project.decisions
    bridge.close()


def configured(tmp_path):
    for name in ("hab-pilot", "hab-final"):
        (tmp_path / name).mkdir(exist_ok=True)
    return options(
        tmp_path,
        project_reference_planet_continuation=True,
        browser_planet_pilot=str(tmp_path / "planet"),
        browser_planet_final_evaluation=str(tmp_path / "planet-final"),
        browser_habitability_pilot=str(tmp_path / "hab-pilot"),
        browser_habitability_final_evaluation=str(tmp_path / "hab-final"),
        browser_habitability_gas_candidates=["CO2", "H2O"],
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("browser_habitability_pilot", None),
        ("browser_habitability_final_evaluation", None),
        ("browser_habitability_gas_candidates", None),
        ("browser_habitability_gas_candidates", []),
        ("browser_habitability_gas_candidates", ["CO2", "CO2"]),
        ("browser_habitability_gas_candidates", ["fictional"]),
        ("project_reference_planet_continuation", False),
        ("browser_planet_pilot", None),
    ],
)
def test_partial_or_unsupported_terrestrial_configuration_is_rejected(tmp_path, key, value):
    payload = configured(tmp_path)
    payload[key] = value
    with pytest.raises(ValueError):
        parse_run_options(payload)


@pytest.mark.parametrize(
    "key,value",
    [
        ("browser_habitability_max_seconds", 0),
        ("browser_habitability_max_seconds", 1801),
        ("browser_habitability_max_seconds", float("inf")),
        ("browser_habitability_max_seconds", True),
        ("browser_habitability_save_settle_seconds", 31),
        ("browser_habitability_save_settle_seconds", "30"),
        ("browser_habitability_seed", True),
    ],
)
def test_strict_fixed_budgets(key, value):
    with pytest.raises(ValueError):
        RunOptions(**{key: value})


def test_valid_configuration_loads_graph_offline_and_forwards_explicit_options(integrated, monkeypatch):
    payload = configured(integrated.root)
    loads = []
    from habfly.data import graphs

    monkeypatch.setattr(graphs, "load_graph", lambda path: loads.append(path) or {"fixture_graph": True})
    bridge = start(integrated, **payload)
    assert len(loads) == 1 and not integrated.calls and bridge.status == "paused"
    assert bridge.terrestrial_options["graph"] == {"fixture_graph": True}
    assert bridge.terrestrial_options["candidates"] == ["CO2", "H2O"]
    assert bridge.scope["terrestrial_continuation"] is True
    assert bridge.scope["automatic_gas_identification"] is False
