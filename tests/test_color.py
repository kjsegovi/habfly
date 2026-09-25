"""Color reference, honest abstention, learned interface and offline replay."""

import io

import pytest
import torch

from habfly.color_reference import (
    COLOR_CONTROL,
    ColorReferenceError,
    color_features,
    load_color_reference,
    validate_color_reference,
)
from habfly.contracts import Action
from habfly.data import make_demo_graph
from habfly.environments.color import ColorEnv, color_cases
from habfly.model import ConnectomePolicy
from habfly.runtime import Runtime, read_trace
from habfly.training.checkpoints import load_checkpoint, save_checkpoint
from habfly.training.color import record, validate_splits
from habfly.training.train import supervised_loss


@pytest.fixture
def reference():
    return load_color_reference()


def act(env, key, value=None):
    obs = env.observe()
    control = next(c for c in obs.controls if c.id.endswith(":" + key))
    return env.step(
        Action(
            kind="CLICK" if key == "check" else "SELECT",
            target=control.id,
            value=value,
            observation_revision=obs.revision,
        )
    )


def test_color_reference_golden_and_hash_covers_provenance(reference):
    assert validate_color_reference(reference)["golden_cases"] == 13
    changed = reference.model_copy(deep=True)
    changed.sources[0]["evidence"] += " amended"
    assert changed.checksum != reference.checksum
    assert reference.private_label(379.999) == "UV"
    assert reference.private_label(744.001) == "IR"


@pytest.mark.parametrize("value", [450, 475, 570, 590, 620, 494.1, 494.5, 494.999])
def test_boundaries_are_not_silently_resolved(reference, value):
    with pytest.raises(ColorReferenceError, match="ambiguous_color_boundary|uncovered_color_gap"):
        reference.guard(value, "nm")


@pytest.mark.parametrize("value", [None, "0", 0, -1, float("inf"), float("nan"), True])
def test_invalid_values_are_not_zero_or_repaired(reference, value):
    with pytest.raises(ColorReferenceError):
        reference.private_label(value)


def test_units_and_schema(reference):
    with pytest.raises(ColorReferenceError, match="incompatible_wavelength_unit"):
        reference.private_label(500, "K")
    assert reference.guard(500, "nm") is None


def test_expert_completes_100_and_splits_are_deterministic_and_separate(reference):
    cases = {
        split: color_cases(split, 100, reference)
        for split in ("train", "calibration", "development", "test", "manual", "expert")
    }
    validate_splits(cases)
    assert cases["train"] == color_cases("train", 100, reference)
    for case in cases["expert"]:
        examples, result = record(reference, case)
        assert result["completed"] and result["steps"] == 3
        for e in examples:
            raw = e.observation.model_dump_json()
            assert "expected_color" not in raw and "expected_source" not in raw and "case_id" not in raw
            assert len(e.observation.calculation["reference_card"]["bands"]) == 9
    with pytest.raises(ValueError, match="overlap"):
        validate_splits({"a": cases["train"], "b": cases["train"]})


def test_wrong_source_and_color_are_not_fixed_and_reset_isolated(reference):
    case = color_cases("train", 1, reference)[0]
    env = ColorEnv(reference, [case])
    initial, _ = env.reset(seed=case["seed"])
    wrong = next(k for k, m in case["measurements"].items() if m["source"] == "reference star")
    act(env, "source", wrong)
    wrong_color = reference.private_label(case["measurements"][wrong]["value"])
    act(env, "color", wrong_color)
    assert env.source == wrong and env.answer == wrong_color
    act(env, "check")
    assert not env.completed
    reset, _ = env.reset(seed=case["seed"])
    assert initial == reset
    act(env, "source", case["expected_source"])
    act(env, "color", "IR")
    act(env, "source", wrong)
    assert not env.answer


def test_ambiguity_stops_without_answer(reference):
    case = color_cases("train", 1, reference)[0]
    case["measurements"][case["expected_source"]]["value"] = 494.5
    env = ColorEnv(reference, [case])
    env.reset(seed=case["seed"])
    act(env, "source", case["expected_source"])
    act(env, "color", "Green")
    assert env.error == "uncovered_color_gap" and env.terminated and not env.answer


def test_stale_targets_do_not_write(reference):
    case = color_cases("train", 1, reference)[0]
    env = ColorEnv(reference, [case])
    env.reset(seed=case["seed"])
    stale = env.expert_action(env.observe())
    env.step(stale)
    env.step(stale)
    assert env.error == "invalid_action" and env.metrics["invalid_actions"] == 1


@pytest.mark.parametrize("readout", ["linear_v1", "ordinal_v2"])
def test_policy_receives_raw_numeric_payload_not_band_oracle(reference, monkeypatch, tmp_path, readout):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    graph = make_demo_graph()
    policy = ConnectomePolicy(
        graph,
        hidden_size=16,
        observation_encoding="structured_color_v1",
        selection_mode="measurement_result_v3",
        control_encoding="semantic_color_v1",
        color_readout=readout,
    )
    examples, _ = record(reference, color_cases("train", 1, reference)[0])

    def forbidden(*args, **kwargs):
        raise AssertionError("Oracle accessed by learned policy")

    monkeypatch.setattr(type(reference), "private_label", forbidden)
    monkeypatch.setattr(type(reference), "guard", forbidden)
    state, parts = None, []
    for example in examples:
        loss, out = supervised_loss(policy, [example], state)
        assert out.state.shape == (1, len(graph.body_ids), 16)
        state = out.state
        parts.append(loss)
    loss = torch.stack(parts).mean()
    loss.backward()
    assert torch.isfinite(loss) and policy.color_projection.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    obs = examples[1].observation
    changed = obs.model_copy(deep=True)
    changed.values["measurements"][changed.calculation["source"]]["value"] *= 2
    assert color_features(obs.model_dump()) != color_features(changed.model_dump())
    policy.eval()
    with torch.no_grad():
        out = policy([obs])
        other = policy([changed])
        assert not torch.allclose(out.pooled, other.pooled)
        control = next(c for c in obs.controls if c.label == COLOR_CONTROL)
        scores = policy.option_scores(obs, control, out.pooled)
        reversed_control = control.model_copy(update={"options": list(reversed(control.options))})
        assert torch.equal(scores.flip(0), policy.option_scores(obs, reversed_control, out.pooled))
    ckpt = tmp_path / "color.pt"
    save_checkpoint(ckpt, policy, stage="color", seed=0, content_pack={"reference": reference.checksum})
    loaded, _ = load_checkpoint(ckpt, graph, content_pack={"reference": reference.checksum})
    assert loaded.color_readout == readout
    with torch.no_grad():
        assert torch.equal(policy([obs]).pooled, loaded([obs]).pooled)
        assert torch.equal(policy.color_head(out.pooled), loaded.color_head(out.pooled))
    with pytest.raises(ValueError, match="content-pack mismatch"):
        load_checkpoint(ckpt, graph, content_pack={"reference": "wrong"})


def test_color_replay_offline_requires_no_model_reference_or_credentials(reference, tmp_path, monkeypatch):
    trace = tmp_path / "color.jsonl"
    record(reference, color_cases("train", 1, reference)[0], trace=trace)
    events = read_trace(trace)
    assert sum(e.event == "action_proposed" for e in events) == 3
    from habfly.training import color

    monkeypatch.setattr(color, "load_color_experiment", lambda *a, **k: pytest.fail("Replay loaded policy"))
    runtime = Runtime(io.StringIO())
    runtime.command({"command": "replay", "payload": {"path": str(trace)}})
    for _ in range(len(events) + 1):
        runtime.tick()
    assert runtime.status == "completed"
    runtime.close()
