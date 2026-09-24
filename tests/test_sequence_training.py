"""Regression checks for the distance diagnostic's representation/credit assignment."""

import json

import pytest
import torch

from habfly.contracts import Action, Control, Observation
from habfly.data import make_demo_graph
from habfly.model import ACTION_KINDS, CharacterTokenizer, ConnectomePolicy
from habfly.model.policy import legal_action_mask, legal_target_mask
from habfly.training.curriculum import Example
from habfly.training.train import supervised_loss, train_behavioral_cloning


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    yield
    torch.set_num_threads(previous)


def policy():
    return ConnectomePolicy(make_demo_graph(8), hidden_size=8, max_answer_length=2)


def controls():
    return [
        Control(id="click", label="Execute"),
        Control(id="select", label="Operation", actions=["SELECT"], options=["distance"]),
        Control(id="disabled", label="Unavailable", enabled=False),
    ]


def test_all_controls_reach_core_and_targets_without_order_or_id_leakage(monkeypatch):
    model = policy()
    items = [Control(id=str(i), label=f"Visible control {i}") for i in range(20)]
    obs = Observation(controls=items)
    assert len(json.dumps([c.model_dump(mode="json") for c in items])) > model.tokenizer.max_length
    texts = []
    original = model.encode_text

    def capture(values):
        texts.extend(values)
        return original(values)

    monkeypatch.setattr(model, "encode_text", capture)
    with torch.no_grad():
        baseline = model([obs])
        assert all(any(c.label in text for text in texts) for c in items)
        assert all(len(text) + 2 <= model.tokenizer.max_length for text in texts)
        altered = obs.model_copy(deep=True)
        altered.controls[-1].label = "Last control changed"
        changed = model([altered])
        assert not torch.equal(baseline.state, changed.state)
        reversed_output = model([Observation(controls=list(reversed(items)))])
        torch.testing.assert_close(baseline.state, reversed_output.state)
        torch.testing.assert_close(baseline.target_logits, reversed_output.target_logits.flip(1))
        renamed = obs.model_copy(deep=True)
        for c in renamed.controls:
            c.id = "revision-" + c.id
        torch.testing.assert_close(baseline.state, model([renamed]).state, rtol=0, atol=0)


def test_long_control_is_split_and_oversized_leaf_is_rejected(monkeypatch):
    model = policy()
    model.tokenizer = CharacterTokenizer(max_length=128)
    item = Control(id="long", label="Choices", options=[f"option-{i}" for i in range(50)])
    texts = []
    original = model.encode_text

    def capture(values):
        texts.extend(values)
        return original(values)

    monkeypatch.setattr(model, "encode_text", capture)
    for target in (False, True):
        texts.clear()
        encoded = model.encode_controls([item], target=target)
        assert encoded.shape == (1, 8)
        assert all(len(text) + 2 <= 128 for text in texts)
        assert all(any(option in text for text in texts) for option in item.options)
    with pytest.raises(ValueError, match="token budget"):
        model.encode_controls([Control(id="bad", label="x" * 129)])


def test_supervision_masks_match_inference_and_exclude_padding():
    model = policy()
    obs = Observation(controls=controls())
    click = Example(obs, Action(kind="CLICK", target="click"))
    select = Example(obs, Action(kind="SELECT", target="select", value="distance"))
    stop = Example(Observation(), Action(kind="STOP"))
    loss, output = supervised_loss(model, [click, select, stop])
    output.action_logits.retain_grad()
    output.target_logits.retain_grad()
    loss.backward()
    for row, example in enumerate((click, select, stop)):
        mask = legal_action_mask(example.observation)
        assert torch.equal(
            output.action_logits.grad[row, ~mask], torch.zeros_like(output.action_logits.grad[row, ~mask])
        )
        target_mask = legal_target_mask(example.observation, str(example.action.kind), width=3)
        assert torch.equal(
            output.target_logits.grad[row, ~target_mask],
            torch.zeros_like(output.target_logits.grad[row, ~target_mask]),
        )
    # Only one legal target per expert action, so target CE is exactly zero,
    # even when the model predicts the other kind. WAIT/STOP have no target CE.
    assert output.target_logits.grad.count_nonzero() == 0
    assert torch.isfinite(loss) and torch.isfinite(model.embedding.weight.grad).all()
    with torch.no_grad():
        model.action_head.weight.zero_()
        model.action_head.bias.fill_(-100)
        model.action_head.bias[ACTION_KINDS.index("CLICK")] = 100
    action, _, _ = model.act(obs)
    assert action.kind == "CLICK" and action.target == "click"


@pytest.mark.parametrize(
    "action",
    [
        Action(kind="CLICK", target="disabled"),
        Action(kind="CLICK", target="select"),
        Action(kind="CLICK", target="missing"),
        Action(kind="TYPE", target="click", value="1"),
    ],
)
def test_invalid_demonstrations_fail_closed(action):
    with pytest.raises(ValueError, match="Expert"):
        supervised_loss(policy(), [Example(Observation(controls=controls()), action)])


def test_later_loss_reaches_earlier_observation_encoder(monkeypatch):
    model = policy()
    encoded = []
    original = model.propagate

    def capture(value, state):
        value.retain_grad()
        encoded.append(value)
        return original(value, state)

    monkeypatch.setattr(model, "propagate", capture)
    example = Example(Observation(controls=controls()), Action(kind="CLICK", target="click"))
    _, first = supervised_loss(model, [example])
    loss, _ = supervised_loss(model, [example], first.state)
    loss.backward()  # Deliberately do NOT backpropagate the first step's loss.
    assert encoded[0].grad is not None and encoded[0].grad.abs().sum() > 0
    assert torch.isfinite(encoded[0].grad).all()


def test_bounded_sequences_update_only_at_boundaries_and_reset_episodes(tmp_path, monkeypatch):
    import habfly.training.train as module

    model = policy()
    events = []
    progress = []
    original_loss = module.supervised_loss
    original_step = torch.optim.AdamW.step

    def observed_loss(policy, examples, state=None):
        events.append(
            ("forward", None if state is None else state.grad_fn is not None, policy.cell.weight_hh._version)
        )
        return original_loss(policy, examples, state)

    def observed_step(optimizer, *args, **kwargs):
        events.append(("update",))
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(module, "supervised_loss", observed_loss)
    monkeypatch.setattr(torch.optim.AdamW, "step", observed_step)
    monkeypatch.setattr(module, "evaluate_examples", lambda *args, **kwargs: {})

    def episode(instruction):
        return [
            {
                "observation": Observation(instruction=instruction, controls=controls()).model_dump(
                    mode="json"
                ),
                "action": Action(kind="CLICK", target="click").model_dump(mode="json"),
            }
            for _ in range(3)
        ]

    report = train_behavioral_cloning(
        [episode("train-a"), episode("train-b")],
        model.graph,
        tmp_path / "sequence",
        validation_episodes=[episode("validation")],
        policy=model,
        epochs=2,
        sequence_length=2,
        progress_callback=progress.append,
    )
    assert report["sequence_length"] == 2
    assert report["optimizer_steps"] == 8 and report["supervised_decisions"] == 12
    assert [row["epoch"] for row in progress] == [1, 2]
    assert [row["optimizer_steps"] for row in progress] == [4, 8]
    assert [row["supervised_decisions"] for row in progress] == [6, 12]
    assert [row["loss"] for row in progress] == report["losses"]
    for offset in range(0, len(events), 5):
        first, second, update, third, final_update = events[offset : offset + 5]
        assert first[:2] == ("forward", None)  # Fresh state every episode.
        assert second[:2] == ("forward", True)  # History connected within chunk.
        assert first[2] == second[2]  # Stable weights until the full chunk ends.
        assert update == final_update == ("update",)
        assert third[:2] == ("forward", False)  # Detached only at chunk boundary.
        assert third[2] > first[2]


@pytest.mark.parametrize("length", [0, -1, 1.5, True])
def test_invalid_sequence_budget_rejected_before_writes(tmp_path, length):
    with pytest.raises(ValueError, match="sequence length"):
        train_behavioral_cloning(
            [], None, tmp_path / "unused", validation_episodes=[], sequence_length=length
        )
    assert not (tmp_path / "unused").exists()
