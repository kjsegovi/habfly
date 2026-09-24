"""Read-only weight inspection; writes a NEW audit JSON, never trains or reads test cases."""

import argparse
import hashlib
import json
import socket
from pathlib import Path

import torch
from torch.nn import functional as F

from habfly.contracts import Observation
from habfly.data import load_graph
from habfly.model import ACTION_KINDS
from habfly.model.policy import legal_action_mask, legal_target_mask
from habfly.spreadsheet import SpreadsheetAdapter
from habfly.training.checkpoints import load_checkpoint, source_hash
from habfly.training.train import episodes_to_examples, value_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh audit output")

    def deny(*args, **kwargs):
        raise RuntimeError("Audit attempted network or spreadsheet access")

    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    SpreadsheetAdapter.__init__ = deny
    torch.set_num_threads(1)
    manifest = json.loads((args.experiment / "manifest.json").read_text())
    dataset = Path(manifest.get("source", args.experiment))
    data = json.loads((dataset / "train.json").read_text())
    if source_hash(data) != manifest["content"]["dataset_splits"]["train"]["sha256"]:
        raise ValueError("Training record hash mismatch")
    checkpoint = args.experiment / "training/checkpoint.pt"
    model, saved = load_checkpoint(
        checkpoint, load_graph("data/processed/graphs-v2/graph-2000"), content_pack=manifest["content"]
    )
    losses, gradients = {}, {}
    count = 0
    for episode in data["episodes"]:
        state = None
        for e in episodes_to_examples([episode]):
            v, a = value_tokens(model, [e.action.value or ""]), value_tokens(model, [e.answer])
            output = model([e.observation], state, teacher_values=v, teacher_answers=a)
            state = output.state.detach()
            ai = ACTION_KINDS.index(str(e.action.kind))
            ti = next(i for i, c in enumerate(e.observation.controls) if c.id == e.action.target)
            parts = {
                "action": F.cross_entropy(
                    output.action_logits.masked_fill(~legal_action_mask(e.observation), -1e9),
                    torch.tensor([ai]),
                ),
                "target": F.cross_entropy(
                    output.target_logits.masked_fill(
                        ~legal_target_mask(e.observation, str(e.action.kind)), -1e9
                    ),
                    torch.tensor([ti]),
                ),
            }
            if model.selection_mode == "characters":
                parts["typed_string"] = F.cross_entropy(
                    output.typed_value_logits.flatten(0, 1), v.flatten(), ignore_index=0
                )
                parts["answer_string"] = F.cross_entropy(
                    output.answer_logits.flatten(0, 1), a.flatten(), ignore_index=0
                )
            elif e.action.kind == "SELECT":
                c = e.observation.controls[ti]
                parts["option"] = F.cross_entropy(
                    model.option_scores(e.observation, c, output.pooled)[None],
                    torch.tensor([c.options.index(e.action.value)]),
                )
            for name, part in parts.items():
                losses[name] = losses.get(name, 0.0) + float(part.detach())
                gradient = torch.autograd.grad(part, output.pooled, retain_graph=True)[0]
                gradients[name] = gradients.get(name, 0.0) + float(gradient.norm())
            count += 1
    obs = [Observation.model_validate(s["observation"]) for s in data["episodes"][0]]
    with torch.no_grad():
        state = None
        for o in obs[:3]:
            state = model([o], state).state
        scores = torch.stack([model([o], state).action_logits[0] for o in obs])
        current_range = float((scores.amax(0) - scores.amin(0)).max())
        repeated, state = [], None
        for _ in range(10):
            output = model([obs[0]], state)
            state = output.state
            repeated.append(float(output.action_logits[0, [0, 2]].softmax(0)[0]))
        invariant = total = 0
        for episode in data["episodes"]:
            state = None
            for step in episode:
                original = Observation.model_validate(step["observation"])
                action, next_state, _ = model.act(original, state)
                changed = original.model_copy(deep=True)
                changed.controls.reverse()
                identities = {}
                for i, control in enumerate(changed.controls):
                    replacement = f"opaque-control-{i}"
                    identities[replacement] = control.id
                    control.id = replacement
                alternative, _, _ = model.act(changed, state)
                invariant += int(
                    (action.kind, action.target, action.value)
                    == (alternative.kind, identities.get(alternative.target), alternative.value)
                )
                total += 1
                state = next_state
    source_changes = [
        p
        for p, h in saved["provenance"]["code_hashes"].items()
        if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
    ]
    audit = {
        "experiment": str(args.experiment),
        "model": model.configuration(),
        "training_decisions": count,
        "mean_loss_per_decision": {k: v / count for k, v in losses.items()},
        "mean_pooled_gradient_norm_per_decision": {k: v / count for k, v in gradients.items()},
        "fixed_history_observation_swap_max_action_logit_range": current_range,
        "fixed_history_observation_swap_kinds": [ACTION_KINDS[int(s.argmax())] for s in scores],
        "repeated_initial_observation_click_probability_among_click_select": repeated,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "changed_source_files_since_checkpoint": source_changes,
        "optimizer_updates": 0,
        "control_reorder_and_id_rename": {"consistent_decisions": invariant, "decisions": total},
        "note": "One seed. Sensitivity probes are evidence of responsiveness, not causal attribution or task completion. Option losses averaged over all decisions; non-SELECT contributes zero. Critic excluded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
