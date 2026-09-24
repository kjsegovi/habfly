"""Small local training jobs and honest held-out reports.

The defaults are development experiments, not claims of solved arithmetic or
HabWorlds competence. Checkpoints contain measured scores and gates explicitly.
"""

from __future__ import annotations

import json
import random
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from habfly.contracts import Action, Observation
from habfly.model import ACTION_KINDS, ConnectomePolicy
from habfly.model.policy import legal_action_mask, legal_target_mask

from .calibration import fit_temperature
from .checkpoints import save_checkpoint, source_hash
from .curriculum import Example, arithmetic_ood_examples, curriculum_examples


def value_tokens(policy: ConnectomePolicy, values: list[str]) -> torch.Tensor:
    result = torch.zeros((len(values), policy.max_answer_length), dtype=torch.long, device=policy.device)
    for row, value in enumerate(values):
        ids = policy.tokenizer.encode(value, policy.max_answer_length + 1)[1:]
        result[row, : len(ids)] = torch.tensor(ids, device=policy.device)
    return result


def prepare_output_directory(path):
    directory = Path(path)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Experiment directory is not empty: {directory}; choose a fresh directory")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def peak_process_rss_bytes():
    """Process high-water mark, not a misleading per-experiment allocation."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _labels(examples: list[Example], device):
    actions = torch.tensor([ACTION_KINDS.index(str(e.action.kind)) for e in examples], device=device)
    targets = torch.tensor(
        [
            next((i for i, c in enumerate(e.observation.controls) if c.id == e.action.target), -100)
            for e in examples
        ],
        device=device,
    )
    return actions, targets


def supervised_loss(policy: ConnectomePolicy, examples: list[Example], state=None):
    action_labels, target_labels = _labels(examples, policy.device)
    values = value_tokens(policy, [e.action.value or "" for e in examples])
    answers = value_tokens(policy, [e.answer for e in examples])
    output = policy([e.observation for e in examples], state, teacher_values=values, teacher_answers=answers)
    action_masks = torch.stack([legal_action_mask(e.observation, policy.device) for e in examples])
    target_masks = torch.stack(
        [
            legal_target_mask(
                e.observation, str(e.action.kind), policy.device, width=output.target_logits.shape[1]
            )
            for e in examples
        ]
    )
    for row, example in enumerate(examples):
        if not action_masks[row, action_labels[row]]:
            raise ValueError("Expert action is not available in the observation")
        if example.action.kind in ("WAIT", "STOP"):
            target_labels[row] = -100
        elif target_labels[row] < 0 or not target_masks[row, target_labels[row]]:
            raise ValueError("Expert target is not available for its action kind")
    loss = F.cross_entropy(output.action_logits.masked_fill(~action_masks, -1e9), action_labels)
    if (target_labels >= 0).any():
        loss = loss + F.cross_entropy(
            output.target_logits.masked_fill(~target_masks, -1e9), target_labels, ignore_index=-100
        )
    if policy.selection_mode != "characters":
        option_losses = []
        for row, example in enumerate(examples):
            if example.action.kind == "SELECT":
                control = example.observation.controls[int(target_labels[row])]
                if example.action.value not in control.options:
                    raise ValueError("Expert value is not a visible option")
                scores = policy.option_scores(example.observation, control, output.pooled[row : row + 1])
                option_losses.append(
                    F.cross_entropy(
                        scores[None],
                        scores.new_tensor([control.options.index(example.action.value)], dtype=torch.long),
                    )
                )
        if option_losses:
            loss = loss + torch.stack(option_losses).mean()
        text_rows = [i for i, e in enumerate(examples) if e.action.kind in ("TYPE", "KEYPRESS")]
        # SELECT is an option decision, not two copies of a string-generation task.
        # Untyped CLICK actions do not need an empty-string prediction either.
        if text_rows:
            loss = loss + F.cross_entropy(
                output.typed_value_logits[text_rows].flatten(0, 1),
                values[text_rows].flatten(),
                ignore_index=0,
            )
            loss = loss + F.cross_entropy(
                output.answer_logits[text_rows].flatten(0, 1), answers[text_rows].flatten(), ignore_index=0
            )
    else:
        loss = loss + F.cross_entropy(
            output.typed_value_logits.flatten(0, 1), values.flatten(), ignore_index=0
        )
        loss = loss + F.cross_entropy(output.answer_logits.flatten(0, 1), answers.flatten(), ignore_index=0)
    pointer_target, pointer_prediction = [], []
    for row, example in enumerate(examples):
        for col, key in enumerate(("x", "y", "dx", "dy")):
            value = getattr(example.action, key)
            if value is not None:
                pointer_target.append(value if col < 2 else (value + 1) / 2)
                pointer_prediction.append(output.pointer[row, col])
    if pointer_target:
        loss = loss + F.mse_loss(torch.stack(pointer_prediction), output.pointer.new_tensor(pointer_target))
    critic_rows = [row for row, example in enumerate(examples) if example.value_target is not None]
    if critic_rows:
        critic_targets = output.value.new_tensor([examples[row].value_target for row in critic_rows])
        loss = loss + 0.1 * F.mse_loss(output.value[critic_rows], critic_targets)
    return loss, output


@torch.no_grad()
def evaluate_examples(policy: ConnectomePolicy, examples: list[Example], *, episode_lengths=None) -> dict:
    policy.eval()
    correct_action = correct_target = correct_value = correct_answer = correct_exact = 0
    boundaries = set(np.cumsum([0] + episode_lengths).tolist()) if episode_lengths else set()
    state = None
    for index, example in enumerate(examples):
        if not episode_lengths or index in boundaries:
            state = None
        action, state, diagnostics = policy.act(example.observation, state)
        value = action.value or ""
        answer = diagnostics["answer"]
        a = action.kind == example.action.kind
        t = action.target == example.action.target or example.action.target is None
        v = value == (example.action.value or "")
        correct_action += a
        correct_target += t
        correct_value += v
        correct_answer += answer == example.answer
        correct_exact += a and t and (v or example.action.value is None)
    total = max(len(examples), 1)
    return {
        "examples": len(examples),
        "action_accuracy": correct_action / total,
        "target_accuracy": correct_target / total,
        "value_exact_accuracy": correct_value / total,
        "answer_exact_accuracy": correct_answer / total,
        "exact_action_accuracy": correct_exact / total,
    }


@torch.no_grad()
def calibrate_policy(policy: ConnectomePolicy, examples: list[Example], *, episode_lengths=None) -> dict:
    if not examples:
        return {"status": "uncalibrated", "reason": "empty_calibration_split"}
    action_logits, target_logits, action_labels, target_labels = [], [], [], []
    max_targets = max(1, max(len(e.observation.controls) for e in examples))
    boundaries = set(np.cumsum([0] + episode_lengths).tolist()) if episode_lengths else set()
    state = None
    for index, example in enumerate(examples):
        if not episode_lengths or index in boundaries:
            state = None
        output = policy([example.observation], state)
        state = output.state.detach()
        labels, targets = _labels([example], policy.device)
        mask = legal_action_mask(example.observation, policy.device)
        action_logits.append(output.action_logits[0].masked_fill(~mask, -1e9).cpu())
        action_labels.append(int(labels[0]))
        if int(targets[0]) >= 0 and int(action_logits[-1].argmax()) == int(labels[0]):
            padded = torch.full((max_targets,), -1e9)
            mask = legal_target_mask(example.observation, str(example.action.kind), policy.device)
            padded[: output.target_logits.shape[1]] = output.target_logits[0].masked_fill(~mask, -1e9).cpu()
            target_logits.append(padded)
            target_labels.append(int(targets[0]))
    action = fit_temperature(torch.stack(action_logits), torch.tensor(action_labels))
    target = (
        fit_temperature(torch.stack(target_logits), torch.tensor(target_labels)) if target_logits else None
    )
    policy.action_temperature = action["temperature"]
    policy.target_temperature = target["temperature"] if target else 1.0
    policy.calibration = {
        "status": "calibrated" if target else "action_only",
        "action": action,
        "target": target,
        "scope": "heldout_recurrent_expert_trajectories"
        if episode_lengths
        else "heldout_independent_examples",
        "data_hash": source_hash([e.observation.model_dump() for e in examples]),
    }
    return policy.calibration


def _fit(policy, examples, *, epochs, seed, learning_rate, batch_size=8):
    if epochs < 1 or not examples:
        raise ValueError("Training needs positive epochs and nonempty examples")
    policy.calibration = {"status": "uncalibrated", "reason": "weights_updated"}
    policy.action_temperature = policy.target_temperature = 1.0
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    losses = []
    for _ in range(epochs):
        policy.train()
        order = list(examples)
        rng.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = supervised_loss(policy, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
        losses.append(epoch_loss / len(order))
    policy.eval()
    policy.vision_trained = policy.vision_trained or any(e.observation.chart_crop for e in examples)
    return optimizer, losses


def run_curriculum(
    stage: str,
    output_dir: str | Path,
    graph: Any,
    *,
    seed: int = 0,
    epochs: int = 3,
    count: int = 96,
    hidden_size: int = 32,
    learning_rate: float = 0.003,
    policy: ConnectomePolicy | None = None,
) -> dict:
    directory = prepare_output_directory(output_dir)
    torch.manual_seed(seed)
    train, heldout = curriculum_examples(stage, seed=seed, count=count)
    # Calibration and reported evaluation are distinct slices of held-out data.
    calibration, test = heldout[::2], heldout[1::2]
    for split, examples in (("train", train), ("calibration", calibration), ("test", test)):
        records = [
            {
                "observation": e.observation.model_dump(mode="json"),
                "action": e.action.model_dump(mode="json"),
                "answer": e.answer,
                "group": e.group,
            }
            for e in examples
        ]
        (directory / f"{split}-examples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    policy = policy or ConnectomePolicy(graph, hidden_size=hidden_size)
    started = time.perf_counter()
    optimizer, losses = _fit(policy, train, epochs=epochs, seed=seed, learning_rate=learning_rate)
    calibrate_policy(policy, calibration)
    report = {
        "stage": stage,
        "seed": seed,
        "epochs": epochs,
        "losses": losses,
        "train": evaluate_examples(policy, train),
        "heldout": evaluate_examples(policy, test),
        "calibration": policy.calibration,
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": peak_process_rss_bytes(),
        "parameter_count": sum(p.numel() for p in policy.parameters()),
        "graph_hash": policy.graph_hash,
        "graph_nodes": len(graph.body_ids),
        "graph_source_kind": graph.manifest.get("source_kind", "unspecified"),
        "split": "disjoint templates and/or semantic groups; calibration separate from test",
        "gate_passed": False,
    }
    report["gate_passed"] = report["heldout"]["exact_action_accuracy"] >= 0.95
    if stage == "arithmetic":
        report["two_digit_ood_evaluation_only"] = evaluate_examples(
            policy, arithmetic_ood_examples(seed, min(count, 32))
        )
    save_checkpoint(
        directory / "checkpoint.pt", policy, stage=stage, seed=seed, optimizer=optimizer, evaluation=report
    )
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def episodes_to_examples(episodes: list[list[dict]]) -> list[Example]:
    examples = []
    for episode in episodes:
        returns = []
        discounted = 0.0
        for step in reversed(episode):
            discounted = float(step.get("result", {}).get("reward", 0.0)) + 0.99 * discounted
            returns.append(discounted)
        for step, value_target in zip(episode, reversed(returns)):
            action = Action.model_validate(step["action"])
            examples.append(
                Example(
                    Observation.model_validate(step["observation"]),
                    action,
                    action.value or "",
                    value_target=value_target,
                )
            )
    return examples


def train_behavioral_cloning(
    episodes: list[list[dict]],
    graph: Any,
    output_dir: str | Path,
    *,
    validation_episodes: list[list[dict]],
    epochs: int = 3,
    seed: int = 0,
    hidden_size: int = 32,
    learning_rate: float = 0.003,
    content_pack: dict | None = None,
    policy: ConnectomePolicy | None = None,
    sequence_length: int = 1,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    # Opt in explicitly for bounded experiments; long existing stellar/Mini
    # trajectories retain their previous memory budget unless configured here.
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("Behavioral cloning requires a positive integer sequence length")
    if not episodes or not validation_episodes:
        raise ValueError("Behavioral cloning needs separate training and validation episodes")
    train_hashes = {source_hash(episode) for episode in episodes}
    if train_hashes & {source_hash(episode) for episode in validation_episodes}:
        raise ValueError("Training and validation trajectories overlap")
    directory = prepare_output_directory(output_dir)
    for split, trajectories in (("train", episodes), ("validation", validation_episodes)):
        records = (
            {"episode": index, **step} for index, episode in enumerate(trajectories) for step in episode
        )
        (directory / f"{split}-trajectories.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )
    torch.manual_seed(seed)
    policy = policy or ConnectomePolicy(graph, hidden_size=hidden_size)
    policy.calibration = {"status": "uncalibrated", "reason": "weights_updated"}
    policy.action_temperature = policy.target_temperature = 1.0
    train = episodes_to_examples(episodes)
    valid = episodes_to_examples(validation_episodes)
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    losses = []
    optimizer_steps = supervised_decisions = 0
    rng = random.Random(seed)
    if epochs < 1:
        raise ValueError("Behavioral cloning requires positive epochs")
    for epoch in range(epochs):
        policy.train()
        ordered = list(episodes)
        rng.shuffle(ordered)
        total_loss, count = 0.0, 0
        for episode in ordered:
            state = None
            examples = episodes_to_examples([episode])
            for start in range(0, len(examples), sequence_length):
                optimizer.zero_grad(set_to_none=True)
                sequence_losses = []
                for example in examples[start : start + sequence_length]:
                    loss, output = supervised_loss(policy, [example], state)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Behavioral cloning produced nonfinite loss")
                    sequence_losses.append(loss)
                    state = output.state
                    total_loss += float(loss.detach())
                    count += 1
                # Weights stay fixed throughout the unroll. Every decision's
                # gradient can reach earlier observations in this sequence.
                torch.stack(sequence_losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                state = state.detach()
                optimizer_steps += 1
                supervised_decisions += len(sequence_losses)
        losses.append(total_loss / max(count, 1))
        if progress_callback:
            progress_callback(
                {
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "loss": losses[-1],
                    "optimizer_steps": optimizer_steps,
                    "supervised_decisions": supervised_decisions,
                }
            )
    policy.eval()
    policy.vision_trained = policy.vision_trained or any(e.observation.chart_crop for e in train)
    # Calibration uses whole alternate episodes, avoiding timestep-level leakage.
    calibration = episodes_to_examples(validation_episodes[::2])
    test = episodes_to_examples(validation_episodes[1::2])
    if test:
        calibrate_policy(policy, calibration, episode_lengths=[len(ep) for ep in validation_episodes[::2]])
        test_lengths = [len(ep) for ep in validation_episodes[1::2]]
    else:
        test = valid
        test_lengths = [len(ep) for ep in validation_episodes]
    report = {
        "stage": "behavioral_cloning",
        "seed": seed,
        "epochs": epochs,
        "sequence_length": sequence_length,
        "optimizer_steps": optimizer_steps,
        "supervised_decisions": supervised_decisions,
        "losses": losses,
        "training_episodes": len(episodes),
        "validation_episodes": len(validation_episodes),
        "heldout": evaluate_examples(policy, test, episode_lengths=test_lengths),
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": peak_process_rss_bytes(),
        "calibration": policy.calibration,
        "gate_passed": False,
        "training_data_hash": source_hash(episodes),
        "validation_data_hash": source_hash(validation_episodes),
        "critic_target": "discounted expert trajectory rewards, gamma=0.99",
        "gate_note": "Action imitation is measured here; closed-loop completion must be measured separately.",
    }
    if content_pack and content_pack.get("task") == "stellar":
        report.update(task="stellar", calculation_mode=content_pack["calculation_mode"], content=content_pack)
    save_checkpoint(
        directory / "checkpoint.pt",
        policy,
        stage="behavioral_cloning",
        seed=seed,
        optimizer=optimizer,
        content_pack=content_pack,
        evaluation=report,
    )
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def evaluate_policy(
    policy: ConnectomePolicy, env_factory, seeds, *, output_dir: str | Path | None = None
) -> dict:
    from habfly.contracts import task_completed, validate_action

    results = []
    directory = prepare_output_directory(output_dir) if output_dir else None
    started = time.perf_counter()
    for seed in seeds:
        env = env_factory()
        observation, _ = env.reset(seed=int(seed))
        state, total, invalid, steps = None, 0.0, 0, 0
        trajectory = []
        try:
            terminated = truncated = False
            while not (terminated or truncated):
                action, state, diagnostics = policy.act(Observation.model_validate(observation), state)
                try:
                    validate_action(Observation.model_validate(observation), action)
                except ValueError:
                    invalid += 1
                following, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                trajectory.append(
                    {
                        "observation": observation,
                        "action": action.model_dump(mode="json"),
                        "result": info.get("result", {}),
                        "diagnostics": diagnostics,
                    }
                )
                observation = following
            completed = task_completed(Observation.model_validate(observation))
            results.append(
                {
                    "seed": int(seed),
                    "completed": completed,
                    "steps": steps,
                    "reward": total,
                    "invalid_actions": invalid,
                    "truncated": truncated,
                }
            )
            if directory:
                (directory / f"trajectory-{seed}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in trajectory)
                )
        finally:
            env.close()
    return {
        "episodes": results,
        "completion_rate": sum(r["completed"] for r in results) / max(1, len(results)),
        "mean_reward": sum(r["reward"] for r in results) / max(1, len(results)),
        "invalid_actions": sum(r["invalid_actions"] for r in results),
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": peak_process_rss_bytes(),
    }
