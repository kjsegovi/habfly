"""Gated on-policy PPO for action/target selection with a fixed value decoder.

This first PPO phase improves interaction decisions. Typed strings and pointer
values remain deterministic outputs of the behavior-cloned heads. It is not a
general language PPO implementation.
"""

from __future__ import annotations

import json
from functools import wraps

import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from habfly.contracts import Observation
from habfly.model import ACTION_KINDS

from .checkpoints import save_checkpoint
from .train import evaluate_policy, prepare_output_directory


def _restore_gradient_flags(function):
    @wraps(function)
    def guarded(policy, *args, **kwargs):
        flags = {name: parameter.requires_grad for name, parameter in policy.named_parameters()}
        try:
            return function(policy, *args, **kwargs)
        finally:
            for name, parameter in policy.named_parameters():
                parameter.requires_grad_(flags[name])
    return guarded


def _log_probability(policy, observation, action, state=None):
    output = policy([observation], state)
    allowed = {"WAIT", "STOP"}
    for control in observation.controls:
        if control.enabled:
            allowed.update(str(kind) for kind in control.actions)
    mask = torch.tensor([kind in allowed for kind in ACTION_KINDS], device=policy.device)
    action_distribution = Categorical(logits=output.action_logits[0].masked_fill(~mask, -1e9) / policy.action_temperature)
    log_probability = action_distribution.log_prob(torch.tensor(ACTION_KINDS.index(action.kind), device=policy.device))
    entropy = action_distribution.entropy()
    if action.target is not None:
        target = next(i for i, control in enumerate(observation.controls) if control.id == action.target)
        mask = torch.tensor([c.enabled and action.kind in c.actions for c in observation.controls], device=policy.device)
        target_distribution = Categorical(logits=output.target_logits[0].masked_fill(~mask, -1e9) / policy.target_temperature)
        log_probability = log_probability + target_distribution.log_prob(torch.tensor(target, device=policy.device))
        entropy = entropy + target_distribution.entropy()
    return log_probability, entropy, output.value[0]


@_restore_gradient_flags
def train_ppo(policy, env_factory, output_dir, *, expert, seed=0, updates=3, episodes_per_update=2,
              gate_seeds=(10_001, 10_002, 10_003), minimum_completion=0.9, learning_rate=0.0003,
              gamma=0.99, clip=0.2, optimization_epochs=2, content_pack=None):
    """Check both expert and cloned policy in fresh episodes before any update."""
    if (not gate_seeds or not 0 < minimum_completion <= 1 or updates < 1
            or episodes_per_update < 1 or optimization_epochs < 1):
        raise ValueError("PPO requires gate seeds, a positive completion threshold, updates, episodes, and optimization epochs")
    if not 0 < gamma <= 1 or not 0 < clip < 1 or learning_rate <= 0:
        raise ValueError("PPO requires gamma in (0,1], clip in (0,1), and a positive learning rate")
    for gate_seed in gate_seeds:
        env = env_factory()
        observation, _ = env.reset(seed=gate_seed)
        try:
            terminated = truncated = False
            while not (terminated or truncated):
                action = expert(Observation.model_validate(observation))
                observation, _, terminated, truncated, _ = env.step(action)
            if not Observation.model_validate(observation).progress.get("submitted", False):
                raise ValueError("PPO expert gate failed; solve the deterministic expert first")
        finally:
            env.close()
    gate = evaluate_policy(policy, env_factory, gate_seeds)
    if gate["completion_rate"] < minimum_completion:
        raise ValueError(f"PPO behavioral-cloning gate failed: completion {gate['completion_rate']:.3f} < {minimum_completion:.3f}")
    directory = prepare_output_directory(output_dir)
    torch.manual_seed(seed)
    policy.calibration = {"status": "uncalibrated", "reason": "policy_updated_by_ppo"}
    policy.action_temperature = policy.target_temperature = 1.0
    # Freeze the recurrent representation and text/value/pointer decoders so
    # support of the deterministic action arguments stays fixed during PPO.
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith(("action_head.", "target_query.", "value_head.")))
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=learning_rate)
    losses, rewards = [], []
    for update in range(updates):
        samples = []
        for episode in range(episodes_per_update):
            env = env_factory()
            observation, _ = env.reset(seed=seed + update * episodes_per_update + episode)
            state, rollout, total, trajectory = None, [], 0.0, []
            try:
                terminated = truncated = False
                while not (terminated or truncated):
                    obs = Observation.model_validate(observation)
                    previous = state.detach() if state is not None else None
                    action, state, diagnostics = policy.act(obs, previous, sample=True)
                    with torch.no_grad():
                        old_logp, _, value = _log_probability(policy, obs, action, previous)
                    observation, reward, terminated, truncated, info = env.step(action)
                    trajectory.append({"observation": obs.model_dump(mode="json"),
                                       "action": action.model_dump(mode="json"), "diagnostics": diagnostics,
                                       "result": info.get("result", {"reward": reward, "terminated": terminated, "truncated": truncated})})
                    rollout.append((obs, action, previous, old_logp.detach(), float(value), reward))
                    total += reward
                discounted = 0.0
                if truncated and not terminated:
                    with torch.no_grad():
                        discounted = float(policy([Observation.model_validate(observation)], state).value[0])
                for obs, action, previous, old_logp, value, reward in reversed(rollout):
                    discounted = reward + gamma * discounted
                    samples.append((obs, action, previous, old_logp, discounted, discounted - value))
                rewards.append(total)
            finally:
                (directory / f"update-{update}-episode-{episode}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trajectory))
                env.close()
        advantages = torch.tensor([row[-1] for row in samples], device=policy.device)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        for _ in range(optimization_epochs):
            for index, (obs, action, previous, old_logp, discounted, _) in enumerate(samples):
                optimizer.zero_grad(set_to_none=True)
                logp, entropy, value = _log_probability(policy, obs, action, previous)
                ratio = (logp - old_logp).exp()
                advantage = advantages[index]
                actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
                loss = actor_loss + 0.5 * F.mse_loss(value, value.new_tensor(discounted)) - 0.01 * entropy
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite PPO loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
    test_seeds = tuple(seed + 100_000 + i for i in range(3))
    report = {"stage": "ppo", "seed": seed, "gate": gate, "updates": updates,
              "rewards": rewards, "mean_loss": sum(losses) / max(len(losses), 1),
              "heldout": evaluate_policy(policy, env_factory, test_seeds),
              "scope": "action and target PPO; cloned value/pointer decoders remain deterministic"}
    save_checkpoint(directory / "checkpoint.pt", policy, stage="ppo", seed=seed,
                    optimizer=optimizer, content_pack=content_pack, evaluation=report)
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
