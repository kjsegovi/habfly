"""Read-only recurrent decision checks on already-opened demonstration splits.

Counterfactuals test selection invariance, not physically valid calculated answers.
Every branch receives the same recorded observations, never repaired policy actions.
"""

from copy import deepcopy

import torch

from habfly.contracts import Observation


def counterfactual(observation, mode):
    changed = deepcopy(observation)
    if mode == "numeric_payload":
        # Numeric copying is the environment's exact transfer, not a policy
        # choice. Preserve all semantic identity, validity, units and bindings.
        for item in changed.get("values", {}).get("measurements", {}).values():
            item["value"] = 12345.6789
        for item in (changed.get("calculation") or {}).get("results", {}).values():
            item["value"] = 98765.4321
        for key in changed.get("values", {}).get("answers", {}):
            changed["values"]["answers"][key] = 98765.4321
        last = (changed.get("calculation") or {}).get("last_operation") or {}
        if "value" in last:
            last["value"] = 98765.4321
    elif mode == "control_order":
        changed["controls"].reverse()
        for control in changed["controls"]:
            control["options"] = list(reversed(control.get("options", [])))
    else:
        raise ValueError("Unknown planet diagnostic counterfactual")
    return changed


def decision(action, observation):
    """Keep option identity, replacing only observation-local control IDs."""
    if hasattr(action, "model_dump"):
        action = action.model_dump(mode="json")
    controls = {c["id"]: c["label"] for c in observation["controls"]}
    return {
        "kind": action["kind"],
        "target_label": controls.get(action.get("target"), action.get("target")),
        "value": action.get("value"),
    }


@torch.no_grad()
def diagnose_trajectories(policy, episodes):
    """No optimizer, expert queries, grading access, final cases or file writes."""
    policy.eval()
    modes = ("numeric_payload", "control_order")
    totals = {mode: {"decisions": 0, "changed_decisions": 0, "first_change": None} for mode in modes}
    first_mistakes, correct, count = [], 0, 0
    parameters = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    for episode_index, episode in enumerate(episodes):
        states = {mode: None for mode in ("original", *modes)}
        first = None
        for step_index, step in enumerate(episode):
            original = step["observation"]
            action, states["original"], _ = policy.act(
                Observation.model_validate(original), states["original"]
            )
            actual, expected = decision(action, original), decision(step["action"], original)
            correct += actual == expected
            count += 1
            if actual != expected and first is None:
                first = {"episode": episode_index, "step": step_index, "actual": actual, "expected": expected}
            for mode in modes:
                observation = counterfactual(original, mode)
                variant, states[mode], _ = policy.act(Observation.model_validate(observation), states[mode])
                different = decision(variant, observation) != actual
                totals[mode]["decisions"] += 1
                totals[mode]["changed_decisions"] += different
                if different and totals[mode]["first_change"] is None:
                    totals[mode]["first_change"] = {
                        "episode": episode_index,
                        "step": step_index,
                        "original": actual,
                        "counterfactual": decision(variant, observation),
                    }
        first_mistakes.append(first)
    if any(not torch.equal(value, policy.state_dict()[key]) for key, value in parameters.items()):
        raise RuntimeError("Frozen diagnostic modified model parameters")
    return {
        "scope": "teacher_forced_recurrent_selection_diagnostic_not_closed_loop",
        "numeric_counterfactual_scope": "payload_invariance_not_physical_answers",
        "optimizer_updates": 0,
        "parameters_unchanged": True,
        "decisions": count,
        "exact_action_accuracy": correct / count if count else None,
        "first_mistake_per_episode": first_mistakes,
        "counterfactuals": totals,
    }
