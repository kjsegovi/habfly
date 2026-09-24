"""Bounded, selector-only recognition practice; no additional arithmetic episodes."""

from __future__ import annotations

import itertools
import random

import torch
from torch.nn import functional as F

from habfly.model.measurement_identity import MeasurementIdentityPointer

MAX_UPDATES = 200
BATCH_SIZE = 12
TEMPLATES = {
    "train": (
        "Find the {source}'s measurement using the local tool. Copy the result and select its unit.",
        "Use the {source}'s measurements. Ignore the {other}'s readings.",
        "For the {source}, bind the requested quantity and unit from the reference card.",
        "Analyse the {source} only; the {other} provides distractor measurements.",
    ),
    "development": (
        "Complete this input using the {source}'s measurement. Disregard the {other}'s values.",
        "Read the input requirements and choose a measurement for the {source}, not the {other}.",
    ),
}
QUANTITIES = ("parallax", "flux", "wavelength")
UNITS = ("arcsec", "W/m2", "nm")
SOURCES = ("current star", "reference star")


def recognition_records(split: str) -> list[dict]:
    """Factorial field-matching exercises with deliberately counterfactual units.

    These are not stellar cases: e.g. a parallax labelled nm is useful for testing
    unit discrimination, but is never submitted to the calculator as valid data.
    Correct IDs are supervised labels, not part of the policy's input fields.
    """
    if split not in TEMPLATES:
        raise ValueError("Only recognition training and development splits are defined")
    rng = random.Random(1100000 if split == "train" else 1200000)
    combinations = list(itertools.product(QUANTITIES, UNITS, SOURCES))
    records = []
    for template in TEMPLATES[split]:
        for _ in range(2):
            for quantity, unit, source in combinations:
                candidates = [
                    {"kind": kind, "unit": u, "source": s, "value": rng.uniform(0.01, 1e6)}
                    for kind, u, s in combinations
                ]
                rng.shuffle(candidates)
                for candidate in candidates:
                    candidate["id"] = f"{split}-{rng.getrandbits(96):024x}"
                other = next(s for s in SOURCES if s != source)
                records.append(
                    {
                        "instruction": template.format(source=source, other=other),
                        "requirement": {"quantity": quantity, "unit": unit},
                        "candidates": candidates,
                        "expected_id": next(
                            c["id"]
                            for c in candidates
                            if (c["kind"], c["unit"], c["source"]) == (quantity, unit, source)
                        ),
                    }
                )
    return records


def attach_identity_pointer(policy):
    """Explicitly migrate a v1 option checkpoint; leave every old tensor frozen."""
    if policy.selection_mode != "option_pointer_v1" or hasattr(policy, "measurement_identity"):
        raise ValueError("Measurement identity training requires an unmodified option_pointer_v1 parent")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.measurement_identity = MeasurementIdentityPointer(policy.tokenizer, policy.hidden_size).to(
        policy.device
    )
    policy.selection_mode = "measurement_identity_v1"
    policy.action_temperature = policy.target_temperature = 1.0
    policy.calibration = {"status": "uncalibrated", "reason": "new_measurement_identity_head"}


def labels_for(records, device):
    return torch.tensor(
        [next(i for i, c in enumerate(r["candidates"]) if c["id"] == r["expected_id"]) for r in records],
        dtype=torch.long,
        device=device,
    )


def train_identity(policy, records, contexts, *, updates=MAX_UPDATES, progress_callback=None):
    if not 1 <= updates <= MAX_UPDATES:
        raise ValueError("Measurement identity budget is capped at 200 updates")
    if (
        policy.selection_mode != "measurement_identity_v1"
        or not records
        or contexts.ndim != 2
        or contexts.shape[1] != policy.hidden_size
        or not len(contexts)
    ):
        raise ValueError("Invalid identity training inputs")
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad != name.startswith("measurement_identity."):
            raise ValueError("Only the new measurement identity parameters may be trainable")
    optimizer = torch.optim.AdamW(policy.measurement_identity.parameters(), lr=0.003)
    rng, losses, indices = random.Random(0), [], []
    policy.measurement_identity.train()
    # Cached readouts come exclusively from frozen parent training histories.
    contexts = contexts.detach()
    for update in range(updates):
        while len(indices) < BATCH_SIZE:
            cycle = list(range(len(records)))
            rng.shuffle(cycle)
            indices.extend(cycle)
        chosen, indices = indices[:BATCH_SIZE], indices[BATCH_SIZE:]
        batch = [records[i] for i in chosen]
        # Randomize context independently of semantic labels; no sequence-clock labels.
        pooled = contexts[[rng.randrange(len(contexts)) for _ in batch]]
        optimizer.zero_grad(set_to_none=True)
        logits = torch.stack(policy.measurement_identity(batch, pooled))
        loss = F.cross_entropy(logits, labels_for(batch, logits.device))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite identity loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.measurement_identity.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if progress_callback:
            progress_callback({"optimizer_steps": update + 1, "loss": losses[-1]})
    return optimizer, {
        "optimizer_steps": updates,
        "supervised_decisions": updates * BATCH_SIZE,
        "losses": losses,
    }


@torch.no_grad()
def evaluate_identity(policy, records, contexts):
    policy.eval()
    totals = {"exact": 0, "kind": 0, "unit": 0, "source": 0}
    predictions = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        pooled = contexts[[(start + i) % len(contexts) for i in range(len(batch))]]
        scores = policy.measurement_identity(batch, pooled)
        for record, logits in zip(batch, scores):
            selected = record["candidates"][int(logits.argmax())]
            expected = next(c for c in record["candidates"] if c["id"] == record["expected_id"])
            totals["exact"] += int(selected["id"] == expected["id"])
            for key in ("kind", "unit", "source"):
                totals[key] += int(selected[key] == expected[key])
            predictions.append({"selected_id": selected["id"], "expected_id": expected["id"]})
    return {
        "examples": len(records),
        "correct": totals,
        "accuracy": {key: count / max(1, len(records)) for key, count in totals.items()},
        "predictions": predictions,
    }
