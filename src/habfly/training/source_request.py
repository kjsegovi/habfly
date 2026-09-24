"""Paired source-request supervision with an immutable quantity/unit/workflow parent."""

from __future__ import annotations

import itertools
import random
from copy import deepcopy

import torch
from torch.nn import functional as F

from habfly.model.source_request import SourceRequestPointer

from .measurement_identity import QUANTITIES, SOURCES, UNITS, labels_for

MAX_UPDATES = 200
PAIRS_PER_BATCH = 6
TEMPLATES = {
    "train": (
        "Use the {source}'s {quantity}.",
        "Find the {quantity} for the {source}. Ignore readings from the {other}.",
        "Ignore the {other}. Use the {source}'s {quantity}.",
        "From the {source}, select the requested {quantity}; the {other} is a distractor.",
    ),
    "development": (
        "Select {quantity} belonging to the {source}, not the {other}.",
        "Disregard the {other}'s data; obtain {quantity} for the {source}.",
    ),
}


def paired_records(split):
    if split not in TEMPLATES:
        raise ValueError("Only paired training and development are supported")
    rng = random.Random(1300000 if split == "train" else 1400000)
    records = []
    for template in TEMPLATES[split]:
        for _ in range(2):
            for quantity, unit in itertools.product(QUANTITIES, UNITS):
                candidates = [
                    {
                        "kind": k,
                        "unit": u,
                        "source": s,
                        "value": rng.uniform(0.01, 1e6),
                        "id": f"{split}-{rng.getrandbits(96):024x}",
                    }
                    for k, u, s in itertools.product(QUANTITIES, UNITS, SOURCES)
                ]
                rng.shuffle(candidates)
                for source in SOURCES:
                    other = next(s for s in SOURCES if s != source)
                    records.append(
                        {
                            "instruction": template.format(source=source, other=other, quantity=quantity),
                            "requirement": {"quantity": quantity, "unit": unit},
                            "candidates": deepcopy(candidates),
                            "expected_id": next(
                                c["id"]
                                for c in candidates
                                if (c["kind"], c["unit"], c["source"]) == (quantity, unit, source)
                            ),
                        }
                    )
    validate_pairs(records)
    return records


def validate_pairs(records):
    if not records or len(records) % 2:
        raise ValueError("Source training requires complete contrastive pairs")
    for left, right in zip(records[::2], records[1::2]):
        if (
            left["candidates"] != right["candidates"]
            or left["requirement"] != right["requirement"]
            or left["instruction"] == right["instruction"]
        ):
            raise ValueError("Paired requests must differ only in instruction and supervised label")
        expected = [next(c for c in r["candidates"] if c["id"] == r["expected_id"]) for r in (left, right)]
        if (expected[0]["kind"], expected[0]["unit"]) != (expected[1]["kind"], expected[1]["unit"]):
            raise ValueError("A source pair must preserve the requested quantity and unit")
        if expected[0]["source"] == expected[1]["source"]:
            raise ValueError("A source pair must request different sources")


def attach_source_pointer(policy):
    if policy.selection_mode != "measurement_identity_v1" or hasattr(
        policy.measurement_identity, "source_request"
    ):
        raise ValueError("Source stage requires an unmodified measurement_identity_v1 parent")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.measurement_identity.source_request = SourceRequestPointer(
        policy.tokenizer, policy.hidden_size
    ).to(policy.device)
    policy.selection_mode = "measurement_source_v2"
    policy.action_temperature = policy.target_temperature = 1.0
    policy.calibration = {"status": "uncalibrated", "reason": "new_source_request_head"}


def source_loss(scores, records):
    """Supervise only the requested source, never repair or filter inference options."""
    losses = []
    for logits, record in zip(scores, records):
        representatives = {}
        for index, candidate in enumerate(record["candidates"]):
            representatives.setdefault(candidate["source"], index)
        expected = next(c["source"] for c in record["candidates"] if c["id"] == record["expected_id"])
        labels = list(representatives)
        losses.append(
            F.cross_entropy(
                logits[list(representatives.values())][None],
                torch.tensor([labels.index(expected)], device=logits.device),
            )
        )
    return torch.stack(losses).mean()


def source_probe(policy, pair, context):
    """Zero-update evidence: changing the instruction reaches scores and gradients."""
    contexts = context[None].expand(2, -1).detach()
    head = policy.measurement_identity
    source = getattr(head, "source_request", head)
    scores = source(pair, contexts)
    # For the original joint head use exact-candidate CE; the new head has only
    # source scores, so duplicated labels are grouped for a source-only loss.
    loss = (
        source_loss(scores, pair)
        if hasattr(head, "source_request")
        else F.cross_entropy(torch.stack(scores), labels_for(pair, context.device))
    )
    parameter = source.embedding.weight
    gradient = torch.autograd.grad(loss, parameter)[0]
    return {
        "optimizer_updates": 0,
        "instruction_changed": pair[0]["instruction"] != pair[1]["instruction"],
        "max_logit_change": float((scores[0] - scores[1]).abs().max().detach()),
        "embedding_gradient_norm": float(gradient.norm()),
        "finite_gradient": bool(torch.isfinite(gradient).all()),
        "loss": float(loss.detach()),
    }


def train_source(
    policy,
    records,
    contexts,
    *,
    updates=MAX_UPDATES,
    progress_callback=None,
    warmup_records=None,
    warmup_updates=0,
    rehearsal_pairs=0,
    budget_limit=MAX_UPDATES,
):
    if not 1 <= updates <= budget_limit <= 1000:
        raise ValueError("Source training is capped at its explicit budget (default 200, maximum 1000)")
    validate_pairs(records)
    if warmup_records is None:
        if warmup_updates or rehearsal_pairs:
            raise ValueError("Warm-up settings require vocabulary records")
    else:
        validate_pairs(warmup_records)
        if not 0 < warmup_updates < updates or not 0 <= rehearsal_pairs < PAIRS_PER_BATCH:
            raise ValueError("Invalid bounded warm-up schedule")
    if (
        policy.selection_mode != "measurement_source_v2"
        or contexts.ndim != 2
        or contexts.shape[1] != policy.hidden_size
        or not len(contexts)
        or not torch.isfinite(contexts).all()
    ):
        raise ValueError("Invalid source training inputs")
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad != name.startswith("measurement_identity.source_request."):
            raise ValueError("Only the new source request parameters may be trainable")
    head = policy.measurement_identity.source_request
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003)
    rng, queue, losses, gradient_norms = random.Random(0), [], [], []
    warmup_queue, vocabulary_decisions, composition_decisions = [], 0, 0
    head.train()
    contexts = contexts.detach()
    for update in range(updates):
        vocabulary_pairs = (
            (PAIRS_PER_BATCH if update < warmup_updates else rehearsal_pairs)
            if warmup_records is not None
            else 0
        )
        composition_pairs = PAIRS_PER_BATCH - vocabulary_pairs
        while len(queue) < composition_pairs:
            cycle = list(range(len(records) // 2))
            rng.shuffle(cycle)
            queue.extend(cycle)
        chosen, queue = queue[:composition_pairs], queue[composition_pairs:]
        batch = [r for index in chosen for r in records[index * 2 : index * 2 + 2]]
        if vocabulary_pairs:
            while len(warmup_queue) < vocabulary_pairs:
                cycle = list(range(len(warmup_records) // 2))
                rng.shuffle(cycle)
                warmup_queue.extend(cycle)
            vocabulary, warmup_queue = warmup_queue[:vocabulary_pairs], warmup_queue[vocabulary_pairs:]
            batch.extend(r for index in vocabulary for r in warmup_records[index * 2 : index * 2 + 2])
        vocabulary_decisions += vocabulary_pairs * 2
        composition_decisions += composition_pairs * 2
        # The two sides of each pair receive the SAME graph context. Their only
        # predictive difference is the visible instruction, not numbers or IDs.
        context_ids = [index for _ in range(PAIRS_PER_BATCH) for index in [rng.randrange(len(contexts))] * 2]
        optimizer.zero_grad(set_to_none=True)
        loss = source_loss(head(batch, contexts[context_ids]), batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite source loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        gradient_norms.append(float(norm))
        if progress_callback:
            row = {"optimizer_steps": update + 1, "loss": losses[-1], "gradient_norm": gradient_norms[-1]}
            if warmup_records is not None:
                row.update(
                    {
                        "stage": "vocabulary" if update < warmup_updates else "composition",
                        "vocabulary_decisions": vocabulary_decisions,
                        "composition_decisions": composition_decisions,
                    }
                )
            progress_callback(row)
    report = {
        "optimizer_steps": updates,
        "supervised_decisions": updates * PAIRS_PER_BATCH * 2,
        "losses": losses,
        "gradient_norms": gradient_norms,
    }
    if warmup_records is not None:
        report["curriculum"] = {
            "warmup_updates": warmup_updates,
            "rehearsal_pairs": rehearsal_pairs,
            "vocabulary_decisions": vocabulary_decisions,
            "composition_decisions": composition_decisions,
        }
    return optimizer, report


@torch.no_grad()
def paired_accuracy(policy, records, contexts):
    validate_pairs(records)
    rows = []
    for index in range(0, len(records), 2):
        pair = records[index : index + 2]
        pooled = contexts[(index // 2) % len(contexts)][None].expand(2, -1)
        scores = policy.measurement_identity(pair, pooled)
        chosen = [r["candidates"][int(score.argmax())]["id"] for r, score in zip(pair, scores)]
        rows.append(
            {
                "pair": index // 2,
                "selected": chosen,
                "both_correct": all(c == r["expected_id"] for c, r in zip(chosen, pair)),
                "selection_changed": chosen[0] != chosen[1],
            }
        )
    return {
        "pairs": len(rows),
        "both_correct": sum(r["both_correct"] for r in rows),
        "accuracy": sum(r["both_correct"] for r in rows) / len(rows),
        "predictions": rows,
    }
