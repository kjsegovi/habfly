"""Held-out temperature scaling and reliability metrics."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    confidence, predicted = probabilities.max(dim=-1)
    accuracy = predicted.eq(labels).float()
    error = probabilities.new_zeros(())
    for i in range(bins):
        lower, upper = i / bins, (i + 1) / bins
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.float().mean() * (confidence[selected].mean() - accuracy[selected].mean()).abs()
    return float(error)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    if logits.ndim != 2 or len(logits) != len(labels) or not len(labels):
        raise ValueError("Calibration needs a nonempty aligned logit/label matrix")
    if not torch.isfinite(logits).all():
        raise ValueError("Calibration logits must be finite")
    logits, labels = logits.detach().float().cpu(), labels.detach().long().cpu()
    # Bounded grid avoids unstable optimizer behavior on tiny smoke datasets.
    temperatures = torch.logspace(-1, 1.3, 120)
    losses = torch.stack([F.cross_entropy(logits / t, labels) for t in temperatures])
    temperature = float(temperatures[losses.argmin()])
    probabilities = (logits / temperature).softmax(-1)
    return {"temperature": temperature, "examples": len(labels), "nll": float(losses.min()),
            "ece": expected_calibration_error(probabilities, labels), "method": "heldout_temperature_scaling"}
