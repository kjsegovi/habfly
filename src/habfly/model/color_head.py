"""Learned ordered color readout of biological efferent activity only.

Class ordering is architectural prior, not a wavelength lookup. Cutpoints and
the scalar projection are learned from training labels. No reference thresholds,
measurements, or observations are accepted by this module.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class OrdinalColorHead(nn.Module):
    def __init__(self, hidden_size: int, classes: int = 9):
        super().__init__()
        if classes < 3:
            raise ValueError("Ordinal readout requires at least three classes")
        self.register_buffer("feature_mean", torch.zeros(hidden_size))
        self.register_buffer("feature_scale", torch.ones(hidden_size))
        self.register_buffer("normalization_fitted", torch.tensor(False))
        self.score = nn.Linear(hidden_size, 1)
        # Equal spacing in arbitrary latent rank units, never physical units.
        self.first_cut = nn.Parameter(torch.tensor(-(classes - 2) / 2))
        self.raw_gaps = nn.Parameter(torch.full((classes - 2,), math.log(math.expm1(1.0))))
        self.log_softness = nn.Parameter(torch.tensor(0.0))

    @torch.no_grad()
    def fit_normalization(self, training_features: torch.Tensor):
        if self.normalization_fitted:
            raise ValueError("Color feature normalization is already fitted")
        if (
            training_features.ndim != 2
            or training_features.shape[1] != self.feature_mean.numel()
            or len(training_features) < 2
            or not torch.isfinite(training_features).all()
        ):
            raise ValueError("Expected finite training effector features")
        self.feature_mean.copy_(training_features.mean(dim=0))
        self.feature_scale.copy_(training_features.std(dim=0, unbiased=False).clamp_min(1e-4))
        self.normalization_fitted.fill_(True)

    def cutpoints(self):
        gaps = F.softplus(self.raw_gaps) + 1e-4
        return torch.cat((self.first_cut.reshape(1), self.first_cut + gaps.cumsum(0)))

    def ordinal_logits(self, pooled: torch.Tensor):
        standardized = (pooled - self.feature_mean) / self.feature_scale
        score = self.score(standardized)
        softness = self.log_softness.clamp(-5, 3).exp()
        return (score - self.cutpoints()) / softness

    @staticmethod
    def class_log_probabilities(logits):
        # P(y=k) = sigmoid(z[k-1]) - sigmoid(z[k]); evaluate in log space.
        previous, following = logits[..., :-1], logits[..., 1:]
        middle = (
            F.logsigmoid(previous) + F.logsigmoid(-following) + torch.log(-torch.expm1(following - previous))
        )
        return torch.cat((F.logsigmoid(-logits[..., :1]), middle, F.logsigmoid(logits[..., -1:])), -1)

    def forward(self, pooled: torch.Tensor):
        return self.class_log_probabilities(self.ordinal_logits(pooled))

    def loss(self, pooled: torch.Tensor, labels: torch.Tensor):
        logits = self.ordinal_logits(pooled)
        ordinal_targets = (labels[..., None] > torch.arange(logits.shape[-1], device=labels.device)).float()
        nll = F.nll_loss(self.class_log_probabilities(logits), labels)
        ordinal = F.binary_cross_entropy_with_logits(logits, ordinal_targets)
        return nll + 0.25 * ordinal, nll, ordinal
