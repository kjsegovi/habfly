"""Experimental learned measurement pointer, independent of numeric payloads.

This is a factorized option head, not a calculator or a next-action expert.
The biological policy still chooses the action/control and supplies its readout.
"""

from __future__ import annotations

import torch
from torch import nn

from .tokenizer import CharacterTokenizer


def measurement_request(observation: dict, control: dict, *, include_results=False) -> dict | None:
    """Route by the visible control contract, never by opaque IDs or correct answers.

    This first stage covers measurement-only source lists after an input has been
    selected. Mixed measurement/result lists retain the existing option scorer.
    """
    if (
        control.get("surface") != "calculation"
        or control.get("role") != "combobox"
        or control.get("label") != "Measurement or result"
    ):
        return None
    state = observation.get("calculation") or {}
    spec = (state.get("reference_card") or {}).get("inputs", {}).get(state.get("parameter"))
    measurements = (observation.get("values") or {}).get("measurements", {})
    if include_results:
        from .tool_state import visible_sources

        measurements = visible_sources(observation)
    options = control.get("options", [])
    if not spec or not options or any(option not in measurements for option in options):
        return None
    return {
        "instruction": observation.get("instruction", ""),
        "requirement": {"quantity": spec["quantity"], "unit": spec["unit"]},
        "candidates": [measurements[option] for option in options],
    }


class MeasurementIdentityPointer(nn.Module):
    """Learned character features for quantity, unit and source; no equality rules."""

    def __init__(self, tokenizer: CharacterTokenizer, hidden_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(tokenizer.vocab_size, hidden_size, padding_idx=0)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(hidden_size, hidden_size, kernel, padding=kernel // 2) for kernel in (3, 5)]
        )
        self.encoding = nn.Linear(hidden_size * 4, hidden_size)
        self.queries = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.graph_context = nn.Linear(hidden_size, hidden_size * 3, bias=False)

    def encode(self, texts: list[str]) -> torch.Tensor:
        if any(
            not isinstance(text, str) or not text or len(text) + 2 > self.tokenizer.max_length
            for text in texts
        ):
            raise ValueError("Measurement identity fields must be nonempty strings within the token budget")
        tokens, lengths = self.tokenizer.batch(texts, self.embedding.weight.device)
        embedded = self.embedding(tokens).transpose(1, 2)
        lengths = lengths.to(tokens.device)
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None, None, :] < lengths[:, None, None]
        features = []
        for convolution in self.convolutions:
            sequence = convolution(embedded).tanh()
            features.extend(
                [
                    (sequence * mask).sum(-1) / lengths[:, None],
                    sequence.masked_fill(~mask, -torch.inf).amax(-1),
                ]
            )
        return self.encoding(torch.cat(features, dim=-1))

    def forward(self, requests: list[dict], pooled: torch.Tensor) -> list[torch.Tensor]:
        if hasattr(self, "source_request"):
            base = self.quantity_unit_scores(requests, pooled)
            source = self.source_request(requests, pooled)
            return [left + right for left, right in zip(base, source)]
        return self._field_scores(requests, pooled, fields=3)

    def quantity_unit_scores(self, requests, pooled):
        """Existing first two contributions, without the failed source contribution."""
        return self._field_scores(requests, pooled, fields=2)

    def _field_scores(self, requests, pooled, *, fields):
        if not requests or pooled.shape != (len(requests), self.hidden_size):
            raise ValueError("Measurement requests and graph readouts must be aligned")
        texts, counts = [], []
        for request in requests:
            candidates = request["candidates"]
            if not candidates:
                raise ValueError("Measurement selection requires visible candidates")
            counts.append(len(candidates))
            texts.extend(
                [request["requirement"]["quantity"], request["requirement"]["unit"], request["instruction"]]
            )
            # Deliberately never tokenize values, IDs, grading labels, or candidate positions.
            for candidate in candidates:
                texts.extend([candidate["kind"], candidate["unit"], candidate["source"]])
        encoded = self.encode(texts)
        context = self.graph_context(pooled).reshape(len(requests), 3, self.hidden_size)
        scores, offset = [], 0
        for row, count in enumerate(counts):
            queries = torch.stack(
                [layer(encoded[offset + field]) for field, layer in enumerate(self.queries)]
            )
            keys = encoded[offset + 3 : offset + 3 + count * 3].reshape(count, 3, self.hidden_size)
            scores.append(
                (keys[:, :fields] * (queries + context[row])[:fields]).sum(dim=(1, 2)) / self.hidden_size**0.5
            )
            offset += 3 + count * 3
        return scores
