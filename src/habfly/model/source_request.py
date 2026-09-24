"""Experimental source-only option scores from visible instructions and source labels."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .tokenizer import CharacterTokenizer


class SourceRequestPointer(nn.Module):
    """Independent ordered character features; no parsing rules or correct-source mask."""

    def __init__(self, tokenizer: CharacterTokenizer, hidden_size: int):
        super().__init__()
        self.tokenizer, self.hidden_size = tokenizer, hidden_size
        self.embedding = nn.Embedding(tokenizer.vocab_size, hidden_size, padding_idx=0)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1, bias=False)
        self.query = nn.Linear(hidden_size * 4, hidden_size)
        self.key = nn.Linear(hidden_size * 4, hidden_size)
        self.graph_context = nn.Linear(hidden_size, hidden_size, bias=False)

    def encode(self, texts):
        if any(not isinstance(t, str) or not t or len(t) + 2 > self.tokenizer.max_length for t in texts):
            raise ValueError("Source request fields must be nonempty strings within the token budget")
        tokens, lengths = self.tokenizer.batch(texts, self.embedding.weight.device)
        packed = pack_padded_sequence(
            self.embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        sequence, final = self.encoder(packed)
        sequence, _ = pad_packed_sequence(sequence, batch_first=True)
        mask = (
            torch.arange(sequence.shape[1], device=tokens.device)[None, :]
            < lengths.to(tokens.device)[:, None]
        )
        weights = self.attention(sequence).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(-1)
        pooled = (weights[:, :, None] * sequence).sum(1)
        return torch.cat([pooled, final.transpose(0, 1).reshape(len(texts), -1)], dim=-1)

    def forward(self, requests, pooled):
        if not requests or pooled.shape != (len(requests), self.hidden_size):
            raise ValueError("Source requests and graph readouts must be aligned")
        # Deduplicate identical visible labels, not correct answers. A source is
        # scored identically across quantities and units, preserving factorization.
        sources = list(dict.fromkeys(c["source"] for r in requests for c in r["candidates"]))
        if not sources or any(not r["candidates"] for r in requests):
            raise ValueError("Source selection requires visible candidates")
        encoded = self.encode([r["instruction"] for r in requests] + sources)
        query = self.query(encoded[: len(requests)]) + self.graph_context(pooled)
        keys = self.key(encoded[len(requests) :])
        indices = {source: i for i, source in enumerate(sources)}
        return [
            (keys[[indices[c["source"]] for c in request["candidates"]]] * query[i]).sum(-1)
            / self.hidden_size**0.5
            for i, request in enumerate(requests)
        ]
