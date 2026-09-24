"""Exact memoization for an explicitly frozen text encoder during CPU training."""

from contextlib import contextmanager

import torch


@contextmanager
def frozen_text_cache(policy, max_entries=50000):
    parameters = list(policy.embedding.parameters()) + list(policy.text_encoder.parameters())
    if any(p.requires_grad for p in parameters):
        raise ValueError("Text memoization requires a frozen embedding and encoder")
    original, cache = policy.encode_text, {}

    def cached(texts):
        if any(p.requires_grad for p in parameters):
            raise ValueError("Cannot unfreeze a cached text encoder")
        missing = list(dict.fromkeys(text for text in texts if text not in cache))
        if len(cache) + len(missing) > max_entries:
            cache.clear()
            missing = list(dict.fromkeys(texts))
        if missing:
            with torch.no_grad():
                for text, encoded in zip(missing, original(missing)):
                    cache[text] = encoded.detach().clone()
        return torch.stack([cache[text] for text in texts])

    policy.encode_text = cached
    try:
        yield
    finally:
        del policy.encode_text
