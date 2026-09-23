"""A deterministic character vocabulary; no downloaded models or tokenizers."""

from __future__ import annotations

import hashlib
import json
import string

import torch


class CharacterTokenizer:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self, characters: str | None = None, max_length: int = 1024):
        self.characters = "".join(dict.fromkeys(characters or (string.printable + "°±×÷≤≥μλπ")))
        self.max_length = max_length
        self._ids = {char: index + 4 for index, char in enumerate(self.characters)}

    @property
    def vocab_size(self) -> int:
        return len(self.characters) + 4

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        limit = self.max_length if max_length is None else max_length
        if limit < 2:
            raise ValueError("Character sequences need space for BOS and EOS")
        return [self.BOS] + [self._ids.get(c, self.UNK) for c in str(text)[: limit - 2]] + [self.EOS]

    def decode(self, ids: list[int]) -> str:
        result = []
        for index in ids:
            if index == self.EOS:
                break
            if index >= 4 and index - 4 < len(self.characters):
                result.append(self.characters[index - 4])
            elif index == self.UNK:
                result.append("�")
        return "".join(result)

    def batch(self, texts: list[str], device=None, max_length: int | None = None):
        sequences = [self.encode(text, max_length) for text in texts]
        if not sequences:
            raise ValueError("Cannot encode an empty batch")
        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
        result = torch.zeros((len(sequences), int(lengths.max())), dtype=torch.long, device=device)
        for row, sequence in enumerate(sequences):
            result[row, : len(sequence)] = torch.tensor(sequence, device=device)
        return result, lengths

    def config(self) -> dict:
        return {"characters": self.characters, "max_length": self.max_length}

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.config(), sort_keys=True).encode()).hexdigest()
