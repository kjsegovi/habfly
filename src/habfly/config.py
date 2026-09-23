"""Portable local profiles loaded explicitly by CLI and runtime."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from .contracts import Contract


class Profile(Contract):
    name: str = "mac-smoke"
    graph: Path | None = None
    stars: int = Field(default=1, ge=1, le=30)
    seed: int = 0
    hidden_size: int = Field(default=8, ge=4)
    epochs: int = Field(default=3, gt=0)
    examples: int = Field(default=96, ge=4)
    device: Literal["cpu"] = "cpu"
    content_pack: Path | None = None
    task: Literal["mini_habworlds", "stellar"] = "mini_habworlds"
    spreadsheet_config: Path | None = None
    calculation_backend: Literal["local", "google_sheets"] | None = None
    knowledge_pack: Path | None = None
    dataset: Path | None = None
    training_episodes: int = Field(default=64, ge=2)
    calibration_episodes: int = Field(default=16, ge=2)
    development_episodes: int = Field(default=16, ge=2)
    test_episodes: int = Field(default=100, ge=1)

    @property
    def backend(self):
        return self.calculation_backend or ("google_sheets" if self.spreadsheet_config else "local")


def calculation_config(settings, spreadsheet_override=None):
    if settings.backend == "local":
        from .knowledge import load_knowledge_pack

        return load_knowledge_pack(settings.knowledge_pack)
    from .spreadsheet import load_spreadsheet_config

    path = spreadsheet_override or settings.spreadsheet_config
    if not path:
        raise ValueError("Google Sheets backend requires spreadsheet_config")
    return load_spreadsheet_config(path)


def load_profile(path=None):
    return Profile.model_validate(yaml.safe_load(Path(path).read_text()) or {}) if path else Profile()
