"""YAML experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    name: str = "smoke"
    seed: int = 0
    families: List[str] = field(default_factory=lambda: [
        "tfim", "heisenberg", "molecular_like",
    ])
    sizes: List[int] = field(default_factory=lambda: [3, 4])
    n_per_family: int = 3
    # Trotter parameters.
    time: float = 1.0
    steps: int = 2                       # the headline (fixed-budget) step count
    steps_grid: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6])
    reference_backend: str = "expm"      # "expm" (dense) or "krylov"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)
