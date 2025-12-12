from __future__ import annotations
import json
from pathlib import Path

import torch


def save_tensor(path: str | Path, tensor: torch.Tensor) -> None:
    """
    Save a tensor to disk as a .pt file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def load_tensor(path: str | Path) -> torch.Tensor:
    """
    Load a tensor saved with torch.save.
    """
    return torch.load(Path(path))


def save_json(path: str | Path, obj: dict) -> None:
    """
    Save a dictionary as json.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> dict:
    """
    Load a json file and return a dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)
