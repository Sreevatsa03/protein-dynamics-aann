from __future__ import annotations
import json
from pathlib import Path

import torch


def save_tensor(path: str | Path, tensor: torch.Tensor) -> None:
    """
    Save a tensor to disk as a .pt file.

    :param path: path to save the tensor
    :type path: str or Path
    :param tensor: tensor to save
    :type tensor: torch.Tensor
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def load_tensor(path: str | Path) -> torch.Tensor:
    """
    Load a tensor saved with torch.save.

    :param path: path to the saved tensor
    :type path: str or Path
    :return: loaded tensor
    :rtype: torch.Tensor
    """
    return torch.load(Path(path))


def save_json(path: str | Path, obj: dict) -> None:
    """
    Save a dictionary as json.

    :param path: path to save the json file
    :type path: str or Path
    :param obj: dictionary to save
    :type obj: dict
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> dict:
    """
    Load a json file and return a dictionary.

    :param path: path to the json file
    :type path: str or Path
    :return: loaded dictionary
    :rtype: dict
    """
    with open(path, "r") as f:
        return json.load(f)
