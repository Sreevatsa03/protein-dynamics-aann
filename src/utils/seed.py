import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """set all rng seeds for reproducibility.

    :param seed: base seed to apply across python, numpy and torch
    :type seed: int
    :return: none
    :rtype: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ensure deterministic behavior where possible
    if torch.backends.mps.is_available():
        # mps doesn't have full determinism controls, but setting seed helps
        pass
    elif torch.backends.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
