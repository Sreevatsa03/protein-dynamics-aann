import torch


def make_signed_mask(d: int = 12, min_deg: int = 3, max_deg: int = 5) -> torch.Tensor:
    """
    Generate a signed adjacency matrix S in {-1, 0, +1}^{d x d}.

    Rows:   target proteins (i)\\
    Cols:   regulators (j)\\
    Entry:  S[i, j] = +1 (activation), -1 (inhibition), or 0 (no edge)

    :param d: number of rows/cols, defaults to 12
    :type d: int, optional
    :param min_deg: minimum degree of connectivity for each protein, defaults to 3
    :type min_deg: int, optional
    :param max_deg: maximum degree of connectivity for each protein, defaults to 5
    :type max_deg: int, optional
    :return: signed adjacency matrix
    :rtype: torch.Tensor
    """
    S = torch.zeros((d, d))

    for i in range(d):
        # num of regulators for protein i
        k = torch.randint(min_deg, max_deg + 1, (1,)).item()

        # randomly choose k regulators (without replacement)
        regulators = torch.randperm(d)[:k]

        # assign random signs {-1, +1}
        # sample from {0, 1} and map to {-1, +1}
        signs = torch.randint(0, 2, (k,)) * 2 - 1

        S[i, regulators] = signs.float()

    return S
