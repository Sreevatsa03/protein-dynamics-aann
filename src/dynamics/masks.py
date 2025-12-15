import torch


def make_cell_cycle_mask() -> torch.Tensor:
    """
    Generate a signed adjacency matrix S in {-1, 0, +1}^{12 x 12} representing
    the 12-protein signed wiring in the original paper.

    The regulator list corresponds to the supplementary table S1 bundled in this repo
    (resources/supplementary material.pdf).

    :return: signed adjacency matrix
    :rtype: torch.Tensor
    """
    proteins = [
        "Myc",
        "Cdh1",
        "p27",
        "Rb",
        "CycD",
        "E2F",
        "SCF",
        "CycE",
        "CycA",
        "NFY",
        "CycB",
        "Cdc20",
    ]
    idx = {p: i for i, p in enumerate(proteins)}

    # define regulators for each protein
    regulators: dict[str, list[tuple[str, int]]] = {
        "Myc": [("Myc", +1)],
        "Cdh1": [("CycE", -1), ("CycA", -1), ("CycB", -1)],
        "p27": [("CycD", -1), ("CycE", -1), ("CycA", -1)],
        "Rb": [("CycD", -1), ("CycE", -1), ("CycA", -1)],
        "CycD": [("Myc", +1), ("p27", -1), ("SCF", -1), ("Cdc20", -1)],
        "E2F": [("Rb", -1), ("CycD", +1), ("CycE", +1), ("CycA", -1)],
        "SCF": [("CycE", +1), ("Cdh1", -1)],
        "CycE": [("E2F", +1), ("p27", -1), ("SCF", -1)],
        "CycA": [("E2F", +1), ("Cdh1", -1), ("p27", -1), ("Cdc20", -1), ("NFY", +1)],
        "NFY": [("CycA", +1)],
        "CycB": [("NFY", +1), ("Cdh1", -1), ("Cdc20", -1)],
        "Cdc20": [("CycB", +1), ("Cdh1", -1)],
    }

    # build adjacency matrix
    S = torch.zeros((12, 12), dtype=torch.float32)
    for tgt, regs in regulators.items():
        for reg, sgn in regs:
            S[idx[tgt], idx[reg]] = float(sgn)

    return S, proteins


def make_random_mask(d: int = 12, min_deg: int = 3, max_deg: int = 5) -> torch.Tensor:
    """
    Generate a random signed adjacency matrix S in {-1, 0, +1}^{d x d}.

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
