import torch
import torch.nn.functional as F


def simulate_trajectories(
    W_eff: torch.Tensor,
    T: int = 400,
    n_seqs: int = 5,
    alpha: float = 0.9,
    noise_std: float = 0.02
) -> list[torch.Tensor]:
    """
    Simulate discrete-time protein dynamics using a linear update rule with Gaussian noise.

    x_{t+1} = σ(α * (W_eff @ x_t) + ϵ), ϵ ~ N(0, noise_std^2)

    :param W_eff: effective interaction weight matrix W_eff in R^{d x d}
    :type W_eff: torch.Tensor
    :param T: number of time steps per sequence, defaults to 400
    :type T: int, optional
    :param n_seqs: number of independent trajectories, defaults to 5
    :type n_seqs: int, optional
    :param alpha: update weight parameter, defaults to 0.9
    :type alpha: float, optional
    :param noise_std: standard deviation of Gaussian noise, defaults to 0.02
    :type noise_std: float, optional
    :return: list of simulated protein expression trajectories with shape (T+1, d)
    :rtype: list[torch.Tensor]
    """
    d = W_eff.shape[0]
    sequences = []
    b = torch.zeros(d)  # no bias term

    for _ in range(n_seqs):
        # initialize x ~ Β(2, 2)
        x = torch.distributions.Beta(2., 2.).sample((d,))
        trajectories = [x]

        for _ in range(T):
            noise = torch.randn(d) * noise_std
            z = F.relu(alpha * (W_eff @ x) + b + noise)
            x = torch.sigmoid(z)
            trajectories.append(x)

        sequences.append(torch.stack(trajectories))  # shape (T+1, d)

    return sequences


def make_transition_dataset(sequences: list[torch.Tensor], train_p: float = 0.7, val_p: float = 0.15):
    """
    Flatten sequences into transition pairs (x_t -> x_{t+1}) and split into train, val, test sets.

    :param sequences: list of simulated protein expression trajectories with shape (T+1, d)
    :type sequences: list[torch.Tensor]
    :param train_p: proportion of data for training set, defaults to 0.7
    :type train_p: float, optional
    :param val_p: proportion of data for validation set, defaults to 0.15
    :type val_p: float, optional
    :return: train, val, test datasets as tuples of (X, Y)
    :rtype: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    """
    X_list, Y_list = [], []

    for seq in sequences:
        X_list.append(seq[:-1])  # shape (T, d)
        Y_list.append(seq[1:])   # shape (T, d)

    X = torch.cat(X_list, dim=0)  # shape (n_seqs * T, d)
    Y = torch.cat(Y_list, dim=0)  # shape (n_seqs * T, d)

    N = X.shape[0]
    idx = torch.randperm(N)  # shuffle
    X, Y = X[idx], Y[idx]

    n_train = int(N * train_p)
    n_val = int(N * val_p)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
