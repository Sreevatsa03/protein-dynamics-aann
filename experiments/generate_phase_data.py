import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.dynamics.simulate import (
    simulate_phase_dynamics,
    get_cell_cycle_phase_states,
    make_transition_dataset,
)
from src.utils.seed import set_seed


def plot_phase_states(phases: dict[str, torch.Tensor], save_path: Path) -> None:
    """
    Plot heatmap of canonical phase states.

    :param phases: dict mapping phase names to 12-dim state vectors
    :type phases: dict[str, torch.Tensor]
    :param save_path: path to save the plot
    :type save_path: Path
    """
    proteins = [
        "Myc", "Cdh1", "p27", "Rb", "CycD", "E2F",
        "SCF", "CycE", "CycA", "NFY", "CycB", "Cdc20"
    ]
    phase_order = ["G1", "S", "G2", "M"]

    data = np.array([phases[p].numpy() for p in phase_order])

    plt.figure(figsize=(10, 4))
    plt.imshow(data, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    plt.colorbar(label="Expression Level")
    plt.xticks(range(12), proteins, rotation=45, ha="right")
    plt.yticks(range(4), phase_order)
    plt.xlabel("Protein")
    plt.ylabel("Cell Cycle Phase")
    plt.title("Canonical Cell-Cycle Phase States")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_example_trajectory(
    traj: torch.Tensor,
    save_path: Path,
    proteins_to_plot: list[int] | None = None,
) -> None:
    """
    Plot example trajectory showing phase transitions.

    :param traj: trajectory tensor of shape (T, 12)
    :type traj: torch.Tensor
    :param save_path: path to save the plot
    :type save_path: Path
    :param proteins_to_plot: indices of proteins to plot, defaults to cyclins
    :type proteins_to_plot: list[int], optional
    """
    proteins = [
        "Myc", "Cdh1", "p27", "Rb", "CycD", "E2F",
        "SCF", "CycE", "CycA", "NFY", "CycB", "Cdc20"
    ]

    if proteins_to_plot is None:
        # plot cyclins by default (CycD, CycE, CycA, CycB)
        proteins_to_plot = [4, 7, 8, 10]

    T = traj.shape[0]
    t = np.arange(T)

    plt.figure(figsize=(12, 5))

    for idx in proteins_to_plot:
        plt.plot(t, traj[:, idx].numpy(), label=proteins[idx], linewidth=1.5)

    plt.xlabel("Time Step")
    plt.ylabel("Expression Level")
    plt.title("Example Trajectory: Cyclin Dynamics Across Cell Cycle Phases")
    plt.legend(loc="upper right")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_all_proteins_trajectory(traj: torch.Tensor, save_path: Path) -> None:
    """
    Plot all 12 proteins in a 3x4 subplot grid.

    :param traj: trajectory tensor of shape (T, 12)
    :type traj: torch.Tensor
    :param save_path: path to save the plot
    :type save_path: Path
    """
    proteins = [
        "Myc", "Cdh1", "p27", "Rb", "CycD", "E2F",
        "SCF", "CycE", "CycA", "NFY", "CycB", "Cdc20"
    ]

    T = traj.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        ax.plot(t, traj[:, idx].numpy(), linewidth=1.0)
        ax.set_title(proteins[idx])
        ax.set_ylim(-0.05, 1.05)
        if idx >= 8:
            ax.set_xlabel("Time")
        if idx % 4 == 0:
            ax.set_ylabel("Expression")

    plt.suptitle("All Protein Dynamics (Single Trajectory)", y=1.02)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    seed = 42
    set_seed(seed)

    outdir = Path("data/phase")
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path("experiments/plots/data_generation/phase")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # simulation parameters - stable attractor dynamics with stochastic jumps
    n_seqs = 24
    T_obs = 400
    dwell_time = 80          # minimum steps in each phase before allowing jump
    jump_prob = 0.02         # probability of phase transition per step after dwell
    relaxation_rate = 0.3    # fast relaxation toward phase attractor
    noise_std = 0.015        # small noise around attractor

    print(f"Generating {n_seqs} phase-based trajectories...")
    print(f"  T_obs: {T_obs}")
    print(f"  dwell_time: {dwell_time}")
    print(f"  jump_prob: {jump_prob}")
    print(f"  relaxation_rate: {relaxation_rate}")
    print(f"  noise_std: {noise_std}")

    # get phase states for plotting
    phases = get_cell_cycle_phase_states()
    plot_phase_states(phases, plot_dir / "phase_states_heatmap.png")
    print(f"  Saved phase states heatmap")

    # generate trajectories with stable attractor dynamics
    trajectories = simulate_phase_dynamics(
        n_seqs=n_seqs,
        T_obs=T_obs,
        dwell_time=dwell_time,
        jump_prob=jump_prob,
        relaxation_rate=relaxation_rate,
        noise_std=noise_std,
        seed=seed,
    )

    print(f"  Trajectories shape: {trajectories.shape}")

    # plot example trajectories
    plot_example_trajectory(
        trajectories[0], plot_dir / "example_trajectory_cyclins.png")
    plot_all_proteins_trajectory(
        trajectories[0], plot_dir / "example_trajectory_all.png")
    print(f"  Saved example trajectory plots")

    # extract initial conditions
    init_conditions = trajectories[:, 0, :].clone()

    # build transition dataset
    sequences = [trajectories[i].cpu() for i in range(trajectories.shape[0])]
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = make_transition_dataset(
        sequences,
        train_p=0.8,
        val_p=0.2,
    )

    print(f"  Train transitions: {X_train.shape[0]}")
    print(f"  Val transitions: {X_val.shape[0]}")
    print(f"  Test transitions: {X_test.shape[0]}")

    # save data
    torch.save(trajectories.cpu(), outdir / "trajectories.pt")
    torch.save({"X": X_train.cpu(), "Y": Y_train.cpu()},
               outdir / "transitions_train.pt")
    torch.save({"X": X_val.cpu(), "Y": Y_val.cpu()},
               outdir / "transitions_val.pt")
    torch.save(init_conditions.cpu(), outdir / "init_conditions.pt")
    torch.save(torch.arange(n_seqs), outdir / "sequence_ids.pt")

    proteins = [
        "Myc", "Cdh1", "p27", "Rb", "CycD", "E2F",
        "SCF", "CycE", "CycA", "NFY", "CycB", "Cdc20"
    ]

    meta = {
        "seed": seed,
        "n_seqs": n_seqs,
        "T_obs": T_obs,
        "state_dim": 12,
        "proteins": proteins,
        "simulator": "phase_dynamics_attractor",
        "dwell_time": dwell_time,
        "jump_prob": jump_prob,
        "relaxation_rate": relaxation_rate,
        "noise_std": noise_std,
        "n_train_transitions": int(X_train.shape[0]),
        "n_val_transitions": int(X_val.shape[0]),
        "n_test_transitions": int(X_test.shape[0]),
        "phase_states": {
            name: state.tolist() for name, state in phases.items()
        },
    }

    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDataset saved to {outdir}")
    print(f"Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
