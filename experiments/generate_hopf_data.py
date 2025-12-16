from pathlib import Path
import json
import torch

from src.dynamics.masks import make_cell_cycle_mask
from src.dynamics.ground_truth import build_gt_W, make_cell_cycle_mu
from src.dynamics.simulate import simulate_hopf_oscillator_network, make_transition_dataset
from src.utils.seed import set_seed


def main():
    # fixed seed for reproducibility
    seed = 42
    set_seed(seed)

    # output directory
    outdir = Path("data/hopf")
    outdir.mkdir(parents=True, exist_ok=True)

    # device and dtype
    device = torch.device("cpu")
    dtype = torch.float32

    # dataset parameters
    n_seqs = 24
    T_obs = 350
    total_time = 300.0
    dt = 0.02

    # Hopf parameters
    kappa = 0.85
    n_memories = 4
    noise_std = 0.002
    omega_range = (1.0, 3.0)
    hebbian_scale = 0.8

    # gating parameters
    use_gate = True
    gate_tau = 50.0
    gate_slope = 10.0
    gate_theta = 0.5
    gate_beta = 0.5
    gate_idx = [4, 7, 8, 10]

    # protein observation / regulation
    tau_p = None
    use_slow_mu = True
    slow_mu_rate = 0.1

    # signed biological wiring
    S, proteins = make_cell_cycle_mask()
    base_W = build_gt_W(S, target_radius=1.0, gain=1.0)

    # heterogeneous mu
    mu_vec = make_cell_cycle_mu(
        proteins=proteins,
        cyclin_mu=1.8,
        regulator_mu=1.2,
        input_mu=0.8,
        device=device,
        dtype=dtype,
    )

    print(f"Generating {n_seqs} trajectories with T_obs={T_obs}...")

    # simulate trajectories
    trajectories = simulate_hopf_oscillator_network(
        n_seqs=n_seqs,
        T_obs=T_obs,
        total_time=total_time,
        dt=dt,
        mu=mu_vec,
        kappa=kappa,
        omega_range=omega_range,
        n_memories=n_memories,
        noise_std=noise_std,
        tau_p=tau_p,
        hebbian_scale=hebbian_scale,
        use_slow_mu=use_slow_mu,
        slow_mu_rate=slow_mu_rate,
        use_gate=use_gate,
        gate_tau=gate_tau,
        gate_slope=gate_slope,
        gate_theta=gate_theta,
        gate_beta=gate_beta,
        gate_idx=gate_idx,
        base_W=base_W,
        squash_to_01=True,
        seed=seed,
        device=device,
        dtype=dtype,
    )  # shape (n_seqs, T_obs, 12)

    print(f"Trajectories shape: {trajectories.shape}")

    # extract initial conditions
    init_conditions = trajectories[:, 0, :].clone()  # (n_seqs, 12)

    # convert to list of tensors for make_transition_dataset
    sequences = [trajectories[i].cpu() for i in range(trajectories.shape[0])]

    # build transition dataset
    (X_train, Y_train), (X_val, Y_val), _ = make_transition_dataset(
        sequences,
        train_p=0.8,
        val_p=0.2,
    )

    print(f"Train transitions: {X_train.shape[0]}")
    print(f"Val transitions: {X_val.shape[0]}")

    # save tensors (ensure CPU)
    torch.save(trajectories.cpu(), outdir / "trajectories.pt")
    torch.save({"X": X_train.cpu(), "Y": Y_train.cpu()},
               outdir / "transitions_train.pt")
    torch.save({"X": X_val.cpu(), "Y": Y_val.cpu()},
               outdir / "transitions_val.pt")
    torch.save(init_conditions.cpu(), outdir / "init_conditions.pt")
    torch.save(torch.arange(n_seqs), outdir / "sequence_ids.pt")

    # save metadata
    meta = {
        "seed": seed,
        "n_seqs": n_seqs,
        "T_obs": T_obs,
        "total_time": total_time,
        "dt": dt,
        "state_dim": 12,
        "proteins": proteins,
        "kappa": kappa,
        "n_memories": n_memories,
        "noise_std": noise_std,
        "omega_range": list(omega_range),
        "hebbian_scale": hebbian_scale,
        "use_gate": use_gate,
        "gate_tau": gate_tau,
        "gate_slope": gate_slope,
        "gate_theta": gate_theta,
        "gate_beta": gate_beta,
        "gate_idx": gate_idx,
        "tau_p": tau_p,
        "use_slow_mu": use_slow_mu,
        "slow_mu_rate": slow_mu_rate,
        "mu_cyclin": 1.8,
        "mu_regulator": 1.2,
        "mu_input": 0.8,
        "train_p": 0.8,
        "val_p": 0.2,
        "n_train_transitions": int(X_train.shape[0]),
        "n_val_transitions": int(X_val.shape[0]),
    }

    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset saved to {outdir}")


if __name__ == "__main__":
    main()
