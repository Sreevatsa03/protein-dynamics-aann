import torch
from pathlib import Path
import json
from src.dynamics.simulate import simulate_phase_dynamics_with_basins


def generate_dataset(
    basin_strength: float,
    n_seqs_train: int = 24,
    n_seqs_val: int = 6,
    T_obs: int = 400,
    output_dir: Path = None,
    seed: int = 42,
):
    """Generate a dataset with specified basin strength."""

    print(f"\nGenerating dataset with basin_strength={basin_strength:.2f}")

    # generate training and validation data
    train_data = simulate_phase_dynamics_with_basins(
        n_seqs=n_seqs_train,
        T_obs=T_obs,
        basin_strength=basin_strength,
        noise_std=0.02,
        seed=seed,
    )

    val_data = simulate_phase_dynamics_with_basins(
        n_seqs=n_seqs_val,
        T_obs=T_obs,
        basin_strength=basin_strength,
        noise_std=0.02,
        seed=seed + 1000,
    )

    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # prepare transition pairs (x_t -> x_{t+1})
    train_transitions_x = train_data[:, :-1, :].reshape(-1, 12)
    train_transitions_y = train_data[:, 1:, :].reshape(-1, 12)

    val_transitions_x = val_data[:, :-1, :].reshape(-1, 12)
    val_transitions_y = val_data[:, 1:, :].reshape(-1, 12)

    # save data
    torch.save(train_data, output_dir / "trajectories.pt")
    torch.save(
        {"X": train_transitions_x, "Y": train_transitions_y},
        output_dir / "transitions_train.pt"
    )
    torch.save(
        {"X": val_transitions_x, "Y": val_transitions_y},
        output_dir / "transitions_val.pt"
    )

    # save metadata
    meta = {
        "basin_strength": basin_strength,
        "n_seqs_train": n_seqs_train,
        "n_seqs_val": n_seqs_val,
        "T_obs": T_obs,
        "n_transitions_train": len(train_transitions_x),
        "n_transitions_val": len(val_transitions_x),
        "seed": seed,
    }

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Train transitions: {len(train_transitions_x)}")
    print(f"Val transitions: {len(val_transitions_x)}")
    print(f"Saved to: {output_dir}")

    return meta


def main():
    base_dir = Path("data/relaxation_sweep")

    # test range of basin strengths
    basin_strengths = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("Generating relaxation sweep datasets...")

    all_meta = {}

    for basin_strength in basin_strengths:
        output_dir = base_dir / f"basin_{basin_strength:.1f}"
        meta = generate_dataset(
            basin_strength=basin_strength,
            n_seqs_train=24,
            n_seqs_val=6,
            T_obs=400,
            output_dir=output_dir,
            seed=42,
        )
        all_meta[f"basin_{basin_strength:.1f}"] = meta

    # save summary
    with open(base_dir / "sweep_summary.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"Summary saved to: {base_dir}/sweep_summary.json")


if __name__ == "__main__":
    main()
