"""
Evaluate Hopfield-encoded AANNs vs SGD/Hebbian-trained AANNs on phase data.

This script compares:
1. SGD-trained Linear AANN
2. SGD-trained Sigmoid AANN
3. Hebbian-trained Linear AANN (if available)
4. Hebbian-trained Sigmoid AANN (if available)
5. Hopfield-encoded Linear AANN (analytically computed)
6. Hopfield-encoded Sigmoid AANN (analytically computed)
7. Mean baseline

The key hypothesis: Hopfield-encoded Sigmoid should show pattern recovery,
while learned models collapse to mean.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dynamics.masks import make_cell_cycle_mask
from src.dynamics.simulate import get_cell_cycle_phase_states
from src.models.masked_aann import (
    LinearAANN,
    SigmoidAANN,
    create_hopfield_linear_aann,
    create_hopfield_sigmoid_aann,
    create_hopfield_sigmoid_aann_dense,
)
from src.evaluation.metrics import mse, per_protein_r2, open_loop_rollout
from src.evaluation.plots import (
    plot_rollout_mse_multi,
    plot_recovery_distance_vs_iter,
    plot_final_recovery_bar,
    plot_recovery_ratio_bar,
)


class MeanPredictor:
    """Baseline that always outputs the training mean."""

    def __init__(self, mean_vec: torch.Tensor):
        self.mean_vec = mean_vec

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return self.mean_vec.clone()
        return self.mean_vec.unsqueeze(0).expand(x.shape[0], -1)

    def eval(self):
        pass


def load_model(model_type: str, checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    S, _ = make_cell_cycle_mask()
    state_dim = S.shape[0]

    if model_type == "linear":
        model = LinearAANN(state_dim=state_dim, mask=S, device=device)
    else:
        model = SigmoidAANN(state_dim=state_dim, mask=S, device=device)

    state_dict = torch.load(
        checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def iterative_recovery(model, x0: torch.Tensor, K: int = 100) -> torch.Tensor:
    """Run model K steps from x0, return trajectory (K+1, d)."""
    trajectory = [x0.clone()]
    x = x0.clone()
    with torch.no_grad():
        for _ in range(K):
            x = model(x)
            # clamp for stability (especially for linear)
            x = x.clamp(0, 1)
            trajectory.append(x.clone())
    return torch.stack(trajectory)


def pattern_completion_eval(
    predictors: dict,
    phase_states: dict[str, torch.Tensor],
    mean_state: torch.Tensor,
    noise_std: float = 0.1,
    n_trials_per_phase: int = 25,
    K: int = 100,
    seed: int = 42,
) -> dict:
    """
    Evaluate pattern completion starting from corrupted phase states.

    Unlike the trajectory-based eval, this starts from KNOWN phase states
    with corruption, testing true associative memory.
    """
    rng = np.random.default_rng(seed)
    phase_names = list(phase_states.keys())

    results = {name: {
        "dist_to_ref_per_iter": [],
        "dist_to_mean_per_iter": [],
        "recovered_to_correct_phase": [],
    } for name in predictors.keys()}

    all_patterns = torch.stack([phase_states[p]
                               for p in phase_names])  # (4, 12)

    for phase_name in phase_names:
        x_ref = phase_states[phase_name]

        for _ in range(n_trials_per_phase):
            # corrupt with Gaussian noise
            noise = torch.from_numpy(
                rng.normal(0, noise_std, x_ref.shape).astype(np.float32)
            )
            x0 = (x_ref + noise).clamp(0, 1)

            for pred_name, model in predictors.items():
                traj = iterative_recovery(model, x0, K=K)

                # distances at each step
                dist_to_ref = torch.norm(traj - x_ref, dim=1).numpy()
                dist_to_mean = torch.norm(traj - mean_state, dim=1).numpy()

                results[pred_name]["dist_to_ref_per_iter"].append(dist_to_ref)
                results[pred_name]["dist_to_mean_per_iter"].append(
                    dist_to_mean)

                # check if final state is closest to correct phase
                final_state = traj[-1]
                dists_to_phases = torch.norm(all_patterns - final_state, dim=1)
                closest_phase_idx = dists_to_phases.argmin().item()
                correct_phase_idx = phase_names.index(phase_name)
                results[pred_name]["recovered_to_correct_phase"].append(
                    closest_phase_idx == correct_phase_idx
                )

    # aggregate
    for name in predictors.keys():
        ref_arr = np.array(results[name]["dist_to_ref_per_iter"])
        mean_arr = np.array(results[name]["dist_to_mean_per_iter"])

        results[name]["mean_dist_to_ref"] = ref_arr.mean(axis=0)
        results[name]["std_dist_to_ref"] = ref_arr.std(axis=0)
        results[name]["mean_dist_to_mean"] = mean_arr.mean(axis=0)
        results[name]["std_dist_to_mean"] = mean_arr.std(axis=0)

        results[name]["final_dist_to_ref"] = float(ref_arr[:, -1].mean())
        results[name]["final_dist_to_mean"] = float(mean_arr[:, -1].mean())
        results[name]["recovery_ratio"] = (
            results[name]["final_dist_to_mean"] -
            results[name]["final_dist_to_ref"]
        )
        results[name]["phase_recovery_accuracy"] = float(
            np.mean(results[name]["recovered_to_correct_phase"])
        )

    return results


def plot_phase_recovery_accuracy(
    results: dict,
    save_path: Path,
) -> None:
    """Plot bar chart of phase recovery accuracy for all models."""
    names = list(results.keys())
    accuracies = [results[n]["phase_recovery_accuracy"] for n in names]

    # color code by model type
    colors = []
    for name in names:
        if "Hopfield" in name and "Sigmoid" in name:
            colors.append("#2ecc71")  # green for Hopfield Sigmoid
        elif "Hopfield" in name:
            colors.append("#27ae60")  # darker green for Hopfield Linear
        elif "SGD" in name:
            colors.append("#3498db")  # blue for SGD
        elif "Hebbian" in name:
            colors.append("#9b59b6")  # purple for Hebbian
        else:
            colors.append("#95a5a6")  # gray for baseline

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), accuracies, color=colors)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("Phase Recovery Accuracy")
    plt.title("Pattern Completion: Recovery to Correct Phase")
    plt.ylim(0, 1.05)

    # add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{acc:.1%}", ha="center", va="bottom", fontsize=9)

    # add reference line at chance level (25%)
    plt.axhline(y=0.25, color="red", linestyle="--",
                alpha=0.5, label="Chance (25%)")
    plt.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Hopfield-encoded vs trained AANNs"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/phase"),
        help="Path to phase data directory",
    )
    parser.add_argument(
        "--sgd-results-dir",
        type=Path,
        default=Path("experiments/results/phase"),
        help="Path to SGD training results",
    )
    parser.add_argument(
        "--hebbian-results-dir",
        type=Path,
        default=Path("experiments/results/phase_hebbian"),
        help="Path to Hebbian training results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/hopfield_eval"),
        help="Path to save evaluation outputs",
    )
    parser.add_argument(
        "--hopfield-scale",
        type=float,
        default=4.0,
        help="Scaling factor for Hopfield weights (larger = sharper attractors)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # load phase states
    print("Loading phase states...")
    phase_states = get_cell_cycle_phase_states(device=device)
    patterns = torch.stack([phase_states[p] for p in ["G1", "S", "G2", "M"]])
    print(f"  Patterns shape: {patterns.shape}")

    # compute mean state from training data
    train_data = torch.load(
        args.data_dir / "transitions_train.pt", weights_only=True, map_location="cpu"
    )
    mean_state = train_data["Y"].mean(dim=0)
    print(f"  Mean state computed")

    # load mask
    S, _ = make_cell_cycle_mask()

    # create all predictors
    print("\nCreating predictors...")
    predictors = {}

    # SGD-trained models
    try:
        sgd_linear = load_model(
            "linear", args.sgd_results_dir / "linear" / "best_model.pt", device
        )
        sgd_sigmoid = load_model(
            "sigmoid", args.sgd_results_dir / "sigmoid" / "best_model.pt", device
        )
        predictors["SGD Linear"] = sgd_linear
        predictors["SGD Sigmoid"] = sgd_sigmoid
        print("  Loaded SGD-trained models")
    except FileNotFoundError:
        print("  SGD models not found, skipping")

    # Hebbian-trained models
    try:
        hebbian_linear = load_model(
            "linear", args.hebbian_results_dir / "linear" / "best_model.pt", device
        )
        hebbian_sigmoid = load_model(
            "sigmoid", args.hebbian_results_dir / "sigmoid" / "best_model.pt", device
        )
        predictors["Hebbian Linear"] = hebbian_linear
        predictors["Hebbian Sigmoid"] = hebbian_sigmoid
        print("  Loaded Hebbian-trained models")
    except FileNotFoundError:
        print("  Hebbian models not found, skipping")

    # Hopfield-encoded models (with sparse biological mask)
    hopfield_linear = create_hopfield_linear_aann(
        patterns, S, scale=args.hopfield_scale, device=device
    )
    hopfield_sigmoid = create_hopfield_sigmoid_aann(
        patterns, S, scale=args.hopfield_scale, device=device
    )
    predictors["Hopfield Linear"] = hopfield_linear
    predictors["Hopfield Sigmoid"] = hopfield_sigmoid
    print(
        f"  Created Hopfield-encoded models with mask (scale={args.hopfield_scale})")

    # Hopfield-encoded sigmoid WITHOUT mask (positive control)
    hopfield_sigmoid_dense = create_hopfield_sigmoid_aann_dense(
        patterns, scale=args.hopfield_scale, device=device
    )
    predictors["Hopfield Dense"] = hopfield_sigmoid_dense
    print(f"  Created dense Hopfield Sigmoid (no mask - positive control)")

    # Mean baseline
    predictors["Mean"] = MeanPredictor(mean_state)

    # Pattern completion evaluation
    print("\nPattern Completion Evaluation...")
    print("  (Starting from corrupted phase states)")

    pc_results = pattern_completion_eval(
        predictors=predictors,
        phase_states=phase_states,
        mean_state=mean_state,
        noise_std=0.15,
        n_trials_per_phase=25,
        K=100,
        seed=42,
    )

    # print results
    print("\n" + "=" * 80)
    print("PATTERN COMPLETION RESULTS")
    print("=" * 80)
    print(f"{'Model':<20} {'Dist to Ref':>12} {'Dist to Mean':>12} {'Recovery':>10} {'Phase Acc':>10}")
    print("-" * 80)
    for name, res in pc_results.items():
        print(
            f"{name:<20} {res['final_dist_to_ref']:>12.4f} "
            f"{res['final_dist_to_mean']:>12.4f} {res['recovery_ratio']:>+10.4f} "
            f"{res['phase_recovery_accuracy']:>10.1%}"
        )
    print("=" * 80)

    # plots
    print("\nGenerating plots...")

    # recovery distance vs iteration
    dist_to_ref_dict = {name: res["mean_dist_to_ref"]
                        for name, res in pc_results.items()}
    dist_to_ref_std_dict = {name: res["std_dist_to_ref"]
                            for name, res in pc_results.items()}
    plot_recovery_distance_vs_iter(
        dist_to_ref_dict,
        dist_to_ref_std_dict,
        save_path=args.output_dir / "recovery_dist_vs_iter.png",
    )

    # final recovery bar
    final_dist_to_ref = {name: res["final_dist_to_ref"]
                         for name, res in pc_results.items()}
    final_dist_to_mean = {name: res["final_dist_to_mean"]
                          for name, res in pc_results.items()}
    plot_final_recovery_bar(
        final_dist_to_ref,
        final_dist_to_mean,
        save_path=args.output_dir / "final_recovery_bar.png",
    )

    # recovery ratio bar
    recovery_ratios = {name: res["recovery_ratio"]
                       for name, res in pc_results.items()}
    plot_recovery_ratio_bar(
        recovery_ratios,
        save_path=args.output_dir / "recovery_ratio_bar.png",
    )

    # phase recovery accuracy bar (key plot!)
    plot_phase_recovery_accuracy(
        pc_results,
        save_path=args.output_dir / "phase_recovery_accuracy.png",
    )

    # save summary
    summary = {
        "hopfield_scale": args.hopfield_scale,
        "noise_std": 0.15,
        "n_trials_per_phase": 25,
        "K_iterations": 100,
        "results": {
            name: {
                "final_dist_to_ref": res["final_dist_to_ref"],
                "final_dist_to_mean": res["final_dist_to_mean"],
                "recovery_ratio": res["recovery_ratio"],
                "phase_recovery_accuracy": res["phase_recovery_accuracy"],
            }
            for name, res in pc_results.items()
        },
    }

    with open(args.output_dir / "hopfield_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
