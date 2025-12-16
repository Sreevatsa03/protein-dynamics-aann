import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from src.models.masked_aann import SigmoidAANN
from src.dynamics.masks import make_cell_cycle_mask
from src.dynamics.simulate import get_cell_cycle_phase_states
from src.utils.seed import set_seed


def evaluate_pattern_completion(
    model,
    phases: dict,
    n_trials: int = 100,
    noise_std: float = 0.15,
    n_steps: int = 100,
    device: torch.device = None,
    seed: int = 42,
):
    """Evaluate pattern completion by corrupting phase states and recovering."""

    if device is None:
        device = torch.device("cpu")

    rng = np.random.default_rng(seed)
    phase_order = ["G1", "S", "G2", "M"]
    results = {}

    model.eval()

    # stack all phase patterns for classification
    all_patterns = torch.stack([phases[p] for p in phase_order])  # (4, 12)
    mean_state = all_patterns.mean(dim=0)  # (12,)

    for phase_name in phase_order:
        ref_state = phases[phase_name]

        phase_correct = 0
        total_dist_to_ref = 0.0
        total_dist_to_mean = 0.0

        for _ in range(n_trials):
            # corrupt with noise
            noise = torch.from_numpy(
                rng.normal(0, noise_std, ref_state.shape).astype(np.float32)
            )
            x = (ref_state + noise).clamp(0.0, 1.0)

            # recover for n_steps
            with torch.no_grad():
                for _ in range(n_steps):
                    x = model(x.unsqueeze(0)).squeeze(0)
                    x = x.clamp(0.0, 1.0)

            # measure recovery - use L2 distance like existing code
            dist_to_ref = torch.norm(x - ref_state).item()
            dist_to_mean = torch.norm(x - mean_state).item()

            # classify phase by nearest neighbor
            dists_to_phases = torch.norm(all_patterns - x, dim=1)
            closest_phase_idx = dists_to_phases.argmin().item()
            correct_phase_idx = phase_order.index(phase_name)

            if closest_phase_idx == correct_phase_idx:
                phase_correct += 1

            total_dist_to_ref += dist_to_ref
            total_dist_to_mean += dist_to_mean

        results[phase_name] = {
            "avg_dist_to_ref": total_dist_to_ref / n_trials,
            "avg_dist_to_mean": total_dist_to_mean / n_trials,
            "phase_accuracy": phase_correct / n_trials,
        }

    # compute overall statistics
    avg_dist_to_ref = np.mean([r["avg_dist_to_ref"] for r in results.values()])
    avg_dist_to_mean = np.mean([r["avg_dist_to_mean"]
                               for r in results.values()])
    phase_accuracy = np.mean([r["phase_accuracy"] for r in results.values()])
    recovery = avg_dist_to_mean - avg_dist_to_ref

    summary = {
        "avg_dist_to_ref": avg_dist_to_ref,
        "avg_dist_to_mean": avg_dist_to_mean,
        "recovery": recovery,
        "phase_accuracy": phase_accuracy,
        "per_phase": results,
    }

    return summary


def main():
    device = torch.device("cpu")
    base_model_dir = Path("experiments/results/relaxation_sweep")
    output_dir = Path("experiments/results/relaxation_sweep_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    # get phase states
    phases = get_cell_cycle_phase_states(device=device)

    basin_strengths = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("Evaluating Pattern Completion vs Basin Strength")

    results_by_basin = {}

    for basin_strength in basin_strengths:
        print(f"\nEvaluating basin_strength={basin_strength:.1f}...")

        model_dir = base_model_dir / f"basin_{basin_strength:.1f}"

        # load model
        mask, _ = make_cell_cycle_mask()
        model = SigmoidAANN(state_dim=12, mask=mask, device=device)
        state_dict = torch.load(
            model_dir / "best_model.pt", weights_only=True, map_location=device)
        model.load_state_dict(state_dict)

        # evaluate
        eval_results = evaluate_pattern_completion(
            model=model,
            phases=phases,
            n_trials=100,
            noise_std=0.15,
            n_steps=100,
            device=device,
            seed=42,
        )

        results_by_basin[basin_strength] = eval_results

        print(f"  Phase accuracy: {eval_results['phase_accuracy']:.1%}")
        print(f"  Distance to ref: {eval_results['avg_dist_to_ref']:.4f}")
        print(f"  Recovery: {eval_results['recovery']:.4f}")

    # save results
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results_by_basin, f, indent=2)

    # create plot
    plt.figure(figsize=(10, 6))

    basin_vals = sorted(results_by_basin.keys())
    phase_accs = [results_by_basin[b]
                  ["phase_accuracy"] * 100 for b in basin_vals]
    recoveries = [results_by_basin[b]["recovery"] for b in basin_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # phase accuracy
    ax1.plot(basin_vals, phase_accs, "o-", linewidth=2,
             markersize=8, color="steelblue")
    ax1.axhline(25, color="red", linestyle="--",
                label="Chance (25%)", alpha=0.7)
    ax1.set_xlabel("Basin Strength", fontsize=12)
    ax1.set_ylabel("Phase Recovery Accuracy (%)", fontsize=12)
    ax1.set_title("Learning Enabled by Smooth Attractor Basins",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 100])

    # recovery metric
    ax2.plot(basin_vals, recoveries, "o-", linewidth=2,
             markersize=8, color="darkorange")
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax2.set_xlabel("Basin Strength", fontsize=12)
    ax2.set_ylabel("Recovery (dist_mean - dist_ref)", fontsize=12)
    ax2.set_title("Attractor Recovery vs Basin Strength",
                  fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "relaxation_sweep_results.png",
                dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "relaxation_sweep_results.pdf",
                bbox_inches="tight")
    print(f"\nPlots saved to: {output_dir}")

    # print summary table
    print("Summary Table")
    print(f"{'Basin':<10} {'Phase Acc':<15} {'Recovery':<15}")
    for basin in basin_vals:
        acc = results_by_basin[basin]["phase_accuracy"] * 100
        rec = results_by_basin[basin]["recovery"]
        print(f"{basin:<10.1f} {acc:<15.1f}% {rec:<15.4f}")

    print(f"\nResults saved to: {output_dir}/eval_results.json")


if __name__ == "__main__":
    main()
