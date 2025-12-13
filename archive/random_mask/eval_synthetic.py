from pathlib import Path
import json
import numpy as np
import torch

from src.dynamics.masks import make_signed_mask
from src.dynamics.ground_truth import build_gt_W
from src.dynamics.simulate import simulate_trajectories, make_transition_dataset
from src.models.masked_aann import LinearAANN, SigmoidAANN
from src.evaluation.metrics import (
    mse,
    per_protein_correlation,
    per_protein_r2,
    rollout_mse,
    rollout_mse_vs_time,
)
from src.evaluation.plots import (
    plot_loss_curves,
    plot_per_protein_metrics,
    plot_rollout_mse_vs_time,
    plot_trajectory_overlay,
)
from src.utils.seed import set_seed


def rollout(model, x0, T):
    """
    Roll out a trained model starting from x0 for T steps.
    """
    xs = [x0]
    with torch.no_grad():
        for _ in range(T - 1):
            xs.append(model(xs[-1]))
    return torch.stack(xs)


def main():
    """
    Evaluate linear vs sigmoid AANNs on synthetic dynamics.
    """
    set_seed(42)

    results_dir = Path("experiments/results")
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # regenerate ground-truth system
    S = make_signed_mask(
        d=12,
        min_deg=3,
        max_deg=5
    )

    W_eff = build_gt_W(
        S=S,
        mu=-1.2,
        sigma=0.6,
        target_radius=0.9,
    )

    trajectories = simulate_trajectories(
        W_eff=W_eff,
        T=400,
        n_seqs=5,
        alpha=0.9,
        noise_std=0.02,
    )

    (_, _), (_, _), (X_test, Y_test) = make_transition_dataset(
        trajectories,
        train_p=0.7,
        val_p=0.15,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")

    # load models
    linear = LinearAANN(state_dim=12, mask=S, device=device)
    sigmoid = SigmoidAANN(state_dim=12, mask=S, device=device)

    linear.load_state_dict(torch.load(
        results_dir / "linear/best_model.pt", map_location=device, weights_only=True))
    sigmoid.load_state_dict(torch.load(
        results_dir / "sigmoid/best_model.pt", map_location=device, weights_only=True))

    linear.eval()
    sigmoid.eval()

    # one-step predictions
    with torch.no_grad():
        Y_hat_linear = linear(X_test.to(device)).cpu()
        Y_hat_sigmoid = sigmoid(X_test.to(device)).cpu()

    # one-step metrics
    mse_linear = mse(Y_test, Y_hat_linear)
    mse_sigmoid = mse(Y_test, Y_hat_sigmoid)

    corr_linear = per_protein_correlation(Y_test, Y_hat_linear)
    corr_sigmoid = per_protein_correlation(Y_test, Y_hat_sigmoid)

    r2_linear = per_protein_r2(Y_test, Y_hat_linear)
    r2_sigmoid = per_protein_r2(Y_test, Y_hat_sigmoid)

    print("Test MSE")
    print(f"  Linear:  {mse_linear:.6f}")
    print(f"  Sigmoid: {mse_sigmoid:.6f}")

    # plot per-protein correlation
    plot_per_protein_metrics(
        corr_linear,
        corr_sigmoid,
        metric_name="Pearson Correlation",
        save_path=plot_dir / "per_protein_correlation.png",
    )

    # plot per-protein R^2
    plot_per_protein_metrics(
        r2_linear,
        r2_sigmoid,
        metric_name="$R^2$",
        save_path=plot_dir / "per_protein_r2.png",
        clip_min=-5,  # clip extreme outliers for better visualization
    )

    # rollout evaluation on first test trajectory
    true_traj = trajectories[-1]
    x0 = true_traj[0].to(device)

    pred_linear = rollout(linear, x0, T=true_traj.shape[0]).cpu()
    pred_sigmoid = rollout(sigmoid, x0, T=true_traj.shape[0]).cpu()

    mse_rollout_linear = rollout_mse_vs_time(true_traj, pred_linear)
    mse_rollout_sigmoid = rollout_mse_vs_time(true_traj, pred_sigmoid)

    plot_rollout_mse_vs_time(
        mse_rollout_linear,
        mse_rollout_sigmoid,
        save_path=plot_dir / "rollout_mse_vs_time.png",
    )

    mse_rollout_linear = rollout_mse(true_traj, pred_linear)
    mse_rollout_sigmoid = rollout_mse(true_traj, pred_sigmoid)

    print("Rollout MSE")
    print(f"  Linear:  {mse_rollout_linear:.6f}")
    print(f"  Sigmoid: {mse_rollout_sigmoid:.6f}")

    # trajectory overlays for select proteins
    t = np.arange(true_traj.shape[0])

    for idx in range(0, 12):
        plot_trajectory_overlay(
            t,
            true_traj,
            pred_linear,
            protein_idx=idx,
            label="Linear",
            save_path=plot_dir / f"traj_linear_protein_{idx}.png",
        )
        plot_trajectory_overlay(
            t,
            true_traj,
            pred_sigmoid,
            protein_idx=idx,
            label="Sigmoid",
            save_path=plot_dir / f"traj_sigmoid_protein_{idx}.png",
        )

    # loss curve plots
    with open(results_dir / "linear/losses.json") as f:
        losses_linear = json.load(f)
    with open(results_dir / "sigmoid/losses.json") as f:
        losses_sigmoid = json.load(f)

    plot_loss_curves(
        {"Linear": (losses_linear["train"], losses_linear["val"])},
        save_path=plot_dir / "loss_linear.png",
    )

    plot_loss_curves(
        {"Sigmoid": (losses_sigmoid["train"], losses_sigmoid["val"])},
        save_path=plot_dir / "loss_sigmoid.png",
    )

    plot_loss_curves(
        {
            "Linear": losses_linear["val"],
            "Sigmoid": losses_sigmoid["val"]
        },
        save_path=plot_dir / "loss_comparison.png",
        plot_train=False,
    )


if __name__ == "__main__":
    main()
