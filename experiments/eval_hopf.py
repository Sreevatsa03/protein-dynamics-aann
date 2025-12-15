import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.dynamics.masks import make_cell_cycle_mask
from src.models.masked_aann import LinearAANN, SigmoidAANN
from src.evaluation.metrics import (
    mse,
    per_protein_mse,
    per_protein_r2,
    rollout_mse,
    rollout_mse_vs_time,
    open_loop_rollout,
)
from src.evaluation.plots import (
    plot_per_protein_metrics,
    plot_rollout_mse_vs_time,
    plot_rollout_mse_multi,
    plot_per_protein_r2_multi,
    plot_phase_portrait_comparison,
    plot_multi_trajectory_overlay,
    plot_recovery_distance_vs_iter,
    plot_final_recovery_bar,
    plot_recovery_ratio_bar,
)


class MeanPredictor:
    """Baseline that always outputs the training mean."""

    def __init__(self, mean_vec: torch.Tensor):
        self.mean_vec = mean_vec

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Return mean for each sample in batch
        if x.dim() == 1:
            return self.mean_vec.clone()
        return self.mean_vec.unsqueeze(0).expand(x.shape[0], -1)

    def eval(self):
        pass  # no-op for compatibility


class IdentityPredictor:
    """Baseline that outputs the input unchanged (x_{t+1} = x_t)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    def eval(self):
        pass  # no-op for compatibility


def corrupt_gaussian(
    x_ref: torch.Tensor,
    sigma: float = 0.05,
    rng: np.random.Generator = None,
) -> torch.Tensor:
    """
    Corrupt reference state with Gaussian noise.

    :param x_ref: reference state (state_dim,)
    :param sigma: noise standard deviation
    :param rng: optional random generator
    :return: corrupted state
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = torch.from_numpy(rng.normal(
        0, sigma, x_ref.shape).astype(np.float32))
    return x_ref + noise


def corrupt_masking(
    x_ref: torch.Tensor,
    mean_state: torch.Tensor,
    mask_frac: float = 0.4,
    rng: np.random.Generator = None,
) -> torch.Tensor:
    """
    Corrupt reference state by masking coordinates to dataset mean.

    :param x_ref: reference state (state_dim,)
    :param mean_state: global mean state for masking
    :param mask_frac: fraction of coordinates to mask (0.3-0.5)
    :param rng: optional random generator
    :return: corrupted state
    """
    if rng is None:
        rng = np.random.default_rng()
    d = x_ref.shape[0]
    n_mask = int(d * mask_frac)
    mask_indices = rng.choice(d, size=n_mask, replace=False)

    x_corrupted = x_ref.clone()
    x_corrupted[mask_indices] = mean_state[mask_indices]
    return x_corrupted


def iterative_recovery(
    model,
    x0: torch.Tensor,
    K: int = 100,
) -> torch.Tensor:
    """
    Iterate model K steps from initial state x0.

    :param model: callable that maps x -> x_next
    :param x0: initial state (state_dim,)
    :param K: number of iterations
    :return: trajectory (K+1, state_dim) including x0
    """
    trajectory = [x0.clone()]
    x = x0.clone()

    with torch.no_grad():
        for _ in range(K):
            x = model(x)
            trajectory.append(x.clone())

    return torch.stack(trajectory)


def pattern_completion_eval(
    predictors: dict,
    trajectories: torch.Tensor,
    mean_state: torch.Tensor,
    n_samples_per_traj: int = 5,
    corruption_mode: str = "gaussian",
    sigma: float = 0.05,
    mask_frac: float = 0.4,
    K: int = 100,
    seed: int = 42,
) -> dict:
    """
    Evaluate pattern completion (associative memory) for all predictors.

    :param predictors: dict mapping model names to callable models
    :param trajectories: ground truth trajectories (n_seqs, T, state_dim)
    :param mean_state: global mean state
    :param n_samples_per_traj: number of reference states per trajectory
    :param corruption_mode: "gaussian" or "masking"
    :param sigma: noise std for Gaussian corruption
    :param mask_frac: fraction of coords to mask
    :param K: number of recovery iterations
    :param seed: random seed
    :return: dict with per-model results
    """
    rng = np.random.default_rng(seed)
    n_seqs, T, state_dim = trajectories.shape

    # sample reference states from trajectories
    # skip first 20 steps (transient) and last 20 steps
    start_idx = min(20, T // 4)
    end_idx = max(T - 20, T * 3 // 4)

    all_refs = []
    for seq_idx in range(n_seqs):
        sample_indices = rng.choice(
            range(start_idx, end_idx),
            size=min(n_samples_per_traj, end_idx - start_idx),
            replace=False,
        )
        for t_idx in sample_indices:
            all_refs.append(trajectories[seq_idx, t_idx])

    n_trials = len(all_refs)
    print(
        f"    Pattern completion: {n_trials} trials, K={K} iterations, mode={corruption_mode}")

    results = {name: {
        "dist_to_ref_per_iter": [],  # (n_trials, K+1)
        "dist_to_mean_per_iter": [],  # (n_trials, K+1)
    } for name in predictors.keys()}

    for trial_idx, x_ref in enumerate(all_refs):
        # corrupt the reference
        if corruption_mode == "gaussian":
            x0 = corrupt_gaussian(x_ref, sigma=sigma, rng=rng)
        elif corruption_mode == "masking":
            x0 = corrupt_masking(
                x_ref, mean_state, mask_frac=mask_frac, rng=rng)
        else:
            raise ValueError(f"Unknown corruption_mode: {corruption_mode}")

        for name, model in predictors.items():
            # run iterative recovery
            traj = iterative_recovery(model, x0, K=K)

            # compute distances at each iteration
            dist_to_ref = torch.norm(traj - x_ref, dim=1).numpy()  # (K+1,)
            dist_to_mean = torch.norm(
                traj - mean_state, dim=1).numpy()  # (K+1,)

            results[name]["dist_to_ref_per_iter"].append(dist_to_ref)
            results[name]["dist_to_mean_per_iter"].append(dist_to_mean)

    # aggregate results
    for name in predictors.keys():
        # (n_trials, K+1)
        ref_arr = np.array(results[name]["dist_to_ref_per_iter"])
        # (n_trials, K+1)
        mean_arr = np.array(results[name]["dist_to_mean_per_iter"])

        results[name]["mean_dist_to_ref"] = ref_arr.mean(axis=0)  # (K+1,)
        results[name]["std_dist_to_ref"] = ref_arr.std(axis=0)
        results[name]["mean_dist_to_mean"] = mean_arr.mean(axis=0)
        results[name]["std_dist_to_mean"] = mean_arr.std(axis=0)

        # final distances
        results[name]["final_dist_to_ref"] = float(ref_arr[:, -1].mean())
        results[name]["final_dist_to_mean"] = float(mean_arr[:, -1].mean())

        # recovery ratio: positive means model recovers toward ref, not mean
        results[name]["recovery_ratio"] = (
            results[name]["final_dist_to_mean"] -
            results[name]["final_dist_to_ref"]
        )

    return results


def load_data(data_dir: Path) -> dict:
    """Load all evaluation data from disk."""
    trajectories = torch.load(
        data_dir / "trajectories.pt", weights_only=True, map_location="cpu"
    )
    init_conditions = torch.load(
        data_dir / "init_conditions.pt", weights_only=True, map_location="cpu"
    )
    val_data = torch.load(
        data_dir / "transitions_val.pt", weights_only=True, map_location="cpu"
    )

    with open(data_dir / "meta.json", "r") as f:
        meta = json.load(f)

    return {
        "trajectories": trajectories,
        "init_conditions": init_conditions,
        "X_val": val_data["X"],
        "Y_val": val_data["Y"],
        "meta": meta,
    }


def load_model(model_type: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained AANN model from checkpoint."""
    S, _ = make_cell_cycle_mask()
    state_dim = S.shape[0]

    if model_type == "linear":
        model = LinearAANN(state_dim=state_dim, mask=S, device=device)
    elif model_type == "sigmoid":
        model = SigmoidAANN(state_dim=state_dim, mask=S, device=device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state_dict = torch.load(
        checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def teacher_forced_eval(model, X_val: torch.Tensor, Y_val: torch.Tensor) -> dict:
    """
    Evaluate model in teacher-forced mode (one-step prediction).

    :return: dict with mse, per_protein_mse, per_protein_r2
    """
    with torch.no_grad():
        Y_pred = model(X_val)

    return {
        "mse": mse(Y_val, Y_pred),
        "per_protein_mse": per_protein_mse(Y_val, Y_pred).tolist(),
        "per_protein_r2": per_protein_r2(Y_val, Y_pred).tolist(),
    }


def open_loop_eval(
    model,
    trajectories: torch.Tensor,
    init_conditions: torch.Tensor,
) -> dict:
    """
    Evaluate model in open-loop rollout mode.

    :param model: trained AANN model
    :param trajectories: ground truth trajectories (n_seqs, T, state_dim)
    :param init_conditions: initial conditions (n_seqs, state_dim)
    :return: dict with rollout metrics
    """
    n_seqs, T, state_dim = trajectories.shape

    # rollout each trajectory
    all_mse_vs_time = []
    all_rollout_mse = []
    all_pred_trajs = []

    for i in range(n_seqs):
        x0 = init_conditions[i]
        true_traj = trajectories[i]

        # open-loop rollout
        pred_traj = open_loop_rollout(model, x0, T)
        all_pred_trajs.append(pred_traj)

        # compute metrics
        mse_vs_t = rollout_mse_vs_time(true_traj.numpy(), pred_traj.numpy())
        all_mse_vs_time.append(mse_vs_t)
        all_rollout_mse.append(rollout_mse(true_traj, pred_traj))

    # average across trajectories
    mean_mse_vs_time = np.mean(all_mse_vs_time, axis=0)
    std_mse_vs_time = np.std(all_mse_vs_time, axis=0)

    return {
        "mean_rollout_mse": float(np.mean(all_rollout_mse)),
        "std_rollout_mse": float(np.std(all_rollout_mse)),
        "final_mse": float(mean_mse_vs_time[-1]),
        "mse_vs_time": mean_mse_vs_time,
        "mse_vs_time_std": std_mse_vs_time,
        "pred_trajectories": torch.stack(all_pred_trajs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AANN models on Hopf dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Path to synthetic data directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Path to training results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/hopf_sgd"),
        help="Path to save evaluation outputs",
    )
    parser.add_argument(
        "--n-plot-trajs",
        type=int,
        default=3,
        help="Number of trajectories to plot for geometry comparison",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # load data
    print(f"Loading data from {args.data_dir}...")
    data = load_data(args.data_dir)
    trajectories = data["trajectories"]
    init_conditions = data["init_conditions"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]
    meta = data["meta"]
    proteins = meta.get("proteins", [f"P{i}" for i in range(12)])

    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Validation samples: {X_val.shape[0]}")

    # load training data for computing mean baseline
    train_data = torch.load(
        args.data_dir / "transitions_train.pt", weights_only=True, map_location="cpu"
    )
    Y_train_mean = train_data["Y"].mean(dim=0)
    print(f"  Y_train mean computed: {Y_train_mean.shape}")

    # create baseline predictors
    mean_predictor = MeanPredictor(Y_train_mean)
    identity_predictor = IdentityPredictor()

    # load trained models
    print("Loading trained models...")
    linear_model = load_model(
        "linear",
        args.results_dir / "linear" / "best_model.pt",
        device,
    )
    sigmoid_model = load_model(
        "sigmoid",
        args.results_dir / "sigmoid" / "best_model.pt",
        device,
    )

    # all predictors for unified evaluation
    predictors = {
        "Linear": linear_model,
        "Sigmoid": sigmoid_model,
        "Mean": mean_predictor,
        "Identity": identity_predictor,
    }

    # teacher-forced evaluation
    print("\nTeacher-Forced Evaluation...")

    tf_results = {}
    for name, model in predictors.items():
        tf_results[name] = teacher_forced_eval(model, X_val, Y_val)
        print(f"  {name:10s} MSE: {tf_results[name]['mse']:.6f}, "
              f"mean R²: {np.mean(tf_results[name]['per_protein_r2']):.4f}")

    # plot per-protein R2 for all models
    r2_dict = {name: np.array(res["per_protein_r2"])
               for name, res in tf_results.items()}
    plot_per_protein_r2_multi(
        r2_dict,
        save_path=args.output_dir / "teacher_forced_r2_all.png",
        clip_min=-1.0,
    )

    # also keep legacy 2-model plot for backwards compatibility
    plot_per_protein_metrics(
        np.array(tf_results["Linear"]["per_protein_r2"]),
        np.array(tf_results["Sigmoid"]["per_protein_r2"]),
        metric_name="R²",
        save_path=args.output_dir / "teacher_forced_r2.png",
        clip_min=-1.0,
    )

    # open-loop rollout evaluation
    print("\nOpen-Loop Rollout Evaluation...")

    ol_results = {}
    for name, model in predictors.items():
        ol_results[name] = open_loop_eval(model, trajectories, init_conditions)
        print(f"  {name:10s} mean rollout MSE: {ol_results[name]['mean_rollout_mse']:.6f}, "
              f"final MSE: {ol_results[name]['final_mse']:.6f}")

    # plot rollout MSE vs time for all models
    mse_dict = {name: res["mse_vs_time"] for name, res in ol_results.items()}
    plot_rollout_mse_multi(
        mse_dict,
        save_path=args.output_dir / "rollout_mse_vs_time_all.png",
    )

    # also keep legacy 2-model plot for backwards compatibility
    plot_rollout_mse_vs_time(
        ol_results["Linear"]["mse_vs_time"],
        ol_results["Sigmoid"]["mse_vs_time"],
        save_path=args.output_dir / "rollout_mse_vs_time.png",
    )

    # geometry preservation plots
    print(
        f"\nGeometry Preservation (plotting {args.n_plot_trajs} trajectories)...")

    # cyclin indices for phase portraits
    cyclin_idx = {"CycA": 8, "CycB": 10, "CycD": 4, "CycE": 7}

    for traj_idx in range(min(args.n_plot_trajs, trajectories.shape[0])):
        true_traj = trajectories[traj_idx].numpy()
        linear_traj = ol_results["Linear"]["pred_trajectories"][traj_idx].numpy(
        )
        sigmoid_traj = ol_results["Sigmoid"]["pred_trajectories"][traj_idx].numpy(
        )
        T = true_traj.shape[0]
        t = np.arange(T)

        # phase portrait: CycA vs CycB
        plot_phase_portrait_comparison(
            true_traj,
            linear_traj,
            sigmoid_traj,
            protein_i=cyclin_idx["CycA"],
            protein_j=cyclin_idx["CycB"],
            protein_names=proteins,
            save_path=args.output_dir /
            f"phase_portrait_traj{traj_idx}_CycA_CycB.png",
        )

        # trajectory overlay: CycB
        plot_multi_trajectory_overlay(
            t,
            true_traj,
            linear_traj,
            sigmoid_traj,
            protein_idx=cyclin_idx["CycB"],
            protein_name="CycB",
            save_path=args.output_dir / f"overlay_traj{traj_idx}_CycB.png",
        )

        # trajectory overlay: CycA
        plot_multi_trajectory_overlay(
            t,
            true_traj,
            linear_traj,
            sigmoid_traj,
            protein_idx=cyclin_idx["CycA"],
            protein_name="CycA",
            save_path=args.output_dir / f"overlay_traj{traj_idx}_CycA.png",
        )

    # ========================================================================
    # Pattern Completion (Associative Memory) Evaluation
    # ========================================================================
    print("\nPattern Completion (Associative Memory) Evaluation...")

    # exclude Identity predictor for pattern completion (it doesn't iterate meaningfully)
    pc_predictors = {
        "Linear": linear_model,
        "Sigmoid": sigmoid_model,
        "Mean": mean_predictor,
    }

    pc_output_dir = Path("experiments/results/eval/pattern_completion")
    pc_output_dir.mkdir(parents=True, exist_ok=True)

    # run both corruption modes
    pc_results_all = {}

    for corruption_mode in ["gaussian", "masking"]:
        print(f"\n  Corruption mode: {corruption_mode}")

        pc_results = pattern_completion_eval(
            predictors=pc_predictors,
            trajectories=trajectories,
            mean_state=Y_train_mean,
            n_samples_per_traj=5,
            corruption_mode=corruption_mode,
            sigma=0.05,
            mask_frac=0.4,
            K=100,
            seed=42,
        )

        pc_results_all[corruption_mode] = pc_results

        # print summary
        print(f"\n    {corruption_mode.upper()} Corruption Results:")
        for name in pc_predictors.keys():
            res = pc_results[name]
            print(
                f"      {name:10s} | final dist_to_ref: {res['final_dist_to_ref']:.4f}, "
                f"dist_to_mean: {res['final_dist_to_mean']:.4f}, "
                f"recovery_ratio: {res['recovery_ratio']:+.4f}"
            )

        # plot recovery distance vs iteration
        dist_to_ref_dict = {name: res["mean_dist_to_ref"]
                            for name, res in pc_results.items()}
        dist_to_ref_std_dict = {name: res["std_dist_to_ref"]
                                for name, res in pc_results.items()}
        plot_recovery_distance_vs_iter(
            dist_to_ref_dict,
            dist_to_ref_std_dict,
            save_path=pc_output_dir /
            f"recovery_dist_vs_iter_{corruption_mode}.png",
        )

        # plot final recovery bar
        final_dist_to_ref = {name: res["final_dist_to_ref"]
                             for name, res in pc_results.items()}
        final_dist_to_mean = {name: res["final_dist_to_mean"]
                              for name, res in pc_results.items()}
        plot_final_recovery_bar(
            final_dist_to_ref,
            final_dist_to_mean,
            save_path=pc_output_dir /
            f"final_recovery_bar_{corruption_mode}.png",
        )

        # plot recovery ratio bar
        recovery_ratios = {name: res["recovery_ratio"]
                           for name, res in pc_results.items()}
        plot_recovery_ratio_bar(
            recovery_ratios,
            save_path=pc_output_dir /
            f"recovery_ratio_bar_{corruption_mode}.png",
        )

    print(f"\n  Pattern completion plots saved to {pc_output_dir}")

    # save summary
    summary = {
        "dataset": {
            "seed": meta.get("seed"),
            "n_seqs": meta.get("n_seqs"),
            "T_obs": meta.get("T_obs"),
            "dt": meta.get("dt"),
        },
        "teacher_forced": {},
        "open_loop": {},
        "pattern_completion": {},
    }

    for name in predictors.keys():
        summary["teacher_forced"][name] = {
            "mse": tf_results[name]["mse"],
            "mean_r2": float(np.mean(tf_results[name]["per_protein_r2"])),
            "per_protein_r2": tf_results[name]["per_protein_r2"],
        }
        summary["open_loop"][name] = {
            "mean_rollout_mse": ol_results[name]["mean_rollout_mse"],
            "std_rollout_mse": ol_results[name]["std_rollout_mse"],
            "final_mse": ol_results[name]["final_mse"],
        }

    # add pattern completion results to summary
    for corruption_mode, pc_results in pc_results_all.items():
        summary["pattern_completion"][corruption_mode] = {}
        for name, res in pc_results.items():
            summary["pattern_completion"][corruption_mode][name] = {
                "final_dist_to_ref": res["final_dist_to_ref"],
                "final_dist_to_mean": res["final_dist_to_mean"],
                "recovery_ratio": res["recovery_ratio"],
            }

    with open(args.output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # also save pattern completion summary separately
    with open(pc_output_dir / "pattern_completion_summary.json", "w") as f:
        pc_summary = {}
        for corruption_mode, pc_results in pc_results_all.items():
            pc_summary[corruption_mode] = {}
            for name, res in pc_results.items():
                pc_summary[corruption_mode][name] = {
                    "final_dist_to_ref": res["final_dist_to_ref"],
                    "final_dist_to_mean": res["final_dist_to_mean"],
                    "recovery_ratio": res["recovery_ratio"],
                }
        json.dump(pc_summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
