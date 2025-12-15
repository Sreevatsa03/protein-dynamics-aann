from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dynamics.masks import make_random_mask
from src.dynamics.ground_truth import build_gt_W
from src.dynamics.simulate import simulate_dt_sigmoid_dynamics, make_transition_dataset
from src.models.masked_aann import LinearAANN
from src.training.trainer import Trainer, TrainingConfig, OptimizerConfig
from src.utils.seed import set_seed
from src.utils.io import save_json


def linear():
    """
    Train a masked linear AANN on the discrete-time synthetic protein dynamics dataset.
    Saves the best checkpoint and logs training/validation losses.
    """
    set_seed(42)

    # generate signed mask
    S = make_random_mask(
        d=12,
        min_deg=3,
        max_deg=5
    )

    # generate ground-truth mask and weights
    W_eff = build_gt_W(
        S=S,
        mu=-1.2,
        sigma=0.6,
        target_radius=0.9,
    )

    # simulate trajectories
    trajectories = simulate_dt_sigmoid_dynamics(
        W_eff=W_eff,
        T=400,
        n_seqs=5,
        alpha=0.9,
        noise_std=0.02,
    )

    # build transition dataset
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = make_transition_dataset(
        trajectories,
        train_p=0.7,
        val_p=0.15,
    )

    # dataloaders
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=64,
        shuffle=True)
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=64,
        shuffle=False)

    # model
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    model = LinearAANN(
        state_dim=12,
        mask=S,
        device=device,
    )

    # configs
    checkpoint = Path("experiments/results/linear/best_model.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    opt_cfg = OptimizerConfig(
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.0,
    )
    train_cfg = TrainingConfig(
        num_epochs=300,
        patience=20,
        device=device,
        checkpoint_path=checkpoint,
    )

    # trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_config=opt_cfg,
        training_config=train_cfg,
    )

    results = trainer.train()

    # save loss curves
    save_json(
        "experiments/results/linear/losses.json",
        {
            "train": results["train_losses"],
            "val": results["val_losses"],
        },
    )

    # report
    print("Linear AANN finished.")
    print(f"Best val loss: {results['best_val_loss']:.6f}")
    print(f"Loss histories saved to: experiments/results/linear/losses.json")
    print(f"Checkpoint saved to: {results['checkpoint']}")


def sigmoid():
    """
    Train a masked sigmoid AANN on the discrete-time synthetic protein dynamics dataset.
    Saves the best checkpoint and logs training/validation losses.
    """
    set_seed(42)

    # generate signed mask
    S = make_random_mask(
        d=12,
        min_deg=3,
        max_deg=5
    )

    # generate ground-truth mask and weights
    W_eff = build_gt_W(
        S=S,
        mu=-1.2,
        sigma=0.6,
        target_radius=0.9,
    )

    # simulate trajectories
    trajectories = simulate_dt_sigmoid_dynamics(
        W_eff=W_eff,
        T=400,
        n_seqs=5,
        alpha=0.9,
        noise_std=0.02,
    )

    # build transition dataset
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = make_transition_dataset(
        trajectories,
        train_p=0.7,
        val_p=0.15,
    )

    # dataloaders
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=64,
        shuffle=True)
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=64,
        shuffle=False)

    # model
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    model = SigmoidAANN(
        state_dim=12,
        mask=S,
        device=device,
    )

    # configs
    checkpoint = Path("experiments/results/sigmoid/best_model.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    opt_cfg = OptimizerConfig(
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.0,
    )
    train_cfg = TrainingConfig(
        num_epochs=300,
        patience=20,
        device=device,
        checkpoint_path=checkpoint,
    )

    # train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_config=opt_cfg,
        training_config=train_cfg,
    )

    results = trainer.train()

    # save loss curves
    save_json(
        "experiments/results/sigmoid/losses.json",
        {
            "train": results["train_losses"],
            "val": results["val_losses"],
        },
    )

    print("Sigmoid AANN finished.")
    print(f"Best val loss: {results['best_val_loss']:.6f}")
    print(f"Loss histories saved to: experiments/results/sigmoid/losses.json")
    print(f"Checkpoint saved to: {results['checkpoint']}")


def main():
    linear()
    sigmoid()


if __name__ == "__main__":
    main()
