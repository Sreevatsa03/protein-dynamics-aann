import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json

from src.models.masked_aann import SigmoidAANN
from src.dynamics.masks import make_cell_cycle_mask
from src.training.trainer import Trainer, OptimizerConfig, TrainingConfig
from src.utils.seed import set_seed


def train_model(
    data_dir: Path,
    output_dir: Path,
    n_epochs: int = 1000,
    lr: float = 0.01,
    batch_size: int = 128,
    patience: int = 50,
    seed: int = 42,
):
    """Train a sigmoid AANN on the specified dataset."""

    set_seed(seed)
    device = torch.device("cpu")

    # load data
    train_data = torch.load(
        data_dir / "transitions_train.pt", weights_only=True)
    val_data = torch.load(data_dir / "transitions_val.pt", weights_only=True)

    train_dataset = TensorDataset(train_data["X"], train_data["Y"])
    val_dataset = TensorDataset(val_data["X"], val_data["Y"])

    # data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # create model
    mask, _ = make_cell_cycle_mask()
    model = SigmoidAANN(state_dim=12, mask=mask, device=device)

    # training config
    output_dir.mkdir(parents=True, exist_ok=True)
    opt_cfg = OptimizerConfig(lr=lr, momentum=0.9, weight_decay=0.0)
    train_cfg = TrainingConfig(
        num_epochs=n_epochs,
        patience=patience,
        device=device,
        checkpoint_path=output_dir / "best_model.pt",
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
    losses = {
        "train": results["train_losses"],
        "val": results["val_losses"],
    }

    with open(output_dir / "losses.json", "w") as f:
        json.dump(losses, f, indent=2)

    summary = {
        "best_val_loss": results["best_val_loss"],
        "final_train_loss": results["train_losses"][-1],
        "final_val_loss": results["val_losses"][-1],
        "n_epochs": len(results["train_losses"]),
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    base_data_dir = Path("data/relaxation_sweep")
    base_output_dir = Path("experiments/results/relaxation_sweep")

    basin_strengths = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("=" * 60)
    print("Training Relaxation Sweep Models")
    print("=" * 60)

    all_summaries = {}

    for basin_strength in basin_strengths:
        print(f"\n{'='*60}")
        print(f"Training on basin_strength={basin_strength:.1f}")
        print(f"{'='*60}")

        data_dir = base_data_dir / f"basin_{basin_strength:.1f}"
        output_dir = base_output_dir / f"basin_{basin_strength:.1f}"

        summary = train_model(
            data_dir=data_dir,
            output_dir=output_dir,
            n_epochs=1000,
            lr=0.01,
            batch_size=128,
            patience=50,
            seed=42,
        )

        all_summaries[f"basin_{basin_strength:.1f}"] = summary

        print(f"\nBest val loss: {summary['best_val_loss']:.6f}")
        print(f"Epochs trained: {summary['n_epochs']}")

    # save overall summary
    with open(base_output_dir / "training_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Summary saved to: {base_output_dir}/training_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
