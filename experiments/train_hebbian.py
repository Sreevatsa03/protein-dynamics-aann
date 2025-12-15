import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dynamics.masks import make_cell_cycle_mask
from src.models.masked_aann import LinearAANN, SigmoidAANN
from src.training.hebbian import HebbianTrainer, evaluate_model
from src.utils.seed import set_seed


def load_data(data_dir: Path) -> tuple[TensorDataset, TensorDataset, dict]:
    """Load pre-generated transition data from disk."""
    train_data = torch.load(
        data_dir / "transitions_train.pt", weights_only=True, map_location="cpu"
    )
    val_data = torch.load(
        data_dir / "transitions_val.pt", weights_only=True, map_location="cpu"
    )

    with open(data_dir / "meta.json", "r") as f:
        meta = json.load(f)

    train_dataset = TensorDataset(train_data["X"], train_data["Y"])
    val_dataset = TensorDataset(val_data["X"], val_data["Y"])

    return train_dataset, val_dataset, meta


def train_model(
    model_type: str,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    meta: dict,
    output_dir: Path,
    seed: int = 42,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_epochs: int = 500,
    patience: int = 50,
    update_bias: bool = True,
) -> dict:
    """
    Train a single AANN model using Hebbian learning.

    :param model_type: "linear" or "sigmoid"
    :param train_dataset: training TensorDataset
    :param val_dataset: validation TensorDataset
    :param meta: dataset metadata dict
    :param output_dir: directory to save results
    :param seed: random seed
    :param batch_size: training batch size
    :param lr: Hebbian learning rate β
    :param num_epochs: maximum training epochs
    :param patience: early stopping patience
    :param update_bias: whether to update bias during training
    :return: training results dict
    """
    set_seed(seed)

    device = torch.device("cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    # data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # load connectivity mask
    S, proteins = make_cell_cycle_mask()
    state_dim = S.shape[0]

    # construct model
    if model_type == "linear":
        model = LinearAANN(state_dim=state_dim, mask=S, device=device)
    elif model_type == "sigmoid":
        model = SigmoidAANN(state_dim=state_dim, mask=S, device=device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Training {model_type} AANN with Hebbian learning...")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Learning rate β: {lr}")
    print(f"  Update bias: {update_bias}")

    # train with Hebbian learning
    trainer = HebbianTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        update_bias=update_bias,
        patience=patience,
    )

    results = trainer.train(num_epochs=num_epochs)

    # save model checkpoint
    torch.save(model.state_dict(), output_dir / "best_model.pt")

    # save loss curves
    losses = {
        "train_losses": results["train_losses"],
        "val_losses": results["val_losses"],
        "val_r2s": results["val_r2s"],
    }
    with open(output_dir / "losses.json", "w") as f:
        json.dump(losses, f, indent=2)

    # final evaluation
    final_metrics = evaluate_model(model, val_loader)

    # save summary
    summary = {
        "model_type": model_type,
        "learning_method": "hebbian",
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "update_bias": update_bias,
        "num_epochs": num_epochs,
        "patience": patience,
        "epochs_trained": results["epochs_trained"],
        "final_train_loss": results["train_losses"][-1],
        "final_val_loss": results["val_losses"][-1],
        "final_val_r2": results["val_r2s"][-1],
        "best_val_loss": results["best_val_loss"],
        "per_protein_r2": final_metrics["per_protein_r2"],
        "dataset_seed": meta.get("seed"),
        "dataset_dt": meta.get("dt"),
        "dataset_T_obs": meta.get("T_obs"),
        "dataset_n_seqs": meta.get("n_seqs"),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Epochs trained: {results['epochs_trained']}")
    print(f"  Best val loss: {results['best_val_loss']:.6f}")
    print(f"  Final val R2: {results['val_r2s'][-1]:.4f}")
    print(f"  Results saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train AANN models with Hebbian learning"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "sigmoid", "both"],
        default="both",
        help="Model type to train",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Path to synthetic data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/hebbian"),
        help="Base output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Hebbian learning rate β")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--no-bias-update",
        action="store_true",
        help="Disable bias updates during training",
    )
    args = parser.parse_args()

    # load data once
    print(f"Loading data from {args.data_dir}...")
    train_dataset, val_dataset, meta = load_data(args.data_dir)

    models_to_train = (
        ["linear", "sigmoid"] if args.model == "both" else [args.model]
    )

    for model_type in models_to_train:
        output_dir = args.output_dir / model_type
        train_model(
            model_type=model_type,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            meta=meta,
            output_dir=output_dir,
            seed=args.seed,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.epochs,
            patience=args.patience,
            update_bias=not args.no_bias_update,
        )


if __name__ == "__main__":
    main()
