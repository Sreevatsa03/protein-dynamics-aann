from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dynamics.simulate import make_transition_dataset
from src.dynamics.masks import make_signed_mask
from src.models.masked_aann import SigmoidAANN
from src.training.trainer import Trainer, TrainingConfig, OptimizerConfig
from src.utils.seed import set_seed


def main():
    """
    Train a masked sigmoid AANN on the synthetic protein dynamics dataset.
    Saves the best checkpoint and logs training/validation losses.
    """
    set_seed(42)

    # paths
    checkpoint = Path("experiments/results/sigmoid/best_model.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # data generation
    x_t, x_t1, mask = make_transition_dataset(
        state_dim=12,
        num_sequences=5,
        timesteps=400,
        alpha=0.9,
        noise_std=0.02,
        mask_fn=make_signed_mask,
    )

    # split dataset
    N = x_t.shape[0]
    n_train = int(0.7 * N)
    n_val = int(0.15 * N)

    x_train, y_train = x_t[:n_train], x_t1[:n_train]
    x_val, y_val = x_t[n_train:n_train + n_val], x_t1[n_train:n_train + n_val]

    # dataloaders
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=64,
        shuffle=False,
    )

    # model
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    model = SigmoidAANN(
        state_dim=12,
        mask=mask,
        device=device,
    )

    # configs
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

    print("sigmoid AANN finished.")
    print(f"best val loss: {results['best_val_loss']:.6f}")
    print(f"checkpoint saved to: {results['checkpoint']}")


if __name__ == "__main__":
    main()
