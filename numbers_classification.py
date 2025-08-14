"""Simple MNIST number classifier with mathematical explanations.

This script demonstrates how a small neural network is trained to
classify handwritten digits.  It prints out the mathematical
operations that happen under the hood so that anybody running the file
can follow along.
"""

import argparse
import sys
from pathlib import Path

# Allow running this file without installing the package
sys.path.append(str(Path(__file__).parent / "src"))

import lightning.pytorch as L
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class MNISTClassifier(L.LightningModule):
    """Basic fully connected network for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 10),
        )
        self.criterion = nn.CrossEntropyLoss()
        # We only print explanations for the first training batch
        self.example_shown = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        if not self.example_shown:
            print("\n=== Training step math explanation ===")
            print("Input tensor shape:", x.shape)
            print(
                "Linear layer computes y = W x + b. ReLU applies max(0, y)."
            )
            print(
                "Cross-entropy loss: L = -sum(y_true * log(softmax(logits)))."
            )
            print(
                "Gradient descent updates parameters: theta = theta - lr * dL/dtheta."
            )
            self.example_shown = True

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)


class RandomDigitsDataset(Dataset):
    """Synthetic dataset mimicking MNIST digits."""

    def __init__(self, length: int):
        self.data = torch.rand(length, 1, 28, 28)
        self.targets = torch.randint(0, 10, (length,))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.data[idx], self.targets[idx]


def create_dataloaders(batch_size: int = 64):
    train_ds = RandomDigitsDataset(length=800)
    val_ds = RandomDigitsDataset(length=200)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def explain_model(model: nn.Module):
    """Print a human-readable explanation of the model and mathematics."""

    print("=== Model Architecture ===")
    print(model)
    print("\n=== Math behind the model ===")
    print("1. Each Linear layer computes y = W x + b")
    print("2. ReLU applies σ(z) = max(0, z)")
    print(
        "3. Dropout randomly sets activations to zero during training to prevent overfitting"
    )
    print(
        "4. Cross-entropy loss: L = -sum(y_true * log(softmax(z))) where z are the logits"
    )
    print(
        "5. Adam optimizer performs gradient descent with adaptive learning rates"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST number classification demo")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run a single batch")
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    train_loader, val_loader = create_dataloaders()
    print(
        "Using synthetic dataset with random 28x28 images representing digits 0-9."
    )
    print(
        f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}"
    )
    model = MNISTClassifier()
    explain_model(model)
    trainer = L.Trainer(max_epochs=args.max_epochs, fast_dev_run=args.fast_dev_run)
    trainer.fit(model, train_loader, val_loader)
