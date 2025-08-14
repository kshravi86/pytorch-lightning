# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cats and dogs classifier example using CIFAR10.

This script trains a simple convolutional neural network to distinguish
between cats and dogs from the CIFAR10 dataset and saves a checkpoint.
To run: ``python cats_dogs_classifier.py --trainer.max_epochs=5``
"""

from os import path
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

from lightning.pytorch import LightningDataModule, LightningModule, Trainer, cli_lightning_logo

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")


class CIFAR10CatDog(CIFAR10):
    def __init__(self, root: str, train: bool = True, transform: Optional[transforms.Compose] = None, download: bool = False):
        super().__init__(root, train=train, transform=transform, download=download)
        idx = [i for i, t in enumerate(self.targets) if t in (3, 5)]
        self.data = self.data[idx]
        targets = [self.targets[i] for i in idx]
        self.targets = [0 if t == 3 else 1 for t in targets]


class CatsDogsDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def prepare_data(self) -> None:  # download if needed
        CIFAR10CatDog(DATASETS_PATH, train=True, download=True)
        CIFAR10CatDog(DATASETS_PATH, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        cifar_full = CIFAR10CatDog(DATASETS_PATH, train=True, transform=self.transform)
        self.train_set, self.val_set = random_split(
            cifar_full, [len(cifar_full) - 1000, 1000], generator=torch.Generator().manual_seed(42)
        )
        self.test_set = CIFAR10CatDog(DATASETS_PATH, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size)


class LitCatsDogs(LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def run_training() -> None:
    data = CatsDogsDataModule()
    model = LitCatsDogs()
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)
    trainer.save_checkpoint("cats_dogs_classifier.ckpt")


if __name__ == "__main__":
    cli_lightning_logo()
    run_training()
