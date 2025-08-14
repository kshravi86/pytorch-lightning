"""Train a Transformer-based text classifier using PyTorch Lightning.

This script trains a small Transformer encoder on the AG_NEWS dataset
using the HuggingFace Datasets library. It serves as a minimal example
of how to use Lightning for natural language processing with Transformer
models.
"""

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

# Allow running this file without installing the package
sys.path.append(str(Path(__file__).parent / "src"))

import lightning.pytorch as L
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset


class AGNewsDataModule(L.LightningDataModule):
    """DataModule for the AG_NEWS dataset."""

    def __init__(self, batch_size: int = 64, max_vocab: int = 20_000):
        super().__init__()
        self.batch_size = batch_size
        self.max_vocab = max_vocab
        self.tokenizer = lambda text: re.findall(r"\w+", text.lower())

    def prepare_data(self) -> None:  # type: ignore[override]
        load_dataset("ag_news")

    def setup(self, stage: str | None = None) -> None:  # type: ignore[override]
        dataset = load_dataset("ag_news")
        train_ds = dataset["train"]
        test_ds = dataset["test"]

        counter = Counter()
        for sample in train_ds:
            counter.update(self.tokenizer(sample["text"]))

        self.itos = ["<pad>", "<unk>"] + [tok for tok, _ in counter.most_common(self.max_vocab - 2)]
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}
        self.num_classes = 4
        self.vocab_size = len(self.itos)

        def encode(text: str) -> torch.Tensor:
            unk_idx = self.stoi["<unk>"]
            return torch.tensor([self.stoi.get(tok, unk_idx) for tok in self.tokenizer(text)], dtype=torch.long)

        self.train_data = [(encode(item["text"]), item["label"]) for item in train_ds]
        self.val_data = [(encode(item["text"]), item["label"]) for item in test_ds]

    def collate_fn(self, batch):
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        sequences = [seq for seq, _ in batch]
        seq_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        pad_mask = seq_padded == 0
        return seq_padded, labels, pad_mask

    def train_dataloader(self):  # type: ignore[override]
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):  # type: ignore[override]
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_fn)


class TransformerClassifier(L.LightningModule):
    """A simple Transformer encoder for text classification."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, pad_mask):
        x = self.embedding(x) * math.sqrt(self.hparams.embed_dim)
        x = x.transpose(0, 1)  # (seq_len, batch, embed)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = x.mean(dim=0)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y, pad_mask = batch
        logits = self(x, pad_mask)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, pad_mask = batch
        logits = self(x, pad_mask)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer text classifier with Lightning")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--max-epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    dm = AGNewsDataModule(batch_size=args.batch_size)
    dm.prepare_data()
    dm.setup("fit")
    model = TransformerClassifier(vocab_size=dm.vocab_size, num_classes=dm.num_classes)
    trainer = L.Trainer(max_epochs=args.max_epochs)
    trainer.fit(model, dm)
