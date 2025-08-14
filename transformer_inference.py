"""Run inference with a Transformer model trained using PyTorch Lightning.

This script loads a saved model checkpoint from the default Lightning
checkpoint directory (`lightning_logs/version_0/checkpoints/`) and runs
predictions for the provided texts.
"""

import argparse
import re
from typing import List

import torch
from torch import nn

from transformer_training import AGNewsDataModule, TransformerClassifier


def encode_text(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """Tokenize and numericalize a text string using the provided vocabulary."""
    tokens = re.findall(r"\w+", text.lower())
    unk_idx = stoi.get("<unk>", 0)
    return torch.tensor([stoi.get(tok, unk_idx) for tok in tokens], dtype=torch.long)


def predict_texts(model: nn.Module, texts: List[str], stoi: dict[str, int]) -> None:
    """Print model predictions for each input text."""
    model.eval()
    for text in texts:
        tokens = encode_text(text, stoi).unsqueeze(0)
        pad_mask = tokens == 0
        logits = model(tokens, pad_mask)
        pred = torch.argmax(logits, dim=-1).item()
        print(f"Input: {text}")
        print(f"Logits: {logits.tolist()[0]}")
        print(f"Predicted class: {pred}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer text classifier inference")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="lightning_logs/version_0/checkpoints/last.ckpt",
        help="Path to a trained model checkpoint",
    )
    parser.add_argument("--texts", nargs="+", help="One or more texts to classify")
    args = parser.parse_args()

    dm = AGNewsDataModule(batch_size=1)
    dm.prepare_data()
    dm.setup("predict")

    model = TransformerClassifier.load_from_checkpoint(
        args.checkpoint_path, vocab_size=dm.vocab_size, num_classes=dm.num_classes
    )

    predict_texts(model, args.texts, dm.stoi)
