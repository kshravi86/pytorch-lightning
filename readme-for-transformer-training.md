Training Transformer with PyTorch Lightning
==========================================

This repository includes `transformer_training.py`, a minimal example that trains a
Transformer encoder on the AG_NEWS text classification dataset using the
HuggingFace Datasets library.

## Installation

1. Install the required packages:
   ```bash
   pip install torch datasets pytorch-lightning
   ```

## Training

2. Run the training script:
   ```bash
   python transformer_training.py --batch-size 64 --max-epochs 5
   ```
   The script automatically downloads the AG_NEWS dataset and reports
   training and validation metrics.

Adjust the command-line arguments to modify hyperparameters such as the
batch size or number of epochs.
