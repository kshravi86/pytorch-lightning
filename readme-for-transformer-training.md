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

   The trained model checkpoint is saved to `transformer_model.ckpt` by default. Specify a custom path with `--save-path` if desired:
   ```bash
   python transformer_training.py --batch-size 64 --max-epochs 5 --save-path transformer_model.ckpt
   ```
   This checkpoint path can be used directly for inference.

Adjust the command-line arguments to modify hyperparameters such as the
batch size or number of epochs.
