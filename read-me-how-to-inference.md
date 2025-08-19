# Transformer Inference

This document shows how to run the `transformer_inference.py` script to perform text classification with a trained Transformer model.

## Requirements

Install the required packages if they are not already present:

```bash
pip install torch datasets pytorch-lightning
```

Use the checkpoint saved during training (by default `transformer_model.ckpt`).

## Run

Provide one or more texts to classify:

```bash
python transformer_inference.py --checkpoint-path transformer_model.ckpt --texts "A sample news headline" "Another headline"
```

The script will print the model's logits and the predicted class for each provided text.
