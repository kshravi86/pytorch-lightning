# Numbers Classification Training

This example trains a simple neural network to classify handwritten digits from the MNIST dataset using PyTorch Lightning.

## Requirements

- Python with `torch`, `torchvision`, and `lightning` installed. Install them with:

```bash
pip install torch torchvision lightning
```

## Run

Execute the training script:

```bash
python numbers_classification.py
```

The script will download the MNIST dataset (if necessary) and train for five epochs, reporting validation loss and accuracy.
