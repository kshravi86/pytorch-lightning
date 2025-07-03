# ⚡ Lightning: Supercharge Your PyTorch AI Development! 🚀

Welcome to the Lightningverse! 🌌 Lightning is your go-to framework for pretraining, finetuning, and deploying AI models with PyTorch, making your life easier and your code cleaner! Whether you're a seasoned researcher or just starting, Lightning helps you focus on the science, not the boilerplate.

This guide will give you a colorful and insightful overview of how Lightning is structured, how you can use its powerful components, and what amazing things you can build with it! ✨

## 🏗️ Code Structure: What's Inside the Box? 🎁

Lightning is ingeniously designed to offer flexibility and power. It primarily consists of two core packages:

1.  **<img src="https://raw.githubusercontent.com/Lightning-AI/lightning/master/docs/source-pytorch/_static/images/logo.svg" width="20" height="20" alt="PyTorch Lightning logo"> PyTorch Lightning (`src/lightning/pytorch/`)**:
    *   The OG! 👑 PyTorch Lightning (PL) is all about organized PyTorch. It introduces the `LightningModule` (your model's blueprint) and the `Trainer` (which handles all the engineering heavy-lifting like training loops, hardware interactions, and scaling).
    *   Think of it as PyTorch on autopilot for most common research workflows. You define the what (your model, data, and optimization), and PL handles the how.
    *   Key sub-directories:
        *   `core/`: Fundamental building blocks like `LightningModule`.
        *   `trainer/`: The powerful `Trainer` class.
        *   `callbacks/`: Reusable components for tasks like early stopping, model checkpointing.
        *   `loggers/`: Integrations with popular logging frameworks (TensorBoard, W&B, etc.).
        *   `strategies/`: Code for different training strategies (DDP, FSDP, etc.).

2.  **<img src="https://raw.githubusercontent.com/Lightning-AI/lightning/master/docs/source-fabric/_static/images/logo.svg" width="20" height="20" alt="Fabric logo"> Lightning Fabric (`src/lightning/fabric/`)**:
    *   For the experts who want more control! 🎛️ Fabric gives you the power to scale your PyTorch models to any hardware (CPUs, GPUs, TPUs, multi-node) with minimal code changes, *without* hiding your training loop.
    *   You keep your PyTorch training loop, and Fabric supercharges it with features like distributed training, mixed precision, and optimized checkpointing.
    *   Key components:
        *   `fabric.py`: The main `Fabric` class that you'll interact with.
        *   `accelerators/`, `strategies/`, `plugins/`: Modules that enable Fabric's magic across different hardware and scaling setups.

**Other Important Directories:**

*   **🧪 `examples/`**: A treasure trove of practical implementations!
    *   `examples/pytorch/`: Shows how to use PyTorch Lightning for various tasks.
    *   `examples/fabric/`: Demonstrates the power and flexibility of Lightning Fabric.
    *   Perfect for finding starter code or inspiration for your next project!
*   **📚 `docs/`**: Your comprehensive guide to everything Lightning.
    *   `docs/source-pytorch/`: In-depth documentation for PyTorch Lightning.
    *   `docs/source-fabric/`: Detailed guides and API references for Lightning Fabric.
    *   When in doubt, check the docs!

## 🛠️ How to Use Lightning: Get Started in a Flash! ⚡

Lightning is designed to be intuitive. Here’s a glimpse of how you can get started:

### 1. PyTorch Lightning: The Batteries-Included Way 🔋

Organize your PyTorch code into a `LightningModule` and let the `Trainer` do the rest!

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning as L

# 1️⃣ Define your LightningModule (your model, data processing, and optimization logic)
class LitAutoEncoder(L.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters() # So you can load it later!
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss) # Easy logging!
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

# 2️⃣ Prepare your data (Standard PyTorch DataLoaders)
train_dataset = torchvision.datasets.MNIST(".", download=True, transform=torchvision.transforms.ToTensor())
# For a real case, split your data or use a separate validation dataset!
# For this example, we'll just reuse the train_dataset for validation.
val_dataset = torchvision.datasets.MNIST(".", download=True, transform=torchvision.transforms.ToTensor()) # Reusing for simplicity
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)


# 3️⃣ Train with the Trainer!
model = LitAutoEncoder()
trainer = L.Trainer(max_epochs=10, accelerator="auto", devices="auto") # Handles GPUs, TPUs, etc. automatically!
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

print("🎉 Training Complete! 🎉")
```

### 2. Lightning Fabric: Your Code, Scaled Up! 📈

Keep your PyTorch code and use Fabric to easily add distributed training, mixed precision, and more.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import lightning as L

# Your standard PyTorch model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10) # MNIST images are 28x28 = 784 pixels

    def forward(self, x):
        # Raw logits are often preferred for loss functions like CrossEntropyLoss
        return self.linear(x.view(x.size(0), -1))

# 1️⃣ Initialize Fabric (choose your accelerator, strategy, etc.)
fabric = L.Fabric(accelerator="auto", devices="auto", strategy="auto") # Simpler setup
fabric.launch() # Important for distributed settings

# 2️⃣ Setup your model and optimizer with Fabric
model = MyModel()
optimizer = optim.Adam(model.parameters())
model, optimizer = fabric.setup(model, optimizer)

# 3️⃣ Setup your DataLoader with Fabric
train_dataset = torchvision.datasets.MNIST(".", download=True, transform=torchvision.transforms.ToTensor())
# In a real scenario, you'd have a separate validation dataloader too.
# For simplicity, we'll just use the training dataloader here.
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
dataloader = fabric.setup_dataloaders(dataloader)

# Your PyTorch training loop, with Fabric's magic ✨
model.train()
for epoch in range(3): # Loop over epochs
    for batch_idx, (data, target) in enumerate(dataloader): # Loop over batches
        optimizer.zero_grad()
        output = model(data) # Forward pass
        loss = F.cross_entropy(output, target) # Calculate loss (CrossEntropyLoss expects raw logits)
        fabric.backward(loss) # Fabric handles backward pass (e.g., for mixed precision)
        optimizer.step() # Update weights
        if batch_idx % 100 == 0: # Log progress
            fabric.print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

print("🚀 Fabric Training Done! 🚀")
```

## 🎯 Common Use Cases: What Can You Build? 🌍

Lightning is versatile! Here are some common tasks where it shines, with links to get you started:

*   🖼️ **Image Classification/Vision Tasks**:
    *   Train state-of-the-art vision models, finetune backbones, or build complex image processing pipelines.
    *   Example: [Image Classifier (Fabric)](examples/fabric/image_classifier/README.md), [Computer Vision Fine-tuning (PyTorch)](examples/pytorch/domain_templates/computer_vision_fine_tuning.py)
*   🗣️ **Natural Language Processing (NLP)**:
    *   Finetune transformers, build text classifiers, summarizers, translation models, and more.
    *   Example: [Language Model (Fabric)](examples/fabric/language_model/README.md), [Transformer (PyTorch Basics)](examples/pytorch/basics/transformer.py)
*   🎭 **Generative Models (GANs, VAEs, Diffusion)**:
    *   Easily structure complex training loops for GANs or other generative architectures.
    *   Example: [DCGAN (Fabric)](examples/fabric/dcgan/README.md), [Autoencoder (PyTorch Basics)](examples/pytorch/basics/autoencoder.py)
*   🎮 **Reinforcement Learning (RL)**:
    *   Implement RL agents and manage their training environments. Fabric can be particularly useful here for custom loops.
    *   Example: [Reinforcement Learning (Fabric)](examples/fabric/reinforcement_learning/README.md), [Reinforce Learn QNet (PyTorch)](examples/pytorch/domain_templates/reinforce_learn_Qnet.py)
*   📈 **Time Series Analysis & Forecasting**:
    *   Build models for predicting future values from sequential data.
*   🧩 **Self-Supervised Learning**:
    *   Implement cutting-edge self-supervised methods with ease.
    *   Example: [Meta Learning (Fabric)](examples/fabric/meta_learning/README.md)
*   🔬 **Any Custom Research!**:
    *   If it's in PyTorch, you can make it cleaner with PyTorch Lightning or scale it with Fabric.

## 🚀 Advanced Features: Power Up! 🔋⚡

Lightning comes packed with features to accelerate your research and production workflows:

*   **Multi-GPU & TPU Training**: Scale across multiple devices and nodes with minimal code changes. 🤯 (`Trainer(devices=8, num_nodes=4)`)
*   **Mixed Precision (16-bit, bfloat16)**: Train faster and use less memory. 💨 (`Trainer(precision="16-mixed")` or `Fabric(precision="bf16-mixed")`)
*   **Flexible Checkpointing**: Save and load your models, experiments, and states easily. 💾
*   **Powerful Logging**: Integrate with TensorBoard, Weights & Biases, MLflow, and more. 📊
*   **Model Pruning & Quantization**: Optimize your models for deployment. ✂️
*   **And much more!** Explore the [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/) and [Lightning Fabric Docs](https://lightning.ai/docs/fabric/stable/) for all the goodies.

## 🤗 Community & Contribution: Join the Movement! 🤝

Lightning has a vibrant and active community!

*   **Get Help**: Have questions? Ask on [GitHub Discussions](https://github.com/Lightning-AI/lightning/discussions) or join the [Discord Server](https://discord.com/invite/tfXFetEZxv).
*   **Contribute**: Want to make Lightning even better? Check out the [Contribution Guide](.github/CONTRIBUTING.md) and become a part of the Lightning family! We welcome contributions of all sizes.

---

We hope this colorful guide helps you understand the power and beauty of Lightning! Now go forth and build amazing AI! 🌟
Happy Coding! 🎉
