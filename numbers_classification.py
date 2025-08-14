import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTClassifier(L.LightningModule):
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

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)


def create_dataloaders(batch_size: int = 64):
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders()
    model = MNISTClassifier()
    trainer = L.Trainer(max_epochs=5)
    trainer.fit(model, train_loader, val_loader)
