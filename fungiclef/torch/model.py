import torch
import pytorch_lightning as pl
import torch.nn as nn
from fungiclef.config import get_device


class DINOv2LightningModel(pl.LightningModule):
    """PyTorch Lightning module for training a classifier on pre-extracted embeddings from parquet files."""

    def __init__(
        self,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.model_device = get_device()
        self.num_classes = 2427  # total fungi species, 0 to 2426 category ids
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.emb_dim = 768
        self.hidden_dim = 1363  # geometric mean of 768 and 2427

        # Trainable Logistic Regression head
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.emb_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.num_classes),
        # )

        self.classifier = nn.Linear(self.emb_dim, self.num_classes)

    def forward(self, embeddings):
        """Extract embeddings using the [CLS] token."""
        # embeddings = embeddings.to(self.model_device)
        # # forward pass
        # logits = self.classifier(embeddings)
        return self.classifier(embeddings)

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self(embeddings)  # forward pass
        loss = nn.functional.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self(embeddings)  # forward pass
        loss = nn.functional.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        """Runs inference on batch and returns embeddings and top-K logits."""
        if isinstance(batch, tuple) and len(batch) == 2:
            embeddings, _ = batch  # Ignore labels if present
        else:
            embeddings = batch  # Use just embeddings if no labels
        # Move data to gpu if available
        embeddings = embeddings.to(self.model_device)
        logits = self(embeddings)
        probabilities = torch.softmax(logits, dim=1)
        return embeddings, probabilities

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
