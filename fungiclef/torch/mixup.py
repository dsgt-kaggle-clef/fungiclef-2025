import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fungiclef.config import get_device


class MixupClassifier(pl.LightningModule):
    def __init__(
        self,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        alpha: float = 2.0,  # typically 1.0 or 2.0
    ):
        super().__init__()
        self.model_device = get_device()
        self.num_classes = 2427  # total fungi species, 0 to 2426 category ids
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.emb_dim = 768
        self.hidden_dim = 1363  # geometric mean of 768 and 2427
        self.alpha = alpha

        # initialize the model
        self.classifier = nn.Linear(self.emb_dim, self.num_classes)

    def forward(self, embeddings):
        """Extract embeddings using the [CLS] token."""
        return self.classifier(embeddings)

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        device = embeddings.device

        # sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # shuffle the batch for mixup
        index = torch.randperm(embeddings.size(0)).to(device)
        embeddings_shuffled = embeddings[index]
        labels_shuffled = labels[index]

        # Mix embeddings
        mixed_embeddings = lam * embeddings + (1 - lam) * embeddings_shuffled

        # forward pass
        logits = self.classifier(mixed_embeddings)

        # Mixup loss
        loss = lam * F.cross_entropy(logits, labels) + (1 - lam) * F.cross_entropy(
            logits, labels_shuffled
        )
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
