import torch
import pytorch_lightning as pl
import torch.nn as nn
from fungiclef.config import get_device, get_class_mappings_file


class DINOv2LightningModel(pl.LightningModule):
    """PyTorch Lightning module for training a classifier on pre-extracted embeddings from parquet files."""

    def __init__(
        self,
        top_k: int = 10,
    ):
        super().__init__()
        self.model_device = get_device()
        self.num_classes = 2427  # total fungi species, 0 to 2426 category ids
        self.top_k = top_k
        self.learning_rate = 1e-3

        # emb_dim = self._get_embedding_dim()
        self.emb_dim = 768

        # Trainable Logistic Regression head
        self.classifier = nn.Linear(self.emb_dim, self.num_classes)

        # class mappings file for classification
        self.class_mappings_file = get_class_mappings_file()
        # load class mappings
        self.cid_to_spid = self._load_class_mappings()

    def _load_class_mappings(self):
        with open(self.class_mappings_file, "r") as f:
            class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
        return class_index_to_class_name

    def forward(self, embeddings):
        """Extract embeddings using the [CLS] token."""
        embeddings = embeddings.to(self.model_device)
        # forward pass
        logits = self.classifier(embeddings)
        return logits

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        embeddings = embeddings.to(self.model_device)
        labels = labels.to(self.model_device)
        # Forward pass
        logits = self(embeddings)
        loss = nn.functional.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        # Move data to gpu if available
        embeddings = embeddings.to(self.model_device)
        labels = labels.to(self.model_device)
        # Forward pass
        logits = self(embeddings)
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
        top_probs, top_indices = torch.topk(probabilities, k=self.top_k, dim=1)

        # map class indices to species names
        batch_logits = []
        for i in range(len(logits)):
            species_probs = {
                self.cid_to_spid.get(int(top_indices[i, j].item()), "Unknown"): float(
                    top_probs[i, j].item()
                )
                for j in range(self.top_k)
            }
            batch_logits.append(species_probs)

        return embeddings, batch_logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    pass
