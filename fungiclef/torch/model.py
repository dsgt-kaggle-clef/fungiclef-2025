import torch
import pytorch_lightning as pl
import torch.nn as nn


from transformers import AutoImageProcessor, AutoModel
from fungiclef.config import get_device, get_class_mappings_file


class DINOv2LightningModel(pl.LightningModule):
    """PyTorch Lightning module for extracting embeddings from a fine-tuned DINOv2 model."""

    def __init__(
        self,
        model_name="facebook/dinov2-base",
        top_k: int = 10,
    ):
        super().__init__()
        self.model_device = get_device()
        self.num_classes = 2427  # total fungi species, 0 to 2426 category ids
        self.top_k = top_k
        self.learning_rate = 1e-3

        # Load model
        self.model = AutoModel.from_pretrained(model_name)
        # move model to device
        self.model.to(self.model_device)
        self.model.eval()

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Load transform for image preprocessing
        self.transform = AutoImageProcessor.from_pretrained(model_name)

        emb_dim = self._get_embedding_dim()

        if emb_dim != 768:
            print(f"embed_dim is {emb_dim}, expected 768")

        # Trainable Logistic Regression head
        self.classifier = nn.Linear(emb_dim, self.num_classes)

        # class mappings file for classification
        self.class_mappings_file = get_class_mappings_file()
        # load class mappings
        self.cid_to_spid = self._load_class_mappings()

    def _get_embedding_dim(self):
        # Dynamically determine embedding size, should be 768
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_processed = self.transform(images=dummy_input, return_tensors="pt")
        with torch.no_grad():
            dummy_output = self.model(**dummy_processed)
            return dummy_output.last_hidden_state[:, 0, :].shape[-1]

    def _load_class_mappings(self):
        with open(self.class_mappings_file, "r") as f:
            class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
        return class_index_to_class_name

    def forward(self, batch):
        """Extract embeddings using the [CLS] token."""
        with torch.no_grad():
            batch = batch.to(self.model_device)  # move to device
            outputs = self.model(batch)
            embeddings = outputs.last_hidden_state[:, 0, :]  # extract [CLS] token
            # forward pass
            logits = self.classifier(embeddings)

        return embeddings, logits

    def extract_embeddings(self, batch):
        """Returns the [CLS] token embeddings"""
        batch = batch.to(self.model_device)
        with torch.no_grad():
            outputs = self.model(batch)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings, logits = self(images)
        loss = nn.functional.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        embeddings, logits = self(images)
        loss = nn.functional.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

    def predict_step(self, batch, batch_idx):
        """Runs inference on batch and returns embeddings and top-K logits."""
        embeddings, logits = self(batch)
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


if __name__ == "__main__":
    pass
