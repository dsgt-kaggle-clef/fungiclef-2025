import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class MultiModalClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.num_classes = 2427  # total fungi species, 0 to 2426
        self.emb_dim = 768
        self.text_dim = 1024

        # Projection layers
        self.image_proj = nn.Linear(self.emb_dim, 256)
        self.text_proj = nn.Linear(self.text_dim, 256)

        # Normalization and final classifier
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Linear(512, self.num_classes)

    def forward(self, image_embed, text_embed):
        image_feat = F.relu(self.image_proj(image_embed))
        text_feat = F.relu(self.text_proj(text_embed))
        combined = torch.cat((image_feat, text_feat), dim=1)
        combined = self.norm(combined)
        logits = self.classifier(combined)
        return logits

    def training_step(self, batch, batch_idx):
        image_embed, text_embed, label = batch
        logits = self(image_embed, text_embed)
        loss = F.cross_entropy(logits, label)
        acc = (logits.argmax(dim=1) == label).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        image_embed, text_embed, label = batch
        logits = self(image_embed, text_embed)
        loss = F.cross_entropy(logits, label)
        acc = (logits.argmax(dim=1) == label).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        """Returns probabilities for use during inference."""
        if isinstance(batch, tuple) and len(batch) == 3:
            image_embed, text_embed, _ = batch
        else:
            image_embed, text_embed = batch

        logits = self(image_embed, text_embed)
        probs = torch.softmax(logits, dim=1)
        return probs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
