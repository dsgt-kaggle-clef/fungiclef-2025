import torch
import pytorch_lightning as pl
import torch.nn as nn
from fungiclef.config import (
    get_device,
    get_poison_mapping,
    get_genus_mapping,
    get_species_mapping,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn.functional as F


class LinearClassifier(pl.LightningModule):
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

        # initialize the model
        self.classifier = nn.Linear(self.emb_dim, self.num_classes)

    def forward(self, embeddings):
        """Extract embeddings using the [CLS] token."""
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


class MultiModalClassifier(pl.LightningModule):
    ### copied from murilogustineli /multimodal/classifier.py can delete this after merging with his branch. see if need to delete the XXXXX_steps from classifier.py
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


class MultiObjectiveClassifier(pl.LightningModule):
    """PyTorch Lightning module for training a multi-objective classifier on pre-extracted embeddings from parquet files."""

    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        alpha: float = 1.5,
        poison_weighting: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.model_device = get_device()
        self.num_classes = 2427  # total fungi species, 0 to 2426 category ids
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.emb_dim = 768
        self.hidden_dim = 1363  # geometric mean of 768 and 2427
        self.alpha = alpha
        self.poison_weighting = poison_weighting

        # initialize the model
        self.poison_mapping = torch.Tensor(get_poison_mapping()).to(self.model_device)
        self.genus_mapping = torch.Tensor(get_genus_mapping()).to(self.model_device)
        self.species_mapping = torch.Tensor(get_species_mapping()).to(self.model_device)

        self.loss_poison = BCEWithLogitsLoss(
            pos_weight=torch.Tensor([poison_weighting]).to(self.model_device)
        )
        self.loss_category_id = CrossEntropyLoss()
        self.loss_genus = CrossEntropyLoss()
        self.loss_species = CrossEntropyLoss()

        self.task_weights = nn.Parameter(torch.ones(4))
        self.initial_losses = torch.zeros(4, device=self.model_device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_category_id, y_poison, y_genus, y_species = (
            batch  ## this currently only works with linear model. need to modify data loader to include text_embed
        )
        logits = self.model(x)
        pred_species = logits.argmax(dim=1)

        # Loss 1: category_id classification
        loss_category_id = self.loss_category_id(logits, y_category_id)

        # Loss 2: poison classification (infer from predicted class)
        pred_poison_logits = self.poison_mapping[pred_species]  # shape: (B,)
        loss_poison = self.loss_poison(pred_poison_logits.float(), y_poison.float())

        # Loss 3: genus classification (infer from predicted class)
        pred_genus_id = self.genus_mapping[pred_species]  # shape: (B,)
        num_genus_classes = int(self.genus_mapping.max().item()) + 1
        pred_genus_logits = F.one_hot(
            pred_genus_id.long(), num_classes=num_genus_classes
        ).float()
        loss_genus = self.loss_genus(pred_genus_logits, y_genus)

        # Loss 4: species classification (infer from predicted class)
        pred_species_id = self.species_mapping[pred_species]  # shape: (B,)
        num_species_classes = int(self.species_mapping.max().item()) + 1
        pred_species_logits = F.one_hot(
            pred_species_id.long(), num_classes=num_species_classes
        ).float()
        loss_species = self.loss_species(pred_species_logits, y_species)

        losses = torch.stack([loss_category_id, loss_poison, loss_genus, loss_species])
        weighted_losses = self.task_weights * losses
        total_loss = weighted_losses.sum()

        # Set initial losses
        if self.current_epoch == 0 and batch_idx == 0:
            self.initial_losses = losses.detach()

        # GradNorm
        shared_params = list(self.model.parameters())[
            0
        ]  # first param (e.g. embedding layer)
        shared_params = [p for p in self.model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(total_loss, shared_params, create_graph=True)
        norms = torch.stack([g.norm() for g in grads])
        norm = norms.mean()

        # Relative loss ratio
        loss_ratios = losses.detach() / self.initial_losses
        avg_ratio = loss_ratios.mean()
        inverse_train_rates = loss_ratios / avg_ratio

        # Target gradient norms
        target_norms = norm.detach() * inverse_train_rates**self.alpha
        gradnorm_loss = (torch.abs(norm - target_norms)).sum()
        total_loss += gradnorm_loss

        self.log_dict(
            {
                "loss_category_id": loss_category_id,
                "loss_poison": loss_poison,
                "loss_genus": loss_genus,
                "loss_species": loss_species,
                "gradnorm_loss": gradnorm_loss,
                "total_loss": total_loss,
            }
        )

        for i, w in enumerate(self.task_weights):
            self.log(f"task_weight_{i}", w.item(), on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y_category_id, y_poison, y_genus, y_species = batch
        logits = self.model(x)
        pred_species = logits.argmax(dim=1)

        # Loss 1: category_id classification
        loss_category_id = self.loss_category_id(logits, y_category_id)
        acc_category_id = (pred_species == y_category_id).float().mean()

        # Loss 2: poison classification
        pred_poison_logits = self.poison_mapping[pred_species]
        loss_poison = self.loss_poison(pred_poison_logits.float(), y_poison.float())
        pred_poison = (torch.sigmoid(pred_poison_logits) > 0.5).float()
        acc_poison = (pred_poison == y_poison.float()).float().mean()

        # Loss 3: genus classification
        pred_genus_id = self.genus_mapping[pred_species]
        num_genus_classes = int(self.genus_mapping.max().item()) + 1
        pred_genus_logits = F.one_hot(
            pred_genus_id.long(), num_classes=num_genus_classes
        ).float()
        loss_genus = self.loss_genus(pred_genus_logits, y_genus)
        acc_genus = (pred_genus_id == y_genus).float().mean()

        # Loss 4: species classification
        pred_species_id = self.species_mapping[pred_species]
        num_species_classes = int(self.species_mapping.max().item()) + 1
        pred_species_logits = F.one_hot(
            pred_species_id.long(), num_classes=num_species_classes
        ).float()
        loss_species = self.loss_species(pred_species_logits, y_species)
        acc_species = (pred_species_id == y_species).float().mean()

        val_loss = loss_category_id + loss_poison + loss_genus + loss_species
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_loss_category_id": loss_category_id,
                "val_loss_poison": loss_poison,
                "val_loss_genus": loss_genus,
                "val_loss_species": loss_species,
                "val_acc_category_id": acc_category_id,
                "val_acc_poison": acc_poison,
                "val_acc_genus": acc_genus,
                "val_acc_species": acc_species,
            },
            prog_bar=True,
        )

        return {
            "val_loss": val_loss,
            "acc_category_id": acc_category_id,
            "acc_poison": acc_poison,
            "acc_genus": acc_genus,
            "acc_species": acc_species,
        }

    def predict_step(self, batch, batch_idx):
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
        return torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.task_weights, "lr": 1e-3},
            ],
            lr=self.learning_rate,
        )
