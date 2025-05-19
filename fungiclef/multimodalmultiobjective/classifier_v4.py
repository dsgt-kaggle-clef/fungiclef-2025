import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from fungiclef.config import (
    get_device,
    get_poison_mapping,
    get_genus_mapping,
    get_species_mapping,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


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

        # Classification heads
        self.category_head = nn.Linear(
            512, self.num_classes
        )  # species-level label (category_id)
        self.poison_head = nn.Linear(512, 1)  # binary classification
        self.genus_head = nn.Linear(
            512, int(torch.tensor(get_genus_mapping()).max().item()) + 1
        )
        self.species_head = nn.Linear(
            512, int(torch.tensor(get_species_mapping()).max().item()) + 1
        )

    def forward(self, image_embed, text_embed):
        image_feat = F.relu(self.image_proj(image_embed))
        text_feat = F.relu(self.text_proj(text_embed))
        combined = torch.cat((image_feat, text_feat), dim=1)
        combined = self.norm(combined)
        # logits = self.classifier(combined)
        return {
            "category_logits": self.category_head(combined),
            "poison_logits": self.poison_head(combined).squeeze(-1),  # shape (B,)
            "genus_logits": self.genus_head(combined),
            "species_logits": self.species_head(combined),
        }

    def training_step(self, batch, batch_idx):
        image_embed, text_embed, label = batch
        logits = self(image_embed, text_embed)["category_logits"]
        loss = F.cross_entropy(logits, label)
        acc = (logits.argmax(dim=1) == label).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        image_embed, text_embed, label = batch
        logits = self(image_embed, text_embed)["category_logits"]
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

    def forward(self, image_embed, text_embed):
        return self.model(image_embed, text_embed)

    def training_step(self, batch, batch_idx):
        image_embed, text_embed, y_category_id, y_poison, y_genus, y_species = batch
        outputs = self(image_embed, text_embed)
        loss_category_id = self.loss_category_id(
            outputs["category_logits"], y_category_id
        )
        loss_poison = self.loss_poison(outputs["poison_logits"], y_poison.float())
        loss_genus = self.loss_genus(outputs["genus_logits"], y_genus)
        loss_species = self.loss_species(outputs["species_logits"], y_species)

        # Accuracy
        pred_species = outputs["category_logits"].argmax(dim=1)
        train_acc = (pred_species == y_category_id).float().mean()
        self.log("train_acc", train_acc, prog_bar=True)

        losses = torch.stack([loss_category_id, loss_poison, loss_genus, loss_species])
        weighted_losses = self.task_weights * losses
        total_loss = weighted_losses.sum()

        if self.current_epoch == 0 and batch_idx == 0:
            self.initial_losses = losses.detach()

        # GradNorm
        shared_params = [p for p in self.model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            total_loss, shared_params, create_graph=True, allow_unused=True
        )
        norms = torch.stack([g.norm() for g in grads if g is not None])
        norm = (
            norms.mean()
            if len(norms) > 0
            else torch.tensor(0.0, device=self.model_device)
        )

        for i, (param, grad) in enumerate(zip(shared_params, grads)):
            if grad is None:
                print(f"Grad {i} ({param.shape}) is None")
            else:
                print(f"Grad {i} ({param.shape}) norm = {grad.norm().item():.4f}")

        loss_ratios = losses.detach() / (self.initial_losses + 1e-8)
        avg_ratio = loss_ratios.mean()
        inverse_train_rates = loss_ratios / avg_ratio
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
        image_embed, text_embed, y_category_id, y_poison, y_genus, y_species = batch
        outputs = self(image_embed, text_embed)

        pred_species = outputs["category_logits"].argmax(dim=1)
        acc_category_id = (pred_species == y_category_id).float().mean()

        loss_category_id = self.loss_category_id(
            outputs["category_logits"], y_category_id
        )
        loss_poison = self.loss_poison(outputs["poison_logits"], y_poison.float())
        loss_genus = self.loss_genus(outputs["genus_logits"], y_genus)
        loss_species = self.loss_species(outputs["species_logits"], y_species)

        pred_poison = (torch.sigmoid(outputs["poison_logits"]) > 0.5).float()
        acc_poison = (pred_poison == y_poison.float()).float().mean()
        acc_genus = (outputs["genus_logits"].argmax(dim=1) == y_genus).float().mean()
        acc_species = (
            (outputs["species_logits"].argmax(dim=1) == y_species).float().mean()
        )

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
        if isinstance(batch, tuple) and len(batch) == 3:
            image_embed, text_embed, _ = batch
        else:
            image_embed, text_embed = batch
        outputs = self(image_embed, text_embed)
        probs = torch.softmax(outputs["category_logits"], dim=1)
        return probs

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.task_weights, "lr": 1e-3},
            ],
            lr=self.learning_rate,
        )
