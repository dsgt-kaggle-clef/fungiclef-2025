import timm
import torch
import pytorch_lightning as pl

from fungiclef.config import get_device
from fungiclef.model_setup import setup_fine_tuned_model


class DINOv2LightningModel(pl.LightningModule):
    """PyTorch Lightning module for extracting embeddings from a fine-tuned DINOv2 model."""

    def __init__(
        self,
        model_path: str = setup_fine_tuned_model(),
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
    ):
        super().__init__()
        self.model_device = get_device()
        self.num_classes = 7806  # total plant species

        # load the fine-tuned model
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=self.num_classes,
            checkpoint_path=model_path,
        )

        # load transform
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(
            **self.data_config, is_training=False
        )

        # move model to device
        self.model.to(self.model_device)
        self.model.eval()

    def forward(self, batch):
        """Extract [CLS] token embeddings using fine-tuned model."""
        with torch.no_grad():
            batch = batch.to(self.model_device)  # move to device

            if batch.dim() == 5:  # (B, grid_size**2, C, H, W)
                B, G, C, H, W = batch.shape
                batch = batch.view(B * G, C, H, W)  # (B * grid_size**2, C, H, W)
            # forward pass
            features = self.model.forward_features(batch)
            embeddings = features[:, 0, :]  # extract [CLS] token

        return embeddings

    def predict_step(self, batch, batch_idx):
        """Runs inference on batch and returns embeddings and top-K logits."""
        return self(batch)  # [CLS] token embeddings
