import timm
import torch
import pytorch_lightning as pl

from torchvision import transforms as T
from fungiclef.config import get_device
from fungiclef.model_setup import setup_fine_tuned_model


class EmbedModel(pl.LightningModule):
    """PyTorch Lightning module for extracting embeddings from a fine-tuned DINOv2 model."""

    def __init__(
        self,
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        resize_size: int = 384,
    ):
        super().__init__()
        self.model_name = model_name
        self.resize_size = resize_size
        self.model_device = get_device()

        # load the fine-tuned model
        self.model = self._get_model(model_name)
        self.model.to(self.model_device)
        self.model.eval()

        # set up transform
        self.transform = self._build_transform(resize_size)

    def _build_transform(self, resize_size):
        """Returns the image transform based on resize size."""
        return T.Compose(
            [
                T.Resize((resize_size, resize_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _get_model(self, model_name: str):
        """Load the model from the specified path."""
        if model_name == "vit_base_patch14_reg4_dinov2.lvd142m":
            self.num_classes = 7806  # total plant species
            model_path = setup_fine_tuned_model()
            # load the fine-tuned plantclef model
            return timm.create_model(
                model_name,
                pretrained=False,
                num_classes=self.num_classes,
                checkpoint_path=model_path,
            )
        else:
            # load fine-tuned model from timm
            model = timm.create_model(
                model_name, pretrained=True, img_size=self.resize_size
            )
            return model

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
