import timm
import pytorch_lightning as pl

from torchvision import transforms as T
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from fungiclef.serde import deserialize_image
from fungiclef.lightning.model import EmbedModel


class FungiDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        col_name: str = "data",
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
    ):
        self.df = df
        self.transform = transform
        self.col_name = col_name
        self.model_name = model_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_bytes = self.df.iloc[idx][self.col_name]
        img = deserialize_image(img_bytes)  # PIL image

        if self.transform:
            return self.transform(img)  # (C, H, W)
        return ToTensor()(img)  # (C, H, W)


class FungiDataModule(pl.LightningDataModule):
    """LightningDataModule for handling dataset loading and preparation."""

    def __init__(
        self,
        pandas_df,
        batch_size=32,
        num_workers=4,
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        resize_size: int = 224,
    ):
        super().__init__()
        self.pandas_df = pandas_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.resize_size = resize_size

    def setup(self, stage=None):
        """Set up dataset and transformations."""

        if self.resize_size == 384:
            transform = T.Compose(
                [
                    T.Resize((384, 384)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif self.resize_size == 224:
            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            model = EmbedModel(self.model_name)
            data_config = timm.data.resolve_model_data_config(model)
            transform = timm.data.create_transform(**data_config, is_training=False)

        self.dataset = FungiDataset(
            self.pandas_df,
            transform=transform,
        )

    def predict_dataloader(self):
        """Returns DataLoader for inference."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
