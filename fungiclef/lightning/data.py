import pytorch_lightning as pl

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
        img = deserialize_image(img_bytes)

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
    ):
        super().__init__()
        self.pandas_df = pandas_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name

    def setup(self, stage=None):
        """Set up dataset and transformations."""

        self.model = EmbedModel(model_name=self.model_name)
        self.dataset = FungiDataset(
            self.pandas_df,
            self.model.transform,  # Use the model's transform
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
