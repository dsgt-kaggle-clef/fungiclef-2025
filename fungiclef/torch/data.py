import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from fungiclef.serde import deserialize_image
from fungiclef.torch.model import DINOv2LightningModel


class FungiDataset(Dataset):
    """Custom PyTorch Dataset for loading fungi images from a Pandas DataFrame."""

    def __init__(
        self,
        df,
        transform=None,
        col_name: str = "data",
        label_col: str = "category_id",
        training_mode: bool = False,
    ):
        """
        Args:
            df (pd.DataFrame): Pandas DataFrame containing image binary data.
            transform: Image transformation function.
            col_name (str): Column name containing image bytes.
            label_col (str): Column name containing class labels.
            training_mode (bool): If True, return image-label pairs for training.
        """
        self.df = df
        self.transform = transform
        self.col_name = col_name
        self.label_col = label_col
        self.training_mode = training_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """returns torch tensor"""
        img_bytes = self.df.iloc[idx][self.col_name]  # column with image bytes
        img = deserialize_image(img_bytes)  # convert from bytes to PIL image
        # single image, shape: (C, H, W)
        if self.transform:
            processed = self.transform(images=img, return_tensors="pt")  # (B, C, H, W)
            image_tensor = processed["pixel_values"].squeeze(0)  # (C, H, W)
        else:
            image_tensor = ToTensor()(img)  # (C, H, W)

        if self.training_mode and self.label_col in self.df.columns:
            label = self.df.iloc[idx][self.label_col]
            return image_tensor, label
        return image_tensor


class FungiDataModule(pl.LightningDataModule):
    """LightningDataModule for handling dataset loading and preparation."""

    def __init__(
        self,
        train_df,
        val_df,
        test_df,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = None

    def setup(self, stage=None):
        """Set up dataset and transformations."""

        if self.model is None:
            self.model = DINOv2LightningModel()

        # Create datasets for each split
        if stage == "fit" or stage is None:
            self.train_dataset = FungiDataset(self.train_df, self.model.transform)
            self.val_dataset = FungiDataset(self.val_df, self.model.transform)

        if stage == "test" or stage is None:
            self.test_dataset = FungiDataset(self.test_df, self.model.transform)

        if stage == "predict" or stage is None:
            # For prediction, typically use the test dataset
            self.predict_dataset = FungiDataset(self.test_df, self.model.transform)

    def predict_dataloader(self):
        """Returns DataLoader for inference."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def train_dataloader(self):
        """Returns DataLoader for training data."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # consider shuffling
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Returns DataLoader for validation data."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """Returns DataLoader for test data."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
