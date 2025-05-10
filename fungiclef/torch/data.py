import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader


class FungiDataset(Dataset):
    """Custom PyTorch Dataset for loading embeddings from parquet files."""

    def __init__(
        self,
        df,
        embedding_col: str = "embeddings",
        label_col: str = None,
        has_labels: bool = True,
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
        self.embedding_col = embedding_col
        self.label_col = label_col
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Return embedding-label pair or just embedding if no labels."""
        embedding_data = self.df.iloc[idx][self.embedding_col]
        embedding_tensor = torch.from_numpy(embedding_data.copy()).float()
        if self.has_labels:
            label = self.df.iloc[idx][self.label_col]
            return embedding_tensor, label
        return embedding_tensor


class FungiDataModule(pl.LightningDataModule):
    """LightningDataModule for handling dataset loading and preparation."""

    def __init__(
        self,
        train_df,
        val_df,
        test_df,
        batch_size=64,
        num_workers=6,
        embedding_col="embedding",
        label_col="category_id",
    ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.embedding_col = embedding_col
        self.label_col = label_col
        # Check if test data has labels
        self.test_has_labels = test_df is not None and label_col in test_df.columns

    def setup(self, stage=None):
        """Set up dataset."""

        # Create datasets for each split
        if stage == "fit" or stage is None:
            self.train_dataset = FungiDataset(
                self.train_df, self.embedding_col, self.label_col, has_labels=True
            )
            self.val_dataset = FungiDataset(
                self.val_df, self.embedding_col, self.label_col, has_labels=True
            )
        if stage == "test" or stage is None:
            self.test_dataset = FungiDataset(
                self.test_df,
                self.embedding_col,
                self.label_col,
                has_labels=self.test_has_labels,
            )

        if stage == "predict" or stage is None:
            # For prediction, typically use the test dataset
            self.predict_dataset = FungiDataset(
                self.test_df,
                self.embedding_col,
                self.label_col,
                has_labels=self.test_has_labels,
            )

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
