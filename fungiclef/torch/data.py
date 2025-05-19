import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_sample_weight
from fungiclef.config import get_genus_mapping, get_species_mapping


class FungiDataset(Dataset):
    """Custom PyTorch Dataset for loading embeddings from parquet files."""

    def __init__(
        self,
        df,
        embedding_col: str = "embeddings",
        label_col: str = None,
        has_labels: bool = True,
        multi_objective: bool = False,
    ):
        """
        Args:
            df (pd.DataFrame): Pandas DataFrame containing image binary data.
            transform: Image transformation function.
            col_name (str): Column name containing image bytes.
            label_col (str): Column name containing class labels.
            has_labels (bool): If True, return image-label pairs for training.
        """
        self.df = df
        self.embedding_col = embedding_col
        self.label_col = label_col
        self.has_labels = has_labels
        self.multi_objective = multi_objective

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Return embedding-label pair or just embedding if no labels."""
        embedding_data = self.df.iloc[idx][self.embedding_col]
        embedding_tensor = torch.from_numpy(embedding_data.copy()).float()
        if self.has_labels:
            label = self.df.iloc[idx][self.label_col]
            if not self.multi_objective:
                return embedding_tensor, label
            else:
                poisonous = self.df.iloc[idx]["poisonous"]
                genus = self.df.iloc[idx]["genus_id"]
                species = self.df.iloc[idx]["species_id"]
                return embedding_tensor, label, poisonous, genus, species
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
        multi_objective=False,
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
        self.genus_ids = None  # Initialize mapping attributes to None
        self.species_ids = None  # Initialize mapping attributes to None
        self.multi_objective = (
            multi_objective  # flag for multi-objective lossb True or False
        )

    def setup(self, stage=None):
        """Set up dataset."""
        # Create datasets for each split
        if stage == "fit" or stage is None:
            # 2024 Weight Sampler Method
            # class_labels = sorted(set(self.train_df[self.label_col]) | set(self.val_df[self.label_col]))
            # label_to_index = {label: idx for idx, label in enumerate(class_labels)}
            # num_classes = len(class_labels)

            # # 2. Compute normalized train/val class distributions
            # train_dist = np.ones(num_classes)  # initialize with 1 for smoothing
            # val_dist = np.ones(num_classes)

            # train_counts = self.train_df[self.label_col].value_counts()
            # val_counts = self.val_df[self.label_col].value_counts()

            # for label, count in train_counts.items():
            #     train_dist[label_to_index[label]] += count

            # for label, count in val_counts.items():
            #     val_dist[label_to_index[label]] += count

            # train_dist /= train_dist.sum()
            # val_dist /= val_dist.sum()

            # # 3. Compute class weights: val distribution over train distribution
            # dist_weights = val_dist / train_dist

            # # 4. Assign each training sample a weight based on its class
            # _train_weights = [
            #     dist_weights[label_to_index[label]] for label in self.train_df[self.label_col]
            # ]

            # self.sampler = WeightedRandomSampler(_train_weights, len(_train_weights), replacement=True)

            # Step 3: Create WeightedRandomSampler
            sample_weights = compute_sample_weight(
                class_weight="balanced", y=self.train_df[self.label_col]
            )
            self.sampler = WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=True
            )
            # conditional for multi-objective
            if self.multi_objective:
                self.train_df = self._add_mappings(self.train_df)
                self.val_df = self._add_mappings(self.val_df)

            self.train_dataset = FungiDataset(
                self.train_df,
                self.embedding_col,
                self.label_col,
                has_labels=True,
                multi_objective=self.multi_objective,
            )
            self.val_dataset = FungiDataset(
                self.val_df,
                self.embedding_col,
                self.label_col,
                has_labels=True,
                multi_objective=self.multi_objective,
            )
        if stage == "test" or stage is None:
            self.test_dataset = FungiDataset(
                self.test_df,
                self.embedding_col,
                self.label_col,
                has_labels=self.test_has_labels,
                multi_objective=False,
            )

        if stage == "predict" or stage is None:
            # For prediction, typically use the test dataset
            self.predict_dataset = FungiDataset(
                self.test_df,
                self.embedding_col,
                self.label_col,
                has_labels=self.test_has_labels,
                multi_objective=False,
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
            sampler=self.sampler,
            # shuffle=True,  # shuffle if no sampler is used
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

    def _add_mappings(self, df):
        """Add genus_id, species_id based on category_id"""
        df = df.copy()
        if self.genus_ids is None:
            self.genus_ids = get_genus_mapping()
        if self.species_ids is None:
            self.species_ids = get_species_mapping()
        if "genus_id" not in df.columns:
            df["genus_id"] = df["category_id"].map(lambda cid: self.genus_ids[cid])
        if "species_id" not in df.columns:
            df["species_id"] = df["category_id"].map(lambda cid: self.species_ids[cid])

        return df
