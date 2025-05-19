import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import WeightedRandomSampler
from fungiclef.config import get_genus_mapping, get_species_mapping


class MultiModalEmbeddingDataset(Dataset):
    def __init__(
        self,
        image_df,
        text_df,
        embedding_col: str = "embeddings",
        label_col: str = "category_id",
        has_labels: bool = True,
        multi_objective: bool = False,
    ):
        # rename embeddings to avoid collision before merge
        image_df = image_df.rename(columns={embedding_col: "image_embedding"})
        text_df = text_df.rename(columns={embedding_col: "text_embedding"})

        # merge on 'filename'
        self.df = pd.merge(image_df, text_df, on="filename")
        self.has_labels = has_labels
        self.label_col = label_col

        self.image_embeddings = self.df["image_embedding"].tolist()
        self.text_embeddings = self.df["text_embedding"].tolist()
        if has_labels:
            self.labels = self.df[label_col].tolist()

        self.multi_objective = multi_objective

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_embed = torch.tensor(self.image_embeddings[idx], dtype=torch.float32)
        text_embed = torch.tensor(self.text_embeddings[idx], dtype=torch.float32)
        if self.has_labels:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            if not self.multi_objective:
                return image_embed, text_embed, label
            else:
                poisonous = self.df.iloc[idx]["poisonous"]
                genus = self.df.iloc[idx]["genus_id"]
                species = self.df.iloc[idx]["species_id"]
                return image_embed, text_embed, label, poisonous, genus, species
        return image_embed, text_embed


class MultiModalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_image_df,
        val_image_df,
        test_image_df,
        train_text_df,
        val_text_df,
        test_text_df,
        batch_size=64,
        num_workers=6,
        embedding_col="embeddings",
        label_col="category_id",
        multi_objective=False,
    ):
        super().__init__()
        self.train_image_df = train_image_df
        self.val_image_df = val_image_df
        self.test_image_df = test_image_df

        self.train_text_df = train_text_df
        self.val_text_df = val_text_df
        self.test_text_df = test_text_df

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.embedding_col = embedding_col
        self.label_col = label_col
        self.genus_ids = None  # Initialize mapping attributes to None
        self.species_ids = None  # Initialize mapping attributes to None

        self.test_has_labels = (
            test_image_df is not None and label_col in test_image_df.columns
        )
        self.multi_objective = (
            multi_objective  # flag for multi-objective lossb True or False
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            sample_weights = compute_sample_weight(
                class_weight="balanced", y=self.train_image_df[self.label_col]
            )
            self.sampler = WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=True
            )

            if self.multi_objective:
                self.train_image_df = self._add_mappings(self.train_image_df)
                self.val_image_df = self._add_mappings(self.val_image_df)

            self.train_dataset = MultiModalEmbeddingDataset(
                self.train_image_df,
                self.train_text_df,
                embedding_col=self.embedding_col,
                label_col=self.label_col,
                has_labels=True,
                multi_objective=self.multi_objective,
            )
            self.val_dataset = MultiModalEmbeddingDataset(
                self.val_image_df,
                self.val_text_df,
                embedding_col=self.embedding_col,
                label_col=self.label_col,
                has_labels=True,
                multi_objective=self.multi_objective,
            )

        if stage == "test" or stage is None:
            self.test_dataset = MultiModalEmbeddingDataset(
                self.test_image_df,
                self.test_text_df,
                embedding_col=self.embedding_col,
                label_col=self.label_col,
                has_labels=self.test_has_labels,
                multi_objective=False,
            )

        if stage == "predict" or stage is None:
            self.predict_dataset = MultiModalEmbeddingDataset(
                self.test_image_df,
                self.test_text_df,
                embedding_col=self.embedding_col,
                label_col=self.label_col,
                has_labels=self.test_has_labels,
                multi_objective=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            # shuffle=True, # use shuffle if no sampler is used
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
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
