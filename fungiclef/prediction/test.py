import torch
import pandas as pd
import pytorch_lightning as pl
from fungiclef.torch.data import (
    FungiDataset,
    FungiDataModule,
)
from fungiclef.torch.model import DINOv2LightningModel
from fungiclef.config import get_device
from torch.utils.data import DataLoader
from tqdm import tqdm


def torch_pipeline(
    pandas_df: pd.DataFrame,
    batch_size: int = 32,
    cpu_count: int = 1,
    top_k: int = 10,
):
    """Pipeline to extract embeddings and top-K logits using PyTorch Lightning."""

    # initialize model
    model = DINOv2LightningModel(top_k=top_k)

    # create Dataset
    dataset = FungiDataset(pandas_df, model.transform, training_mode=False)
    # create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpu_count,
    )

    # run inference and collect embeddings with tqdm progress bar
    all_embeddings = []
    all_logits = []
    for batch in tqdm(
        dataloader, desc="Extracting embeddings and logits", unit="batch"
    ):
        embeddings, logits = model.predict_step(
            batch, batch_idx=0
        )  # batch: List[Tuple[embeddings, logits]]
        all_embeddings.append(embeddings)  # keep embeddings as tensors
        all_logits.extend(logits)  # preserve batch structure

    # convert embeddings to tensor
    embeddings = torch.cat(all_embeddings, dim=0)  # shape: [len(df), 768]

    embeddings = embeddings.view(-1, 1, 768)

    return embeddings, all_logits


def pl_trainer_pipeline(
    pandas_df: pd.DataFrame,
    batch_size: int = 32,
    cpu_count: int = 1,
    top_k: int = 10,
):
    """Pipeline to extract embeddings and top-k logits using PyTorch Lightning."""

    # initialize DataModule
    data_module = FungiDataModule(
        train_df=None,
        val_df=None,
        test_df=pandas_df,
        batch_size=batch_size,
        num_workers=cpu_count,
    )

    # initialize Model
    model = DINOv2LightningModel(top_k=top_k)

    # define Trainer (inference mode)
    trainer = pl.Trainer(
        accelerator=get_device(),
        devices=1,
        enable_progress_bar=True,
    )

    # run Inference
    predictions = trainer.predict(model, datamodule=data_module)

    all_embeddings = []
    all_logits = []
    for batch in predictions:
        embed_batch, logits_batch = batch  # batch: List[Tuple[embeddings, logits]]
        all_embeddings.append(embed_batch)  # keep embeddings as tensors
        all_logits.extend(logits_batch)  # preserve batch structure

    # convert embeddings to tensor
    embeddings = torch.cat(all_embeddings, dim=0)  # shape: [len(df), 768]

    embeddings = embeddings.view(-1, 1, 768)  # reshape to [len(df), 1, 768]

    return embeddings, all_logits
