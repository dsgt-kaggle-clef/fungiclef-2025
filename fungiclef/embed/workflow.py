import typer
import torch
import pandas as pd
import pytorch_lightning as pl

from fungiclef.config import get_device
from fungiclef.lightning.data import FungiDataModule
from fungiclef.lightning.model import DINOv2LightningModel


def pl_trainer_pipeline(
    pandas_df: pd.DataFrame,
    batch_size: int = 32,
    cpu_count: int = 1,
):
    """Pipeline to extract embeddings and top-k logits using PyTorch Lightning."""

    # initialize DataModule
    data_module = FungiDataModule(
        pandas_df,
        batch_size=batch_size,
        num_workers=cpu_count,
    )

    # initialize Model
    model = DINOv2LightningModel()

    # define Trainer (inference mode)
    trainer = pl.Trainer(
        accelerator=get_device(),
        devices=1,
        enable_progress_bar=True,
    )

    # run inference
    predictions = trainer.predict(model, datamodule=data_module)

    all_embeddings = []
    for batch in predictions:
        embed_batch = batch  # batch: List[embeddings]
        all_embeddings.append(embed_batch)  # keep embeddings as tensors

    # convert embeddings to tensor
    embeddings = torch.cat(all_embeddings, dim=0)  # shape: [len(df), grid_size**2, 768]
    return embeddings


def main(
    input_path: str = typer.Argument(..., help="Path to input CSV file."),
    output_path: str = typer.Argument(..., help="Path to output CSV file."),
    batch_size: int = typer.Option(32, help="Batch size for inference."),
    cpu_count: int = typer.Option(1, help="Number of CPU cores to use."),
):
    """Main function to run the embedding pipeline."""
    # load the DataFrame
    df = pd.read_parquet(input_path)

    # run the pipeline
    embeddings = pl_trainer_pipeline(df, batch_size, cpu_count)

    # save the embeddings to a parquet file
    embeddings_df = pd.DataFrame(embeddings.numpy())
    embeddings_df.to_parquet(output_path, index=False)
