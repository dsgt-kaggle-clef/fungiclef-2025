import typer
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from fungiclef.config import get_device
from .datamodule import MultiModalDataModule
from .classifier import MultiModalClassifier


def load_and_merge_embeddings(
    parquet_path: str, embed_path: str, columns: list, embedding_col: str = "embeddings"
) -> pd.DataFrame:
    """Load and merge metadata and embeddings"""
    df_meta = pd.read_parquet(parquet_path, columns=columns)
    df_embed = pd.read_parquet(embed_path, columns=["filename", embedding_col])
    return df_meta.merge(df_embed, on="filename", how="inner")


def train_multimodal_classifier(
    train_parquet: str,
    train_image_embed: str,
    train_text_embed: str,
    val_parquet: str,
    val_image_embed: str,
    val_text_embed: str,
    batch_size: int = 64,
    num_workers: int = 6,
    max_epochs: int = 10,
    learning_rate: float = 1e-3,
    output_dir: str = "model",
    model_name: str = "fungi-multimodal",
    embedding_col: str = "embeddings",
    early_stopping_patience: int = 3,
):
    # Required columns
    image_columns = [
        "filename",
        "category_id",
        "species",
        "genus",
        "family",
        "order",
        "class",
        "poisonous",
    ]
    text_columns = ["filename"]

    # Load training data
    train_image_df = load_and_merge_embeddings(
        train_parquet, train_image_embed, image_columns, embedding_col
    )
    train_text_df = load_and_merge_embeddings(
        train_parquet, train_text_embed, text_columns, embedding_col
    )

    val_image_df = load_and_merge_embeddings(
        val_parquet, val_image_embed, image_columns, embedding_col
    )
    val_text_df = load_and_merge_embeddings(
        val_parquet, val_text_embed, text_columns, embedding_col
    )

    # Setup data module
    data_module = MultiModalDataModule(
        train_image_df=train_image_df,
        val_image_df=val_image_df,
        test_image_df=None,
        train_text_df=train_text_df,
        val_text_df=val_text_df,
        test_text_df=None,
        batch_size=batch_size,
        num_workers=num_workers,
        embedding_col=embedding_col,
        label_col="category_id",
    )

    # Model
    model = MultiModalClassifier(lr=learning_rate)

    # Logger
    logger = TensorBoardLogger("logs", name=model_name)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator=get_device(),
    )

    # Train
    trainer.fit(model, data_module)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


def main(
    train_parquet: str = typer.Argument(...),
    train_image_embed: str = typer.Argument(...),
    train_text_embed: str = typer.Argument(...),
    val_parquet: str = typer.Argument(...),
    val_image_embed: str = typer.Argument(...),
    val_text_embed: str = typer.Argument(...),
    batch_size: int = typer.Option(64, help="Batch size"),
    cpu_count: int = typer.Option(6, help="Number of workers"),
    max_epochs: int = typer.Option(10),
    learning_rate: float = typer.Option(1e-3),
    output_model_path: str = typer.Option("models"),
    model_name: str = typer.Option("fungi-multimodal"),
    embedding_col: str = typer.Option("embeddings"),
    early_stopping_patience: int = typer.Option(3),
):
    Path(output_model_path).mkdir(parents=True, exist_ok=True)

    train_multimodal_classifier(
        train_parquet=train_parquet,
        train_image_embed=train_image_embed,
        train_text_embed=train_text_embed,
        val_parquet=val_parquet,
        val_image_embed=val_image_embed,
        val_text_embed=val_text_embed,
        batch_size=batch_size,
        num_workers=cpu_count,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        output_dir=output_model_path,
        model_name=model_name,
        embedding_col=embedding_col,
        early_stopping_patience=early_stopping_patience,
    )
