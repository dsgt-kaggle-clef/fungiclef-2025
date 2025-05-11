import typer
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from fungiclef.config import get_device
from fungiclef.torch.data import FungiDataModule
from fungiclef.torch.model import DINOv2LightningModel

## training pipeline
### example usage: python clef/fungiclef-2025/fungiclef/prediction/train.py --train scratch/fungiclef/embeddings/dinov2/train_embeddings.parquet --val scratch/fungiclef/embeddings/dinov2/val_embeddings.parquet --batch-size 64 --max-epochs 20 --output-dir scratch/fungiclef/model/base_model --learning-rate 1e-3

# python clef/fungiclef-2025/fungiclef/prediction/train.py --train scratch/fungiclef/embeddings/dinov2/train_augment_embeddings.parquet --val scratch/fungiclef/embeddings/dinov2/val_embeddings.parquet --batch-size 64 --max-epochs 20 --output-dir scratch/fungiclef/model/base_model --learning-rate 1e-3

### example usage: python clef/fungiclef-2025/fungiclef/prediction/train.py --train scratch/fungiclef/embeddings/plantclef/train_embeddings.parquet --val scratch/fungiclef/embeddings/plantclef/val_embeddings.parquet --batch-size 64 --max-epochs 20 --output-dir scratch/fungiclef/model/plantclef --learning-rate 1e-3


def load_and_merge_embeddings(
    parquet_path: str,
    embed_path: str,
    columns: list,
    embedding_col: str = "embeddings",
) -> pd.DataFrame:
    """Load and merge metadata and embeddings"""
    df_meta = pd.read_parquet(parquet_path, columns=columns)
    df_embed = pd.read_parquet(embed_path, columns=["filename", embedding_col])
    return df_meta.merge(df_embed, on="filename", how="inner")


def train_fungi_classifier(
    train_parquet_path: str,
    train_embed_path: str,
    val_parquet_path: str,
    val_embed_path: str,
    batch_size: int = 64,
    num_workers: int = 6,
    max_epochs: int = 10,
    learning_rate: float = 1e-3,
    output_dir: str = "model",
    model_name: str = "fungi-classifier",
    embedding_col: str = "embeddings",
    early_stopping_patience: int = 3,
):
    """
    Train a fungi classifier based on DINOv2 features.

    Args:
        train_parquet: Path to training data parquet
        val_parquet: Path to validation data parquet
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        output_dir: Directory to save model checkpoints
    """
    # get category ID
    columns = [
        "filename",
        "category_id",
        "species",
        "genus",
        "family",
        "order",
        "class",
        "poisonous",
    ]
    # load data
    train_df = load_and_merge_embeddings(
        train_parquet_path, train_embed_path, columns, embedding_col
    )
    val_df = load_and_merge_embeddings(
        val_parquet_path, val_embed_path, columns, embedding_col
    )
    print(f"Train DF cols: {train_df.columns}")
    print(f"Val DF cols: {val_df.columns}")

    # create data module
    data_module = FungiDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        batch_size=batch_size,
        num_workers=num_workers,
        embedding_col=embedding_col,
        label_col="category_id",
    )

    # create model
    model = DINOv2LightningModel(learning_rate=learning_rate)

    # set up callbacks
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

    # Set up logger
    logger = TensorBoardLogger("logs", name="fungi-classifier")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator=get_device(),
    )

    # Train model
    print("Starting training...")
    trainer.fit(model, data_module)

    print(f"Model saved to {checkpoint_callback.best_model_path}")


def main(
    train_parquet_path: str = typer.Argument(..., help="Path to training data parquet"),
    train_embed_path: str = typer.Argument(..., help="Path to training data parquet"),
    val_parquet_path: str = typer.Argument(..., help="Path to validation data parquet"),
    val_embed_path: str = typer.Argument(..., help="Path to validation data parquet"),
    cpu_count: int = typer.Option(6, help="Number of workers for data loading"),
    batch_size: int = typer.Option(64, help="Batch size for training"),
    max_epochs: int = typer.Option(10, help="Maximum number of training epochs"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate for optimizer"),
    output_model_path: str = typer.Option(
        "models", help="Directory to save model checkpoints"
    ),
    model_name: str = typer.Option(
        "fungi-classifier", help="Name of the model to save checkpoints"
    ),
    embedding_col: str = typer.Option(
        "embeddings", help="Column name containing embeddings"
    ),
    early_stopping_patience: int = typer.Option(3, help="Patience for early stopping"),
):
    # Create output directory if it doesn't exist
    Path(output_model_path).mkdir(parents=True, exist_ok=True)

    # Train model
    train_fungi_classifier(
        train_parquet_path=train_parquet_path,
        train_embed_path=train_embed_path,
        val_parquet_path=val_parquet_path,
        val_embed_path=val_embed_path,
        num_workers=cpu_count,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        output_dir=output_model_path,
        model_name=model_name,
        embedding_col=embedding_col,
        early_stopping_patience=early_stopping_patience,
    )
