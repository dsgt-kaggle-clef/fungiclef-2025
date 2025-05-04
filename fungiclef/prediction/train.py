import argparse
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from fungiclef.torch.data import FungiDataModule
from fungiclef.torch.model import DINOv2LightningModel

## training pipeline
### example usage: python clef/fungiclef-2025/fungiclef/prediction/train.py --train scratch/fungiclef/embeddings/train_embeddings.parquet --val scratch/fungiclef/embeddings/val_embeddings.parquet --batch-size 64 --max-epochs 20 --output-dir scratch/fungiclef/model --learning-rate 1e-3


def train_fungi_classifier(
    train_parquet: str,
    val_parquet: str,
    batch_size: int = 64,
    num_workers: int = 6,
    max_epochs: int = 10,
    learning_rate: float = 1e-3,
    output_dir: str = "model",
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
    # Load data
    print(f"Loading data from {train_parquet}, {val_parquet}")

    train_df = pd.read_parquet(train_parquet)
    val_df = pd.read_parquet(val_parquet)

    # Check if category_id column exists
    if "category_id" not in train_df.columns or "category_id" not in val_df.columns:
        raise ValueError("Expected 'category_id' column in parquet files")

    if "embedding" not in train_df.columns or "embedding" not in val_df.columns:
        raise ValueError("Expected 'embedding' column in parquet files")

    # Create data module
    data_module = FungiDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=None,
        batch_size=batch_size,
        num_workers=num_workers,
        embedding_col="embedding",
        label_col="category_id",
    )

    # Create model
    model = DINOv2LightningModel()
    model.learning_rate = learning_rate  # Set learning rate

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="fungi-classifier-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    # Set up logger
    logger = TensorBoardLogger("logs", name="fungi-classifier")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="auto",  # Automatically select GPU if available
    )

    # Train model
    print("Starting training...")
    trainer.fit(model, data_module)

    print(f"Model saved to {checkpoint_callback.best_model_path}")

    return model, checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser(description="Train a fungi classifier")
    parser.add_argument(
        "--train", type=str, required=True, help="Path to training data CSV"
    )
    parser.add_argument(
        "--val", type=str, required=True, help="Path to validation data CSV"
    )

    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers", type=int, default=6, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=10, help="Maximum number of training epochs"
    )

    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Train model
    train_fungi_classifier(
        train_parquet=args.train,
        val_parquet=args.val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
