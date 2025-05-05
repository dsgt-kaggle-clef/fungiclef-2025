"""Script to extract embeddings from serialized images and save them to a parquet file with metadata."""

import torch
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
import timm
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader, Dataset
from plantclef_model_setup import setup_fine_tuned_model
from fungiclef.config import get_device

### example usage python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/dataset/processed/train_serialized.parquet --output scratch/fungiclef/embeddings/train_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6

### example usage python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/dataset/processed/val_serialized.parquet --output scratch/fungiclef/embeddings/val_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6

### example usage python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/dataset/processed/test_serialized.parquet --output scratch/fungiclef/embeddings/test_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6

### example usage python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/dataset/processed/train_serialized.parquet --output scratch/fungiclef/embeddings/plantclef/train_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6 --model-name vit_base_patch14_reg4_dinov2.lvd142m

### example usage python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/dataset/processed/val_serialized.parquet --output scratch/fungiclef/embeddings/plantclef/val_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6 --model-name vit_base_patch14_reg4_dinov2.lvd142m

### example usage python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/dataset/processed/test_serialized.parquet --output scratch/fungiclef/embeddings/plantclef/test_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6 --model-name vit_base_patch14_reg4_dinov2.lvd142m


class SerializedImageDataset(Dataset):
    """Dataset for loading serialized images from a parquet file."""

    def __init__(self, parquet_path, transform=None):
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_data = row["data"]

        # Convert binary data to image
        img = Image.open(io.BytesIO(img_data))

        # Convert grayscale to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        # Extract class label if available
        label = row.get("category_id", -1)

        return {"image": img, "label": label, "idx": idx}


def extract_embeddings(
    parquet_path: str,
    output_path: str,
    model_name: str = "facebook/dinov2-base",
    batch_size: int = 64,
    num_workers: int = 6,
    device: str = None,
):
    """
    Extract ViT embeddings from serialized images and save them.

    Args:
        parquet_path: Path to the parquet file with serialized images
        output_path: Path to save the embeddings
        model_name: Name of the Vit model to use
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        device: Device to use for inference ('cuda', 'cpu', or None for auto-detection)
    """
    # Determine device
    device = get_device()
    print(f"Using device: {device}")

    # Load model and processor
    print(f"Loading model: {model_name}")

    if model_name != "facebook/dinov2-base":
        # used for timm models
        model_path = setup_fine_tuned_model()
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=7806,
            checkpoint_path=model_path,
        )

        # load transform/processor
        data_config = timm.data.resolve_model_data_config(model)
        transform_fn = timm.data.create_transform(**data_config, is_training=False)

        # move model to device
        model.to(device)
        model.eval()
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        # Create dataset and dataloader
        def transform_fn(image):
            return processor(images=image, return_tensors="pt")["pixel_values"][0]

    dataset = SerializedImageDataset(parquet_path, transform=transform_fn)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # Load the original dataframe to maintain metadata
    original_df = pd.read_parquet(parquet_path)

    # Extract embeddings
    all_embeddings = []
    all_indices = []

    print(f"Extracting embeddings from {parquet_path}...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch["image"].to(device)
            indices = batch["idx"]

            # Extract features - differently based on model type
            if model_name != "facebook/dinov2-base":
                features = model.forward_features(
                    images
                )  # Get features before classifier
                embeddings = features[:, 0, :]  # CLS token for ViT models
            else:
                outputs = model(images)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

            # Convert embeddings to numpy arrays for storage in parquet
            embeddings_np = embeddings.cpu().numpy()
            for i, idx in enumerate(indices):
                all_embeddings.append(embeddings_np[i])
                all_indices.append(idx.item())

    # Create a new dataframe with embeddings
    embeddings_df = pd.DataFrame(
        {
            "idx": all_indices,
            "embedding": all_embeddings,
            "model_name": [model_name] * len(all_indices),
        }
    )

    # Merge with original dataframe to preserve metadata
    # Sort by indices to ensure correct order
    embeddings_df = embeddings_df.sort_values("idx")

    # Add embeddings to original dataframe
    result_df = original_df.copy()
    result_df["embedding"] = embeddings_df["embedding"].values
    result_df["model_name"] = model_name

    # Save to parquet
    print(f"Saving embeddings to {output_path}")
    result_df.to_parquet(output_path, index=False)

    print(f"Done! Extracted {len(all_embeddings)} embeddings.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ViT embeddings from serialized images"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input parquet file with serialized images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the extracted embeddings",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-base",
        help="Name of the ViT model to use",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for processing"
    )
    parser.add_argument(
        "--num-workers", type=int, default=6, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (cuda, cpu, or None for auto)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Extract embeddings
    extract_embeddings(
        parquet_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
