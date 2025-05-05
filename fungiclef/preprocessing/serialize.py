"""Script to preprocess images and save them in a CSV file with serialized image data."""

import os
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

# Import from your serde module
from fungiclef.serde import read_image_bytes

### example usage: python clef/fungiclef-2025/fungiclef/preprocessing/serialize.py --csv scratch/fungiclef/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv --image-dir scratch/fungiclef/dataset/images/FungiTastic-FewShot/train/fullsize --output scratch/fungiclef/dataset/processed/train_serialized.parquet

### example usage: python clef/fungiclef-2025/fungiclef/preprocessing/serialize.py --csv scratch/fungiclef/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Val.csv --image-dir scratch/fungiclef/dataset/images/FungiTastic-FewShot/val/fullsize --output scratch/fungiclef/dataset/processed/val_serialized.parquet

### example usage: python clef/fungiclef-2025/fungiclef/preprocessing/serialize.py --csv scratch/fungiclef/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Test.csv --image-dir scratch/fungiclef/dataset/images/FungiTastic-FewShot/test/fullsize --output scratch/fungiclef/dataset/processed/test_serialized.parquet


def add_serialized_images_to_csv(
    csv_path: str,
    image_dir: str,
    output_parquet: str,
    image_col: str = "filename",  # Column containing image filenames
):
    """
    Reads images from disk as raw bytes and adds them to a new column in the metadata and saves to Parquet.

    Args:
        csv_path: Path to the input CSV file
        image_dir: Directory containing the images
        output_parquet: Path to save the output Parquet file
        image_col: Column name containing image filenames
    """
    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    if image_col not in df.columns:
        raise ValueError(
            f"Column '{image_col}' not found in CSV. Available columns: {df.columns.tolist()}"
        )

    print(f"Processing {len(df)} images...")
    # Create a new column to store image data
    df["data"] = None

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_filename = row[image_col]
        img_path = os.path.join(image_dir, img_filename)

        try:
            # Read image bytes directly without re-encoding
            img_bytes = read_image_bytes(img_path)

            # Store the binary data
            df.at[idx, "data"] = img_bytes
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            # Set a placeholder to avoid NaN values
            df.at[idx, "data"] = b""

    # Save the updated dataframe
    print(f"Saving processed data to {output_parquet}")
    df.to_parquet(output_parquet)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Serialize images and add them to metadata and saves to Parquet"
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument(
        "--image-dir", type=str, required=True, help="Directory containing the images"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the output Parquet file"
    )
    parser.add_argument(
        "--image-col",
        type=str,
        required=False,
        default="filename",
        help="Column name containing image filenames",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Process images and update CSV
    add_serialized_images_to_csv(
        csv_path=args.csv,
        image_dir=args.image_dir,
        output_parquet=args.output,
        image_col=args.image_col,
    )


if __name__ == "__main__":
    main()
