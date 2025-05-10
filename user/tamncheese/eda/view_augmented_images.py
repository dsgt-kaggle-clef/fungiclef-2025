"""
Script to display the head of augmented images from a parquet file.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
from pathlib import Path

### example usage python clef/fungiclef-2025/user/tamncheese/eda/view_augmented_images.py --parquet /home/jasonkhtam7/Documents/DSGT/scratch/fungiclef/processed/train_augment_serialized.parquet --num 10 --output /home/jasonkhtam7/Documents/DSGT/clef/fungiclef-2025/user/tamncheese/eda/augment.png


def display_head_images(
    parquet_path,
    num_images=5,
    figsize=(15, 12),
    include_original=True,
    include_augmented=True,
    target_col="category_id",
    output_path=None,
):
    """
    Display the first few images from a parquet file containing serialized image data.

    Parameters:
    -----------
    parquet_path : str
        Path to the parquet file containing serialized images
    num_images : int
        Number of images to display
    figsize : tuple
        Figure size for the matplotlib plot
    include_original : bool
        Whether to include original images
    include_augmented : bool
        Whether to include augmented images
    target_col : str
        Column name for class/category labels
    """
    # Load the parquet file
    print(f"Loading parquet file from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Check if the dataframe has the expected columns
    if "data" not in df.columns:
        raise ValueError(
            "The parquet file does not contain a 'data' column with image data"
        )

    # Check if augmentation info is available
    has_augmentation_info = "augmented" in df.columns

    # Filter images based on original/augmented if specified
    if has_augmentation_info:
        display_df = pd.DataFrame()

        if include_original:
            original_df = df[not df["augmented"]]
            if len(original_df) > 0:
                display_df = pd.concat([display_df, original_df.head(num_images)])

        if include_augmented:
            augmented_df = df[df["augmented"]]
            if len(augmented_df) > 0:
                display_df = pd.concat([display_df, augmented_df.head(num_images)])

        if len(display_df) == 0:
            display_df = df.head(num_images)
    else:
        display_df = df.head(num_images)

    # Determine the total number of images to display
    total_images = len(display_df)
    if total_images == 0:
        print("No images to display!")
        return

    # Print metadata about the dataset
    print(f"Dataset contains {len(df)} total images")
    if has_augmentation_info:
        num_original = sum(~df["augmented"]) if "augmented" in df.columns else "Unknown"
        num_augmented = sum(df["augmented"]) if "augmented" in df.columns else "Unknown"
        print(f"- Original images: {num_original}")
        print(f"- Augmented images: {num_augmented}")

    print(f"\nDisplaying {total_images} images from the dataset")

    # Calculate grid dimensions
    n_cols = min(5, total_images)
    n_rows = (total_images + n_cols - 1) // n_cols  # Ceiling division

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Make axes iterable even for a single image
    if total_images == 1:
        axes = np.array([axes])

    # Flatten axes for easy iteration
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Display images
    for i, (idx, row) in enumerate(display_df.iterrows()):
        if i >= len(axes):
            break

        try:
            # Load image from binary data
            img_data = row["data"]
            img = Image.open(io.BytesIO(img_data))

            # Display image
            axes[i].imshow(img)

            # Create title with relevant metadata
            title_parts = []

            # Add class/category if available
            if target_col in row and row[target_col] is not None:
                title_parts.append(f"Class: {row[target_col]}")

            # Add original/augmented info if available
            if has_augmentation_info:
                is_augmented = row["augmented"]
                aug_info = "Augmented" if is_augmented else "Original"
                title_parts.append(aug_info)

                # Add augmentation ID if available and it's an augmented image
                if is_augmented and "aug_id" in row and row["aug_id"] is not None:
                    title_parts.append(f"ID: {row['aug_id']}")

            # Set title
            axes[i].set_title(" | ".join(title_parts))
            axes[i].axis("off")
        except Exception as e:
            print(f"Error displaying image at index {idx}: {e}")
            axes[i].text(
                0.5,
                0.5,
                f"Error: {str(e)[:50]}...",
                horizontalalignment="center",
                verticalalignment="center",
            )
            axes[i].axis("off")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)


def main():
    parser = argparse.ArgumentParser(description="Display head of augmented images")
    parser.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="Path to parquet file with serialized images",
    )
    parser.add_argument(
        "--num", type=int, default=5, help="Number of images to display of each type"
    )
    parser.add_argument(
        "--original", action="store_true", default=True, help="Include original images"
    )
    parser.add_argument(
        "--augmented",
        action="store_true",
        default=True,
        help="Include augmented images",
    )
    parser.add_argument(
        "--original-only", action="store_true", help="Display only original images"
    )
    parser.add_argument(
        "--augmented-only", action="store_true", help="Display only augmented images"
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="category_id",
        help="Column name for class/category labels",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save display"
    )

    args = parser.parse_args()

    # Handle mutually exclusive options
    if args.original_only:
        include_original = True
        include_augmented = False
    elif args.augmented_only:
        include_original = False
        include_augmented = True
    else:
        include_original = args.original
        include_augmented = args.augmented

    # Check if parquet file exists
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}")
        return

    # Display images
    display_head_images(
        parquet_path=args.parquet,
        num_images=args.num,
        include_original=include_original,
        include_augmented=include_augmented,
        target_col=args.target_col,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
