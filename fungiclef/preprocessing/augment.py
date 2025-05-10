"""Script to augment fungi dataset using PyTorch transforms and save as serialized parquet."""

import pandas as pd
import os
import io
import random
from PIL import Image
from collections import Counter
from tqdm import tqdm
from transform import get_transform_pipeline


def augment_and_serialize(
    csv_path,
    image_dir,
    output_parquet,
    target_col="category_id",
    image_col="filename",
    img_size=224,
    scale_factor=10,  # Scale factor parameter instead of balance
):
    """
    Augment fungi dataset by a specified scale factor and serialize directly to parquet

    Parameters:
    -----------
    csv_path : str
        Path to the metadata CSV file
    image_dir : str
        Directory containing the original images
    output_parquet : str
        Path to save the serialized augmented dataset
    target_col : str
        Column name for the target classes
    image_col : str
        Column containing image filenames
    img_size : int
        Size to use for image transforms
    scale_factor : float
        Factor by which to multiply the dataset size (e.g., 10 for 10x more images)
    """
    print(f"Loading metadata from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples with {df[target_col].nunique()} classes")

    # Analyze class distribution (just for information)
    class_counts = Counter(df[target_col])
    total_samples = len(df)
    num_classes = len(class_counts)
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count

    print("Class distribution statistics:")
    print(f"- Total samples: {total_samples}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Smallest class: {min_count} samples")
    print(f"- Largest class: {max_count} samples")
    print(f"- Target scale factor: {scale_factor}x")
    print(f"- Expected output size: ~{int(total_samples * scale_factor)} samples")
    print(f"- Imbalance ratio: {imbalance_ratio:.2f}")

    # Create a list to store serialized data
    serialized_data = []

    # First, include ALL original images
    print("\nSerializing all original images first...")
    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Serializing original images"
    ):
        img_path = os.path.join(image_dir, row[image_col])

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        try:
            # Load and convert image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size))

            # Serialize image
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            img_data = img_bytes.getvalue()

            # Create row data
            row_data = row.to_dict()
            row_data["data"] = img_data
            row_data["augmented"] = False  # Flag as original

            serialized_data.append(row_data)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Calculate how many augmented images to create per original image
    aug_per_image = scale_factor - 1  # We already added the originals

    # Get the transform pipeline
    transform = get_transform_pipeline()

    # Process each image and create augmented copies
    print(f"\nGenerating {aug_per_image:.1f}x augmented images per original image...")
    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Generating augmented images"
    ):
        img_path = os.path.join(image_dir, row[image_col])

        if not os.path.exists(img_path):
            tqdm.write(f"Warning: Image not found: {img_path}")
            continue

        try:
            # Load the image using PIL
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size))
        except Exception as e:
            tqdm.write(f"Warning: Could not read image: {img_path} - {e}")
            continue

        # Determine exact number of augmentations for this image
        # This accounts for non-integer scale factors (e.g., 9.5x)
        n_augmentations = int(aug_per_image)
        # Probabilistic rounding for the fractional part
        if random.random() < (aug_per_image - n_augmentations):
            n_augmentations += 1

        # Apply augmentations
        for i in range(n_augmentations):
            try:
                # Apply the transformation
                augmented = transform(img)

                # Serialize augmented image
                img_bytes = io.BytesIO()
                augmented.save(img_bytes, format="JPEG")
                img_data = img_bytes.getvalue()

                # Create augmented row data
                aug_row = row.to_dict()
                aug_row["data"] = img_data
                aug_row["augmented"] = True
                aug_row["aug_id"] = i + 1  # Augmentation ID (1-based)

                serialized_data.append(aug_row)
            except Exception as e:
                tqdm.write(f"Warning: Failed to augment image {img_path}: {e}")
                continue

    # Create DataFrame and save to parquet
    serialized_df = pd.DataFrame(serialized_data)

    # Print final statistics
    print("\nAugmentation completed!")
    print(f"Original dataset: {total_samples} samples")
    print(f"Augmented dataset: {len(serialized_df)} samples")
    print(f"Actual scale factor: {len(serialized_df) / total_samples:.2f}x")

    # Save to parquet
    serialized_df.to_parquet(output_parquet, index=False)
    print(f"Saved {len(serialized_df)} serialized images to {output_parquet}")


if __name__ == "__main__":
    pass
    # parser = argparse.ArgumentParser(description='Augment and serialize fungi dataset')
    # parser.add_argument('--metadata', type=str, required=True,
    #                     help='Path to metadata CSV file')
    # parser.add_argument('--images', type=str, required=True,
    #                     help='Directory containing original images')
    # parser.add_argument('--output', type=str, required=True,
    #                     help='Path to save serialized augmented dataset')
    # parser.add_argument('--target-col', type=str, default='category_id',
    #                     help='Target column for classification')
    # parser.add_argument('--img-col', type=str, default='filename',
    #                     help='Column containing image filenames')
    # parser.add_argument('--img-size', type=int, default=224,
    #                     help='Image size for transforms')
    # parser.add_argument('--scale', type=float, default=10.0,
    #                     help='Scale factor for augmentation (e.g., 10 for 10x)')
    # args = parser.parse_args()

    # # Create output directory if needed
    # Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # augment_and_serialize(
    #     csv_path=args.metadata,
    #     image_dir=args.images,
    #     output_parquet=args.output,
    #     target_col=args.target_col,
    #     image_col=args.img_col,
    #     img_size=args.img_size,
    #     scale_factor=args.scale,
    # )
