"""Script to validate embeddings stored in parquet format."""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

### example usage python clef/fungiclef-2025/fungiclef/preprocessing/validate_embeddings.py --input scratch/fungiclef/embeddings/train_embeddings.parquet


def validate_embeddings(parquet_path: str):
    """
    Validate embeddings stored in a parquet file.

    Args:
        parquet_path: Path to the parquet file with embeddings
    """
    print(f"Loading embeddings from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Check if embedding column exists
    if "embedding" not in df.columns:
        print("ERROR: 'embedding' column not found in the parquet file")
        return

    # Extract embeddings as numpy arrays
    embeddings = np.stack(df["embedding"].values)

    # Get labels if available
    if "category_id" in df.columns:
        labels = df["category_id"].values
        has_labels = True
    else:
        labels = None
        has_labels = False

    # Basic information
    print(f"Total samples: {len(df)}")
    print(f"Embeddings shape: {embeddings.shape}")
    if has_labels:
        print(f"Number of unique labels: {len(np.unique(labels))}")

    # Check embedding statistics
    print("\nEmbedding Statistics:")
    print(f"Mean: {embeddings.mean():.6f}")
    print(f"Std: {embeddings.std():.6f}")
    print(f"Min: {embeddings.min():.6f}")
    print(f"Max: {embeddings.max():.6f}")

    # Check for NaN or infinity values
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    print("\nQuality Checks:")
    print(f"NaN values: {nan_count} ({nan_count/(embeddings.size)*100:.6f}%)")
    print(f"Infinity values: {inf_count} ({inf_count/(embeddings.size)*100:.6f}%)")

    # Check for zeros (potential issues)
    zero_embeddings = (np.abs(embeddings).sum(axis=1) == 0).sum()
    print(
        f"Zero embeddings: {zero_embeddings} ({zero_embeddings/len(embeddings)*100:.6f}%)"
    )

    # Check for duplicate embeddings
    # This might be slow for large datasets, so we'll check just a sample if needed
    if len(embeddings) > 10000:
        sample_size = 10000
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        print(f"\nChecking for duplicates on a sample of {sample_size} embeddings...")

        # Simple check: count unique dot products in the sample
        dot_products = np.dot(sample_embeddings, sample_embeddings.T)
        np.fill_diagonal(dot_products, 0)  # Remove self-similarity
        high_similarity = (dot_products > 0.99).sum()
        print(f"Highly similar embedding pairs (>0.99 similarity): {high_similarity}")
    else:
        print("\nChecking for duplicates across all embeddings...")
        dot_products = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(dot_products, 0)  # Remove self-similarity
        high_similarity = (dot_products > 0.99).sum()
        print(f"Highly similar embedding pairs (>0.99 similarity): {high_similarity}")

    # Embedding dimensionality check
    embedding_dim = embeddings.shape[1]
    print(f"\nEmbedding dimensionality: {embedding_dim}")

    # Get model name if available
    if "model_name" in df.columns:
        model_name = df["model_name"].iloc[0]
        print(f"Model used: {model_name}")

    print("\nValidation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Validate embeddings stored in parquet format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input parquet file with embeddings",
    )

    args = parser.parse_args()

    # Validate that the file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {args.input}")
        return

    # Validate embeddings
    validate_embeddings(args.input)


if __name__ == "__main__":
    main()
