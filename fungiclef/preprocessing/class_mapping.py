import pandas as pd
import argparse
from pathlib import Path
### example usage: python clef/fungiclef-2025/fungiclef/preprocessing/class_mapping.py --metadata scratch/fungiclef/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv --output clef/fungiclef-2025/fungiclef/class_mapping.txt


def generate_class_mappings(
    metadata_path: str, output_path: str = "class_mappings.txt"
):
    """
    Generates a class mappings file from a metadata CSV.
    Each line in the file corresponds to a class index and contains a category_id.

    Args:
        metadata_path (str): Path to the CSV file containing a 'category_id' column.
        output_path (str): Path to save the class mappings file.
    """
    df = pd.read_csv(metadata_path)

    if "genus" not in df.columns:
        raise ValueError("Expected column 'category_id' in metadata CSV.")

    # Get sorted unique species IDs
    unique_ids = sorted(df["genus"].unique())
    # index_to_category_id = {i: cid for i, cid in enumerate(unique_ids)}

    # Write them to the output file
    with open(output_path, "w") as f:
        for cid in unique_ids:
            f.write(f"{cid}\n")

    print(f"Saved {len(unique_ids)} class mappings to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate class mappings from metadata CSV"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("../class_mappings.txt"),
        required=True,
        help="Path to metadata CSV file (must contain 'category_id' column)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../class_mappings.txt"),
        help="Output path for class mappings file",
    )

    args = parser.parse_args()
    generate_class_mappings(args.metadata, args.output)


if __name__ == "__main__":
    main()
