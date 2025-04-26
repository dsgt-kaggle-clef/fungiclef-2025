import pandas as pd
import argparse
from pathlib import Path


def main():
    # Load embeddings from parquet file
    parser = argparse.ArgumentParser(description="validate embeddings")
    parser.add_argument(
        "--embeddings", type=Path, required=True, help="path to embeddings parquet file"
    )

    args = parser.parse_args()

    df_embeddings = pd.read_parquet(args.embeddings)
    print(df_embeddings["category_id"].unique())
    print(df_embeddings.loc[df_embeddings["poisonous"] == 1])


if __name__ == "__main__":
    main()
