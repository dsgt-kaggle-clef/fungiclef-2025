import pandas as pd
import os

IMAGE_SIZE = "fullsize"


def merge_embedding_metadata(dataset):
    # Load embeddings from parquet file
    embeddings_path = os.path.join(
        os.environ["HOME"],
        f"scratch/fungiclef/embeddings/images_only/{IMAGE_SIZE}_fungi_{dataset}_embeddings.parquet",
    )
    df_embeddings = pd.read_parquet(embeddings_path)
    if dataset == "train":
        metadata_name = "Train"
    elif dataset == "val":
        metadata_name = "Val"
    else:
        metadata_name = "Test"

    # Load metadata file
    metadata_path = os.path.join(
        os.environ["HOME"],
        f"scratch/fungiclef/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-{metadata_name}.csv",
    )
    metadata = pd.read_csv(metadata_path)
    merged_df = pd.merge(df_embeddings, metadata, on="filename", how="inner")

    merged_embeddings_dir = os.path.join(
        os.environ["HOME"],
        f"scratch/fungiclef/embeddings/images_with_metadata/merged_{IMAGE_SIZE}_fungi_{dataset}_embeddings.parquet",
    )

    # Save the merged DataFrame to a parquet file
    merged_df.to_parquet(merged_embeddings_dir, index=False)


names = ["train", "val", "test"]
for name in names:
    merge_embedding_metadata(name)
