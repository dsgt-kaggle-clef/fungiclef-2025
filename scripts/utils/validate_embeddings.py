import pandas as pd
import os

# Load embeddings from parquet file
embeddings_path = os.path.join(
    os.environ["HOME"],
    "scratch/fungiclef/embeddings/stratified_720p_fungi_train_embeddings.parquet",
)
df_embeddings = pd.read_parquet(embeddings_path)

print(df_embeddings["category_id"].unique())
print(df_embeddings.loc[df_embeddings["poisonous"] == 1])
