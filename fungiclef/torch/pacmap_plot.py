import pandas as pd
import os
import matplotlib.pyplot as plt
import pacmap

# Load embeddings from parquet file
embeddings_path = os.path.join(
    os.environ["HOME"],
    "scratch/fungiclef/embeddings/images_only/fullsize_fungi_train_embeddings.parquet",
)
df = pd.read_parquet(embeddings_path)

# Separate embeddings and labels
image_names = df["filename"].values
embeddings = df.drop(columns=["filename"]).values
# embeddings were normalized from embedding.py

# Apply PaCMAP for dimensionality reduction to 2D
print("Applying PaCMAP...")
embedding_reducer = pacmap.PaCMAP(n_components=2)
reduced_embeddings = embedding_reducer.fit_transform(embeddings)

# Plot the clustering chart
plt.figure(figsize=(8, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=5, alpha=0.8)
plt.title("PaCMAP of DINOv2 CLS Features, Training Images")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plot_path = os.path.join(
    os.environ["HOME"],
    "clef/fungiclef-2025/fungiclef/embedding/train_pacmap.png",
)
plt.savefig(plot_path)
