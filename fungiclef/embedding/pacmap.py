import pandas as pd
import os
import matplotlib.pyplot as plt
import pacmap  # Efficient for visualizing high-dimensional data
from sklearn.preprocessing import StandardScaler

# Load embeddings from parquet file
# Load embeddings from parquet file
embeddings_path = os.path.join(
    os.environ["HOME"],
    "scratch/fungiclef/embeddings/720p_fungi_train_embeddings.parquet",
)
df = pd.read_parquet(embeddings_path)

# Separate embeddings and labels
image_names = df["filename"].values
embeddings = df.drop(columns=["filename"]).values

# Standardize the embeddings (important for PaCMAP)
scaler = StandardScaler()
embeddings = scaler.fit_transform(embeddings)

# Apply PaCMAP for dimensionality reduction to 2D
print("Applying PaCMAP...")
embedding_reducer = pacmap.PaCMAP(n_components=2)
reduced_embeddings = embedding_reducer.fit_transform(embeddings)

# Plot the clustering chart
plt.figure(figsize=(8, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=5, alpha=0.8)
plt.title("PaCMAP of DINOv2 CLS Features")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
