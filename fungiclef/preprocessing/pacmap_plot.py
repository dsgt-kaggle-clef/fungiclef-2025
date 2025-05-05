import pandas as pd
import numpy as np
import matplotlib

# Force non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pacmap
import argparse
from pathlib import Path
import os

### example usage: python clef/fungiclef-2025/fungiclef/preprocessing/pacmap_plot.py --input scratch/fungiclef/embeddings/train_embeddings.parquet --output clef/fungiclef-2025/fungiclef/preprocessing/elbow_analysis.png --analyze-only
### example usage: python clef/fungiclef-2025/fungiclef/preprocessing/pacmap_plot.py --input scratch/fungiclef/embeddings/train_embeddings.parquet --output clef/fungiclef-2025/fungiclef/preprocessing/train_pacmap.png --clusters 8


def plot_fungi_pacmap(
    df: pd.DataFrame,
    embedding_col: str = "embedding",
    label_col: str = "category_id",
    title: str = "PaCMAP of Fungi Embeddings",
    n_clusters: int = 5,
    output_file: str = None,
):
    """
    Create a PaCMAP visualization of fungi embeddings with cluster coloring.

    Args:
        df: DataFrame containing embeddings and labels
        embedding_col: Column name for embeddings
        label_col: Column name for labels/categories
        title: Plot title
        n_clusters: Number of clusters for K-means
        output_file: Path to save the output image
    """
    # Extract embeddings and stack them into a 2D array
    print("Extracting embeddings...")
    embeddings = np.stack(df[embedding_col].values)

    # Apply K-means clustering
    print(f"Applying K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    # Create PaCMAP instance
    reducer = pacmap.PaCMAP(
        n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42
    )

    # Reduce dimensions
    print("Performing PaCMAP dimensionality reduction...")
    embedded = reducer.fit_transform(embeddings)

    # Create figure
    plt.figure(figsize=(12, 12))

    # Create scatter plot with clusters as colors
    scatter = plt.scatter(
        embedded[:, 0], embedded[:, 1], c=clusters, cmap="tab10", alpha=0.7, s=10
    )

    # Add legend for clusters
    legend1 = plt.legend(
        *scatter.legend_elements(), loc="upper right", title="Clusters"
    )
    plt.gca().add_artist(legend1)

    # Add label information if available
    if label_col in df.columns:
        # Create a mapping between clusters and most common label in each cluster
        cluster_label_map = {}
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_labels = df.loc[cluster_mask, label_col].values
            if len(cluster_labels) > 0:
                # Find most frequent label in this cluster
                unique, counts = np.unique(cluster_labels, return_counts=True)
                most_common_label = unique[np.argmax(counts)]
                cluster_label_map[cluster_id] = most_common_label

        # Add a text annotation about cluster compositions
        cluster_info = "\n".join(
            [f"Cluster {k}: Mostly {v}" for k, v in cluster_label_map.items()]
        )
        plt.figtext(
            0.02,
            0.02,
            cluster_info,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    plt.title(title, fontsize=18)
    plt.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Save figure to output file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        print(f"Visualization saved to {output_file}")

    # Close the plot to free resources
    plt.close()

    return embedded, clusters


def analyze_optimal_clusters(embeddings, max_k=15, output_file=None):
    """
    Use the elbow method to find the optimal number of clusters and save the plot.
    Only analyzes cluster counts without creating the full PaCMAP visualization.

    Args:
        embeddings: Numpy array of embeddings
        max_k: Maximum number of clusters to consider
        output_file: Path to save the elbow curve plot
    """
    print(f"Analyzing optimal cluster count (max k={max_k})...")
    distortions = []
    K = range(1, max_k + 1)

    # Calculate distortion for different cluster counts
    for k in K:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)

    # Print distortion values to console for reference
    print("\nElbow Method Results:")
    print("Number of Clusters | Distortion")
    print("-" * 30)
    for k, d in zip(K, distortions):
        print(f"{k:^16} | {d:.2f}")

    # Calculate rate of change (first derivative)
    derivatives = []
    for i in range(1, len(distortions)):
        derivatives.append(distortions[i - 1] - distortions[i])

    # Calculate second derivative to find the elbow point
    second_derivatives = []
    for i in range(1, len(derivatives)):
        second_derivatives.append(derivatives[i - 1] - derivatives[i])

    # Find potential elbow points
    elbow_points = [i + 2 for i, val in enumerate(second_derivatives) if val > 0]
    if elbow_points:
        suggested_k = elbow_points[0]
        print(f"\nSuggested optimal cluster count: {suggested_k}")
        print(f"All potential elbow points: {elbow_points}")
    else:
        suggested_k = 5
        print("\nNo clear elbow point detected. Defaulting to 5 clusters.")

    # Plot the elbow curve
    plt.figure(figsize=(12, 8))

    # Main plot: distortion vs. number of clusters
    plt.subplot(2, 1, 1)
    plt.plot(K, distortions, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Distortion", fontsize=12)
    plt.title("Elbow Method for Optimal k", fontsize=14)
    plt.grid(True)

    # Mark the suggested optimal k
    plt.axvline(
        x=suggested_k, color="r", linestyle="--", label=f"Suggested k={suggested_k}"
    )
    plt.legend()

    # Second plot: rate of change (first derivative)
    plt.subplot(2, 1, 2)
    plt.plot(K[1:], derivatives, "go-", linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Rate of Change", fontsize=12)
    plt.title("Rate of Improvement with Increasing Clusters", fontsize=14)
    plt.grid(True)

    # Mark the suggested optimal k on the derivative plot too
    plt.axvline(
        x=suggested_k, color="r", linestyle="--", label=f"Suggested k={suggested_k}"
    )
    plt.legend()

    plt.tight_layout()

    # Save figure if output file provided
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        plt.savefig(output_file)
        print(f"Elbow analysis saved to {output_file}")

    # Always close plot to free resources
    plt.close()

    return suggested_k


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Create PaCMAP visualization for fungi embeddings"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input parquet file containing embeddings",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output visualization image",
    )
    parser.add_argument(
        "--embedding-col",
        type=str,
        default="embedding",
        help="Column name containing embedding vectors (default: embedding)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="category_id",
        help="Column name containing category or species labels (default: category_id)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of clusters for K-means (default: 5)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="PaCMAP of Fungi Embeddings",
        help="Title for the plot (default: PaCMAP of Fungi Embeddings)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=15,
        help="Maximum number of clusters to consider in optimization (default: 15)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only perform cluster analysis without generating PaCMAP visualization",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} does not exist")
        return

    # Load embeddings
    print(f"Loading embeddings from {args.input}")
    df = pd.read_parquet(args.input)

    # Check if embedding column exists
    if args.embedding_col not in df.columns:
        print(f"Error: Embedding column '{args.embedding_col}' not found in the data")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Extract embeddings
    embeddings = np.stack(df[args.embedding_col].values)

    # Analysis-only mode: Just find optimal clusters and exit
    if args.analyze_only:
        analyze_optimal_clusters(embeddings, max_k=args.max_k, output_file=args.output)
        print(
            "\nAnalysis complete. Run the script again with your chosen cluster count:"
        )
        print(
            f"python pacmap_plot.py --input {args.input} --output path/to/pacmap.png --clusters [YOUR_CHOSEN_K]"
        )
        return

    # Full visualization mode: Generate PaCMAP visualization
    plot_fungi_pacmap(
        df=df,
        embedding_col=args.embedding_col,
        label_col=args.label_col,
        title=args.title,
        n_clusters=args.clusters,
        output_file=args.output,
    )

    print("\nPaCMAP visualization complete!")


if __name__ == "__main__":
    main()
