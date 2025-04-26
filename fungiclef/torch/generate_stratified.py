import pandas as pd
import numpy as np
import os
import math
from sklearn.cluster import KMeans

IMAGE_SIZE = "fullsize"

# https://www.geeksforgeeks.org/how-to-implement-stratified-sampling-with-scikit-learn/
# Generates a stratified sample for training and validation dataset


def generate_stratified_sample(
    df, dataset_name, n_species=10, rare_species_proportion=0.2
):
    """
    Generate a stratified sample of species based on visual similarity and poisonous classification

    Parameters:
    df : Dataframe
    n_species : int
        Number of species to include in the sample

    Returns:
    list
        Category IDs selected for the stratified sample
    """
    species_avg_embeddings = {}
    species_list = []

    # get embedding columns
    metadata_cols = [
        "filename",
        "category_id",
        "poisonous",
        "eventDate",
        "year",
        "month",
        "day",
        "habitat",
        "countryCode",
        "scientificName",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "specificEpithet",
        "hasCoordinate",
        "species",
        "iucnRedListCategory",
        "substrate",
        "latitude",
        "longitude",
        "coorUncert",
        "observationID",
        "region",
        "district",
        "metaSubstrate",
        "elevation",
        "landcover",
        "biogeographicalRegion",
    ]
    embeddings_cols = [col for col in df.columns if col not in metadata_cols]

    species_counts = df["category_id"].value_counts()

    count_threshold = species_counts.quantile(0.2)
    rare_species = species_counts[species_counts <= count_threshold].index.tolist()

    # Get proportion of species that are poisonous
    species_poisonous_status = {
        cat_id: group["poisonous"].mode()[0]
        for cat_id, group in df.groupby("category_id")
    }
    natural_poisonous_proportion = sum(species_poisonous_status.values()) / len(
        species_poisonous_status
    )
    print(sum(species_poisonous_status.values()))
    print("Natural poisonous proportion:", natural_poisonous_proportion)
    # Calculate target number of poisonous/non-poisonous
    n_poisonous = math.ceil(n_species * natural_poisonous_proportion)
    print(f"n_poisonous is {n_poisonous}")
    n_non_poisonous = n_species - n_poisonous
    print(f"n_non_poisonous is {n_non_poisonous}")
    # Calculate the target number of rare species
    n_rare = int(n_species * rare_species_proportion)
    n_common = n_species - n_rare

    # Determine how many rare species should be poisonous/non-poisonous
    # (using the same natural proportion)
    n_rare_poisonous = int(n_rare * natural_poisonous_proportion)
    n_rare_non_poisonous = n_rare - n_rare_poisonous
    n_common_poisonous = n_poisonous - n_rare_poisonous
    n_common_non_poisonous = n_common - n_common_poisonous
    print(f"n_common_non_poisonous is {n_common_non_poisonous}")
    for category_id, group in df.groupby("category_id"):
        # Calculate mean embedding for this species (centroid)
        avg_embedding = group[embeddings_cols].values.mean(axis=0)
        species_avg_embeddings[category_id] = avg_embedding
        species_list.append(category_id)

    # Get poisonous status for each species
    species_poisonous = {}
    for category_id, group in df.groupby("category_id"):
        # Handle poisonous misclassification in the metadata
        is_poisonous = group["poisonous"].mode()[0]
        species_poisonous[category_id] = is_poisonous

    # Clustering
    species_embeddings = np.array(
        [species_avg_embeddings[sp_id] for sp_id in species_list]
    )

    # Normalize
    # species_embeddings = species_embeddings / np.linalg.norm(species_embeddings, axis=1, keepdims=True)

    # Get poisonous status for each species
    poisonous_status = np.array([species_poisonous[sp_id] for sp_id in species_list])

    # Calculate target distribution
    rare_poisonous_indices = [
        i
        for i, sp_id in enumerate(species_list)
        if sp_id in rare_species and poisonous_status[i] == 1
    ]
    rare_non_poisonous_indices = [
        i
        for i, sp_id in enumerate(species_list)
        if sp_id in rare_species and poisonous_status[i] == 0
    ]
    common_poisonous_indices = [
        i
        for i, sp_id in enumerate(species_list)
        if sp_id not in rare_species and poisonous_status[i] == 1
    ]
    common_non_poisonous_indices = [
        i
        for i, sp_id in enumerate(species_list)
        if sp_id not in rare_species and poisonous_status[i] == 0
    ]
    print("Rare poisonous indices:", rare_poisonous_indices)
    print("Common poisonous indices:", common_poisonous_indices)
    selected_indices = []
    print(f"n_rare_poisonous is: {n_rare_poisonous}")
    if rare_poisonous_indices and n_rare_poisonous > 0:
        if len(rare_poisonous_indices) <= n_rare_poisonous:
            selected_indices.extend(rare_poisonous_indices)
            n_common_poisonous += n_rare_poisonous - len(rare_poisonous_indices)
        else:
            # clustering
            rare_poison_embeddings = species_embeddings[rare_poisonous_indices]
            rp_kmeans = KMeans(n_cluster=n_rare_poisonous, random_state=42).fit(
                rare_poison_embeddings
            )
            rp_centers = rp_kmeans.cluster_centers_

        for center in rp_centers:
            distances = np.linalg.norm(rare_poison_embeddings - center, axis=1)
            closest_idx = rare_poisonous_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)
    else:
        n_common_poisonous += n_rare_poisonous

    # Select rare non-poisonous species
    if rare_non_poisonous_indices and n_rare_non_poisonous > 0:
        if len(rare_non_poisonous_indices) <= n_rare_non_poisonous:
            selected_indices.extend(rare_non_poisonous_indices)
            n_common_non_poisonous += n_rare_non_poisonous - len(
                rare_non_poisonous_indices
            )
        else:
            rnp_embeddings = species_embeddings[rare_non_poisonous_indices]
            rnp_kmeans = KMeans(n_clusters=n_rare_non_poisonous, random_state=42).fit(
                rnp_embeddings
            )
            rnp_centers = rnp_kmeans.cluster_centers_

            for center in rnp_centers:
                distances = np.linalg.norm(rnp_embeddings - center, axis=1)
                closest_idx = rare_non_poisonous_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
    else:
        # If no rare non-poisonous species, allocate to common non-poisonous
        n_common_non_poisonous += n_rare_non_poisonous
    print(f"n_common_poisonous is: {n_common_poisonous}")
    # Select common poisonous species
    if common_poisonous_indices and n_common_poisonous > 0:
        if len(common_poisonous_indices) <= n_common_poisonous:
            selected_indices.extend(common_poisonous_indices)
        else:
            cp_embeddings = species_embeddings[common_poisonous_indices]
            cp_kmeans = KMeans(n_clusters=n_common_poisonous, random_state=42).fit(
                cp_embeddings
            )
            cp_centers = cp_kmeans.cluster_centers_

            for center in cp_centers:
                distances = np.linalg.norm(cp_embeddings - center, axis=1)
                closest_idx = common_poisonous_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
    # Select common non-poisonous species
    if common_non_poisonous_indices and n_common_non_poisonous > 0:
        if len(common_non_poisonous_indices) <= n_common_non_poisonous:
            selected_indices.extend(common_non_poisonous_indices)
        else:
            cnp_embeddings = species_embeddings[common_non_poisonous_indices]
            cnp_kmeans = KMeans(n_clusters=n_common_non_poisonous, random_state=42).fit(
                cnp_embeddings
            )
            cnp_centers = cnp_kmeans.cluster_centers_

            for center in cnp_centers:
                distances = np.linalg.norm(cnp_embeddings - center, axis=1)
                closest_idx = common_non_poisonous_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)

    # Convert indices back to species IDs
    selected_species = [species_list[i] for i in selected_indices]

    stratified_embeddings_dir = os.path.join(
        os.environ["HOME"],
        f"scratch/fungiclef/embeddings/stratified/stratified_{IMAGE_SIZE}_fungi_{dataset_name}_embeddings.parquet",
    )

    # Create a DataFrame with just the stratified sample
    stratified_df = df[df["category_id"].isin(selected_species)]

    # Save the stratified DataFrame to a parquet file
    stratified_df.to_parquet(stratified_embeddings_dir, index=False)

    return selected_species


names = ["train", "val"]
for name in names:
    # Load embeddings from parquet file
    df_path = os.path.join(
        os.environ["HOME"],
        f"scratch/fungiclef/embeddings/images_with_metadata/merged_{IMAGE_SIZE}_fungi_{name}_embeddings.parquet",
    )
    df = pd.read_parquet(df_path)

    selected_species = generate_stratified_sample(
        df, name, n_species=10, rare_species_proportion=0.2
    )  # using 20% for rare species proportion, 20% is the number stated in the Motivation of the Kaggle dataset
    print(selected_species)
