import pandas as pd
from pathlib import Path


def count(path=None):
    # Load the CSV file
    df = pd.read_csv(path)

    # Count unique genus
    unique_genus = df["genus"].nunique()
    print(f"Number of unique genus: {unique_genus}")

    # Count unique species
    unique_species = df["species"].nunique()
    print(f"Number of unique species: {unique_species}")
    print("Unique species:", df["species"].unique())
    unique_family = df["family"].nunique()
    print(f"Number of unique families: {unique_family}")


if __name__ == "__main__":
    path = Path(
        "~/scratch/fungiclef/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv"
    )
    count(path)
