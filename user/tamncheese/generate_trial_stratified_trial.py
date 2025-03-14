import pandas as pd
from sklearn.model_selection import train_test_split

# https://www.geeksforgeeks.org/how-to-implement-stratified-sampling-with-scikit-learn/
# generating a smaller subset for
seed = 2025

# stratified training set 10%
df = pd.read_csv(
    "/storage/scratch1/5/jtam30/FungiCLEF2023_train_metadata_PRODUCTION.csv"
)
label = "species"
print(f"Dataset shape: {df.shape}")
df = df.dropna(subset=[label]).reset_index(drop=True)
print(f"Dataset shape after dropping NaN: {df.shape}")
print(f"Target column shape: {df['species'].shape}")  # Ensure both match

#
trial_df, temp_df = train_test_split(
    df, test_size=0.99, random_state=seed, stratify=df[label]
)

print(f"Train set size: {len(trial_df)}")

print("Class distribution in training set:")
print(trial_df[label].value_counts(normalize=True))
print("Class distribution in full set:")
print(df[label].value_counts(normalize=True))

trial_df.to_csv(
    "/storage/home/hcoda1/5/jtam30/clef/fungiclef-2025/user/tamncheese/_dataset_trial/metadata_trial.csv",
    index=False,
)
