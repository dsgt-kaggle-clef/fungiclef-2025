import pandas as pd
from pathlib import Path

all_images_df = pd.read_csv("all.csv", dtype=str)
selected_images = list(Path("training_laptop").glob("[0-9]*-[0-9]*.jpg"))
print(selected_images)
print(len(selected_images))


filtered_df = all_images_df[all_images_df["image_path"].isin(selected_images)]
print(filtered_df)
filtered_df.to_csv("training_laptop.csv", index=False)
