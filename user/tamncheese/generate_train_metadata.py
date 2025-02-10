import pandas as pd
import os

all_images_df = pd.read_csv("all.csv", dtype=str)
selected_images = []
for root, dirs, files, in os.walk("training"):
    for file in files:
        selected_images.append(file.replace(".jpg", ".JPG"))
print(selected_images)
print(len(selected_images))



filtered_df = all_images_df[all_images_df['image_path'].isin(selected_images)]
print(filtered_df)
filtered_df.to_csv("training.csv", index=False)