import torch
from transformers import AutoImageProcessor, AutoModel
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

print("finish import")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/dinov2-base"
print("defining processor")
model = AutoModel.from_pretrained(model_name).to(device)
processor = AutoImageProcessor.from_pretrained(model_name)
model.eval()
print("finished setting model to eval")

for param in model.parameters():
    param.requires_grad = False


def preprocess_image_cv2(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return image


def extract_embeddings(image_paths, batch_size=8):
    embeddings = []
    image_names = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[i : i + batch_size]
        print("just before preprocess image")
        batch_images = [preprocess_image_cv2(img_paths) for img_paths in batch_paths]

        model_inputs = processor(images=batch_images, return_tensors="pt")

        # Move inputs to GPU if available

        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

        with torch.no_grad():
            outputs = model(**model_inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings.extend(batch_embeddings.cpu().numpy())
        image_names.extend(batch_paths)
    return np.array(embeddings), image_names


image_dir = os.path.join(os.environ["HOME"], "scratch/fungi2024")
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

batch_size = 64
embeddings, image_names = extract_embeddings(image_files, batch_size=batch_size)

df_embeddings = pd.DataFrame(embeddings)
df_embeddings.insert(0, "image_name", image_names)
df_embeddings.to_parquet("fungi_train_embeddings.parquet", index=False)

print("finished, embeddings saved to parquet file")
