import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

### This assume the images are saved in scratch/fungiclef/dataset
### extract embeddings from the images using DINO into a Pandas Dataframe and saves as parquet files


def extract_embeddings(image_paths, model, processor, device, batch_size=8):
    """Function to generate embeddings for the images"""
    embeddings = []
    image_names = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[i : i + batch_size]
        print("Loading Images")
        batch_images = [cv2.imread(img_path)[:, :, ::-1] for img_path in batch_paths]
        print("Finish Loading Images")
        model_inputs = processor(images=batch_images, return_tensors="pt")

        # Move inputs to GPU if available

        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

        with torch.no_grad():
            outputs = model(**model_inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

            # Normalize the embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

        embeddings.extend(batch_embeddings.cpu().numpy())
        filenames = [os.path.basename(path) for path in batch_paths]
        image_names.extend(filenames)
    return np.array(embeddings), image_names


def train_embeddings(image_size):
    image_dir = os.path.join(
        os.environ["HOME"],
        f"scratch/fungiclef/dataset/images/FungiTastic-FewShot/train/{image_size}",
    )
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    batch_size = 256
    embeddings, image_names = extract_embeddings(image_files, batch_size=batch_size)

    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.insert(0, "filename", image_names)
    embeddings_dir = os.path.join(
        os.environ["HOME"],
        f"scratch/fungiclef/embeddings/images_only/{image_size}_fungi_train_embeddings.parquet",
    )
    df_embeddings.to_parquet(embeddings_dir, index=False)

    print("finished, training embeddings saved to parquet file")


def val_embeddings(image_size):
    image_dir = os.path.join(
        os.environ["HOME"],
        f"scratch/fungiclef/dataset/images/FungiTastic-FewShot/val/{image_size}",
    )
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    batch_size = 256
    embeddings, image_names = extract_embeddings(image_files, batch_size=batch_size)

    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.insert(0, "filename", image_names)
    embeddings_dir = os.path.join(
        os.environ["HOME"],
        f"scratch/fungiclef/embeddings/images_only/{image_size}_fungi_val_embeddings.parquet",
    )
    df_embeddings.to_parquet(embeddings_dir, index=False)

    print("finished, validation embeddings saved to parquet file")


def test_embeddings(image_size):
    image_dir = os.path.join(
        os.environ["HOME"],
        f"scratch/fungiclef/dataset/images/FungiTastic-FewShot/test/{image_size}",
    )
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    batch_size = 256
    embeddings, image_names = extract_embeddings(image_files, batch_size=batch_size)

    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.insert(0, "filename", image_names)
    embeddings_dir = os.path.join(
        os.environ["HOME"],
        f"scratch/fungiclef/embeddings/images_only/{image_size}_fungi_test_embeddings.parquet",
    )
    df_embeddings.to_parquet(embeddings_dir, index=False)

    print("finished, testing embeddings saved to parquet file")


if __name__ == "__main__":
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

    image_size = "fullsize"
    train_embeddings(image_size)
    val_embeddings(image_size)
    test_embeddings(image_size)
