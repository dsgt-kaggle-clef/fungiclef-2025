import torch
from tqdm import tqdm


## work in progress, training pipeline


def extract_and_save_embeddings(model, dataloader, df, output_path):
    """Extract embeddings and save them with labels."""
    model.eval()

    # Extract embeddings
    all_embeddings = []
    all_labels = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, labels = batch
            images = images.to(model.model_device)
            embeddings = model.extract_embeddings(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    # Concatenate embeddings and labels
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Save embeddings and labels
    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
            "label_mapping": model.cid_to_spid,
        },
        output_path,
    )

    print(f"Embeddings saved to {output_path}")
