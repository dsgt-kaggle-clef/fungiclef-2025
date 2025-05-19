import typer
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from fungiclef.config import get_class_mappings_file
from .classifier import MultiModalClassifier, MultiObjectiveClassifier
from .datamodule import MultiModalDataModule


def load_class_mappings(class_mappings_file: str = None) -> dict:
    with open(class_mappings_file, "r") as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_and_merge_embeddings(
    parquet_path: str,
    embed_path: str,
    columns: list,
    embedding_col: str = "embeddings",
) -> pd.DataFrame:
    df_meta = pd.read_parquet(parquet_path, columns=columns)
    df_embed = pd.read_parquet(embed_path, columns=["filename", embedding_col])
    return df_meta.merge(df_embed, on="filename", how="inner")


def generate_predictions(
    test_image_parquet: str,
    test_image_embed: str,
    test_text_embed: str,
    model_path: str,
    output_path: str = "predictions.csv",
    batch_size: int = 64,
    num_workers: int = 6,
    embedding_col: str = "embeddings",
    id_col: str = "observationID",
    top_k: int = 10,
    multi_objective: bool = False,
):
    # Load class mapping
    cid_to_spid = load_class_mappings(get_class_mappings_file())

    # Load and merge test data
    columns = ["filename", id_col]
    test_image_df = load_and_merge_embeddings(
        test_image_parquet, test_image_embed, columns, embedding_col
    )
    test_text_df = load_and_merge_embeddings(
        test_image_parquet,
        test_text_embed,
        columns=["filename"],
        embedding_col=embedding_col,
    )

    # Setup data module
    data_module = MultiModalDataModule(
        train_image_df=None,
        val_image_df=None,
        test_image_df=test_image_df,
        train_text_df=None,
        val_text_df=None,
        test_text_df=test_text_df,
        batch_size=batch_size,
        num_workers=num_workers,
        embedding_col=embedding_col,
        label_col=None,
    )
    data_module.setup(stage="predict")

    # Load model
    print(f"Loading model from {model_path}")
    if multi_objective:
        model = MultiObjectiveClassifier.load_from_checkpoint(
            model_path, model=MultiModalClassifier()
        )
    else:
        model = MultiModalClassifier.load_from_checkpoint(model_path)

    model.eval()
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Run inference
    print("Generating predictions...")
    all_predictions = []
    test_df = test_image_df  # to access IDs
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_module.predict_dataloader())):
            # batch = (image_embed, text_embed)
            image_embed, text_embed = batch
            image_embed = image_embed.to(device)
            text_embed = text_embed.to(device)
            probs = model.predict_step((image_embed, text_embed), batch_idx)

            # Top-k predictions
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
            batch_preds = [
                {
                    cid_to_spid.get(int(top_indices[i, j].item()), "Unknown"): float(
                        top_probs[i, j].item()
                    )
                    for j in range(top_k)
                }
                for i in range(len(probs))
            ]
            all_predictions.extend(batch_preds)

    # Group predictions by observation ID
    print("Formatting predictions...")
    observations = {}
    for i, pred in enumerate(all_predictions):
        obs_id = test_df.iloc[i][id_col]
        observations.setdefault(obs_id, []).append(pred)

    results = []
    for obs_id, preds in observations.items():
        combined = {}
        for pred in preds:
            for species, score in pred.items():
                combined[species] = combined.get(species, 0) + score
        if len(preds) > 1:
            for species in combined:
                combined[species] /= len(preds)

        top_species_ids = [
            sp
            for sp, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        results.append(
            {"observationId": obs_id, "predictions": " ".join(top_species_ids)}
        )

    # Save CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return results_df


def main(
    test_image_parquet: str = typer.Argument(...),
    test_image_embed: str = typer.Argument(...),
    test_text_embed: str = typer.Argument(...),
    model_path: str = typer.Argument(...),
    output_path: str = typer.Argument("predictions.csv"),
    batch_size: int = typer.Option(64),
    cpu_count: int = typer.Option(6),
    embedding_col: str = typer.Option("embeddings"),
    id_col: str = typer.Option("observationID"),
    multi_objective: bool = typer.Option(
        False, help="True or False to use multi-objective"
    ),
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    generate_predictions(
        test_image_parquet=test_image_parquet,
        test_image_embed=test_image_embed,
        test_text_embed=test_text_embed,
        model_path=model_path,
        output_path=output_path,
        batch_size=batch_size,
        num_workers=cpu_count,
        embedding_col=embedding_col,
        id_col=id_col,
        multi_objective=multi_objective,
    )
