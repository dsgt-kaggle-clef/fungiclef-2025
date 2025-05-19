import typer
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from fungiclef.config import get_class_mappings_file
from fungiclef.torch.data import FungiDataModule
from fungiclef.torch.model import LinearClassifier, MultiObjectiveClassifier
from fungiclef.torch.mixup import MixupClassifier


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
    """Load and merge metadata and embeddings"""
    df_meta = pd.read_parquet(parquet_path, columns=columns)
    df_embed = pd.read_parquet(embed_path, columns=["filename", embedding_col])
    return df_meta.merge(df_embed, on="filename", how="inner")


def get_classifier_class(model_type: str):
    """Get the model class based on the model type."""
    if model_type == "mixup":
        return MixupClassifier
    elif model_type == "linear":
        return LinearClassifier


def generate_predictions(
    test_parquet_path: str,
    test_embed_path: str,
    model_path: str,
    output_path: str = "predictions.csv",
    model_type: str = "linear",
    num_workers: int = 6,
    batch_size: int = 64,
    embedding_col: str = "embedding",
    id_col: str = "observationID",
    top_k: int = 10,
    multi_objective: bool = False,
):
    """
    Generate predictions using a trained model.

    Args:
        test_parquet: Path to test data parquet file
        model_path: Path to trained model checkpoint
        output_path: Path to save predictions
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        embedding_col: Column name containing embeddings
        id_col: Column name containing observation IDs
    """
    # class mappings file for classification
    class_mappings_file = get_class_mappings_file()
    # load class mappings
    cid_to_spid = load_class_mappings(class_mappings_file)

    # load test data
    columns = ["filename", "observationID"]
    test_df = load_and_merge_embeddings(
        test_parquet_path, test_embed_path, columns, embedding_col
    )

    # create data module
    data_module = FungiDataModule(
        train_df=None,
        val_df=None,
        test_df=test_df,
        num_workers=num_workers,
        batch_size=batch_size,
        embedding_col=embedding_col,
        label_col=None,  # No labels for prediction
    )

    # set up test dataset
    data_module.setup(stage="predict")

    # load trained model
    print(f"Loading model from {model_path}")
    classifier_cls = get_classifier_class(model_type)
    if multi_objective:
        model = MultiObjectiveClassifier.load_from_checkpoint(
            model_path, model=LinearClassifier()
        )
    else:
        model = classifier_cls.load_from_checkpoint(model_path)
    model.eval()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # generate predictions
    print("Generating predictions...")
    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_module.predict_dataloader())):
            # Handle batch output
            batch_predictions = model.predict_step(batch, batch_idx)

            # Get top probabilities and indices
            top_probs, top_indices = torch.topk(batch_predictions, k=top_k, dim=1)

            # map class indices to species names
            batch_preds = []
            for i in range(len(batch_predictions)):
                species_probs = {
                    cid_to_spid.get(int(top_indices[i, j].item()), "Unknown"): float(
                        top_probs[i, j].item()
                    )
                    for j in range(top_k)
                }
                batch_preds.append(species_probs)
            all_predictions.extend(batch_preds)

    # Format predictions in the required format
    print("Formatting predictions...")

    # Group predictions observation ID
    observations = {}
    for i, predictions in enumerate(all_predictions):
        # Get observation ID
        obs_id = test_df.iloc[i][id_col]

        if obs_id not in observations:
            observations[obs_id] = []

        # Add the predictions for this image to the observation
        observations[obs_id].append(predictions)

    results = []

    for obs_id, prediction in observations.items():
        # Get observation ID if available, otherwise use index
        combined_preds = {}

        # sum up all prediction scores for each species across all images
        for pred_dict in prediction:
            for species, score in pred_dict.items():
                if species not in combined_preds:
                    combined_preds[species] = 0
                combined_preds[species] += score

        # If there are multiple images, average the scores
        if len(prediction) > 1:
            for species in combined_preds:
                combined_preds[species] /= len(prediction)

        # Get top 10 predictions with highest confidence
        top_predictions = sorted(
            combined_preds.items(), key=lambda x: x[1], reverse=True
        )[:10]
        # Extract just the species IDs
        top_species_ids = [species for species, _ in top_predictions]
        # Join species IDs with spaces
        formatted_predictions = " ".join(top_species_ids)

        # Add to results
        results.append({"observationId": obs_id, "predictions": formatted_predictions})

    # Save predictions to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    return results_df


def main(
    test_parquet_path: str = typer.Argument(..., help="Path to test data parquet file"),
    test_embed_path: str = typer.Argument(..., help="Path to test embed parquet file"),
    model_path: str = typer.Argument(..., help="Path to trained model checkpoint"),
    output_path: str = typer.Argument(
        "predictions.csv", help="Path to save predictions"
    ),
    model_type: str = typer.Option("linear", help="Model type: 'linear' or 'mixup'"),
    cpu_count: int = typer.Option(6, help="Number of workers for data loading"),
    batch_size: int = typer.Option(64, help="Batch size for inference"),
    embedding_col: str = typer.Option(
        "embeddings", help="Column name containing embeddings"
    ),
    id_col: str = typer.Option(
        "observationID", help="Column name containing observation IDs"
    ),
    multi_objective: bool = typer.Option(
        False, help="True or False to use multi-objective"
    ),
):
    # create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Generate predictions
    generate_predictions(
        test_parquet_path=test_parquet_path,
        test_embed_path=test_embed_path,
        model_path=model_path,
        output_path=output_path,
        model_type=model_type,
        num_workers=cpu_count,
        batch_size=batch_size,
        embedding_col=embedding_col,
        id_col=id_col,
        multi_objective=multi_objective,
    )
