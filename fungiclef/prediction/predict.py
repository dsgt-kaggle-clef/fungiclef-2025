import torch
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

from fungiclef.torch.data import FungiDataModule
from fungiclef.torch.model import DINOv2LightningModel

### example usage: python clef/fungiclef-2025/fungiclef/prediction/predict.py --test scratch/fungiclef/embeddings/test_embeddings.parquet --model scratch/fungiclef/model/base_model/base_fungi-classifier-epoch=05-val_loss=5.17.ckpt --output clef/fungiclef-2025/fungiclef/prediction/base_predictions.csv


def generate_predictions(
    test_parquet: str,
    model_path: str,
    output_path: str = "predictions.csv",
    batch_size: int = 64,
    num_workers: int = 6,
    embedding_col: str = "embedding",
    id_col: str = "observationID",
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
    # Load test data
    print(f"Loading test data from {test_parquet}")
    test_df = pd.read_parquet(test_parquet)

    # Create data module
    data_module = FungiDataModule(
        train_df=None,
        val_df=None,
        test_df=test_df,
        batch_size=batch_size,
        num_workers=num_workers,
        embedding_col=embedding_col,
        label_col=None,  # No labels for prediction
    )

    # Set up test dataset
    data_module.setup(stage="predict")

    # Load trained model
    print(f"Loading model from {model_path}")
    model = DINOv2LightningModel.load_from_checkpoint(model_path)
    model.eval()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate predictions
    print("Generating predictions...")
    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_module.predict_dataloader())):
            # Forward pass
            _, batch_predictions = model.predict_step(batch, batch_idx)
            all_predictions.extend(batch_predictions)

    # Format predictions in the required format
    print("Formatting predictions...")

    # Group predictions observation ID
    observations = {}
    for i, pred_dict in enumerate(all_predictions):
        # Get observation ID
        obs_id = test_df.iloc[i][id_col]

        if obs_id not in observations:
            observations[obs_id] = []

        # Add the predictions for this image to the observation
        observations[obs_id].append(pred_dict)

    results = []

    for obs_id, pred_dicts in observations.items():
        # Get observation ID if available, otherwise use index
        combined_preds = {}

        # sum up all prediction scores for each species across all images
        for pred_dict in pred_dicts:
            for species, score in pred_dict.items():
                if species not in combined_preds:
                    combined_preds[species] = 0
                combined_preds[species] += score

        # If there are multiple images, average the scores
        if len(pred_dicts) > 1:
            for species in combined_preds:
                combined_preds[species] /= len(pred_dicts)

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


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for test data")
    parser.add_argument(
        "--test", type=str, required=True, help="Path to test data parquet file"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="predictions.csv", help="Path to save predictions"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers", type=int, default=6, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--embedding-col",
        type=str,
        default="embedding",
        help="Column name containing embeddings",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="observationID",
        help="Column name containing observation IDs",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Generate predictions
    generate_predictions(
        test_parquet=args.test,
        model_path=args.model,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        embedding_col=args.embedding_col,
        id_col=args.id_col,
    )


if __name__ == "__main__":
    main()
