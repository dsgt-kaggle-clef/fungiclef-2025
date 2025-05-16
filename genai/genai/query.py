import requests
import os
import yaml
import base64
from textwrap import dedent
import pandas as pd
from pathlib import Path
import typer
import json
from tqdm import tqdm
import dotenv
import multiprocessing
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
import logging
from datetime import datetime

dotenv.load_dotenv()
app = typer.Typer()

TAXONOMY_COLUMNS = [
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    # specificEpithet is the latter half of the binomial name
]


def extract_taxonomy_df(metadata):
    df = (
        metadata[["observationID", "category_id", "poisonous", *TAXONOMY_COLUMNS]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df


def get_taxonomy_children(
    taxonomy: pd.DataFrame, class_type: str, parents: list[str] = []
):
    """
    Get the children of a taxonomy class type. Some of this code
    is hacked together so that we have binomial names available.
    """
    # get parent taxon column
    parent_column = TAXONOMY_COLUMNS[TAXONOMY_COLUMNS.index(class_type) - 1]
    if not parents:
        return taxonomy[class_type].dropna().unique().tolist()
    return (
        taxonomy[taxonomy[parent_column].isin(parents)][class_type]
        .dropna()
        .unique()
        .tolist()
    )


def encode_image(path):
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def ask_llm(
    row: pd.Series,
    image_root: Path,
    class_type: str,
    classes: list[str] = [],
    api_key: str | None = os.getenv("OPENROUTER_API_KEY"),
    verbose: bool = False,
) -> dict:
    if not api_key:
        raise ValueError("API key is required")

    PROMPT = dedent(f"""
    Accurately identify and assign the correct {class_type} label to each image of
    fungi, protozoa, or chromista utilizing all provided image views and associated
    metadata (location, substrate, season) to ensure precision, especially for
    fine-grained distinctions. Choose the top twenty most relevant labels ranked in
    order from the available class labels, a confidence on the likert scale between
    1-5 on not-confident to confident and provide short reasoning (in under 50
    words) for your selection.
    """)

    contents = (
        f"available {class_type} labels:\n"
        + "\n".join(classes)
        + "\n"
        + yaml.safe_dump(row.to_dict(), sort_keys=False)
        + "\n"
        + PROMPT
    )
    if verbose:
        print(f"Contents: {contents}")

    completion = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=dict(
            model="google/gemini-2.0-flash-001",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": contents},
                        {
                            "type": "image_url",
                            "image_url": encode_image(image_root / row["filename"]),
                        },
                    ],
                },
            ],
            # object with predict list of strings and a reason sting
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "fungiclef",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "predictions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                        },
                                        "confidence": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 5,
                                        },
                                    },
                                },
                                "maxItems": 20,
                                "minItems": 20,
                            },
                            "reason": {
                                "type": "string",
                                "maxLength": 200,
                            },
                        },
                    },
                },
            },
        ),
    )
    return completion.json()


def get_logger(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log.jsonl"
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode="a")
    handler.setFormatter(logging.Formatter("%(message)s"))
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger


def process_row(row, taxonomy_df, image_root, output_path, logger):
    root = output_path / str(row.observationID)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "_SUCESS").exists():
        print("skipping", row.observationID)
        logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "observationID": row.observationID,
                    "status": "skipped",
                    "message": "Already processed (_SUCESS exists)",
                }
            )
        )
        return "skipped"
    try:
        parent_list = []
        for class_type in ["family", "genus", "species"]:
            children = get_taxonomy_children(taxonomy_df, class_type, parent_list)
            completion = ask_llm(row, image_root, class_type, children)
            (root / f"completion_{class_type}.json").write_text(
                json.dumps(completion, indent=2),
            )
            content = json.loads(completion["choices"][0]["message"]["content"])
            (root / f"predictions_{class_type}.json").write_text(
                json.dumps(content, indent=2)
            )
            pred = pd.DataFrame(content["predictions"])
            matched_labels = []
            mismatches = 0
            for label in pred.label.tolist():
                match = rf_process.extractOne(
                    label, children, scorer=rf_fuzz.WRatio, score_cutoff=90
                )
                if match:
                    matched_labels.append(match[0])
                else:
                    mismatches += 1
            if mismatches > 5:
                raise ValueError(
                    f"Too many unmatched predictions for {class_type}",
                    {
                        "error_type": "too_many_mismatches",
                        "class_type": class_type,
                        "mismatches": mismatches,
                        "total": len(pred),
                        "observationID": row.observationID,
                    },
                )
            # write the parent list and children to a file
            (root / f"labels_{class_type}.json").write_text(
                json.dumps({
                    "parent": parent_list,
                    "children": matched_labels,
                }, indent=2),
            )
            parent_list = matched_labels

        # now we need to map the binomial names to the taxonomy
        species_names = parent_list
        mapping = {row.species: row.category_id for _, row in taxonomy_df.iterrows()}
        observations = []
        for label in species_names:
            observations.append(mapping[label])
        (root / "predictions.json").write_text(
            json.dumps(
                {
                    "observationId": row.observationID,
                    "predictions": " ".join([str(x) for x in observations][:10]),
                },
                indent=2,
            )
        )
        (root / "_SUCESS").touch()
        logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "observationID": row.observationID,
                    "status": "success",
                    "message": "Processed successfully",
                }
            )
        )
        return "success"
    except json.JSONDecodeError as e:
        logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "observationID": row.observationID,
                    "status": "error",
                    "error_type": "bad_json_response",
                    "message": str(e),
                }
            )
        )
        return "error"
    except ValueError as e:
        err_type = (
            "too_many_mismatches"
            if isinstance(e.args, tuple)
            and len(e.args) > 1
            and isinstance(e.args[1], dict)
            and e.args[1].get("error_type")
            else "value_error"
        )
        logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "observationID": row.observationID,
                    "status": "error",
                    "error_type": err_type,
                    "message": str(e),
                }
            )
        )
        return "error"
    except Exception as e:
        logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "observationID": row.observationID,
                    "status": "error",
                    "error_type": type(e).__name__,
                    "message": str(e),
                }
            )
        )
        return "error"


def process_row_wrapper(args):
    row, taxonomy_df, image_root, output_path, logger = args
    return process_row(row, taxonomy_df, image_root, output_path, logger)


@app.command()
def extract_labels(
    root: Path,
    output_path: Path,
    num_workers: int = 4,
    limit: int = 0,
):
    """
    Extract labels from the metadata and images using the OpenRouter API.
    """
    image_root = (
        Path(root).expanduser()
        / "raw"
        / "images"
        / "FungiTastic-FewShot"
        / "test"
        / "300p"
    )
    train_metadata_path = list(
        Path(root).expanduser().glob("raw/metadata/*/*Train.csv")
    )[0]
    test_metadata_path = list(Path(root).expanduser().glob("raw/metadata/*/*Test.csv"))[
        0
    ]
    train_df = pd.read_csv(train_metadata_path)
    test_df = pd.read_csv(test_metadata_path)
    test_df = test_df.drop_duplicates(subset=["observationID"], keep="first")

    taxonomy_df = extract_taxonomy_df(train_df)

    logger = get_logger(output_path / "logs")

    # Prepare arguments for each row
    args_list = [
        (row, taxonomy_df, image_root, output_path / "llm", logger)
        for _, row in test_df.iterrows()
    ]
    if limit > 0:
        args_list = args_list[:limit]

    results = []
    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap(process_row_wrapper, args_list), total=len(args_list)
        ):
            results.append(result)

    # Print summary
    success_count = sum(1 for r in results if r == "success")
    error_count = sum(1 for r in results if r == "error")
    skipped_count = sum(1 for r in results if r == "skipped")
    print(f"Success: {success_count}, Errors: {error_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    app()
