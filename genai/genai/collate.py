from pathlib import Path
import typer
import json
import pandas as pd

app = typer.Typer()


@app.command()
def collate_files(root: Path):
    """Collate prediction files into a table."""
    predictions = root.glob("llm/**/predictions.json")
    data = []
    for pred in predictions:
        data.append(json.loads(pred.read_text()))
    df = pd.DataFrame(data).sort_values(by=["observationId"])
    print(df.head())
    df.to_csv(root / "predictions.csv", index=False)


if __name__ == "__main__":
    app()
