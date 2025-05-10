import typer
from .train import main as train_main
from .predict import main as predict_main

app = typer.Typer()
app.command("train")(train_main)
app.command("predict")(predict_main)
