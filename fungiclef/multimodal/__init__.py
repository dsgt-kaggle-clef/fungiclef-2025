import typer
from .train import main as train_main

app = typer.Typer()
app.command("train")(train_main)
