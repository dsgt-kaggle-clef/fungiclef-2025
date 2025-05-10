import typer
from .train import main as workflow_main

app = typer.Typer()
app.command("train")(workflow_main)
