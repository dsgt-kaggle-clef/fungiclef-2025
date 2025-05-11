import typer
from .augment import main as augment_main

app = typer.Typer()
app.command("augment")(augment_main)
