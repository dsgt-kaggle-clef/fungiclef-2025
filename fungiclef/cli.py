from typer import Typer
from fungiclef.embed import app as embed_app

app = Typer()
app.add_typer(embed_app, name="embed", help="Embed images using DINOv2.")
