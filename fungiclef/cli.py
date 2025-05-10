from typer import Typer
from fungiclef.embed import app as embed_app
from fungiclef.prediction import app as prediction_app

app = Typer()
app.add_typer(embed_app, name="embed", help="Embed images using DINOv2.")
app.add_typer(prediction_app, name="prediction", help="Train and evaluate models.")
