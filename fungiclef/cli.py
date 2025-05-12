from typer import Typer
from fungiclef.embed import app as embed_app
from fungiclef.prediction import app as prediction_app
from fungiclef.preprocessing import app as preprocessing_app
from fungiclef.multimodal import app as multimodal_app

app = Typer()
app.add_typer(embed_app, name="embed", help="Embed images using DINOv2.")
app.add_typer(prediction_app, name="prediction", help="Train and evaluate models.")
app.add_typer(preprocessing_app, name="preprocessing", help="Preprocess images.")
app.add_typer(
    multimodal_app, name="multimodal", help="Train and evaluate multimodal models."
)
