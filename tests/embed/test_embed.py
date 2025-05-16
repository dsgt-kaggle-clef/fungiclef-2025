import pytest
import torch
import pytorch_lightning as pl
from fungiclef.config import get_device
from fungiclef.serde import deserialize_image
from fungiclef.lightning.model import EmbedModel
from fungiclef.lightning.data import FungiDataModule


def test_data_module(pandas_df):
    data_module = FungiDataModule(
        pandas_df=pandas_df,
        batch_size=1,
        num_workers=1,
        model_name="hf-hub:BVRA/vit_base_patch16_384.in1k_ft_fungitastic_384",
        resize_size=384,
    )
    data_module.setup()
    img_bytes = pandas_df.iloc[0]["data"]
    pil_img = deserialize_image(img_bytes)
    transform = data_module.dataset.transform
    transformed_tensor = transform(pil_img)
    assert transformed_tensor.shape == (3, 384, 384)
    assert transformed_tensor.min().item() == 1.0
    assert transformed_tensor.max().item() == 1.0
    mean_per_channel = transformed_tensor.mean(dim=(1, 2))
    expected = torch.ones(3)
    assert torch.allclose(mean_per_channel, expected, atol=1e-6)


@pytest.mark.parametrize(
    "model_name, resize_size",
    [
        ("hf-hub:BVRA/vit_base_patch16_384.in1k_ft_fungitastic_384", 384),
    ],
)
def test_embed_model(pandas_df, model_name, resize_size):
    data_module = FungiDataModule(
        pandas_df=pandas_df,
        batch_size=1,
        num_workers=1,
        model_name="hf-hub:BVRA/vit_base_patch16_384.in1k_ft_fungitastic_384",
        resize_size=384,
    )

    model = EmbedModel(model_name=model_name, resize_size=resize_size)
    trainer = pl.Trainer(
        accelerator=get_device(),
        devices=1,
        enable_progress_bar=True,
    )

    predictions = trainer.predict(model, datamodule=data_module)

    all_embeddings = []
    for batch in predictions:
        embed_batch = batch  # batch: List[embeddings]
        all_embeddings.append(embed_batch)  # keep embeddings as tensors

    embeddings = torch.cat(all_embeddings, dim=0)  # shape: [len(df), grid_size**2, 768]
    assert embeddings.shape == (len(pandas_df), 768)
