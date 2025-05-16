import torch
from fungiclef.lightning.model import EmbedModel
from fungiclef.lightning.data import FungiDataModule


def test_data_module(pandas_df):
    data_module = FungiDataModule(
        pandas_df=pandas_df,
        batch_size=1,
        num_workers=0,
        model_name="hf-hub:BVRA/vit_base_patch16_384.in1k_ft_fungitastic_384",
        resize_size=384,
    )
    data_module.setup()
    transform = data_module.dataset.transform
    transformed_tensor = transform(pandas_df.iloc[0]["data"])
    assert transformed_tensor.shape == (3, 384, 384)
    assert transformed_tensor.min().item() == 1.0
    assert transformed_tensor.max().item() == 1.0
    mean_per_channel = transformed_tensor.mean(dim=(1, 2))
    expected = torch.ones(3)
    assert torch.allclose(mean_per_channel, expected, atol=1e-6)
