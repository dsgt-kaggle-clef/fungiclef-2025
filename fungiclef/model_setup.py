import os
from pathlib import Path


def get_model_dir() -> str:
    """
    Get the model directory in the plantclef shared project for the current user on PACE
    """
    home_dir = Path(os.path.expanduser("~"))
    return f"{home_dir}/p-dsgt_clef2025-0/shared/fungiclef/model"


def setup_fine_tuned_model() -> str:
    """
    Downloads and unzips a model from PACE and returns the path to the specified model file.
    Checks if the model already exists and skips download and extraction if it does.

    :return: Absolute path to the model file.
    """
    model_base_path = get_model_dir()
    tar_filename = "model_best.pth.tar"
    pretrained_model = (
        "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all"
    )
    relative_model_path = (
        f"plantclef/pretrained_models/{pretrained_model}/{tar_filename}"
    )
    full_model_path = os.path.join(model_base_path, relative_model_path)

    # Check if the model file exists
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model file not found at: {full_model_path}")

    # Return the path to the model file
    return full_model_path


if __name__ == "__main__":
    # Example usage
    model_path = setup_fine_tuned_model()
    print(f"Model path: {model_path}")
