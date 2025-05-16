import pytest
from fungiclef.spark import spark_resource


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("spark_data")
    with spark_resource(local_dir=tmp_path.as_posix()) as spark:
        yield spark


@pytest.fixture(scope="session")
def pandas_df():
    import io
    import numpy as np
    import pandas as pd
    from PIL import Image

    dummy_image = Image.fromarray(np.ones((384, 384, 3), dtype=np.uint8) * 255)

    # serialize to bytes (PNG)
    buf = io.BytesIO()
    dummy_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # store raw bytes in the DataFrame
    return pd.DataFrame({"data": [img_bytes]})
