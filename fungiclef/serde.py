"""Module for encoding and decoding data structures to and from raw bytes"""

from PIL import Image
import io


def deserialize_image(buffer: bytes | bytearray) -> Image.Image:
    """Decode the image from raw bytes using PIL."""
    buffer = io.BytesIO(bytes(buffer))
    return Image.open(buffer).convert("RGB")


def serialize_image(image: Image.Image) -> bytes:
    """Encode the image as raw bytes using PIL."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPG")
    buffer.seek(0)
    return buffer.getvalue()


def read_image_bytes(image_path: str) -> bytes:
    """Read image file directly as bytes without re-encoding.

    This is more efficient when you just want to store the original image bytes.

    Args:
        image_path: Path to the image file

    Returns:
        The raw bytes of the image file
    """
    with open(image_path, "rb") as f:
        return f.read()
