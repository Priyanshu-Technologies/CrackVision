
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image and return it as a BGR NumPy array."""

    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {path.suffix}"
        )

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Could not decode image: {path}")

    return image

def get_image_info(image: np.ndarray) -> dict:
    """Return basic metadata for an image loaded with OpenCV."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.dtype.kind not in {"u", "i", "f"}:
        raise TypeError("image must contain numeric pixel values")

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    elif image.ndim == 3:
        height, width, channels = image.shape
    else:
        raise ValueError(
            f"Expected a 2D or 3D image array, got {image.ndim} dimensions"
        )

    if height == 0 or width == 0:
        raise ValueError("Image must have non-zero width and height")

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": str(image.dtype),
    }

