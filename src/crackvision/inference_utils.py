
import cv2
import numpy as np
from typing import Any 

def prepare_image_for_inference(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or BGRA image to three-channel BGR."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.dtype.kind not in {"u", "i", "f"}:
        raise TypeError("image must contain numeric pixel values")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected a grayscale, BGR, or BGRA image"
        )

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image must have non-zero width and height")

    return image


def predict_image(
    model: Any,
    image: np.ndarray,
    image_size: int = 640,
    confidence: float = 0.25,
) -> Any:
    """Run segmentation inference on a prepared image."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    prepared = prepare_image_for_inference(image)

    results = model.predict(
        source=prepared,
        imgsz=image_size,
        conf=confidence,
        save=False,
        verbose=False,
    )

    if not results:
        raise RuntimeError("The model returned no prediction results")

    return results[0]
