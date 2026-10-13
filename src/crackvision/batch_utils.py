
from pathlib import Path

from crackvision.image_utils import get_image_info, load_image


def inspect_images(image_paths: list[str | Path]) -> list[dict]:
    """Load images and return their metadata in the original order."""

    results = []

    for image_path in image_paths:
        image = load_image(image_path)
        info = get_image_info(image)

        results.append({
            "path": str(image_path),
            **info,
        })

    return results
