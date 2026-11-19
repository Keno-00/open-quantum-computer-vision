"""Loading images from disk."""

from __future__ import annotations

from os import PathLike

import numpy as np


def load_image(
    path: str | PathLike, size: int | tuple[int, int] | None = None
) -> np.ndarray:
    """Load an image file as a 2-D ``uint8`` grayscale array.

    Args:
        path: Image file (PNG, JPEG, ...).
        size: Optional resize target: an int for a square image, or
            ``(height, width)`` to match the returned array's shape.
            Quantum encodings get expensive fast, so small sizes such as 8
            or 16 are a good start.

    Requires Pillow (``pip install "openqcv[image]"``).
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "load_image needs Pillow. "
            'Install it with: pip install "openqcv[image]"'
        ) from None

    with Image.open(path) as image:
        gray = image.convert("L")
        if size is not None:
            height, width = (size, size) if isinstance(size, int) else size
            gray = gray.resize((width, height), Image.Resampling.BOX)
        return np.asarray(gray, dtype=np.uint8)
