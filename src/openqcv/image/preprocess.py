"""Classical image preparation shared by all representations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# ITU-R BT.601 luma weights, the same ones Pillow uses for mode "L".
_LUMA = np.array([0.299, 0.587, 0.114])


def as_grayscale(image: ArrayLike) -> np.ndarray:
    """Return a 2-D float64 grayscale copy of ``image``.

    2-D arrays are used as-is. ``(H, W, 1)`` arrays are squeezed and
    ``(H, W, 3|4)`` RGB(A) arrays are converted with ITU-R BT.601 luma
    weights (alpha is ignored). Pixels are indexed row-major:
    pixel ``i`` is ``(i // width, i % width)``.

    Raises:
        TypeError: If the values are not real numbers.
        ValueError: If the image is empty, not finite, or has an
            unsupported shape.

    References:
        Recommendation ITU-R BT.601-7, "Studio encoding parameters of
        digital television for standard 4:3 and wide-screen 16:9 aspect
        ratios" (2011).
    """
    values = np.asarray(image)
    if values.size == 0:
        raise ValueError("Image cannot be empty")
    if not (np.issubdtype(values.dtype, np.number) or values.dtype == bool):
        raise TypeError(
            f"Image values must be numbers, got dtype {values.dtype}"
        )
    if np.iscomplexobj(values):
        raise TypeError("Image values must be real numbers")

    if values.ndim == 2:
        gray = values.astype(np.float64)
    elif values.ndim == 3 and values.shape[2] == 1:
        gray = values[..., 0].astype(np.float64)
    elif values.ndim == 3 and values.shape[2] in (3, 4):
        gray = values[..., :3].astype(np.float64) @ _LUMA
    elif values.ndim == 3:
        raise ValueError("Color images must have 1, 3 or 4 channels")
    else:
        raise ValueError(
            f"Image must be a 2-D or 3-D array, got {values.ndim}-D"
        )

    if not np.isfinite(gray).all():
        raise ValueError("Image contains NaN or infinite values")
    return gray


def position_qubits(pixel_count: int) -> int:
    """Qubits needed to index ``pixel_count`` pixels (at least 1)."""
    if pixel_count < 1:
        raise ValueError("pixel_count must be at least 1")
    return max(1, (pixel_count - 1).bit_length())
