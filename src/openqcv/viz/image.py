"""Draw images and compare an original with its decoded version.

Black-and-white figures with pixel-index axes; images up to
:data:`ANNOTATE_MAX_PIXELS` pixels get their values written in each cell.
"""

from __future__ import annotations

import numpy as np
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from openqcv.image.preprocess import as_grayscale
from openqcv.viz.drawing import PAPER_STYLE, Drawing

ANNOTATE_MAX_PIXELS = 64  # write values into cells up to 8x8


def draw_image(
    image: ArrayLike,
    *,
    title: str | None = None,
    value_range: tuple[float, float] | None = None,
) -> Drawing:
    """Draw a grayscale image with pixel-index axes and a color bar.

    Args:
        image: 2-D grayscale or ``(H, W, 3|4)`` color array.
        title: Optional caption above the image.
        value_range: Intensities shown as black and white. Defaults to the
            image's minimum and maximum.
    """
    pixels = as_grayscale(image)
    low, high = value_range or _range(pixels)
    with rc_context(PAPER_STYLE):
        figure = Figure(figsize=(3.6, 3.2))
        ax = figure.add_axes((0.14, 0.14, 0.62, 0.74))
        _panel(figure, ax, pixels, low, high, title or "", low_is_dark=True)
        ax.set_ylabel("$y$")
    return Drawing(figure)


def compare(original: ArrayLike, decoded: ArrayLike) -> Drawing:
    """Draw (a) the original, (b) the decoded image and (c) their difference.

    (a) and (b) share the original's gray scale, so differences are real;
    (c) has its own scale, white = 0. The footer gives the maximum and mean
    absolute error and the PSNR, with the original's peak absolute value as
    the signal maximum.
    """
    a, b = as_grayscale(original), as_grayscale(decoded)
    if a.shape != b.shape:
        raise ValueError(f"Shapes differ: {a.shape} vs {b.shape}")
    low, high = _range(a)
    difference = np.abs(a - b)

    with rc_context(PAPER_STYLE):
        figure = Figure(figsize=(10.2, 3.5))
        axes = figure.subplots(1, 3)
        figure.subplots_adjust(
            left=0.05, right=0.95, bottom=0.2, top=0.88, wspace=0.5
        )
        _panel(figure, axes[0], a, low, high, "(a) original", low_is_dark=True)
        _panel(figure, axes[1], b, low, high, "(b) decoded", low_is_dark=True)
        _panel(
            figure,
            axes[2],
            difference,
            0.0,
            max(float(difference.max()), 1e-12),
            r"(c) $|\Delta|$",
            low_is_dark=False,
        )
        axes[0].set_ylabel("$y$")
        figure.text(
            0.05,
            0.03,
            rf"max $|\Delta|$ = {difference.max():.4g},   "
            rf"mean $|\Delta|$ = {difference.mean():.4g},   "
            rf"PSNR = ${_psnr(a, b)}$ dB",
            fontsize=10,
        )
    return Drawing(figure)


def _panel(
    figure: Figure,
    ax: Axes,
    pixels: np.ndarray,
    low: float,
    high: float,
    title: str,
    *,
    low_is_dark: bool,
) -> None:
    """Draw one panel: pixels, ticks, color bar and, if small, values."""
    rows, cols = pixels.shape
    shown = ax.imshow(
        pixels,
        cmap="gray" if low_is_dark else "Greys",
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("$x$")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.tick_params(length=2, labelsize=8)
    bar = figure.colorbar(shown, cax=ax.inset_axes((1.04, 0.0, 0.05, 1.0)))
    bar.ax.tick_params(labelsize=8, direction="in", length=2)
    bar.outline.set_linewidth(0.8)

    if pixels.size > ANNOTATE_MAX_PIXELS:
        return
    integers = np.allclose(pixels, np.round(pixels))
    fontsize = 9 if pixels.size <= 16 else 6.5
    span = (high - low) or 1.0
    for (r, c), value in np.ndenumerate(pixels):
        level = (value - low) / span
        dark_cell = level < 0.5 if low_is_dark else level > 0.5
        ax.text(
            c,
            r,
            f"{value:.0f}" if integers else f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="white" if dark_cell else "black",
        )


def _range(pixels: np.ndarray) -> tuple[float, float]:
    low, high = float(pixels.min()), float(pixels.max())
    return (low, high) if high > low else (low, low + 1.0)


def _psnr(original: np.ndarray, decoded: np.ndarray) -> str:
    r"""Return the PSNR in dB as mathtext; ``\infty`` if identical."""
    mse = float(np.mean((original - decoded) ** 2))
    peak = float(np.abs(original).max()) or 1.0
    return r"\infty" if mse == 0 else f"{10 * np.log10(peak**2 / mse):.2f}"
