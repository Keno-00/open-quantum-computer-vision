"""Figure container and the shared black-and-white publication style."""

from __future__ import annotations

from io import BytesIO
from os import PathLike
from pathlib import Path

from matplotlib import rc_context
from matplotlib.figure import Figure

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}
"""LaTeX-like, black-and-white matplotlib style. Use it for your own plots
with ``matplotlib.pyplot.rcParams.update(PAPER_STYLE)``."""


class Drawing:
    """A matplotlib figure; ``save`` picks the format from the extension."""

    def __init__(self, figure: Figure) -> None:
        self.figure = figure

    def save(self, path: str | PathLike, dpi: int = 300) -> Path:
        """Save as PNG, PDF, SVG, ... depending on the file extension."""
        target = Path(path)
        with rc_context(PAPER_STYLE):
            self.figure.savefig(
                target, dpi=dpi, bbox_inches="tight", pad_inches=0.05
            )
        return target

    def _repr_png_(self) -> bytes:
        buffer = BytesIO()
        with rc_context(PAPER_STYLE):
            self.figure.savefig(
                buffer, format="png", dpi=120, bbox_inches="tight"
            )
        return buffer.getvalue()
