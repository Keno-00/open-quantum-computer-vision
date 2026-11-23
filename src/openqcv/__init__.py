"""openqcv: quantum image representations.

The public workflow is :func:`encode`, :func:`run` and :func:`decode`;
:func:`draw`, :func:`draw_image` and :func:`compare` make figures.
Circuits are built, lowered and simulated with Qiskit.

References:
    A. Javadi-Abhari et al., "Quantum computing with Qiskit" (2024).
    arXiv:2405.08810
"""

from openqcv import qir
from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.image import load_image
from openqcv.pipeline import (
    decode,
    encode,
    list_backends,
    list_representations,
    run,
)
from openqcv.viz import compare, draw, draw_image

__version__ = "0.1.0"

__all__ = [
    "ExecutionResult",
    "QasmProgram",
    "compare",
    "decode",
    "draw",
    "draw_image",
    "encode",
    "list_backends",
    "list_representations",
    "load_image",
    "qir",
    "run",
]
