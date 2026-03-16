"""
Open Quantum Computer Vision Library

A unified library for quantum image representation and processing,
inspired by OpenCV but for quantum computing.

This library unifies various quantum image representation algorithms
and provides quantum image processing operations.

Encoding Classifications:
- Amplitude Encoded: QARQI, FRQI (uses RY rotations on amplitudes)
- Computational Basis Encoded: MHRQI, NEQR (uses basis states for intensities)

Authors: Unified from MHRQI, QARQI, FRQI, and NEQR projects
License: Apache 2.0
"""

__version__ = "0.1.0"

from .representations import *
from .processing import *
from .utils import *