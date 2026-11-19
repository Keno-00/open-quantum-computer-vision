"""Novel Enhanced Quantum Representation (NEQR)."""

from __future__ import annotations

from math import prod

import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XGate

from openqcv.core.blocks import Block
from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.image.preprocess import as_grayscale, position_qubits
from openqcv.representations.base import (
    EncodedImage,
    Representation,
    controlled,
    image_shape,
    marginal,
    register_representation,
)


@register_representation
class NEQR(Representation):
    r"""Novel enhanced quantum representation: intensities as bit strings.

    Each intensity is an integer :math:`f(i) \in \{0, \dots, 2^q - 1\}`
    with binary digits :math:`f(i) = \sum_{b=0}^{q-1} f_b(i) \, 2^b`, where
    :math:`q` is ``bit_depth``. The image is encoded on :math:`n + q` qubits
    as

    .. math::

        |I\rangle = \frac{1}{2^{n/2}} \sum_{i=0}^{2^n - 1}
            |f(i)\rangle_{\mathrm{intensity}} \otimes
            |i\rangle_{\mathrm{position}},

    where :math:`f(i) = 0` for the padding positions :math:`i \ge N`.

    Hadamard gates on the position register create the uniform
    superposition. For every pixel :math:`i` and every bit with
    :math:`f_b(i) = 1`, a NOT gate on ``intensity[b]`` is controlled on the
    position register holding :math:`i`, with open controls for the zero
    bits of :math:`i`. Pixels with :math:`f(i) = 0` require no gates.

    Every outcome :math:`(i, v)` with non-zero probability has
    :math:`v = f(i)`, so a single observation of position :math:`i`
    recovers its intensity exactly; the decoder takes the most frequent
    value at each position. With :math:`S` shots, a given position is never
    observed with probability :math:`(1 - 2^{-n})^S`, in which case its
    pixel decodes to 0.

    Args:
        bit_depth: The number of bits :math:`q` per intensity. Intensities
            must be integers in :math:`[0, 2^q)`; the default, 8, fits
            ``uint8`` images.

    References:
        Y. Zhang, K. Lu, Y. Gao and M. Wang, "NEQR: a novel enhanced
        quantum representation of digital images", Quantum Inf. Process.
        12, 2833-2860 (2013). doi:10.1007/s11128-013-0567-z
    """

    name = "neqr"

    def __init__(self, bit_depth: int = 8) -> None:
        if not isinstance(bit_depth, int) or bit_depth < 1:
            raise ValueError("bit_depth must be a positive integer")
        self.bit_depth = bit_depth

    def qubits_required(self, shape: tuple[int, ...]) -> int:
        """Return ``ceil(log2 N) + bit_depth``."""
        return position_qubits(prod(shape[:2])) + self.bit_depth

    def encode(self, image: ArrayLike) -> EncodedImage:
        """Write each pixel's binary intensity under its position.

        Raises:
            ValueError: If intensities are not integers in
                ``[0, 2**bit_depth)``.
        """
        pixels = as_grayscale(image)
        values = np.rint(pixels)
        top = (1 << self.bit_depth) - 1
        if not np.allclose(values, pixels):
            raise ValueError(
                "NEQR needs integer intensities; round or scale the image "
                "first, e.g. (image * 255).round()"
            )
        if values.min() < 0 or values.max() > top:
            raise ValueError(
                f"NEQR with bit_depth={self.bit_depth} needs values "
                f"in 0..{top}"
            )

        n, q = position_qubits(pixels.size), self.bit_depth
        position = QuantumRegister(n, "position")
        intensity = QuantumRegister(q, "intensity")
        circuit = QuantumCircuit(position, intensity, name=self.name)
        circuit.h(position)
        for index, value in enumerate(values.ravel().astype(int)):
            if value:  # a 0 intensity needs no gates
                body = _pixel_body(index, int(value), n, q)
                circuit.append(
                    Block(f"pixel_{index}", body), [*position, *intensity]
                )

        return EncodedImage(
            self.name, circuit, {"shape": list(pixels.shape), "bit_depth": q}
        )

    @classmethod
    def decode(
        cls, result: ExecutionResult, program: QasmProgram
    ) -> np.ndarray:
        """Decode the intensities as described in the class docstring."""
        shape, count = image_shape(program)
        image = np.zeros(count)
        best = np.zeros(count)
        joint = marginal(result, program, "position", "intensity")
        for (index, value), p in joint.items():
            if index < count and p > best[index]:
                best[index], image[index] = p, value
        return image.reshape(shape)


def _pixel_body(index: int, value: int, n: int, q: int) -> QuantumCircuit:
    """Write ``value`` into the intensity qubits when position == ``index``.

    Qubits ``0..n-1`` are position, ``n..n+q-1`` are intensity.
    """
    body = QuantumCircuit(n + q)
    for bit in range(q):
        if (value >> bit) & 1:
            controlled(body, XGate(), range(n), index, n + bit)
    return body
