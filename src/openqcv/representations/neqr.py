"""Novel Enhanced Quantum Representation (NEQR), Zhang et al. 2013."""

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
    """Each pixel's intensity is stored as a binary number in basis states.

    ``|I> = 1/sqrt(2^n) sum_i |f(i)>_intensity |i>_position`` on
    ``n + bit_depth`` qubits. After Hadamards on the position register, every
    set bit of ``f(i)`` is one controlled-NOT, controlled on ``position == i``.
    Because intensities are basis states, a single observation of a position
    recovers its value exactly; a pixel that was never observed decodes to 0.

    Args:
        bit_depth: Bits per intensity. Values must be integers in
            ``[0, 2**bit_depth)``; the default fits ``uint8`` images.
    """

    name = "neqr"

    def __init__(self, bit_depth: int = 8) -> None:
        if not isinstance(bit_depth, int) or bit_depth < 1:
            raise ValueError("bit_depth must be a positive integer")
        self.bit_depth = bit_depth

    def qubits_required(self, shape: tuple[int, ...]) -> int:
        """``ceil(log2 N)`` position qubits plus ``bit_depth`` qubits."""
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
        """The most likely intensity observed at each position."""
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
