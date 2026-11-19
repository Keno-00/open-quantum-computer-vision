"""Flexible Representation of Quantum Images (FRQI)."""

from __future__ import annotations

from math import pi, prod

import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate

from openqcv.core.blocks import Block
from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.image.preprocess import as_grayscale, position_qubits
from openqcv.representations.base import (
    IDENTITY_ANGLE,
    EncodedImage,
    Representation,
    controlled,
    image_shape,
    marginal,
    register_representation,
)


@register_representation
class FRQI(Representation):
    """Each pixel's intensity becomes a rotation angle on one colour qubit.

    ``|I> = 1/sqrt(2^n) sum_i (cos t_i |0> + sin t_i |1>)_color |i>_position``
    with ``t_i`` in ``[0, pi/2]``, on ``n + 1`` qubits. After Hadamards on the
    position register, pixel ``i`` is one ``R_y(2 t_i)`` on the colour qubit,
    controlled on ``position == i``. Intensities are read back statistically,
    so precision improves with the number of shots; a pixel that was never
    observed decodes to the bottom of the range.

    Args:
        value_range: ``(low, high)`` intensities mapped to angles ``0`` and
            ``pi/2``. Defaults to the image's own minimum and maximum.

    References:
        P. Q. Le, F. Dong and K. Hirota, "A flexible representation of
        quantum images for polynomial preparation, image compression, and
        processing operations", Quantum Inf. Process. 10, 63-84 (2011).
        doi:10.1007/s11128-010-0177-y
    """

    name = "frqi"

    def __init__(self, value_range: tuple[float, float] | None = None) -> None:
        if value_range is not None:
            low, high = value_range
            if not high > low:
                raise ValueError("value_range must satisfy low < high")
        self.value_range = value_range

    def qubits_required(self, shape: tuple[int, ...]) -> int:
        """``ceil(log2 N)`` position qubits plus one colour qubit."""
        return position_qubits(prod(shape[:2])) + 1

    def encode(self, image: ArrayLike) -> EncodedImage:
        """Map intensities to angles and rotate the colour qubit per pixel.

        Raises:
            ValueError: If an intensity is outside ``value_range``.
        """
        pixels = as_grayscale(image)
        if self.value_range is not None:
            low, high = map(float, self.value_range)
            if pixels.min() < low or pixels.max() > high:
                raise ValueError(
                    f"Intensities fall outside value_range {(low, high)}"
                )
        else:
            low, high = float(pixels.min()), float(pixels.max())
            if high == low:
                high = low + 1.0  # constant image: every angle is 0

        n = position_qubits(pixels.size)
        angles = (pixels.ravel() - low) / (high - low) * (pi / 2)

        position = QuantumRegister(n, "position")
        color = QuantumRegister(1, "color")
        circuit = QuantumCircuit(position, color, name=self.name)
        circuit.h(position)
        for index, angle in enumerate(angles):
            if angle > IDENTITY_ANGLE:
                # qubits 0..n-1: position, n: color
                body = QuantumCircuit(n + 1)
                controlled(body, RYGate(2 * angle), range(n), index, n)
                circuit.append(
                    Block(f"pixel_{index}", body), [*position, color[0]]
                )

        meta = {"shape": list(pixels.shape), "value_range": [low, high]}
        return EncodedImage(self.name, circuit, meta)

    @classmethod
    def decode(
        cls, result: ExecutionResult, program: QasmProgram
    ) -> np.ndarray:
        """Angles ``arctan(sqrt(p1 / p0))`` per pixel, mapped back to range."""
        shape, count = image_shape(program)
        low, high = program.meta["value_range"]
        p = np.zeros((2, count))  # p[color_bit, pixel]
        joint = marginal(result, program, "position", "color")
        for (index, color), probability in joint.items():
            if index < count:
                p[color, index] = probability
        angles = np.arctan2(np.sqrt(p[1]), np.sqrt(p[0]))
        return (low + angles / (pi / 2) * (high - low)).reshape(shape)
