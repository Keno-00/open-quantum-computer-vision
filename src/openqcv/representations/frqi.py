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
    r"""Flexible representation of quantum images: intensities as angles.

    Each intensity :math:`x_i` is mapped linearly onto an angle

    .. math::

        \theta_i = \frac{\pi}{2} \, \frac{x_i - a}{b - a}
            \in \Big[ 0, \frac{\pi}{2} \Big],

    where :math:`[a, b]` is ``value_range``, and the image is encoded on
    :math:`n + 1` qubits as

    .. math::

        |I\rangle = \frac{1}{2^{n/2}} \sum_{i=0}^{2^n - 1}
            \big( \cos\theta_i \, |0\rangle + \sin\theta_i \, |1\rangle
            \big)_{\mathrm{color}} \otimes |i\rangle_{\mathrm{position}},

    where :math:`\theta_i = 0` for the padding positions :math:`i \ge N`.

    Hadamard gates on the position register create the uniform
    superposition. Pixel :math:`i` is then one :math:`R_y(2\theta_i)` on the
    color qubit, controlled on the position register holding :math:`i`, with
    open controls for the zero bits of :math:`i`. Each controlled rotation
    acts only on the :math:`|i\rangle` term, so the rotations commute.

    Measuring both registers yields :math:`P(i, 0) = 2^{-n} \cos^2\theta_i`
    and :math:`P(i, 1) = 2^{-n} \sin^2\theta_i`, from which the decoder
    recovers

    .. math::

        \theta_i = \arctan \sqrt{\frac{P(i, 1)}{P(i, 0)}}, \qquad
        x_i = a + \frac{2 \theta_i}{\pi} \, (b - a).

    With :math:`S` shots, each angle is estimated from the roughly
    :math:`S / 2^n` observations of its own position. A position that is
    never observed decodes to :math:`a`.

    Args:
        value_range: The interval :math:`[a, b]` mapped onto the angles
            :math:`0` and :math:`\pi/2`. Defaults to the minimum and maximum
            of the image.

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
        """Return ``ceil(log2 N) + 1``: position qubits and one color qubit."""
        return position_qubits(prod(shape[:2])) + 1

    def encode(self, image: ArrayLike) -> EncodedImage:
        """Map intensities to angles and rotate the color qubit per pixel.

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
        """Decode the intensities as described in the class docstring."""
        shape, count = image_shape(program)
        low, high = program.meta["value_range"]
        p = np.zeros((2, count))  # p[color_bit, pixel]
        joint = marginal(result, program, "position", "color")
        for (index, color), probability in joint.items():
            if index < count:
                p[color, index] = probability
        angles = np.arctan2(np.sqrt(p[1]), np.sqrt(p[0]))
        return (low + angles / (pi / 2) * (high - low)).reshape(shape)
