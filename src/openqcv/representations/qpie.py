"""Quantum Probability Image Encoding (QPIE)."""

from __future__ import annotations

from math import acos, prod, sqrt

import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate

from openqcv.core.blocks import Block
from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.image.preprocess import as_grayscale, index_qubits
from openqcv.representations.base import (
    IDENTITY_ANGLE,
    QIR,
    EncodedImage,
    controlled,
    image_shape,
    marginal,
    register,
)


@register
class QPIE(QIR):
    r"""Quantum probability image encoding: intensities as amplitudes.

    An image with non-negative intensities :math:`x_0, \dots, x_{N-1}` is
    encoded on :math:`n = \lceil \log_2 N \rceil` position qubits as

    .. math::

        |I\rangle = \frac{1}{\lVert x \rVert}
            \sum_{i=0}^{2^n - 1} x_i \, |i\rangle,
        \qquad
        \lVert x \rVert = \Big( \sum_{i} x_i^2 \Big)^{1/2},

    where :math:`x_i = 0` for the padding positions :math:`i \ge N`.

    The state is prepared with the rotation tree of Grover and Rudolph. For
    a prefix :math:`v` of the :math:`k` most significant position bits, let
    :math:`w(v)` be the sum of :math:`x_i^2` over all indices :math:`i` that
    begin with :math:`v`, and let :math:`v0` denote :math:`v` followed by a
    zero bit. Level :math:`k` applies :math:`R_y(\theta_v)` to
    ``position[n-1-k]``, controlled on the higher position qubits holding
    :math:`v`, with

    .. math::

        \cos\frac{\theta_v}{2} = \sqrt{\frac{w(v0)}{w(v)}}.

    Because :math:`R_y(\theta)|0\rangle = \cos\frac{\theta}{2}|0\rangle +
    \sin\frac{\theta}{2}|1\rangle`, the amplitude of :math:`|i\rangle` after
    all :math:`n` levels is :math:`\sqrt{w(i) / w(\varnothing)} =
    x_i / \lVert x \rVert`. All amplitudes are non-negative, so no phase
    gates are required. Prefixes with :math:`w(v) = 0` are skipped.

    Measuring the position register yields :math:`p_i = x_i^2 /
    \lVert x \rVert^2`, so the decoder returns :math:`x_i = \lVert x \rVert
    \sqrt{p_i}`. The norm is not part of the quantum state; it is carried in
    ``meta``. With :math:`S` shots the estimate :math:`\hat p_i` has
    variance :math:`p_i (1 - p_i) / S`, so for :math:`p_i > 0` the decoded
    intensity has, to first order, the standard error

    .. math::

        \sigma(\hat x_i) \approx
            \lVert x \rVert \sqrt{\frac{1 - p_i}{4 S}}.

    References:
        X.-W. Yao et al., "Quantum Image Processing and Its Application to
        Edge Detection: Theory and Experiment", Phys. Rev. X 7, 031041
        (2017). doi:10.1103/PhysRevX.7.031041

        L. Grover and T. Rudolph, "Creating superpositions that correspond
        to efficiently integrable probability distributions" (2002).
        arXiv:quant-ph/0208112
    """

    name = "qpie"

    def qubits_required(self, shape: tuple[int, ...]) -> int:
        """Return ``ceil(log2 N)``, the number of position qubits."""
        return index_qubits(prod(shape[:2]))

    def encode(self, image: ArrayLike) -> EncodedImage:
        """Prepare the normalized intensities as amplitudes.

        Raises:
            ValueError: If any intensity is negative or all are zero.
        """
        pixels = as_grayscale(image)
        if (pixels < 0).any():
            raise ValueError("QPIE needs non-negative intensities")

        n = index_qubits(pixels.size)
        weights = np.zeros(1 << n)
        weights[: pixels.size] = pixels.ravel() ** 2
        norm = sqrt(float(weights.sum()))
        if norm == 0.0:
            raise ValueError("QPIE cannot encode an all-zero image")

        position = QuantumRegister(n, "position")
        circuit = QuantumCircuit(position, name=self.name)
        for level in range(n):
            body = _grover_rudolph_level(weights, n, level)
            if body.size():
                circuit.append(Block(f"level_{level}", body), position)

        return EncodedImage(
            circuit, {"shape": list(pixels.shape), "norm": norm}
        )

    @classmethod
    def decode(
        cls, result: ExecutionResult, program: QasmProgram
    ) -> np.ndarray:
        """Decode the intensities as described in the class docstring."""
        shape, count = image_shape(program)
        probabilities = np.zeros(count)
        for (index,), p in marginal(result, program, "position").items():
            if index < count:
                probabilities[index] = p
        return (np.sqrt(probabilities) * program.meta["norm"]).reshape(shape)


def _grover_rudolph_level(
    weights: np.ndarray, n: int, level: int
) -> QuantumCircuit:
    """Return the rotations that split each prefix over the target.

    The target is ``position[n - 1 - level]``. ``weights`` are squared
    amplitudes indexed by pixel; qubit ``j`` is bit ``j`` of the index. A
    prefix is the value of the qubits above the target.
    """
    target = n - 1 - level
    controls = list(range(target + 1, n))
    body = QuantumCircuit(n)
    # by_prefix[v, b] = total weight with high bits v and target bit b.
    by_prefix = weights.reshape(-1, 2, 1 << target).sum(axis=2)
    for prefix, (zero, one) in enumerate(by_prefix):
        total = zero + one
        if total == 0:
            continue
        angle = 2 * acos(min(1.0, sqrt(zero / total)))
        if angle > IDENTITY_ANGLE:
            controlled(body, RYGate(angle), controls, prefix, target)
    return body
