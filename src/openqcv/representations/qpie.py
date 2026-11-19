"""Quantum Probability Image Encoding (QPIE)."""

from __future__ import annotations

from math import acos, prod, sqrt

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
class QPIE(Representation):
    """Pixel intensities become the amplitudes of the state.

    ``|I> = sum_i (x_i / ||x||) |i>`` on ``n = ceil(log2 N)`` qubits: the most
    qubit-efficient representation, but reading it back needs many shots.

    The state is prepared with the Grover–Rudolph rotation tree: level ``k``
    rotates ``position[n-1-k]`` by ``R_y`` angles conditioned on the
    already-prepared higher qubits. Amplitudes are non-negative, so no phase
    gates are needed.

    The norm ``||x||`` is not in the quantum state, so it travels in
    ``meta`` and decoding multiplies it back.
    """

    name = "qpie"

    def qubits_required(self, shape: tuple[int, ...]) -> int:
        """``ceil(log2 N)`` position qubits for ``N`` pixels."""
        return position_qubits(prod(shape[:2]))

    def encode(self, image: ArrayLike) -> EncodedImage:
        """Prepare the normalised intensities as amplitudes.

        Raises:
            ValueError: If any intensity is negative or all are zero.
        """
        pixels = as_grayscale(image)
        if (pixels < 0).any():
            raise ValueError("QPIE needs non-negative intensities")

        n = position_qubits(pixels.size)
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
            self.name, circuit, {"shape": list(pixels.shape), "norm": norm}
        )

    @classmethod
    def decode(
        cls, result: ExecutionResult, program: QasmProgram
    ) -> np.ndarray:
        """Intensities ``sqrt(p_i) * norm`` from position probabilities."""
        shape, count = image_shape(program)
        probabilities = np.zeros(count)
        for (index,), p in marginal(result, program, "position").items():
            if index < count:
                probabilities[index] = p
        return (np.sqrt(probabilities) * program.meta["norm"]).reshape(shape)


def _grover_rudolph_level(
    weights: np.ndarray, n: int, level: int
) -> QuantumCircuit:
    """Rotations that split each prefix's weight over the target qubit.

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
