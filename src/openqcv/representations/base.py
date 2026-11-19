"""The contract every quantum image representation follows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Qubit

from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.core.registry import Registry


@dataclass(frozen=True)
class EncodedImage:
    """A representation's output, before it is exported to QASM.

    Attributes:
        representation: Name of the representation, e.g. ``"neqr"``.
        circuit: The logical circuit, without measurements. Each quantum
            register's name is its role (``"position"``, ``"color"``, ...),
            it is built from standard (controlled) gates, and high-level steps
            are wrapped in :class:`~openqcv.core.Block` gates.
        meta: JSON-serialisable classical information needed to decode.
    """

    representation: str
    circuit: QuantumCircuit
    meta: dict[str, Any]

    @property
    def registers(self) -> dict[str, list[int]]:
        """Role -> qubit indices, taken from the circuit's register names."""
        return {
            register.name: [self.circuit.find_bit(q).index for q in register]
            for register in self.circuit.qregs
        }


class Representation(ABC):
    """Base class for quantum image representations.

    To add one: subclass this, set ``name``, implement the three methods and
    decorate the class with ``@register_representation``.

    ``decode`` is a classmethod on purpose: it may use only the result and
    ``program.meta``, never encoding options. That is what lets a program
    received as JSON be decoded by anyone.
    """

    name: ClassVar[str]

    @abstractmethod
    def qubits_required(self, shape: tuple[int, ...]) -> int:
        """Number of qubits needed to encode an image of ``shape``."""

    @abstractmethod
    def encode(self, image: ArrayLike) -> EncodedImage:
        """Build the logical circuit for ``image``."""

    @classmethod
    @abstractmethod
    def decode(
        cls, result: ExecutionResult, program: QasmProgram
    ) -> np.ndarray:
        """Reconstruct the image (float64, original shape) from a result."""


IDENTITY_ANGLE = 1e-12
"""Rotations by at most this angle (radians) are the identity and omitted."""

REPRESENTATIONS: Registry[Representation] = Registry(
    "representation", Representation
)
register_representation = REPRESENTATIONS.register


def controlled(
    circuit: QuantumCircuit,
    gate: Gate,
    controls: Sequence[Qubit | int],
    value: int,
    target: Qubit | int,
) -> None:
    """Apply ``gate`` to ``target`` only when ``controls`` read ``value``.

    ``controls[0]`` is the least significant bit of ``value``; 0-bits become
    open controls (``negctrl`` in logical QASM). Representations should build
    circuits from standard gates this way so they stay textbook-readable.
    """
    if not 0 <= value < (1 << len(controls)):
        raise ValueError(
            f"value {value} does not fit in {len(controls)} control qubits"
        )
    if controls:
        # annotated=False gives a ControlledGate (base gate + ctrl_state),
        # which is what logical QASM writes as ctrl/negctrl modifiers.
        gate = gate.control(len(controls), ctrl_state=value, annotated=False)
    circuit.append(gate, [*controls, target])


def marginal(
    result: ExecutionResult, program: QasmProgram, *roles: str
) -> dict[tuple[int, ...], float]:
    """Probability of each joint value of the registers ``roles``.

    Example: ``marginal(result, program, "position", "color")`` maps
    ``(pixel_index, color_bit)`` to its probability.
    """
    totals: dict[tuple[int, ...], float] = defaultdict(float)
    for bits, p in result.probabilities.items():
        totals[tuple(program.value(bits, role) for role in roles)] += p
    return dict(totals)


def image_shape(program: QasmProgram) -> tuple[tuple[int, ...], int]:
    """The original image shape from ``program.meta`` and its pixel count."""
    shape = tuple(int(s) for s in program.meta["shape"])
    return shape, int(np.prod(shape))
