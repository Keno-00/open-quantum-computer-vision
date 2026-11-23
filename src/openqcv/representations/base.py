"""The contract every quantum image representation (QIR) follows."""

from __future__ import annotations

import inspect
import re
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

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""Register names must be valid OpenQASM identifiers."""


@dataclass(frozen=True)
class EncodedImage:
    """A QIR's output, before it is exported to QASM.

    Attributes:
        circuit: The logical circuit, without measurements. Every qubit
            belongs to exactly one named quantum register; the QIR chooses
            the names. It is built from standard (controlled) gates, and
            high-level steps are wrapped in :class:`~openqcv.core.Block`
            gates.
        meta: JSON-serializable classical information needed to decode.

    Raises:
        TypeError: If ``circuit`` is not a ``QuantumCircuit`` or ``meta`` is
            not a ``dict``.
        ValueError: If the circuit has classical bits, has a qubit outside
            every register, or a register name is not an identifier.
    """

    circuit: QuantumCircuit
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError(
                "circuit must be a QuantumCircuit, got "
                f"{type(self.circuit).__name__}"
            )
        if not isinstance(self.meta, dict):
            raise TypeError(
                f"meta must be a dict, got {type(self.meta).__name__}"
            )
        if self.circuit.num_clbits:
            raise ValueError(
                "The circuit must have no classical bits; every qubit is "
                "measured when the program is exported"
            )
        registered = sum(register.size for register in self.circuit.qregs)
        if registered != self.circuit.num_qubits:
            raise ValueError(
                "Every qubit must belong to exactly one named register"
            )
        for register in self.circuit.qregs:
            if not _IDENTIFIER.fullmatch(register.name):
                raise ValueError(
                    f"Register name {register.name!r} is not an identifier "
                    "(letters, digits and _, not starting with a digit)"
                )

    @property
    def registers(self) -> dict[str, list[int]]:
        """Register name -> qubit indices, in register order."""
        return {
            register.name: [self.circuit.find_bit(q).index for q in register]
            for register in self.circuit.qregs
        }


class QIR(ABC):
    """Base class for quantum image representations.

    A QIR is a class with a ``name``, an :meth:`encode` method and a
    :meth:`decode` classmethod, registered with :func:`register`. Nothing
    else is prescribed: it chooses its own registers, metadata and readout.
    :func:`openqcv.pipeline.check` verifies that it works with the rest of
    openqcv.

    ``decode`` is a classmethod on purpose: it may use only the result and
    ``program.meta``, never encoding options. That is what lets a program
    received as JSON be decoded by anyone.
    """

    name: ClassVar[str]

    @abstractmethod
    def encode(self, image: ArrayLike) -> EncodedImage:
        """Build the logical circuit for ``image``."""

    @classmethod
    @abstractmethod
    def decode(
        cls, result: ExecutionResult, program: QasmProgram
    ) -> np.ndarray:
        """Reconstruct the image from a result."""

    def qubits_required(self, shape: tuple[int, ...]) -> int:
        """Return the number of qubits needed for an image of ``shape``.

        Optional; :func:`openqcv.pipeline.check` compares it with the encoded
        program when a QIR implements it.

        Raises:
            NotImplementedError: If the QIR does not predict its qubit
                count.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not predict its qubit count"
        )


Representation = QIR
"""The former name of :class:`QIR`, kept so existing code keeps working."""

IDENTITY_ANGLE = 1e-12
"""Rotations by at most this angle (radians) are the identity and omitted."""

REPRESENTATIONS: Registry[QIR] = Registry(
    "representation", QIR, entry_point_group="openqcv.representations"
)


def register(cls: type[QIR]) -> type[QIR]:
    """Register a QIR under its ``name``; a class decorator.

    Raises:
        TypeError: If ``cls`` is not a :class:`QIR` subclass, has no
            ``name``, or its ``decode`` is not a classmethod.
        ValueError: If the name is taken.
    """
    if isinstance(cls, type) and not isinstance(
        inspect.getattr_static(cls, "decode", None), classmethod
    ):
        raise TypeError(
            f"{cls.__name__}.decode must be a classmethod, so that it can "
            "use only the result and the program"
        )
    return REPRESENTATIONS.register(cls)


register_representation = register
"""The former name of :func:`register`."""


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
    result: ExecutionResult, program: QasmProgram, *registers: str
) -> dict[tuple[int, ...], float]:
    """Return the probability of each joint value of ``registers``.

    Example: ``marginal(result, program, "position", "color")`` maps
    ``(pixel_index, color_bit)`` to its probability.
    """
    totals: dict[tuple[int, ...], float] = defaultdict(float)
    for bits, p in result.probabilities.items():
        totals[tuple(program.value(bits, name) for name in registers)] += p
    return dict(totals)


def image_shape(program: QasmProgram) -> tuple[tuple[int, ...], int]:
    """Return ``meta["shape"]`` and its pixel count."""
    shape = tuple(int(s) for s in program.meta["shape"])
    return shape, int(np.prod(shape))
