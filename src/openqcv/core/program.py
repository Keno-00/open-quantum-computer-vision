"""The two data types every part of openqcv passes around.

Bitstring convention
--------------------
Every bitstring in openqcv is written in *qubit order*: character ``j`` is the
value of qubit ``j``. (Qiskit prints the reverse; backend adapters convert.)
Inside a register the first qubit is the least significant bit, so
``program.value(bits, "position")`` is the pixel index.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class QasmProgram:
    """An encoded image, ready to send to any backend.

    Representations produce it, backends consume it, and it can be saved or
    sent as JSON. Decoding needs nothing but this object and a result.

    Attributes:
        representation: Name of the representation that produced it, e.g.
            ``"frqi"``.
        logical_qasm: Textbook-level OpenQASM 3: named registers, standard
            gates with ``ctrl``/``negctrl`` modifiers, one ``gate`` per
            high-level step. Readable; used for drawing.
        qasm2: The circuit lowered to basis gates, as OpenQASM 2.
        qasm3: The same lowered circuit, as OpenQASM 3.
        registers: Role -> qubit indices, e.g.
            ``{"position": [0, 1], "color": [2]}``.
        meta: Classical information needed to decode (image shape, norm,
            ...).
    """

    representation: str
    logical_qasm: str
    qasm2: str
    qasm3: str
    registers: dict[str, list[int]]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        """Total number of qubits over all registers."""
        return sum(len(qubits) for qubits in self.registers.values())

    def qasm(self, version: Literal[2, 3] = 3) -> str:
        """The lowered circuit as OpenQASM ``version`` (2 or 3)."""
        if version == 2:
            return self.qasm2
        if version == 3:
            return self.qasm3
        raise ValueError("version must be 2 or 3")

    def value(self, bits: str, role: str) -> int:
        """The integer held by register ``role`` in a qubit-order bitstring."""
        return sum(
            1 << k
            for k, qubit in enumerate(self.registers[role])
            if bits[qubit] == "1"
        )

    def to_json(self, indent: int | None = None) -> str:
        """Serialise to JSON; :meth:`from_json` restores an equal program."""
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> QasmProgram:
        """Restore a program written by :meth:`to_json`."""
        return cls(**json.loads(text))

    def __repr__(self) -> str:
        registers = ", ".join(
            f"{role}={len(q)}" for role, q in self.registers.items()
        )
        return (
            f"QasmProgram({self.representation!r}, "
            f"qubits={self.num_qubits} [{registers}])"
        )


@dataclass(frozen=True)
class ExecutionResult:
    """What a backend returns, in a backend-independent form.

    Attributes:
        backend: Name of the backend that produced it.
        probabilities: Qubit-order bitstring -> probability, exact or
            estimated from shots. Outcomes with zero probability are omitted.
        shots: Number of shots, or ``None`` for exact simulation.
        counts: Qubit-order counts when shots were used.
        statevector: Final state from exact simulators, else ``None``.
            Index ``i`` has qubit ``j`` as bit ``2**j`` (Qiskit's ordering).
    """

    backend: str
    probabilities: dict[str, float]
    shots: int | None = None
    counts: dict[str, int] | None = None
    statevector: np.ndarray | None = None

    @classmethod
    def from_counts(
        cls, backend: str, counts: dict[str, int]
    ) -> ExecutionResult:
        """Build a result from qubit-order counts.

        Raises:
            ValueError: If any count is negative or there are no shots at all.
        """
        if any(n < 0 for n in counts.values()):
            raise ValueError("Counts cannot be negative")
        shots = sum(counts.values())
        if shots == 0:
            raise ValueError("A result needs at least one shot")
        probabilities = {bits: n / shots for bits, n in counts.items() if n}
        return cls(backend, probabilities, shots=shots, counts=dict(counts))

    def __repr__(self) -> str:
        mode = "exact" if self.shots is None else f"shots={self.shots}"
        return (
            f"ExecutionResult({self.backend!r}, {mode}, "
            f"outcomes={len(self.probabilities)})"
        )
