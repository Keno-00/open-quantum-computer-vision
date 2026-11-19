"""Named high-level steps inside a representation's circuit."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Gate


class Block(Gate):
    """A named, high-level step of a representation, e.g. ``"frqi_rotation"``.

    A block stays a single named gate in the logical QASM. That is what lets
    the drawer show one labelled box instead of hundreds of basis gates.

    Args:
        name: Gate name in QASM. Must be a valid identifier (lowercase, ``_``).
        body: The circuit the block stands for.
    """

    def __init__(self, name: str, body: QuantumCircuit) -> None:
        super().__init__(name, body.num_qubits, [])
        self.definition = body
