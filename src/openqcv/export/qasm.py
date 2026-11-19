"""Turning encoded images into OpenQASM, and reading OpenQASM back.

Each program carries two levels of QASM:

* **logical** (OpenQASM 3): textbook gates with named registers, open and
  filled controls, and one ``gate`` per :class:`Block`. Readable, and what
  the drawer uses. See :mod:`openqcv.export.logical`.
* **lowered** (OpenQASM 2 and 3): only basis gates and measurements. What
  backends run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, qasm2, qasm3, transpile

from openqcv.core.program import QasmProgram
from openqcv.export import logical

if TYPE_CHECKING:
    from openqcv.representations.base import EncodedImage

DEFAULT_BASIS: tuple[str, ...] = ("rx", "ry", "rz", "cx")
"""Gates found in both OpenQASM 2's ``qelib1.inc`` and OpenQASM 3's
``stdgates.inc``, so the lowered QASM parses almost anywhere."""

MEASUREMENT_REGISTER = "meas"


def to_program(
    encoded: EncodedImage, basis: Sequence[str] = DEFAULT_BASIS
) -> QasmProgram:
    """Export an :class:`EncodedImage` as a :class:`QasmProgram`.

    Every qubit ``j`` is measured into bit ``j`` of the ``meas`` register.
    """
    circuit = encoded.circuit.copy()
    circuit.add_register(
        ClassicalRegister(circuit.num_qubits, MEASUREMENT_REGISTER)
    )
    circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))

    # Blocks can sit anywhere, so never let synthesis assume a qubit is still
    # |0> and borrow it as a clean ancilla: that silently corrupts the state.
    lowered = transpile(
        circuit,
        basis_gates=[*basis, "measure"],
        optimization_level=1,
        qubits_initially_zero=False,
    )

    return QasmProgram(
        representation=encoded.representation,
        logical_qasm=logical.dumps(circuit),
        qasm2=qasm2.dumps(lowered),
        qasm3=qasm3.dumps(lowered),
        registers=encoded.registers,
        meta=_jsonable(encoded.meta),
    )


def load_circuit(program: QasmProgram) -> QuantumCircuit:
    """Parse a program's lowered OpenQASM 2 into a Qiskit circuit."""
    return qasm2.loads(program.qasm2)


def _jsonable(value: Any) -> Any:
    """Convert tuples and NumPy scalars/arrays so ``meta`` survives JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
