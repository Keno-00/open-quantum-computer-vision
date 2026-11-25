"""Turning encoded images into OpenQASM, and reading OpenQASM back.

Each program carries two levels of QASM:

* **logical** (OpenQASM 3): textbook gates with named registers, open and
  filled controls, and one ``gate`` per :class:`Block`. Readable, and what
  the drawer uses. See :mod:`openqcv.export.logical`.
* **lowered** (OpenQASM 2 and 3): only basis gates and measurements. What
  backends run.
"""

from __future__ import annotations

import json
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
    encoded: EncodedImage,
    representation: str,
    basis: Sequence[str] = DEFAULT_BASIS,
) -> QasmProgram:
    """Export an :class:`EncodedImage` as a :class:`QasmProgram`.

    Every qubit ``j`` is measured into bit ``j`` of the ``meas`` register.

    Args:
        encoded: The circuit and metadata.
        representation: The registered name of the QIR that produced it;
            :func:`openqcv.decode` looks the decoder up by this name.
        basis: The gates the lowered circuit is built from.

    Raises:
        ValueError: If a register is called ``meas``, the circuit has a gate
            with no logical QASM form, or ``meta`` cannot be written as JSON.
    """
    if MEASUREMENT_REGISTER in encoded.registers:
        raise ValueError(
            f"The register name {MEASUREMENT_REGISTER!r} is reserved for "
            "measurement results"
        )
    meta = _jsonable(encoded.meta)
    try:
        json.dumps(meta, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"meta must be JSON-serializable with finite numbers: {error}"
        ) from None

    circuit = measured(encoded.circuit)
    lowered = lower(circuit, basis)

    return QasmProgram(
        representation=representation,
        logical_qasm=logical.dumps(circuit),
        qasm2=qasm2.dumps(lowered),
        qasm3=qasm3.dumps(lowered),
        registers=encoded.registers,
        meta=meta,
    )


def measured(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``circuit`` measuring qubit ``j`` into ``meas[j]``."""
    circuit = circuit.copy()
    circuit.add_register(
        ClassicalRegister(circuit.num_qubits, MEASUREMENT_REGISTER)
    )
    circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))
    return circuit


def lower(
    circuit: QuantumCircuit, basis: Sequence[str] = DEFAULT_BASIS
) -> QuantumCircuit:
    """Rewrite ``circuit`` in ``basis`` gates and measurements.

    Parameters may stay unbound; they end up in the angles of the rotation
    gates. Qubits keep their order and registers.
    """
    # Blocks can sit anywhere, so never let synthesis assume a qubit is still
    # |0> and borrow it as a clean ancilla: that silently corrupts the state.
    return transpile(
        circuit,
        basis_gates=[*basis, "measure"],
        optimization_level=1,
        qubits_initially_zero=False,
    )


def load_circuit(program: QasmProgram) -> QuantumCircuit:
    """Parse a program's lowered OpenQASM 2 into a Qiskit circuit."""
    return qasm2.loads(program.qasm2)


def _jsonable(value: Any) -> Any:
    """Convert tuples and NumPy values so ``meta`` survives JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
