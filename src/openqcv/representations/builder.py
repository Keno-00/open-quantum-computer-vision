"""Building a QIR's circuit by register name."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import (
    HGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    SdgGate,
    SGate,
    SXGate,
    TdgGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)

from openqcv.core.blocks import Block
from openqcv.representations.base import _IDENTIFIER, EncodedImage, controlled

Targets = str | Qubit | Sequence[Qubit]
"""A register name, a qubit, or a sequence of qubits."""

Condition = Mapping[str | Qubit, int]
"""Register name or qubit -> the value it must hold."""

_Operation = tuple[Gate, list[Qubit], int, Qubit]
"""A gate, its controls, the value they must read, and its target."""


class Circuit:
    """A quantum circuit addressed by register name; optional for a QIR.

    Gates take their targets as a register name (every qubit of it), a
    qubit or a sequence of qubits, and an optional condition ``when``
    that maps register names or single qubits to the value they must hold::

        c = Circuit(row=1, col=2, value=4)
        c.h("row", "col")
        c.x(c.bits("value", 5), when={"row": 1, "col": 2})

    Register values are read with the register's first qubit as the least
    significant bit, as everywhere in openqcv. Inside :meth:`block`, gates
    are collected into one named :class:`~openqcv.core.Block`.

    Args:
        **registers: Register name -> number of qubits, in circuit order.

    Raises:
        TypeError: If a size is not an ``int``.
        ValueError: If there are no registers, a size is less than 1, or a
            name is not an identifier.
    """

    def __init__(self, **registers: int) -> None:
        if not registers:
            raise ValueError("A circuit needs at least one register")
        for name, size in registers.items():
            if not _IDENTIFIER.fullmatch(name):
                raise ValueError(
                    f"Register name {name!r} is not an identifier"
                )
            if not isinstance(size, int) or isinstance(size, bool):
                raise TypeError(
                    f"Register {name!r} size must be an int, got "
                    f"{type(size).__name__}"
                )
            if size < 1:
                raise ValueError(f"Register {name!r} needs at least 1 qubit")
        self._registers = {
            name: QuantumRegister(size, name)
            for name, size in registers.items()
        }
        self._circuit = QuantumCircuit(*self._registers.values())
        self._block: list[_Operation] | None = None
        self._block_names: set[str] = set()

    def __getitem__(self, name: str) -> QuantumRegister:
        """Return the register called ``name``.

        Raises:
            ValueError: If there is no such register.
        """
        try:
            return self._registers[name]
        except KeyError:
            available = ", ".join(self._registers)
            raise ValueError(
                f"No register {name!r}; the registers are {available}"
            ) from None

    def bits(self, name: str, value: int) -> list[Qubit]:
        """Return the qubits of register ``name`` that are 1 in ``value``.

        Raises:
            ValueError: If ``value`` does not fit in the register.
        """
        register = self[name]
        _check_fits(name, value, register.size)
        return [q for k, q in enumerate(register) if (value >> k) & 1]

    def apply(
        self, gate: Gate, *targets: Targets, when: Condition | None = None
    ) -> None:
        """Apply the one-qubit ``gate`` to every target qubit.

        Raises:
            ValueError: If ``gate`` acts on more than one qubit, a target is
                also a control, or a target or condition is invalid.
        """
        if gate.num_qubits != 1:
            raise ValueError(
                f"apply takes one-qubit gates; {gate.name!r} acts on "
                f"{gate.num_qubits}. Express it with conditions instead"
            )
        controls, value = self._condition(when or {})
        for target in self._qubits(targets):
            if target in controls:
                raise ValueError("A qubit cannot be both target and control")
            if self._block is None:
                controlled(self._circuit, gate, controls, value, target)
            else:
                self._block.append((gate, controls, value, target))

    def x(self, *targets: Targets, when: Condition | None = None) -> None:
        """Apply X (NOT)."""
        self.apply(XGate(), *targets, when=when)

    def y(self, *targets: Targets, when: Condition | None = None) -> None:
        """Apply Y."""
        self.apply(YGate(), *targets, when=when)

    def z(self, *targets: Targets, when: Condition | None = None) -> None:
        """Apply Z."""
        self.apply(ZGate(), *targets, when=when)

    def h(self, *targets: Targets, when: Condition | None = None) -> None:
        """Apply the Hadamard gate."""
        self.apply(HGate(), *targets, when=when)

    def s(self, *targets: Targets, when: Condition | None = None) -> None:
        """Apply S."""
        self.apply(SGate(), *targets, when=when)

    def sdg(self, *targets: Targets, when: Condition | None = None) -> None:
        """Apply S-dagger."""
        self.apply(SdgGate(), *targets, when=when)

    def t(self, *targets: Targets, when: Condition | None = None) -> None:
        """Apply T."""
        self.apply(TGate(), *targets, when=when)

    def tdg(self, *targets: Targets, when: Condition | None = None) -> None:
        """Apply T-dagger."""
        self.apply(TdgGate(), *targets, when=when)

    def sx(self, *targets: Targets, when: Condition | None = None) -> None:
        """Apply the square root of X."""
        self.apply(SXGate(), *targets, when=when)

    def rx(
        self, theta: float, *targets: Targets, when: Condition | None = None
    ) -> None:
        """Apply a rotation by ``theta`` about X."""
        self.apply(RXGate(theta), *targets, when=when)

    def ry(
        self, theta: float, *targets: Targets, when: Condition | None = None
    ) -> None:
        """Apply a rotation by ``theta`` about Y."""
        self.apply(RYGate(theta), *targets, when=when)

    def rz(
        self, theta: float, *targets: Targets, when: Condition | None = None
    ) -> None:
        """Apply a rotation by ``theta`` about Z."""
        self.apply(RZGate(theta), *targets, when=when)

    def p(
        self, theta: float, *targets: Targets, when: Condition | None = None
    ) -> None:
        """Apply the phase gate ``diag(1, exp(i theta))``."""
        self.apply(PhaseGate(theta), *targets, when=when)

    @contextmanager
    def block(self, name: str) -> Iterator[None]:
        """Collect the gates applied inside into one block called ``name``.

        The block acts on the qubits its gates use, in circuit order. A
        block with no gates is left out.

        Raises:
            ValueError: If ``name`` is not an identifier or is already used,
                or blocks are nested.
        """
        if self._block is not None:
            raise ValueError("Blocks cannot be nested")
        if not _IDENTIFIER.fullmatch(name):
            raise ValueError(f"Block name {name!r} is not an identifier")
        if name in self._block_names:
            raise ValueError(f"A block called {name!r} already exists")
        self._block = []
        try:
            yield
            operations = self._block
        finally:
            self._block = None
        self._block_names.add(name)
        if operations:
            self._append_block(name, operations)

    def finish(self, **meta: Any) -> EncodedImage:
        """Return the circuit and ``meta`` as an :class:`EncodedImage`.

        Raises:
            ValueError: If a block is still open.
        """
        if self._block is not None:
            raise ValueError("finish was called inside a block")
        return EncodedImage(self._circuit.copy(), meta)

    def _qubits(self, targets: Sequence[Targets]) -> list[Qubit]:
        qubits: list[Qubit] = []
        for target in targets:
            if isinstance(target, str):
                qubits.extend(self[target])
            elif isinstance(target, Qubit):
                qubits.append(self._own(target))
            else:
                qubits.extend(self._own(q) for q in target)
        return qubits

    def _condition(self, when: Condition) -> tuple[list[Qubit], int]:
        """Return the control qubits and the value they must read."""
        controls: list[Qubit] = []
        value = 0
        for key, wanted in when.items():
            if not isinstance(wanted, int) or isinstance(wanted, bool):
                raise TypeError(
                    f"Condition values must be ints, got "
                    f"{type(wanted).__name__} for {key!r}"
                )
            qubits = list(self[key]) if isinstance(key, str) else [key]
            _check_fits(key, wanted, len(qubits))
            value |= wanted << len(controls)
            controls.extend(self._own(q) for q in qubits)
        if len(set(controls)) != len(controls):
            raise ValueError("A qubit appears twice in the condition")
        return controls, value

    def _own(self, qubit: Any) -> Qubit:
        """Return ``qubit`` if it belongs to this circuit."""
        try:
            self._circuit.find_bit(qubit)
        except CircuitError:
            raise ValueError(
                f"{qubit!r} is not a qubit of this circuit"
            ) from None
        return qubit

    def _append_block(self, name: str, operations: list[_Operation]) -> None:
        used = {
            q
            for gate, controls, _, target in operations
            for q in (*controls, target)
        }
        qubits = sorted(used, key=lambda q: self._circuit.find_bit(q).index)
        local = {q: k for k, q in enumerate(qubits)}
        body = QuantumCircuit(len(qubits))
        for gate, controls, value, target in operations:
            controlled(
                body,
                gate,
                [local[q] for q in controls],
                value,
                local[target],
            )
        self._circuit.append(Block(name, body), qubits)


def _check_fits(name: object, value: int, size: int) -> None:
    if not 0 <= value < (1 << size):
        raise ValueError(f"{name!r} has {size} qubit(s); {value} does not fit")
