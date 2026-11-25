"""Logical OpenQASM 3: the textbook-level form of a circuit.

The logical QASM keeps what a paper figure would show:

* named registers (``qubit[2] position;``),
* one ``gate`` definition per :class:`~openqcv.core.Block`, which a drawing
  frames as a labeled group,
* controlled gates as OpenQASM 3 modifiers, including open controls::

      negctrl @ ctrl @ ry(0.927) position[0], position[1], color[0];

* classically controlled gates: ``if (m[0] == 1) z bob[0];``

:func:`dumps` writes it from a Qiskit circuit; :func:`parse` reads it back as
a flat list of :class:`LogicalOp`. Only this subset of OpenQASM 3 is
supported: sized ``qubit``/``bit`` registers, parameter-free ``gate``
definitions, gate calls with ``ctrl``/``negctrl`` modifiers, ``measure``
assignments and ``if (bit == value)`` on a single gate.
"""

from __future__ import annotations

import ast
import itertools
import math
import operator
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate, Instruction

from openqcv.core.blocks import Block

_STANDARD = {
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "rx",
    "ry",
    "rz",
    "p",
    "id",
}

# stdgates.inc names that are controlled versions of a standard gate.
_ALIASES = {
    "cx": (1, "x"),
    "cy": (1, "y"),
    "cz": (1, "z"),
    "ch": (1, "h"),
    "crx": (1, "rx"),
    "cry": (1, "ry"),
    "crz": (1, "rz"),
    "cp": (1, "p"),
    "ccx": (2, "x"),
}


# --------------------------------------------------------------------- write


def dumps(circuit: QuantumCircuit) -> str:
    """Write ``circuit`` as logical OpenQASM 3.

    Raises:
        ValueError: For gates outside the textbook set; build them from
            standard (optionally controlled) gates or wrap them in a Block.
    """

    def ref(bit) -> str:
        register, index = circuit.find_bit(bit).registers[0]
        return f"{register.name}[{index}]"

    definitions: dict[str, str] = {}
    body = [f"bit[{c.size}] {c.name};" for c in circuit.cregs]
    body += [f"qubit[{q.size}] {q.name};" for q in circuit.qregs]

    for instruction in circuit.data:
        operation = instruction.operation
        args = [ref(q) for q in instruction.qubits]
        if isinstance(operation, Block):
            definition = _gate_definition(operation)
            if (
                definitions.setdefault(operation.name, definition)
                != definition
            ):
                raise ValueError(
                    f"Two different blocks are both named {operation.name!r}"
                )
            body.append(f"{operation.name} {', '.join(args)};")
        elif operation.name == "measure":
            body.append(f"{ref(instruction.clbits[0])} = measure {args[0]};")
        elif operation.name != "barrier":
            body.append(_statement(operation, args))

    header = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    return "\n".join(header + list(definitions.values()) + body) + "\n"


def _gate_definition(block: Block) -> str:
    formals = [f"q{i}" for i in range(block.num_qubits)]
    inner = block.definition
    lines = [
        "  "
        + _statement(
            i.operation, [formals[inner.find_bit(q).index] for q in i.qubits]
        )
        for i in inner.data
    ]
    return (
        f"gate {block.name} {', '.join(formals)} {{\n"
        + "\n".join(lines)
        + "\n}"
    )


def _statement(operation: Instruction, args: list[str]) -> str:
    modifiers = ""
    if isinstance(operation, ControlledGate):
        state = operation.ctrl_state
        modifiers = "".join(
            "ctrl @ " if (state >> m) & 1 else "negctrl @ "
            for m in range(operation.num_ctrl_qubits)
        )
        operation = operation.base_gate
    if operation.name not in _STANDARD:
        raise ValueError(
            f"Gate {operation.name!r} has no textbook form in logical QASM; "
            "build it from standard (controlled) gates or wrap it in a Block"
        )
    params = ""
    if operation.params:
        params = (
            "(" + ", ".join(format_angle(p) for p in operation.params) + ")"
        )
    return f"{modifiers}{operation.name}{params} {', '.join(args)};"


def format_angle(value: float) -> str:
    """Write a gate parameter as logical QASM does: 12 significant digits."""
    return f"{float(value):.12g}"


# ---------------------------------------------------------------------- read


@dataclass(frozen=True)
class LogicalOp:
    """One textbook operation. Qubits are lane indices (register order).

    Attributes:
        name: Base gate (``"x"``, ``"ry"``, ...) or ``"measure"``.
        targets: Lanes the base gate acts on.
        params: Gate parameters (angles in radians).
        controls: ``(lane, filled)`` pairs; ``filled=False`` is an open
            control.
        condition: ``(bit, value)`` for classically controlled gates.
        clbit: Destination bit of a measurement.
        group: Name of the gate definition this op came from, if any.
        group_id: Distinguishes separate calls of the same gate definition.
    """

    name: str
    targets: tuple[int, ...]
    params: tuple[float, ...] = ()
    controls: tuple[tuple[int, bool], ...] = ()
    condition: tuple[int, int] | None = None
    clbit: int | None = None
    group: str | None = None
    group_id: int | None = None

    @property
    def lanes(self) -> tuple[int, ...]:
        """Every quantum lane the op touches: controls, then targets."""
        return tuple(lane for lane, _ in self.controls) + self.targets


@dataclass(frozen=True)
class LogicalCircuit:
    """A parsed logical program.

    Attributes:
        qregs: ``(name, size)`` per quantum register, in declaration order;
            lanes are numbered through them consecutively.
        cregs: ``(name, size)`` per classical register; bits likewise.
        ops: Operations in program order, gate definitions expanded.
    """

    qregs: tuple[tuple[str, int], ...]
    cregs: tuple[tuple[str, int], ...]
    ops: tuple[LogicalOp, ...]

    @property
    def num_qubits(self) -> int:
        """Total number of quantum lanes."""
        return sum(size for _, size in self.qregs)


_GATE_DEF = re.compile(r"gate\s+(\w+)\s+([\w\s,]+?)\s*\{(.*?)\}", re.S)
# modifiers, gate name, optional parameters (may nest), qubit arguments
_CALL = re.compile(
    r"((?:(?:neg)?ctrl(?:\(\d+\))?\s*@\s*)*)(\w+)\s*(?:\((.*)\))?\s+([^()]+)"
)
_IF = re.compile(r"if\s*\(\s*(\w+\[\d+\])\s*(?:==\s*(\d+))?\s*\)\s*(.+)")


def parse(text: str) -> LogicalCircuit:
    """Read logical OpenQASM 3 written by :func:`dumps` (or by hand).

    Raises:
        ValueError: For statements outside the supported subset, unknown
            gates, registers or bits, and wrong argument counts.
    """
    text = re.sub(r"//[^\n]*", "", text)
    definitions = {
        m[1]: ([a.strip() for a in m[2].split(",")], m[3])
        for m in _GATE_DEF.finditer(text)
    }
    text = _GATE_DEF.sub("", text)

    qregs: list[tuple[str, int]] = []
    cregs: list[tuple[str, int]] = []
    qubits: dict[str, int] = {}
    bits: dict[str, int] = {}
    ops: list[LogicalOp] = []
    group_ids = itertools.count()

    def qubit(ref: str) -> int:
        return _lookup(qubits, ref, "qubit")

    for statement in _statements(text):
        if statement.startswith(("OPENQASM", "include", "barrier")):
            continue
        if m := re.fullmatch(r"qubit\[(\d+)\] (\w+)", statement):
            qregs.append((m[2], int(m[1])))
            _declare(qubits, m[2], int(m[1]))
        elif m := re.fullmatch(r"bit\[(\d+)\] (\w+)", statement):
            cregs.append((m[2], int(m[1])))
            _declare(bits, m[2], int(m[1]))
        elif m := re.fullmatch(
            r"(\w+\[\d+\]) ?= ?measure (\w+\[\d+\])", statement
        ):
            ops.append(
                LogicalOp(
                    "measure", (qubit(m[2]),), clbit=_lookup(bits, m[1], "bit")
                )
            )
        elif m := _IF.fullmatch(statement):
            condition = (_lookup(bits, m[1], "bit"), int(m[2] or 1))
            ops += _call(m[3], qubit, definitions, group_ids, condition)
        else:
            ops += _call(statement, qubit, definitions, group_ids)

    return LogicalCircuit(tuple(qregs), tuple(cregs), tuple(ops))


def _statements(text: str) -> Iterator[str]:
    """Yield the non-empty statements with whitespace normalized."""
    for statement in text.split(";"):
        statement = " ".join(statement.split())
        if statement:
            yield statement


def _declare(table: dict[str, int], name: str, size: int) -> None:
    """Assign the next ``size`` indices to ``name[0]`` to ``name[size-1]``."""
    first = len(table)
    table.update({f"{name}[{i}]": first + i for i in range(size)})


def _lookup(table: dict[str, int], ref: str, kind: str) -> int:
    try:
        return table[ref]
    except KeyError:
        raise ValueError(f"Unknown {kind} {ref!r} in logical QASM") from None


def _call(
    statement: str,
    resolve: Callable[[str], int],
    definitions: dict[str, tuple[list[str], str]],
    group_ids: Iterator[int],
    condition: tuple[int, int] | None = None,
    group: tuple[str, int] | None = None,
) -> list[LogicalOp]:
    """Expand one gate call into ops.

    A call of a gate definition becomes a group; nested calls inherit the
    outermost call's group, so each top-level call is one frame.
    """
    m = _CALL.fullmatch(statement)
    if not m:
        raise ValueError(f"Can't read logical QASM statement: {statement!r}")
    filled = [
        not negated
        for negated, count in re.findall(r"(neg)?ctrl(?:\((\d+)\))?", m[1])
        for _ in range(int(count or 1))
    ]
    name, args = m[2], [a.strip() for a in m[4].split(",")]
    params = tuple(_evaluate(p) for p in m[3].split(",")) if m[3] else ()

    if name in definitions:
        if filled or params:
            raise ValueError(
                f"Modifiers and parameters on gate definitions ({name!r}) "
                "are not supported"
            )
        formals, body = definitions[name]
        if len(args) != len(formals):
            raise ValueError(
                f"{name!r} takes {len(formals)} qubits, got {len(args)}"
            )
        actual = dict(zip(formals, args, strict=True))
        group = group or (name, next(group_ids))

        def inner(ref: str) -> int:
            return resolve(actual.get(ref, ref))

        return [
            op
            for statement in _statements(body)
            for op in _call(
                statement, inner, definitions, group_ids, condition, group
            )
        ]

    if name in _ALIASES:
        extra, name = _ALIASES[name]
        filled += [True] * extra
    if name not in _STANDARD:
        raise ValueError(f"Unknown gate {name!r} in logical QASM")
    if len(args) != len(filled) + 1:
        raise ValueError(
            f"{statement!r}: expected {len(filled)} control(s) and 1 target"
        )
    lanes = [resolve(a) for a in args]
    group_name, group_id = group or (None, None)
    return [
        LogicalOp(
            name,
            (lanes[-1],),
            params,
            tuple(zip(lanes[:-1], filled, strict=True)),
            condition,
            None,
            group_name,
            group_id,
        )
    ]


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(expression: str) -> float:
    """Safely evaluate a numeric parameter such as ``0.927`` or ``pi/2``."""

    def value(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float)
        ):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in ("pi", "π"):
            return math.pi
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](
                value(node.left), value(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](value(node.operand))
        raise ValueError(f"Unsupported parameter expression: {expression!r}")

    return value(ast.parse(expression.strip(), mode="eval").body)
