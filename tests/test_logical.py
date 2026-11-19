"""Logical OpenQASM 3: writing, reading and their agreement."""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate

import openqcv as qcv
from openqcv.core import Block
from openqcv.export import logical

IMAGE = np.array(
    [[0, 32, 64], [96, 128, 160], [192, 224, 255]], dtype=np.uint8
)


def test_open_controls_are_written_as_negctrl():
    # Pixel 1 of a 3x3 image is position 0001: position[0] filled, rest open.
    lines = qcv.encode(IMAGE, "neqr").logical_qasm.splitlines()
    first = next(line for line in lines if "@ x" in line)
    assert first.strip().startswith("ctrl @ negctrl @ negctrl @ negctrl @ x")


def test_dumps_then_parse_keeps_controls_angles_and_groups():
    a, b = QuantumRegister(2, "a"), QuantumRegister(1, "b")
    body = QuantumCircuit(3)
    body.append(
        RYGate(0.5).control(2, ctrl_state=0b10, annotated=False), [0, 1, 2]
    )
    circuit = QuantumCircuit(a, b)
    circuit.h(a[0])
    circuit.append(Block("step", body), [*a, b[0]])

    parsed = logical.parse(logical.dumps(circuit))
    assert parsed.qregs == (("a", 2), ("b", 1))
    h, ry = parsed.ops
    assert (h.name, h.targets, h.group) == ("h", (0,), None)
    assert ry.name == "ry" and math.isclose(ry.params[0], 0.5)
    # ctrl_state bit 0 -> control 0, so 0b10 is (open, filled).
    assert ry.controls == ((0, False), (1, True))
    assert ry.targets == (2,) and ry.group == "step"


def test_parser_reads_hand_written_qasm():
    circuit = logical.parse(
        "OPENQASM 3.0; qubit[2] q; bit[1] m;"
        "h q[0]; cx q[0], q[1]; m[0] = measure q[0];"
        "if (m[0] == 1) ry(pi/2) q[1];"
    )
    h, cx, measure, conditional = circuit.ops
    assert (cx.name, cx.targets, cx.controls) == ("x", (1,), ((0, True),))
    assert (measure.name, measure.clbit) == ("measure", 0)
    assert conditional.condition == (0, 1)
    assert math.isclose(conditional.params[0], math.pi / 2)


def test_each_call_of_a_definition_is_its_own_group():
    circuit = logical.parse(
        "qubit[2] q; gate pair a, b { h a; cx a, b; }"
        "pair q[0], q[1]; pair q[1], q[0];"
    )
    ids = [op.group_id for op in circuit.ops]
    assert ids[0] == ids[1] != ids[2] == ids[3]
    second_cx = circuit.ops[3]
    assert (second_cx.controls, second_cx.targets) == (((1, True),), (0,))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("qubit[1] q; h r[0];", "Unknown qubit 'r\\[0\\]'"),
        ("qubit[1] q; foo q[0];", "Unknown gate 'foo'"),
        ("qubit[2] q; ctrl @ x q[0];", "expected 1 control"),
        ("qubit[1] q; ry(sqrt(2)) q[0];", "Unsupported parameter"),
        ("qubit[1] q; bit[1] c; c[1] = measure q[0];", "Unknown bit"),
    ],
)
def test_parser_errors_are_clear(source, message):
    with pytest.raises(ValueError, match=message):
        logical.parse(source)


def test_dumps_rejects_non_textbook_gates():
    from qiskit.circuit.library import StatePreparation

    circuit = QuantumCircuit(1)
    circuit.append(StatePreparation([0.6, 0.8]), [0])
    with pytest.raises(ValueError, match="no textbook form"):
        logical.dumps(circuit)
