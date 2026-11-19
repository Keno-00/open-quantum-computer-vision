"""Core types, registry, preprocessing and input validation."""

import numpy as np
import pytest

import openqcv as qcv
from openqcv.backends import BACKENDS, Backend, SimulatorBackend
from openqcv.image import as_grayscale, position_qubits
from openqcv.representations import FRQI, NEQR, QPIE, REPRESENTATIONS

IMAGE = np.array([[0, 64], [128, 255]], dtype=np.uint8)


# ------------------------------------------------------------------ program


def test_value_reads_registers_least_significant_bit_first():
    program = qcv.QasmProgram(
        "x", "", "", "", {"position": [0, 1], "color": [2]}
    )
    assert program.value("100", "position") == 1
    assert program.value("010", "position") == 2
    assert program.value("001", "color") == 1
    assert program.num_qubits == 3


def test_qasm_version_selection():
    program = qcv.encode(IMAGE, "qpie")
    assert program.qasm(2) is program.qasm2 and program.qasm() is program.qasm3
    with pytest.raises(ValueError):
        program.qasm(4)


def test_from_counts():
    result = qcv.ExecutionResult.from_counts("b", {"0": 3, "1": 1, "x": 0})
    assert result.shots == 4 and result.probabilities == {"0": 0.75, "1": 0.25}
    with pytest.raises(ValueError, match="at least one shot"):
        qcv.ExecutionResult.from_counts("b", {})
    with pytest.raises(ValueError, match="negative"):
        qcv.ExecutionResult.from_counts("b", {"0": -1, "1": 2})


# ----------------------------------------------------------------- registry


def test_registry_accepts_name_class_and_instance():
    assert isinstance(REPRESENTATIONS.create("NEQR", bit_depth=4), NEQR)
    assert isinstance(REPRESENTATIONS.create(FRQI), FRQI)
    instance = QPIE()
    assert REPRESENTATIONS.create(instance) is instance


def test_registry_errors():
    with pytest.raises(ValueError, match="Available: frqi, neqr, qpie"):
        REPRESENTATIONS.create("nope")
    with pytest.raises(TypeError, match="existing representation instance"):
        REPRESENTATIONS.create(QPIE(), bit_depth=3)
    with pytest.raises(TypeError, match="got int"):
        REPRESENTATIONS.create(42)
    with pytest.raises(TypeError, match="subclasses of Backend"):
        BACKENDS.register(QPIE)
    with pytest.raises(ValueError, match="already registered"):

        @BACKENDS.register
        class Duplicate(Backend):
            name = "simulator"

            def run(self, program, shots=None):
                raise NotImplementedError


# ---------------------------------------------------------------- backends


def test_shot_validation():
    program = qcv.encode(IMAGE, "neqr")
    with pytest.raises(ValueError, match="at least 1"):
        qcv.run(program, SimulatorBackend(), shots=0)
    with pytest.raises(ValueError, match="takes no shots"):
        qcv.run(program, "statevector", shots=100)


def test_simulator_is_reproducible_with_a_seed():
    program = qcv.encode(IMAGE, "frqi")
    first = qcv.run(program, "simulator", shots=500, seed=3)
    second = qcv.run(program, "simulator", shots=500, seed=3)
    assert first.counts == second.counts


# ---------------------------------------------------------- preprocessing


def test_as_grayscale_validation():
    assert as_grayscale(np.ones((2, 2, 1))).shape == (2, 2)
    assert as_grayscale(np.ones((2, 2, 4))).shape == (2, 2)
    for bad, error in [
        (np.array([]), ValueError),
        (np.ones(3), ValueError),
        (np.ones((2, 2, 2)), ValueError),
        (np.array([[np.nan, 1.0]]), ValueError),
        (np.array([["a", "b"]]), TypeError),
        (np.array([[1j, 1.0]]), TypeError),
    ]:
        with pytest.raises(error):
            as_grayscale(bad)


@pytest.mark.parametrize(
    ("pixels", "qubits"), [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (64, 6)]
)
def test_position_qubits(pixels, qubits):
    assert position_qubits(pixels) == qubits


# ------------------------------------------------------- representation input


def test_representation_input_validation():
    with pytest.raises(ValueError, match="non-negative"):
        qcv.encode(-IMAGE.astype(float), "qpie")
    with pytest.raises(ValueError, match="all-zero"):
        qcv.encode(np.zeros((2, 2)), "qpie")
    with pytest.raises(ValueError, match="integer intensities"):
        qcv.encode(IMAGE / 2.5, "neqr")
    with pytest.raises(ValueError, match="0..15"):
        qcv.encode(IMAGE, "neqr", bit_depth=4)
    with pytest.raises(ValueError, match="bit_depth"):
        NEQR(bit_depth=0)
    with pytest.raises(ValueError, match="low < high"):
        FRQI(value_range=(5, 5))
    with pytest.raises(ValueError, match="outside value_range"):
        qcv.encode(IMAGE, "frqi", value_range=(0, 100))


def test_decode_needs_only_the_program():
    # A program decoded with no access to the encoding options (bit_depth=4).
    small = np.array([[0, 5], [9, 15]])
    program = qcv.encode(small, "neqr", bit_depth=4)
    received = qcv.QasmProgram.from_json(program.to_json())
    np.testing.assert_array_equal(
        qcv.decode(qcv.run(received), received), small
    )
