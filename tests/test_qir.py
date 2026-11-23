"""The API for user-defined QIRs: builder, registration, check, plugins."""

import sys
import types
from importlib.metadata import EntryPoint

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import CXGate, StatePreparation

import openqcv as qcv
from openqcv import qir
from openqcv.core import registry
from openqcv.core.registry import Registry
from openqcv.export import logical
from openqcv.representations import REPRESENTATIONS

GRAY_IMAGE = np.array([[3, 9, 1], [15, 0, 6]])


class GrayCode(qir.QIR):
    """The example in the openqcv.qir module docstring."""

    name = "graycode"

    def encode(self, image):
        """Write each pixel's Gray code under its row and column."""
        img = np.asarray(image, dtype=int)
        h, w = img.shape
        c = qir.Circuit(
            row=qir.index_qubits(h), col=qir.index_qubits(w), value=4
        )
        c.h("row", "col")
        for (y, x), v in np.ndenumerate(img):
            with c.block(f"pixel_{y}_{x}"):
                gray = v ^ (v >> 1)
                c.x(c.bits("value", gray), when={"row": y, "col": x})
        return c.finish(shape=[h, w])

    @classmethod
    def decode(cls, result, program):
        """Read each pixel's Gray code and convert it back."""
        h, w = program.meta["shape"]
        image = np.zeros((h, w))
        joint = qir.marginal(result, program, "row", "col", "value")
        for (y, x, gray), p in joint.items():
            if y < h and x < w and p > 0:
                value = gray
                while gray := gray >> 1:
                    value ^= gray
                image[y, x] = value
        return image


class Wrong(qir.QIR):
    """Decodes every pixel as 0."""

    name = "wrong_for_tests"

    def encode(self, image):
        """Encode nothing about the image."""
        c = qir.Circuit(position=1)
        c.h("position")
        return c.finish(shape=list(np.shape(image)))

    @classmethod
    def decode(cls, result, program):
        """Return zeros."""
        return np.zeros(program.meta["shape"])

    def qubits_required(self, shape):
        """Predict one qubit too many."""
        return 2


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    """Register the test QIRs for one test only."""
    monkeypatch.setattr(
        REPRESENTATIONS, "_classes", dict(REPRESENTATIONS._classes)
    )
    qir.register(GrayCode)
    qir.register(Wrong)


# -------------------------------------------------------------- user QIRs


def test_user_qir_works_like_a_builtin():
    qir.check(GrayCode(), GRAY_IMAGE, shots=2000, shot_tolerance=0)
    assert "graycode" in qcv.list_representations()
    program = qcv.encode(GRAY_IMAGE, "graycode")
    assert program.registers == {
        "row": [0],
        "col": [1, 2],
        "value": [3, 4, 5, 6],
    }
    assert qcv.draw(program)._repr_png_().startswith(b"\x89PNG")


def test_check_reports_failures():
    with pytest.raises(AssertionError, match="off by up to 15"):
        qir.check(Wrong(), GRAY_IMAGE)
    with pytest.raises(AssertionError, match="returned shape \\(2, 3\\)"):
        qir.check(Wrong(), GRAY_IMAGE, expected=np.zeros((3, 2)))
    with pytest.raises(
        AssertionError, match="predicts 2 qubits; the program has 1"
    ):
        qir.check(Wrong(), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="given together"):
        qir.check(GrayCode(), GRAY_IMAGE, shots=10)


def test_check_needs_registration():
    class Unregistered(GrayCode):
        name = "unregistered"

    with pytest.raises(ValueError, match="not registered under"):
        qir.check(Unregistered(), GRAY_IMAGE)


def test_decode_must_be_a_classmethod():
    class Plain(qir.QIR):
        name = "plain"

        def encode(self, image):
            raise NotImplementedError

        def decode(self, result, program):
            raise NotImplementedError

    with pytest.raises(TypeError, match="must be a classmethod"):
        qir.register(Plain)


def test_qubits_required_is_optional():
    with pytest.raises(NotImplementedError, match="does not predict"):
        GrayCode().qubits_required((2, 2))


def test_encode_must_return_an_encoded_image():
    class Raw(GrayCode):
        name = "raw_for_tests"

        def encode(self, image):
            return QuantumCircuit(1)

    with pytest.raises(TypeError, match="must return an EncodedImage"):
        qcv.encode(GRAY_IMAGE, Raw())


# ------------------------------------------------------------------ builder


def test_block_acts_only_on_the_qubits_it_uses():
    c = qir.Circuit(a=2, b=3)
    with c.block("step"):
        c.x(c["b"][1], when={"a": 2})
    with c.block("nothing"):
        pass
    circuit = c.finish().circuit
    assert len(circuit.data) == 1
    instruction = circuit.data[0]
    assert instruction.operation.name == "step"
    assert [circuit.find_bit(q).index for q in instruction.qubits] == [0, 1, 3]
    assert "negctrl @ ctrl @ x q0, q1, q2;" in logical.dumps(circuit)


def test_conditions_on_single_qubits_and_bits():
    c = qir.Circuit(a=3, t=1)
    assert c.bits("a", 5) == [c["a"][0], c["a"][2]]
    c.ry(0.5, "t", when={c["a"][2]: 1, c["a"][0]: 0})
    operation = c.finish().circuit.data[0].operation
    assert operation.ctrl_state == 0b01
    assert operation.base_gate.name == "ry"


def test_builder_validation():
    with pytest.raises(ValueError, match="at least one register"):
        qir.Circuit()
    with pytest.raises(ValueError, match="'1a' is not an identifier"):
        qir.Circuit(**{"1a": 1})
    with pytest.raises(TypeError, match="size must be an int, got str"):
        qir.Circuit(a="2")
    with pytest.raises(TypeError, match="size must be an int, got bool"):
        qir.Circuit(a=True)
    with pytest.raises(ValueError, match="needs at least 1 qubit"):
        qir.Circuit(a=0)

    c = qir.Circuit(a=2, t=1)
    with pytest.raises(ValueError, match="No register 'b'; the registers"):
        c.x("b")
    with pytest.raises(ValueError, match="2 qubit\\(s\\); 4 does not fit"):
        c.bits("a", 4)
    with pytest.raises(ValueError, match="2 qubit\\(s\\); 7 does not fit"):
        c.x("t", when={"a": 7})
    with pytest.raises(TypeError, match="must be ints, got bool"):
        c.x("t", when={"a": True})
    with pytest.raises(ValueError, match="one-qubit gates"):
        c.apply(CXGate(), "t")
    with pytest.raises(ValueError, match="both target and control"):
        c.x("a", when={"a": 1})
    with pytest.raises(ValueError, match="appears twice"):
        c.x("t", when={"a": 1, c["a"][0]: 1})
    with pytest.raises(ValueError, match="not a qubit of this circuit"):
        c.x(Qubit())
    with pytest.raises(ValueError, match="'2x' is not an identifier"):
        with c.block("2x"):
            pass
    with c.block("once"):
        c.x("t")
    with pytest.raises(ValueError, match="'once' already exists"):
        with c.block("once"):
            pass
    with pytest.raises(ValueError, match="cannot be nested"):
        with c.block("outer"), c.block("inner"):
            pass
    with c.block("open"), pytest.raises(ValueError, match="inside a block"):
        c.finish()


def test_failed_block_is_discarded():
    c = qir.Circuit(t=1)
    with pytest.raises(RuntimeError), c.block("broken"):
        c.x("t")
        raise RuntimeError
    with c.block("broken"):  # the name was never used
        c.x("t")
    assert [i.operation.name for i in c.finish().circuit.data] == ["broken"]


# ------------------------------------------------- encoded image and export


def test_encoded_image_validation():
    with pytest.raises(TypeError, match="must be a QuantumCircuit, got str"):
        qir.EncodedImage("circuit", {})
    with pytest.raises(TypeError, match="meta must be a dict, got list"):
        qir.EncodedImage(QuantumCircuit(QuantumRegister(1, "a")), [])
    measured = QuantumCircuit(QuantumRegister(1, "a"), ClassicalRegister(1))
    with pytest.raises(ValueError, match="no classical bits"):
        qir.EncodedImage(measured, {})
    with pytest.raises(ValueError, match="exactly one named register"):
        qir.EncodedImage(QuantumCircuit([Qubit()]), {})
    with pytest.raises(ValueError, match="'my-reg' is not an identifier"):
        qir.EncodedImage(QuantumCircuit(QuantumRegister(1, "my-reg")), {})


@pytest.mark.parametrize(
    ("circuit", "meta", "message"),
    [
        (QuantumCircuit(QuantumRegister(1, "meas")), {}, "'meas' is reserved"),
        (QuantumCircuit(QuantumRegister(1, "a")), {"x": object()}, "JSON"),
        (QuantumCircuit(QuantumRegister(1, "a")), {"x": np.nan}, "finite"),
    ],
)
def test_export_validation(circuit, meta, message):
    with pytest.raises(ValueError, match=message):
        qcv.export.to_program(qir.EncodedImage(circuit, meta), "a")


def test_opaque_gates_are_rejected():
    circuit = QuantumCircuit(QuantumRegister(1, "a"))
    circuit.append(StatePreparation([0.6, 0.8]), [0])
    with pytest.raises(ValueError, match="no textbook form"):
        qcv.export.to_program(qir.EncodedImage(circuit, {}), "a")


# ------------------------------------------------------------ entry points


class _Base:
    pass


def _installed(monkeypatch, *entries):
    """Make ``entries`` look like installed entry points."""
    module = types.ModuleType("fake_plugin")
    for name in ("Good", "Misnamed"):
        cls = type(name, (_Base,), {"name": name.lower()})
        setattr(module, name, cls)
    monkeypatch.setitem(sys.modules, "fake_plugin", module)
    found = [EntryPoint(n, v, "test.group") for n, v in entries]
    monkeypatch.setattr(registry, "entry_points", lambda group: found)
    return module


def test_installed_classes_load_on_first_use(monkeypatch):
    module = _installed(monkeypatch, ("good", "fake_plugin:Good"))
    things = Registry("thing", _Base, entry_point_group="test.group")
    assert things.names() == ["good"]
    assert things.get("Good") is module.Good
    assert things.get("good") is module.Good  # registered once


def test_entry_point_name_must_match(monkeypatch):
    _installed(monkeypatch, ("other", "fake_plugin:Misnamed"))
    things = Registry("thing", _Base, entry_point_group="test.group")
    with pytest.raises(ValueError, match="named 'misnamed'; the names must"):
        things.get("other")
