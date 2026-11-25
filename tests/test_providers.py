"""Providers: choosing backends as provider:backend, Aer and IBM devices."""

import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister

import openqcv as qcv
from openqcv import qml
from openqcv.backends import PROVIDERS, Provider, get_backend
from openqcv.backends.base import OpenqcvProvider
from openqcv.export.qasm import to_program
from openqcv.representations import EncodedImage

EXACT = 1e-9
"""Exact simulators agree within 2.4e-11 (Aer's matrix product state, the
loosest, measured); this leaves room for larger circuits."""

SEED = 7
SHOTS = 4000
IMAGE = np.array([[3, 9, 1], [15, 0, 6]])

needs_aer = pytest.mark.skipif(
    not PROVIDERS.get("aer").available(),
    reason="qiskit-aer is not installed (pip install openqcv[aer])",
)
needs_ibm = pytest.mark.skipif(
    not PROVIDERS.get("ibm_fake").available(),
    reason="qiskit-ibm-runtime is not installed (pip install openqcv[ibm])",
)


def _program(gates, qubits=3):
    """A program whose qubit j ends in |1> when gates contains j."""
    circuit = QuantumCircuit(QuantumRegister(qubits, "r"))
    for qubit in gates:
        circuit.x(qubit)
    return to_program(EncodedImage(circuit, {}), "test")


def test_builtin_backends_keep_their_names():
    assert "openqcv" in qcv.list_providers()
    assert qcv.list_backends() == ["simulator", "statevector"]
    program = qcv.encode(IMAGE, "neqr", bit_depth=4)
    plain = qcv.run(program, "statevector")
    prefixed = qcv.run(program, "openqcv:statevector")
    assert plain.probabilities == prefixed.probabilities


@pytest.mark.parametrize(
    ("spec", "error", "message"),
    [
        ("nowhere:statevector", ValueError, "Unknown provider 'nowhere'"),
        ("openqcv:nothing", ValueError, "Unknown backend 'nothing'"),
    ],
)
def test_unknown_provider_or_backend(spec, error, message):
    with pytest.raises(error, match=message):
        get_backend(spec)


def test_a_registered_provider_is_found_by_name(monkeypatch):
    class Mine(Provider):
        name = "mine"

        def backends(self):
            return ["exact"]

        def backend(self, name, **options):
            return OpenqcvProvider().backend("statevector")

    monkeypatch.setattr(PROVIDERS, "_classes", dict(PROVIDERS._classes))
    PROVIDERS.register(Mine)
    assert "mine" in qcv.list_providers()
    assert qcv.list_backends("mine") == ["mine:exact"]
    program = qcv.encode(IMAGE, "neqr", bit_depth=4)
    assert qcv.run(program, "mine:exact").backend == "statevector"


@pytest.mark.parametrize("provider", ["aer", "ibm_fake"])
def test_missing_packages_are_reported(provider, monkeypatch):
    monkeypatch.setitem(sys.modules, "qiskit_aer", None)
    assert provider not in qcv.list_providers()
    with pytest.raises(ImportError, match=r"pip install openqcv\[aer\]"):
        PROVIDERS.create(provider)


@needs_aer
@pytest.mark.parametrize(
    "method", ["statevector", "density_matrix", "matrix_product_state"]
)
def test_exact_aer_methods_match_statevector(method):
    program = qcv.encode(IMAGE, "frqi")
    expected = qcv.run(program, "statevector").probabilities
    result = qcv.run(program, f"aer:{method}")
    assert result.shots is None
    assert result.backend == f"aer:{method}"
    assert set(result.probabilities) == set(expected)
    for bits, p in expected.items():
        assert result.probabilities[bits] == pytest.approx(p, abs=EXACT)


@needs_aer
def test_aer_samples_in_qubit_order():
    result = qcv.run(_program([0]), "aer:statevector", SHOTS, seed=SEED)
    assert result.shots == SHOTS
    assert result.counts == {"100": SHOTS}
    again = qcv.run(_program([0]), "aer:statevector", SHOTS, seed=SEED)
    assert again.counts == result.counts


@needs_aer
def test_aer_qir_round_trip_with_shots():
    program = qcv.encode(IMAGE, "neqr", bit_depth=4)
    result = qcv.run(program, "aer:automatic", SHOTS, seed=SEED)
    assert np.array_equal(qcv.decode(result, program), IMAGE)


@needs_aer
def test_qml_network_runs_on_aer():
    network = qml.SamplerQNN(
        circuit=QuantumCircuit(QuantumRegister(1, "q")),
        qir="qpie",
        backend="aer:statevector",
    )
    exact = qml.SamplerQNN(
        circuit=QuantumCircuit(QuantumRegister(1, "q")), qir="qpie"
    )
    images = [np.array([[1.0, 2.0]])]
    assert network.forward(images, []) == pytest.approx(
        exact.forward(images, []), abs=EXACT
    )


@needs_aer
@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: get_backend("aer:unitary"), "Unknown Aer method 'unitary'"),
        (
            lambda: qcv.run(_program([]), "aer:statevector", shots=0),
            "shots must be at least 1",
        ),
    ],
)
def test_aer_validation(make, message):
    with pytest.raises(ValueError, match=message):
        make()


@needs_aer
@needs_ibm
def test_ibm_device_runs_and_limits_qubits():
    names = qcv.list_backends("ibm_fake")
    assert "ibm_fake:fake_manila" in names
    program = qcv.encode(np.array([[1, 3], [2, 0]]), "neqr", bit_depth=2)
    result = qcv.run(program, "ibm_fake:fake_manila", SHOTS, seed=SEED)
    assert result.backend == "ibm_fake:fake_manila"
    assert np.array_equal(qcv.decode(result, program), [[1, 3], [2, 0]])
    with pytest.raises(ValueError, match="exceeds ibm_fake:fake_manila's"):
        qcv.run(_program([], qubits=6), "ibm_fake:fake_manila")
    with pytest.raises(ValueError, match="Unknown device 'fake_nowhere'"):
        get_backend("ibm_fake:fake_nowhere")
