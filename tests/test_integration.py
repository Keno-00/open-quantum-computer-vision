"""End-to-end paths through several parts of openqcv at once.

The other test files check each part; these check that the parts work
together: QIRs through JSON and every exact or sampling backend, training
on interchangeable backends, fast export end to end, kernels through
providers, a provider installed as a plugin, and a backend in a worker
process.
"""

import sys
import types
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import EntryPoint

import numpy as np
import pytest
from qiskit.circuit.library import real_amplitudes, z_feature_map

import openqcv as qcv
from openqcv import qml
from openqcv.backends import PROVIDERS, Provider, StatevectorBackend
from openqcv.core import registry
from openqcv.qml.template import Template

EXACT = 1e-9
"""Exact backends agree within 2.4e-11 (measured, Aer's matrix product
state the loosest); decoded images within 1.1e-13 of their range."""

TRAINED = 1e-6
"""Weights trained on two exact backends agree to this; their outcome
probabilities differ by ~1e-16, which COBYLA's steps amplify to 3.2e-12
over the 43 evaluations used here (measured)."""

SEED = 7
SHOTS = 8192

IMAGES = {
    "frqi": np.array([[3.0, 9.0, 1.0], [15.0, 0.0, 6.0]]),
    "neqr": np.array([[3, 9, 1], [15, 0, 6]]),
    "qpie": np.array([[3.0, 9.0, 1.0], [15.0, 0.5, 6.0]]),
}
"""Padded, asymmetric images (2x3), which expose bit-order errors."""

EXACT_BACKENDS = [
    "statevector",
    "openqcv:statevector",
    "aer:statevector",
    "aer:density_matrix",
    "aer:matrix_product_state",
]

has_aer = PROVIDERS.get("aer").available()
has_ibm = PROVIDERS.get("ibm_fake").available()
needs_aer = pytest.mark.skipif(
    not has_aer, reason="qiskit-aer is not installed (openqcv[aer])"
)


def _needs(spec):
    provider = spec.partition(":")[0] if ":" in spec else "openqcv"
    available = {"openqcv": True, "aer": has_aer, "ibm_fake": has_ibm}
    return pytest.param(
        spec,
        marks=pytest.mark.skipif(
            not available[provider],
            reason=f"provider {provider!r} is not installed",
        ),
    )


def _digit_images():
    """Two classes of 2x2 images: bright left column or right column."""
    rng = np.random.default_rng(SEED)
    images, labels = [], []
    for _ in range(4):
        bright, dark = rng.uniform(150, 255, 2), rng.uniform(1, 60, 2)
        images += [np.column_stack([bright, dark])]
        images += [np.column_stack([dark, bright])]
        labels += [0, 1]
    return images, np.array(labels)


# ------------------------------------------------- QIRs through providers


@pytest.mark.parametrize("backend", [_needs(b) for b in EXACT_BACKENDS])
@pytest.mark.parametrize("name", sorted(IMAGES))
def test_qir_round_trip_through_json_on_every_exact_backend(name, backend):
    image = IMAGES[name]
    received = qcv.QasmProgram.from_json(qcv.encode(image, name).to_json())
    result = qcv.run(received, backend)
    assert result.shots is None
    decoded = qcv.decode(result, received)
    scale = max(1.0, float(np.abs(image).max()))
    assert np.abs(decoded - image).max() <= EXACT * scale


@pytest.mark.parametrize(
    "backend",
    ["simulator", _needs("aer:statevector"), _needs("aer:automatic")],
)
def test_sampled_neqr_decodes_exactly(backend):
    # NEQR stores bits, so any shot count recovers the image exactly.
    image = IMAGES["neqr"]
    program = qcv.encode(image, "neqr")
    result = qcv.run(program, backend, SHOTS, seed=SEED)
    assert result.shots == SHOTS
    assert np.array_equal(qcv.decode(result, program), image)


@pytest.mark.skipif(
    not (has_aer and has_ibm),
    reason="qiskit-aer and qiskit-ibm-runtime are needed",
)
def test_device_simulation_through_json():
    image = np.array([[1, 3], [2, 0]])
    program = qcv.encode(image, "neqr", bit_depth=2)
    received = qcv.QasmProgram.from_json(program.to_json())
    result = qcv.run(received, "ibm_fake:fake_manila", SHOTS, seed=SEED)
    assert np.array_equal(qcv.decode(result, received), image)


# -------------------------------------------------------- QML end to end


def _vqc(**options):
    return qml.VQC(
        feature_map="qpie",
        ansatz=real_amplitudes(2, reps=1),
        optimizer="COBYLA",
        initial_point=np.full(4, 0.1),
        **options,
    )


@pytest.mark.parametrize("backend", [_needs("aer:statevector")])
def test_vqc_trains_the_same_on_interchangeable_exact_backends(backend):
    images, labels = _digit_images()
    reference = _vqc().fit(images, labels)
    other = _vqc(backend=backend).fit(images, labels)
    assert np.abs(other.weights - reference.weights).max() <= TRAINED
    assert np.array_equal(other.predict(images), reference.predict(images))
    assert other.neural_network.forward(images[:1], other.weights) == (
        pytest.approx(
            reference.neural_network.forward(images[:1], other.weights),
            abs=EXACT,
        )
    )


@pytest.mark.parametrize("backend", ["statevector", _needs("aer:statevector")])
def test_fast_export_trains_identically(backend):
    images, labels = _digit_images()
    plain = _vqc(backend=backend).fit(images, labels)
    fast = _vqc(backend=backend, fast_export=True).fit(images, labels)
    assert np.array_equal(fast.weights, plain.weights)
    assert np.array_equal(fast.predict(images), plain.predict(images))


def test_vqr_with_classical_features_and_fast_export():
    features = np.linspace(0.1, 1.4, 6)[:, None]
    targets = np.cos(2 * features[:, 0])
    models = [
        qml.VQR(
            feature_map=z_feature_map(1, reps=1),
            ansatz=real_amplitudes(1, reps=1),
            initial_point=[0.0, 0.0],
            fast_export=flag,
        ).fit(features, targets)
        for flag in (False, True)
    ]
    assert np.array_equal(models[0].weights, models[1].weights)
    assert np.array_equal(
        models[0].predict(features), models[1].predict(features)
    )


def test_qml_programs_survive_json_and_run_anywhere():
    images, _ = _digit_images()
    network = qml.SamplerQNN(circuit=real_amplitudes(2, reps=1), qir="qpie")
    template, _ = network._templates_for(network._samples(images[:1])[0][0])[0]
    program = template.program(np.linspace(-1, 1, template.num_angles))
    received = qcv.QasmProgram.from_json(program.to_json())
    expected = qcv.run(program, "statevector").probabilities
    for backend in ["simulator", *([("aer:statevector")] if has_aer else [])]:
        shots = SHOTS if backend == "simulator" else None
        result = qcv.run(received, backend, shots, seed=SEED)
        tolerance = 5 * 0.5 / np.sqrt(SHOTS) if shots else EXACT
        for bits, p in expected.items():
            assert result.probabilities.get(bits, 0.0) == pytest.approx(
                p, abs=tolerance
            )


@needs_aer
def test_qsvc_on_images_through_a_provider():
    pytest.importorskip("sklearn")
    images, labels = _digit_images()
    exact = qml.FidelityQuantumKernel(feature_map="qpie")
    aer = qml.FidelityQuantumKernel(
        feature_map="qpie", backend="aer:statevector", fast_export=True
    )
    assert np.abs(aer.evaluate(images) - exact.evaluate(images)).max() <= (
        EXACT
    )
    qsvc = qml.QSVC(aer).fit(images, labels)
    assert qsvc.score(images, labels) == 1.0


# ------------------------------------------------------ plugins, processes


class _PluginProvider(Provider):
    """A provider as an installed package would declare it."""

    name = "plugin"

    def backends(self):
        return ["exact"]

    def backend(self, name, **options):
        if name != "exact":
            raise ValueError(f"Unknown backend {name!r}")
        return StatevectorBackend()


def test_an_installed_provider_is_discovered_and_used(monkeypatch):
    module = types.ModuleType("fake_provider_plugin")
    module.PluginProvider = _PluginProvider
    monkeypatch.setitem(sys.modules, "fake_provider_plugin", module)
    found = [
        EntryPoint(
            "plugin",
            "fake_provider_plugin:PluginProvider",
            "openqcv.providers",
        )
    ]
    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda group: found if group == "openqcv.providers" else [],
    )
    monkeypatch.setattr(PROVIDERS, "_classes", dict(PROVIDERS._classes))
    assert "plugin" in qcv.list_providers()
    assert qcv.list_backends("plugin") == ["plugin:exact"]
    image = IMAGES["neqr"]
    program = qcv.encode(image, "neqr")
    assert np.array_equal(
        qcv.decode(qcv.run(program, "plugin:exact"), program), image
    )


def _run_in_worker(backend):
    program = qcv.encode(IMAGES["neqr"], "neqr")
    return qcv.decode(qcv.run(program, backend), program).tolist()


@pytest.mark.parametrize("backend", ["statevector", _needs("aer:statevector")])
def test_backends_work_in_a_worker_process(backend):
    with ProcessPoolExecutor(max_workers=1) as pool:
        decoded = pool.submit(_run_in_worker, backend).result()
    assert decoded == IMAGES["neqr"].tolist()


def test_template_fast_export_matches_for_every_builtin_qir():
    for name, image in IMAGES.items():
        circuit = qcv.pipeline.encode_circuit(image, name).circuit.copy()
        n = circuit.num_qubits
        circuit.compose(
            real_amplitudes(n, reps=1), qubits=range(n), inplace=True
        )
        slow, fast = Template(circuit), Template(circuit, fast_export=True)
        angles = np.random.default_rng(SEED).uniform(-7, 7, slow.num_angles)
        assert fast.program(angles) == slow.program(angles), name
