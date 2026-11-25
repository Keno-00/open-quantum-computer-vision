"""Quantum machine learning: networks, gradients, models and kernels."""

import functools
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    real_amplitudes,
    z_feature_map,
    zz_feature_map,
)
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from scipy.optimize import minimize

from openqcv import qml
from openqcv.backends import SimulatorBackend
from openqcv.pipeline import encode_circuit
from openqcv.qml import template as template_module
from openqcv.qml.losses import PROBABILITY_FLOOR, get_loss
from openqcv.qml.template import Template
from openqcv.representations import REPRESENTATIONS

EXACT = 1e-12
"""Exact simulation agrees with Qiskit's statevector within 1.1e-15
(measured); this leaves room for larger circuits."""

STEP = 1e-6
FINITE_DIFFERENCE = 1e-8
"""Central differences with step STEP err by about STEP**2 (truncation)
plus 1e-16 / STEP (rounding); measured errors are below 2.5e-10."""

SHOTS = 20_000
SAMPLING = 5 * 0.5 / np.sqrt(SHOTS)
"""Five standard deviations of a probability estimated from SHOTS shots;
the deviation is at most 0.5 / sqrt(SHOTS)."""

FTOL = 1e-6
"""SciPy's SLSQP stops when the objective changes by less than this."""

SEED = 7

# Two clusters that a one-layer VQC separates exactly.
FEATURES = np.array(
    [[0.1, 0.2], [0.2, 0.0], [0.0, 0.1], [1.5, 1.4], [1.4, 1.6], [1.6, 1.5]]
)
LABELS = np.array(["a", "a", "a", "b", "b", "b"])


def _images():
    """2x2 images with a bright left column or a bright right column."""
    rng = np.random.default_rng(SEED)
    images, labels = [], []
    for _ in range(4):
        bright, dark = rng.uniform(150, 255, 2), rng.uniform(0, 60, 2)
        images += [np.column_stack([bright, dark])]
        images += [np.column_stack([dark, bright])]
        labels += ["left", "right"]
    return images, np.array(labels)


def _network(kind=qml.SamplerQNN, **options):
    feature_map = zz_feature_map(2)
    ansatz = real_amplitudes(2, reps=1)
    circuit = QuantumCircuit(2)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    network = kind(
        circuit=circuit,
        input_params=list(feature_map.parameters),
        weight_params=list(ansatz.parameters),
        **options,
    )
    return network, circuit, [*feature_map.parameters, *ansatz.parameters]


def _states(circuit, parameters, features, weights):
    values = [
        dict(zip(parameters, [*x, *weights], strict=True)) for x in features
    ]
    return [Statevector(circuit.assign_parameters(v)) for v in values]


def _finite_difference(network, features, weights):
    steps = np.eye(len(weights)) * STEP
    return np.stack(
        [
            network.forward(features, weights + s)
            - network.forward(features, weights - s)
            for s in steps
        ],
        axis=-1,
    ) / (2 * STEP)


# ------------------------------------------------------------------ networks


def test_sampler_qnn_returns_outcome_probabilities():
    network, circuit, parameters = _network()
    rng = np.random.default_rng(SEED)
    features, weights = rng.random((3, 2)), rng.random(network.num_weights)
    expected = [
        s.probabilities()
        for s in _states(circuit, parameters, features, weights)
    ]
    output = network.forward(features, weights)
    assert network.output_shape == (4,)
    assert np.abs(output - expected).max() <= EXACT


def test_sampler_qnn_interpret_sums_outcomes():
    network, circuit, parameters = _network(
        interpret=lambda x: bin(x).count("1") % 2, output_shape=2
    )
    rng = np.random.default_rng(SEED)
    features, weights = rng.random((2, 2)), rng.random(network.num_weights)
    output = network.forward(features, weights)
    for row, state in zip(
        output, _states(circuit, parameters, features, weights), strict=True
    ):
        p = state.probabilities()  # index 2**j is qubit j
        assert row == pytest.approx([p[0] + p[3], p[1] + p[2]], abs=EXACT)


def test_parameter_shift_gradient_is_exact():
    network, _, _ = _network()
    rng = np.random.default_rng(SEED)
    features, weights = rng.random((3, 2)), rng.random(network.num_weights)
    input_gradient, gradient = network.backward(features, weights)
    assert input_gradient is None
    assert gradient.shape == (3, 4, network.num_weights)
    expected = _finite_difference(network, features, weights)
    assert np.abs(gradient - expected).max() <= FINITE_DIFFERENCE


def test_gradient_of_a_shared_and_scaled_parameter():
    a, b = Parameter("a"), Parameter("b")
    circuit = QuantumCircuit(2)
    circuit.rx(2 * a, 0)
    circuit.cx(0, 1)
    circuit.ry(a, 1)
    circuit.rz(a + b, 0)
    circuit.h(0)
    network = qml.SamplerQNN(circuit=circuit)
    weights = np.array([0.3, -1.1])
    _, gradient = network.backward(None, weights)
    expected = _finite_difference(network, None, weights)
    assert np.abs(gradient - expected).max() <= FINITE_DIFFERENCE


def test_estimator_qnn_measures_pauli_observables():
    observables = [
        SparsePauliOp(["XY", "ZI", "II"], [0.5, -1.0, 0.25]),
        Pauli("YZ"),
    ]
    network, circuit, parameters = _network(
        qml.EstimatorQNN, observables=observables
    )
    rng = np.random.default_rng(SEED)
    features, weights = rng.random((3, 2)), rng.random(network.num_weights)
    expected = [
        [s.expectation_value(SparsePauliOp(o)).real for o in observables]
        for s in _states(circuit, parameters, features, weights)
    ]
    assert np.abs(network.forward(features, weights) - expected).max() <= (
        EXACT
    )
    _, gradient = network.backward(features, weights)
    assert np.abs(
        gradient - _finite_difference(network, features, weights)
    ).max() <= (FINITE_DIFFERENCE)


def test_estimator_qnn_default_observable_is_z_on_every_qubit():
    network, _, _ = _network(qml.EstimatorQNN)
    assert network.observables == [SparsePauliOp("ZZ")]


def test_network_on_qir_encoded_images():
    images, _ = _images()
    ansatz = real_amplitudes(3, reps=1)
    network = qml.SamplerQNN(circuit=ansatz, qir="frqi")
    weights = np.random.default_rng(SEED).random(network.num_weights)
    frqi = REPRESENTATIONS.create("frqi")
    for image, row in zip(
        images[:2], network.forward(images[:2], weights), strict=True
    ):
        circuit = encode_circuit(image, frqi).circuit.copy()
        circuit.compose(ansatz.assign_parameters(weights), inplace=True)
        assert np.abs(row - Statevector(circuit).probabilities()).max() <= (
            EXACT
        )


def test_network_samples_on_the_simulator():
    exact, _, _ = _network()
    sampled, _, _ = _network(backend=SimulatorBackend(seed=SEED), shots=SHOTS)
    rng = np.random.default_rng(SEED)
    features, weights = rng.random((2, 2)), rng.random(exact.num_weights)
    error = np.abs(
        sampled.forward(features, weights) - exact.forward(features, weights)
    ).max()
    assert error <= SAMPLING


def test_network_without_inputs_takes_none():
    network = qml.SamplerQNN(circuit=real_amplitudes(1, reps=0))
    output = network.forward(None, [np.pi / 2])
    assert np.abs(output - [[0.5, 0.5]]).max() <= EXACT


# -------------------------------------------------------------------- losses


@pytest.mark.parametrize("name", ["absolute_error", "squared_error"])
def test_distance_losses(name):
    loss = get_loss(name)
    predict = np.array([[0.2, -0.5], [1.0, 3.0]])
    target = np.array([[0.0, 0.5], [1.0, 1.0]])
    power = 1 if name == "absolute_error" else 2
    expected = (np.abs(predict - target) ** power).sum(axis=1)
    assert np.array_equal(loss(predict, target), expected)


def test_loss_gradients_are_derivatives():
    predict = np.array([[0.2, 0.3, 0.5], [0.7, 0.1, 0.2]])
    target = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    for loss in (qml.L2Loss(), qml.CrossEntropyLoss(), qml.L1Loss()):
        gradient = loss.gradient(predict, target)
        for j in range(predict.shape[1]):
            step = np.zeros_like(predict)
            step[:, j] = STEP
            expected = (
                loss(predict + step, target) - loss(predict - step, target)
            ) / (2 * STEP)
            assert gradient[:, j] == pytest.approx(
                expected, abs=FINITE_DIFFERENCE
            )


def test_cross_entropy_is_natural_log_with_a_floor():
    loss = qml.CrossEntropyLoss()
    predict = np.array([[0.25, 0.75], [0.0, 1.0]])
    target = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert loss(predict, target) == pytest.approx(
        [-np.log(0.75), -np.log(PROBABILITY_FLOOR)]
    )
    assert loss.gradient(predict, target)[1].tolist() == [0.0, 0.0]


# -------------------------------------------------------------------- models


def test_vqc_separates_classical_features():
    vqc = qml.VQC(feature_map=z_feature_map(2, reps=1), initial_point=[0] * 8)
    vqc.fit(FEATURES, LABELS)
    assert vqc.predict(FEATURES).tolist() == LABELS.tolist()
    assert vqc.score(FEATURES, LABELS) == 1.0
    probabilities = vqc.predict_proba(FEATURES)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(6), abs=EXACT)


def test_vqc_on_qir_encoded_images():
    images, labels = _images()
    vqc = qml.VQC(
        feature_map="qpie", optimizer="COBYLA", initial_point=[0] * 8
    )
    assert vqc.num_qubits is None
    vqc.fit(images, labels)
    assert vqc.num_qubits == 2
    assert vqc.score(images, labels) == 1.0


def test_vqc_with_one_hot_labels_predicts_one_hot_rows():
    one_hot = np.eye(2)[(LABELS == "b").astype(int)]
    vqc = qml.VQC(feature_map=z_feature_map(2, reps=1), initial_point=[0] * 8)
    vqc.fit(FEATURES, one_hot)
    assert np.array_equal(vqc.predict(FEATURES), one_hot)
    assert vqc.score(FEATURES, one_hot) == 1.0


def test_vqc_defaults_follow_qiskit_machine_learning():
    vqc = qml.VQC(num_qubits=2)
    assert vqc.feature_map.name == zz_feature_map(2).name
    assert vqc.feature_map.num_parameters == 2
    assert vqc.ansatz.name == real_amplitudes(2).name
    assert vqc.ansatz.num_parameters == real_amplitudes(2).num_parameters
    one = qml.VQC(num_qubits=1).feature_map
    assert one.name == z_feature_map(1).name


def test_classifier_with_one_output_separates_two_classes():
    x, w = Parameter("x"), Parameter("w")
    circuit = QuantumCircuit(1)
    circuit.ry(x, 0)
    circuit.ry(w, 0)
    network = qml.EstimatorQNN(circuit=circuit, input_params=[x])
    classifier = qml.NeuralNetworkClassifier(network, initial_point=[0.0])
    features = np.array([[0.1], [0.3], [2.9], [3.0]])
    labels = np.array([5, 5, 9, 9])
    classifier.fit(features, labels)
    assert classifier.predict(features).tolist() == [5, 5, 9, 9]
    assert classifier.classes_.tolist() == [5, 9]


def test_vqr_fits_a_representable_target():
    x, a = Parameter("x"), Parameter("a")
    feature_map, ansatz = QuantumCircuit(1), QuantumCircuit(1)
    feature_map.ry(x, 0)
    ansatz.ry(a, 0)
    features = np.linspace(0, 1.5, 8)[:, None]
    targets = np.cos(features[:, 0] + 0.7)  # <Z> after ry(x + 0.7)
    vqr = qml.VQR(feature_map=feature_map, ansatz=ansatz, initial_point=[0])
    vqr.fit(features, targets)
    assert vqr.predict(features).shape == (8, 1)
    assert vqr.score(features, targets) >= 1 - FTOL


def test_callback_sees_every_objective_evaluation():
    seen = []
    vqc = qml.VQC(
        feature_map=z_feature_map(2, reps=1),
        initial_point=[0] * 8,
        callback=lambda w, value: seen.append((w.shape, value)),
    )
    vqc.fit(FEATURES, LABELS)
    assert len(seen) == vqc.fit_result.nfev
    assert all(shape == (8,) and value >= 0 for shape, value in seen)


def test_minimizer_callable_and_warm_start():
    lbfgs = functools.partial(
        minimize, method="L-BFGS-B", options={"maxiter": 3}
    )
    evaluated = []
    vqc = qml.VQC(
        feature_map=z_feature_map(2, reps=1),
        optimizer=lbfgs,
        initial_point=[0] * 8,
        warm_start=True,
        callback=lambda w, value: evaluated.append(w),
    )
    vqc.fit(FEATURES, LABELS)
    first, calls = vqc.weights, len(evaluated)
    vqc.fit(FEATURES, LABELS)
    assert np.array_equal(evaluated[calls], first)


def test_programs_are_ordinary_qasm_programs():
    network, _, _ = _network()
    base = network._templates_for(None)[0][0]
    program = base.program(np.zeros(base.num_angles))
    assert program.representation == template_module.QML_PROGRAM
    assert program.num_qubits == 2
    assert "rz" in program.qasm2


# ------------------------------------------------------------------- kernels


def test_fidelity_kernel_is_the_state_overlap():
    feature_map = zz_feature_map(2)
    kernel = qml.FidelityQuantumKernel(feature_map=feature_map)
    states = [Statevector(feature_map.assign_parameters(x)) for x in FEATURES]
    expected = np.array(
        [[abs(a.inner(b)) ** 2 for b in states] for a in states]
    )
    matrix = kernel.evaluate(FEATURES)
    assert np.abs(matrix - expected).max() <= EXACT
    assert np.abs(matrix - matrix.T).max() == 0
    cross = kernel.evaluate(FEATURES[:2], FEATURES[2:])
    assert np.abs(cross - expected[:2, 2:]).max() <= EXACT


def test_qir_kernel_is_the_overlap_of_encoded_images():
    images, _ = _images()
    kernel = qml.FidelityQuantumKernel(feature_map="frqi")
    frqi = REPRESENTATIONS.create("frqi")
    states = [Statevector(encode_circuit(i, frqi).circuit) for i in images[:3]]
    expected = np.array(
        [[abs(a.inner(b)) ** 2 for b in states] for a in states]
    )
    assert np.abs(kernel.evaluate(images[:3]) - expected).max() <= EXACT


@pytest.mark.parametrize("duplicates", ["all", "off_diagonal", "none"])
def test_evaluate_duplicates(duplicates, monkeypatch):
    kernel = qml.FidelityQuantumKernel(
        feature_map=zz_feature_map(2), evaluate_duplicates=duplicates
    )
    runs = []
    real = template_module.probabilities

    def counting(*args):
        runs.append(1)
        return real(*args)

    monkeypatch.setattr("openqcv.qml.kernels.probabilities", counting)
    a, b = FEATURES[0], FEATURES[3]
    # Pairs: 3 diagonal, (0, 1) identical off-diagonal, (0, 2) and (1, 2).
    matrix = kernel.evaluate([a, a, b])
    expected = {"all": 6, "off_diagonal": 3, "none": 2}
    assert len(runs) == expected[duplicates]
    assert matrix[0, 1] == pytest.approx(1.0, abs=EXACT)


def test_kernel_matrix_is_made_positive_semidefinite():
    from openqcv.qml.kernels import _nearest_psd

    indefinite = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.9], [0.1, 0.9, 1.0]])
    assert np.linalg.eigvalsh(indefinite).min() < 0
    fixed = _nearest_psd(indefinite)
    assert np.linalg.eigvalsh(fixed).min() >= -EXACT
    assert np.abs(fixed - fixed.T).max() <= EXACT


def test_qsvc_classifies_features_and_images():
    pytest.importorskip("sklearn")
    kernel = qml.FidelityQuantumKernel(feature_map=z_feature_map(2))
    qsvc = qml.QSVC(kernel).fit(FEATURES, LABELS)
    assert qsvc.score(FEATURES, LABELS) == 1.0
    images, labels = _images()
    qsvc = qml.QSVC(qml.FidelityQuantumKernel(feature_map="qpie"))
    qsvc.fit(images, labels)
    assert qsvc.predict(images).tolist() == labels.tolist()
    assert qsvc.decision_function(images).shape == (len(images),)


# ---------------------------------------------------------------- validation


def _circuit_with_clbit():
    circuit = QuantumCircuit(1, 1)
    circuit.ry(Parameter("w"), 0)
    return circuit


def _one_parameter():
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("w"), 0)
    return circuit


@pytest.mark.parametrize(
    ("make", "error", "message"),
    [
        (
            lambda: qml.SamplerQNN(circuit="ry"),
            TypeError,
            "circuit must be a QuantumCircuit, got str",
        ),
        (
            lambda: qml.SamplerQNN(circuit=_circuit_with_clbit()),
            ValueError,
            "The circuit must have no classical bits",
        ),
        (
            lambda: qml.SamplerQNN(
                circuit=_one_parameter(),
                input_params=list(_one_parameter().parameters),
                weight_params=[],
            ),
            ValueError,
            "neither inputs nor weights",
        ),
        (
            lambda: _shared_parameter_network(),
            ValueError,
            "A parameter cannot be both input and weight",
        ),
        (
            lambda: _qir_with_inputs(),
            ValueError,
            "With a QIR the input is the image; input_params must be empty",
        ),
        (
            lambda: qml.SamplerQNN(circuit=_one_parameter(), output_shape=2),
            ValueError,
            "interpret and output_shape must be given together",
        ),
        (
            lambda: qml.SamplerQNN(
                circuit=_one_parameter(), interpret=abs, output_shape=True
            ),
            TypeError,
            "output_shape must be an int, got bool",
        ),
        (
            lambda: qml.SamplerQNN(
                circuit=_one_parameter(), interpret=abs, output_shape=0
            ),
            ValueError,
            "output_shape must be at least 1",
        ),
        (
            lambda: qml.EstimatorQNN(
                circuit=_one_parameter(), observables=["Z"]
            ),
            TypeError,
            "An observable must be a SparsePauliOp or Pauli, got str",
        ),
        (
            lambda: qml.EstimatorQNN(
                circuit=_one_parameter(), observables=SparsePauliOp("ZZ")
            ),
            ValueError,
            "An observable acts on 2 qubits; the circuit has 1",
        ),
        (
            lambda: qml.EstimatorQNN(
                circuit=_one_parameter(),
                observables=SparsePauliOp("Z", [1j]),
            ),
            ValueError,
            "An observable must be Hermitian",
        ),
    ],
)
def test_network_validation(make, error, message):
    with pytest.raises(error, match=message):
        make()


def _shared_parameter_network():
    circuit = _one_parameter()
    (w,) = circuit.parameters
    return qml.SamplerQNN(circuit=circuit, input_params=[w], weight_params=[w])


def _qir_with_inputs():
    circuit = _one_parameter()
    return qml.SamplerQNN(
        circuit=circuit, input_params=list(circuit.parameters), qir="qpie"
    )


@pytest.mark.parametrize(
    ("inputs", "weights", "message"),
    [
        (None, [0.0, 1.0], r"Expected 4 weights, got shape \(2,\)"),
        (None, [0, 0, 0, np.nan], "Weights must be finite"),
        (None, [0] * 4, "Expected 2 input features, got None"),
        ([1.0, 2.0], [0] * 4, r"Expected input of shape \(n, 2\), got \(2,\)"),
        ([[1.0, np.inf]], [0] * 4, "Input features must be finite"),
    ],
)
def test_forward_validation(inputs, weights, message):
    network, _, _ = _network()
    with pytest.raises(ValueError, match=message):
        network.forward(inputs, weights)


def test_qir_network_needs_images_with_matching_qubits():
    network = qml.SamplerQNN(circuit=real_amplitudes(3), qir="qpie")
    with pytest.raises(ValueError, match="needs input images"):
        network.forward(None, np.zeros(network.num_weights))
    with pytest.raises(
        ValueError,
        match=r"The QIR encodes an image of shape \(2, 2\) on 2 qubits; the "
        "circuit has 3",
    ):
        network.forward([np.ones((2, 2))], np.zeros(network.num_weights))


def test_interpret_out_of_range():
    network = qml.SamplerQNN(
        circuit=real_amplitudes(1, reps=0),
        interpret=lambda x: x + 1,
        output_shape=2,
    )
    with pytest.raises(ValueError, match=r"interpret\(1\) = 2 is outside"):
        network.forward(None, [np.pi / 2])


def test_template_rejects_a_parameter_outside_a_rotation(monkeypatch):
    def unlowered(circuit, basis=None):
        return circuit

    monkeypatch.setattr(template_module, "lower", unlowered)
    circuit = QuantumCircuit(1)
    circuit.p(Parameter("w"), 0)
    with pytest.raises(ValueError, match="A parameter ended up in a 'p' gate"):
        Template(circuit)


@pytest.mark.parametrize(
    ("loss", "error", "message"),
    [
        ("hinge", ValueError, "Unknown loss 'hinge'. Available:"),
        (2, TypeError, "loss must be a name or a Loss, got int"),
    ],
)
def test_loss_validation(loss, error, message):
    with pytest.raises(error, match=message):
        get_loss(loss)


@pytest.mark.parametrize(
    ("make", "error", "message"),
    [
        (
            lambda: qml.NeuralNetworkClassifier("net"),
            TypeError,
            "neural_network must be a NeuralNetwork, got str",
        ),
        (
            lambda: qml.VQC(num_qubits=1, optimizer=3),
            TypeError,
            "optimizer must be a SciPy method name or a callable, got int",
        ),
        (
            lambda: qml.VQC(num_qubits=1, callback=3),
            TypeError,
            "callback must be callable",
        ),
        (
            lambda: qml.VQC(num_qubits=1, interpret=abs),
            ValueError,
            "interpret and output_shape must be given together",
        ),
        (
            lambda: qml.VQC(num_qubits=1.0),
            TypeError,
            "num_qubits must be an int, got float",
        ),
        (
            lambda: qml.VQC(num_qubits=0),
            ValueError,
            "num_qubits must be at least 1",
        ),
        (
            lambda: qml.VQR(ansatz="ry"),
            TypeError,
            "ansatz must be a QuantumCircuit, got str",
        ),
        (
            lambda: qml.VQC(num_qubits=3, feature_map=zz_feature_map(2)),
            ValueError,
            r"disagree on the number of qubits: \[2, 3\]",
        ),
        (
            lambda: qml.VQC(),
            ValueError,
            "Give num_qubits, a feature map or an ansatz",
        ),
        (
            lambda: qml.FidelityQuantumKernel(
                feature_map="qpie", evaluate_duplicates="some"
            ),
            ValueError,
            "evaluate_duplicates must be one of",
        ),
    ],
)
def test_model_validation(make, error, message):
    with pytest.raises(error, match=message):
        make()


def _vqc(**options):
    return qml.VQC(
        feature_map=z_feature_map(2, reps=1), initial_point=[0] * 8, **options
    )


def test_fit_validation():
    with pytest.raises(
        ValueError, match="x has 6 samples and y has 2; they must match"
    ):
        _vqc().fit(FEATURES, LABELS[:2])
    with pytest.raises(
        ValueError, match=r"initial_point has shape \(2,\); the network"
    ):
        qml.VQC(feature_map=z_feature_map(2), initial_point=[0, 0]).fit(
            FEATURES, LABELS
        )
    with pytest.raises(ValueError, match="This VQC is not fitted"):
        _vqc().predict(FEATURES)
    with pytest.raises(ValueError, match="This VQC is not fitted"):
        _ = _vqc().weights
    with pytest.raises(
        TypeError, match="The optimizer must return an object with"
    ):
        _vqc(optimizer=lambda **_: None).fit(FEATURES, LABELS)


def test_classifier_target_validation():
    network = qml.SamplerQNN(
        circuit=real_amplitudes(1), interpret=lambda x: x, output_shape=2
    )
    one_hot = qml.NeuralNetworkClassifier(network, one_hot=True)
    with pytest.raises(ValueError, match="With one_hot, y must be rows"):
        one_hot.fit([None, None], [[1, 1], [0, 1]])
    labels = qml.NeuralNetworkClassifier(network)
    with pytest.raises(ValueError, match=r"y must be 1-D labels, got shape"):
        labels.fit([None], [[0, 1]])
    with pytest.raises(
        ValueError, match="The network has 2 outputs; y has 3 classes"
    ):
        labels.fit([None] * 3, [0, 1, 2])
    estimator = qml.EstimatorQNN(circuit=real_amplitudes(1))
    with pytest.raises(
        ValueError, match="A network with one output separates two classes"
    ):
        qml.NeuralNetworkClassifier(estimator).fit([None] * 3, [0, 1, 2])


def test_warm_start_needs_the_same_classes():
    vqc = _vqc(warm_start=True)
    vqc.fit(FEATURES, LABELS)
    with pytest.raises(ValueError, match="A warm start needs the classes"):
        vqc.fit(FEATURES, np.where(LABELS == "a", "a", "c"))


def test_regressor_target_validation():
    network = qml.EstimatorQNN(circuit=real_amplitudes(1))
    with pytest.raises(
        ValueError, match=r"the network outputs \(1,\) per sample"
    ):
        qml.NeuralNetworkRegressor(network).fit([None], [[1.0, 2.0]])


def test_kernel_validation():
    kernel = qml.FidelityQuantumKernel(feature_map=zz_feature_map(2))
    with pytest.raises(
        ValueError, match=r"Expected features of shape \(n, 2\)"
    ):
        kernel.evaluate([1.0, 2.0])
    with pytest.raises(ValueError, match="Features must be finite"):
        kernel.evaluate([[1.0, np.nan]])
    images = qml.FidelityQuantumKernel(feature_map="qpie")
    with pytest.raises(
        ValueError, match="Two images are encoded on 2 and 3 qubits"
    ):
        images.evaluate([np.ones((2, 2))], [np.ones((2, 3))])


def test_qsvc_validation():
    pytest.importorskip("sklearn")
    kernel = qml.FidelityQuantumKernel(feature_map="qpie")
    with pytest.raises(
        TypeError,
        match="quantum_kernel must be a FidelityQuantumKernel, got str",
    ):
        qml.QSVC("kernel")
    with pytest.raises(
        ValueError,
        match="The kernel is the quantum kernel; do not pass kernel=",
    ):
        qml.QSVC(kernel, kernel="rbf")
    with pytest.raises(ValueError, match="This QSVC is not fitted"):
        qml.QSVC(kernel).predict([np.ones((2, 2))])


def test_qsvc_without_scikit_learn(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.svm", None)
    kernel = qml.FidelityQuantumKernel(feature_map="qpie")
    with pytest.raises(
        ImportError,
        match=r"QSVC needs scikit-learn: pip install openqcv\[qml\]",
    ):
        qml.QSVC(kernel)


# ---------------------------------------------------------------- optimizers


def _quadratic(x):
    return float(np.sum((np.asarray(x) - [1.0, -2.0, 0.5]) ** 2))


def test_spsa_minimizes_a_quadratic():
    result = qml.SPSA(maxiter=300, seed=SEED)(fun=_quadratic, x0=np.zeros(3))
    assert np.abs(result.x - [1.0, -2.0, 0.5]).max() < 1e-2
    assert result.fun == _quadratic(result.x)
    calls = 2 * SPSA_CALIBRATION + 2 * 300 + 1
    assert (result.nit, result.nfev) == (300, calls)


SPSA_CALIBRATION = 25


def test_spsa_step_follows_the_update_rule():
    a, c, x0 = 0.3, 0.1, np.array([0.2, -0.4, 0.9])
    result = qml.SPSA(maxiter=1, learning_rate=a, perturbation=c, seed=SEED)(
        fun=_quadratic, x0=x0
    )
    delta = 1.0 - 2.0 * np.random.default_rng(SEED).binomial(1, 0.5, 3)
    slope = (_quadratic(x0 + c * delta) - _quadratic(x0 - c * delta)) / (2 * c)
    assert np.array_equal(result.x, x0 - a * slope * delta)


def test_spsa_calibration_targets_the_first_step():
    x0 = np.array([0.2, -0.4, 0.9])
    rng = np.random.default_rng(SEED)
    a = qml.SPSA.calibrate(_quadratic, x0, rng)
    rng = np.random.default_rng(SEED)
    c, total = 0.2, 0.0
    for _ in range(SPSA_CALIBRATION):
        d = 1.0 - 2.0 * rng.binomial(1, 0.5, 3)
        total += abs(
            (_quadratic(x0 + c * d) - _quadratic(x0 - c * d)) / (2 * c)
        )
    assert a == pytest.approx(2 * np.pi / 10 / (total / SPSA_CALIBRATION))
    flat = qml.SPSA.calibrate(lambda x: 1.0, x0, np.random.default_rng(SEED))
    assert flat == 2 * np.pi / 10


def test_spsa_last_avg_returns_the_mean_of_the_last_iterates():
    def run(maxiter, last_avg=1):
        spsa = qml.SPSA(
            maxiter=maxiter,
            learning_rate=0.1,
            perturbation=0.1,
            last_avg=last_avg,
            seed=SEED,
        )
        return spsa(fun=_quadratic, x0=np.zeros(3)).x

    averaged = run(2, last_avg=2)
    assert np.array_equal(averaged, (run(1) + run(2)) / 2)


def test_vqc_trains_with_spsa():
    vqc = _vqc(optimizer=qml.SPSA(maxiter=30, seed=SEED))
    vqc.fit(FEATURES, LABELS)
    assert vqc.score(FEATURES, LABELS) == 1.0


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"maxiter": 1.5}, TypeError, "maxiter must be an int, got float"),
        ({"last_avg": 0}, ValueError, "last_avg must be at least 1"),
        (
            {"resamplings": True},
            TypeError,
            "resamplings must be an int, got bool",
        ),
        (
            {"learning_rate": 0.1},
            ValueError,
            "learning_rate and perturbation must be given together",
        ),
    ],
)
def test_spsa_validation(options, error, message):
    with pytest.raises(error, match=message):
        qml.SPSA(**options)


# ---------------------------------------------------------------- fast export


def _qir_network(**options):
    return qml.SamplerQNN(
        circuit=real_amplitudes(3, reps=1), qir="frqi", **options
    )


def test_fast_export_makes_identical_programs():
    images, _ = _images()
    encoded = encode_circuit(images[0], REPRESENTATIONS.create("frqi"))
    circuit = encoded.circuit.copy()
    circuit.compose(real_amplitudes(3, reps=2), qubits=range(3), inplace=True)
    slow, fast = Template(circuit), Template(circuit, fast_export=True)
    rng = np.random.default_rng(SEED)
    special = np.resize(template_module.FAST_EXPORT_CHECKS, slow.num_angles)
    for angles in (rng.uniform(-7, 7, slow.num_angles), special):
        assert fast.program(angles) == slow.program(angles)


def test_fast_export_gives_identical_outputs_and_gradients():
    images, _ = _images()
    weights = np.random.default_rng(SEED).random(6)
    slow, fast = _qir_network(), _qir_network(fast_export=True)
    assert fast.fast_export and not slow.fast_export
    assert np.array_equal(
        fast.forward(images[:2], weights), slow.forward(images[:2], weights)
    )
    assert np.array_equal(
        fast.backward(images[:2], weights)[1],
        slow.backward(images[:2], weights)[1],
    )
    estimator, _, _ = _network(qml.EstimatorQNN)
    quick, _, _ = _network(qml.EstimatorQNN, fast_export=True)
    features = np.random.default_rng(SEED).random((2, 2))
    w = np.random.default_rng(SEED).random(estimator.num_weights)
    assert np.array_equal(
        quick.forward(features, w), estimator.forward(features, w)
    )


def test_models_and_kernels_pass_fast_export_on():
    vqc = _vqc(fast_export=True, optimizer="COBYLA")
    vqc.fit(FEATURES, LABELS)
    assert vqc.neural_network.fast_export
    assert vqc.score(FEATURES, LABELS) == _vqc(optimizer="COBYLA").fit(
        FEATURES, LABELS
    ).score(FEATURES, LABELS)
    feature_map = zz_feature_map(2)
    slow = qml.FidelityQuantumKernel(feature_map=feature_map)
    fast = qml.FidelityQuantumKernel(feature_map=feature_map, fast_export=True)
    assert np.array_equal(fast.evaluate(FEATURES), slow.evaluate(FEATURES))


@pytest.mark.parametrize(
    ("qasm2_format", "message"),
    [
        (
            lambda angle: f"{angle:.3f}",
            "fast_export cannot locate an angle in the qasm2 text",
        ),
        (repr, "fast_export does not reproduce the qasm2 exporter's text"),
    ],
)
def test_fast_export_refuses_text_it_cannot_reproduce(
    qasm2_format, message, monkeypatch
):
    dump, _ = template_module._FORMATS["qasm2"]
    monkeypatch.setitem(
        template_module._FORMATS, "qasm2", (dump, qasm2_format)
    )
    with pytest.raises(ValueError, match=message):
        Template(_one_parameter(), fast_export=True)
