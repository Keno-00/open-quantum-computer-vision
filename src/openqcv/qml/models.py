r"""Trainable models: classifiers and regressors built on a network.

The models follow the API of Qiskit Machine Learning (``fit``,
``predict``, ``score``; ``VQC`` and ``VQR`` with a feature map and an
ansatz) and run on openqcv backends. Training minimizes the mean loss

.. math::

    L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell\big(f(x_i; \theta),
    t_i\big)

over the weights :math:`\theta` of the network :math:`f`, with gradient
:math:`\nabla L = \frac{1}{n} \sum_i \sum_k (\partial \ell / \partial
f_k)\, \nabla_\theta f_k(x_i; \theta)` from the network's parameter-shift
gradients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    real_amplitudes,
    z_feature_map,
    zz_feature_map,
)
from qiskit.quantum_info import Pauli, SparsePauliOp
from scipy.optimize import minimize

from openqcv.pipeline import encode_circuit
from openqcv.qml.losses import Loss, get_loss
from openqcv.qml.qnn import EstimatorQNN, NeuralNetwork, SamplerQNN

if TYPE_CHECKING:
    from openqcv.backends import Backend
    from openqcv.representations import QIR

Minimizer = Callable[..., Any]
"""``minimizer(fun=..., x0=..., jac=...)`` returning an object with ``x``,
such as ``functools.partial(scipy.optimize.minimize, method="L-BFGS-B",
options={"maxiter": 50})``. It is always given ``jac``, the
parameter-shift gradient, which it may ignore."""

GRADIENT_FREE_METHODS = frozenset(
    {"nelder-mead", "powell", "cobyla", "cobyqa"}
)
"""SciPy methods that take no gradient; they are not given one, so the
parameter-shift circuits are not run for nothing."""

Callback = Callable[[np.ndarray, float], None]
"""Called with the weights and the objective at every evaluation."""


class TrainableModel(ABC):
    """A model that fits a network's weights to data.

    Args:
        neural_network: The network; ``None`` for a model that builds its
            own when it is fitted.
        loss: A loss name (``"absolute_error"``, ``"squared_error"``,
            ``"cross_entropy"``) or a :class:`~openqcv.qml.losses.Loss`.
        optimizer: A :func:`scipy.optimize.minimize` method name, or a
            :data:`Minimizer`. SciPy methods that use a gradient get the
            parameter-shift gradient.
        warm_start: Start each fit from the weights of the last one.
        initial_point: The starting weights; random in :math:`[0, 1)` if
            ``None``, as in Qiskit Machine Learning.
        callback: Called with the weights and the objective value at every
            objective evaluation.

    Raises:
        TypeError: If an argument has the wrong type.
    """

    def __init__(
        self,
        neural_network: NeuralNetwork | None,
        loss: str | Loss,
        optimizer: str | Minimizer,
        warm_start: bool,
        initial_point: ArrayLike | None,
        callback: Callback | None,
    ) -> None:
        if neural_network is not None and not isinstance(
            neural_network, NeuralNetwork
        ):
            raise TypeError(
                "neural_network must be a NeuralNetwork, got "
                f"{type(neural_network).__name__}"
            )
        if not (isinstance(optimizer, str) or callable(optimizer)):
            raise TypeError(
                "optimizer must be a SciPy method name or a callable, got "
                f"{type(optimizer).__name__}"
            )
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        self._neural_network = neural_network
        self._loss = get_loss(loss)
        self._optimizer = optimizer
        self._warm_start = bool(warm_start)
        self._initial_point = (
            None
            if initial_point is None
            else np.asarray(initial_point, dtype=float)
        )
        self._callback = callback
        self._fit_result: Any = None

    @property
    def neural_network(self) -> NeuralNetwork | None:
        """The network, or ``None`` before a self-building model is fitted."""
        return self._neural_network

    @property
    def loss(self) -> Loss:
        """The loss."""
        return self._loss

    @property
    def fit_result(self) -> Any:
        """What the optimizer returned in the last fit, or ``None``."""
        return self._fit_result

    @property
    def weights(self) -> np.ndarray:
        """The fitted weights.

        Raises:
            ValueError: If the model is not fitted.
        """
        self._check_fitted()
        return np.asarray(self._fit_result.x, dtype=float)

    def fit(self, x: Any, y: ArrayLike) -> TrainableModel:
        """Fit the weights to inputs ``x`` and targets ``y``.

        Args:
            x: ``(n, num_inputs)`` features, or ``n`` images if the network
                encodes them with a QIR.
            y: The targets; see the model's class.

        Raises:
            ValueError: If ``x`` and ``y`` have different lengths, or the
                targets do not fit the network's output.
            TypeError: If the optimizer's result has no ``x``.
        """
        if len(x) != len(y):
            raise ValueError(
                f"x has {len(x)} samples and y has {len(y)}; they must match"
            )
        targets = self._targets(x, y)
        network = self._neural_network
        weights = self._start(network)
        self._fit_result = self._minimize(network, x, targets, weights)
        return self

    @abstractmethod
    def _targets(self, x: Any, y: ArrayLike) -> np.ndarray:
        """Return the targets as ``(n, *output_shape)``.

        A model that builds its own network builds it here first.
        """

    def _start(self, network: NeuralNetwork) -> np.ndarray:
        if self._warm_start and self._fit_result is not None:
            return self.weights
        if self._initial_point is None:
            return np.random.default_rng().random(network.num_weights)
        if self._initial_point.shape != (network.num_weights,):
            raise ValueError(
                f"initial_point has shape {self._initial_point.shape}; the "
                f"network has {network.num_weights} weights"
            )
        return self._initial_point

    def _minimize(
        self,
        network: NeuralNetwork,
        x: Any,
        targets: np.ndarray,
        start: np.ndarray,
    ) -> Any:
        n = len(targets)
        last: dict[bytes, np.ndarray] = {}

        def outputs(weights: np.ndarray) -> np.ndarray:
            # Optimizers ask for the objective and the gradient at the same
            # weights; run the circuits once.
            key = weights.tobytes()
            if key not in last:
                last.clear()
                last[key] = network.forward(x, weights)
            return last[key]

        def objective(weights: ArrayLike) -> float:
            weights = np.asarray(weights, dtype=float)
            value = float(self._loss(outputs(weights), targets).sum() / n)
            if self._callback is not None:
                self._callback(weights.copy(), value)
            return value

        def gradient(weights: ArrayLike) -> np.ndarray:
            weights = np.asarray(weights, dtype=float)
            _, derivative = network.backward(x, weights)
            slope = self._loss.gradient(outputs(weights), targets)
            return np.einsum("nk,nkp->p", slope, derivative) / n

        if isinstance(self._optimizer, str):
            method = self._optimizer
            jac = None if method.lower() in GRADIENT_FREE_METHODS else gradient
            result = minimize(objective, start, method=method, jac=jac)
        else:
            result = self._optimizer(fun=objective, x0=start, jac=gradient)
        if not hasattr(result, "x"):
            raise TypeError(
                "The optimizer must return an object with the weights as x"
            )
        return result

    def _check_fitted(self) -> None:
        if self._fit_result is None:
            raise ValueError(
                f"This {type(self).__name__} is not fitted; call fit first"
            )

    def _forward(self, x: Any) -> np.ndarray:
        self._check_fitted()
        return self._neural_network.forward(x, self.weights)


class NeuralNetworkClassifier(TrainableModel):
    """A classifier whose scores are a network's outputs.

    With one output, the two classes (in sorted order) are trained toward
    :math:`-1` and :math:`+1`, and an output above 0 predicts the second.
    With :math:`K` outputs, there must be :math:`K` classes; they are
    trained toward one-hot rows, and the largest output wins.

    Args:
        neural_network: The network.
        loss: See :class:`TrainableModel`.
        one_hot: ``y`` is given as one-hot rows, and :meth:`predict`
            returns them; otherwise ``y`` holds labels of any sortable
            type.
        optimizer: See :class:`TrainableModel`.
        warm_start: See :class:`TrainableModel`.
        initial_point: See :class:`TrainableModel`.
        callback: See :class:`TrainableModel`.
    """

    def __init__(
        self,
        neural_network: NeuralNetwork | None,
        loss: str | Loss = "squared_error",
        one_hot: bool = False,
        optimizer: str | Minimizer = "SLSQP",
        warm_start: bool = False,
        initial_point: ArrayLike | None = None,
        callback: Callback | None = None,
    ) -> None:
        super().__init__(
            neural_network,
            loss,
            optimizer,
            warm_start,
            initial_point,
            callback,
        )
        self._one_hot = bool(one_hot)
        self._classes: np.ndarray | None = None

    @property
    def classes_(self) -> np.ndarray | None:
        """The class labels, or column indices with ``one_hot``."""
        return self._classes

    @property
    def num_classes(self) -> int | None:
        """The number of classes, or ``None`` before fitting."""
        return None if self._classes is None else len(self._classes)

    def predict(self, x: Any) -> np.ndarray:
        """Return the predicted labels, or one-hot rows with ``one_hot``."""
        output = self._forward(x)
        if output.shape[1] == 1:
            index = (output[:, 0] > 0).astype(int)
        else:
            index = np.argmax(output, axis=1)
        if self._one_hot:
            return np.eye(len(self._classes))[index]
        return self._classes[index]

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return the network's outputs, shape ``(n, *output_shape)``.

        They are probabilities only if the network makes them so, as a
        :class:`~openqcv.qml.qnn.SamplerQNN` does.
        """
        return self._forward(x)

    def score(
        self,
        x: Any,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> float:
        """Return the (weighted) fraction of samples predicted correctly."""
        return accuracy(self.predict(x), y, sample_weight)

    def _targets(self, x: Any, y: ArrayLike) -> np.ndarray:
        y = np.asarray(y)
        if self._one_hot:
            if (
                y.ndim != 2
                or not np.all((y == 0) | (y == 1))
                or not np.all(y.sum(axis=1) == 1)
            ):
                raise ValueError(
                    "With one_hot, y must be rows of 0s with a single 1"
                )
            classes, index = np.arange(y.shape[1]), np.argmax(y, axis=1)
        else:
            if y.ndim != 1:
                raise ValueError(
                    f"y must be 1-D labels, got shape {y.shape}; pass "
                    "one_hot=True for one-hot rows"
                )
            classes, index = np.unique(y, return_inverse=True)
        if (
            self._warm_start
            and self._classes is not None
            and not np.array_equal(classes, self._classes)
        ):
            raise ValueError(
                "A warm start needs the classes of the last fit, "
                f"{self._classes.tolist()}; got {classes.tolist()}"
            )
        self._classes = classes
        outputs = self._network(x, len(classes)).output_shape[0]
        if outputs == 1:
            if len(classes) != 2:
                raise ValueError(
                    "A network with one output separates two classes; y "
                    f"has {len(classes)}"
                )
            return np.where(index == 1, 1.0, -1.0)[:, None]
        if outputs != len(classes):
            raise ValueError(
                f"The network has {outputs} outputs; y has {len(classes)} "
                "classes"
            )
        return np.eye(outputs)[index]

    def _network(self, x: Any, num_classes: int) -> NeuralNetwork:
        """Return the network to train for ``num_classes`` classes."""
        return self._neural_network


class NeuralNetworkRegressor(TrainableModel):
    """A regressor whose predictions are a network's outputs.

    Args:
        neural_network: The network.
        loss: See :class:`TrainableModel`.
        optimizer: See :class:`TrainableModel`.
        warm_start: See :class:`TrainableModel`.
        initial_point: See :class:`TrainableModel`.
        callback: See :class:`TrainableModel`.
    """

    def __init__(
        self,
        neural_network: NeuralNetwork | None,
        loss: str | Loss = "squared_error",
        optimizer: str | Minimizer = "SLSQP",
        warm_start: bool = False,
        initial_point: ArrayLike | None = None,
        callback: Callback | None = None,
    ) -> None:
        super().__init__(
            neural_network,
            loss,
            optimizer,
            warm_start,
            initial_point,
            callback,
        )

    def predict(self, x: Any) -> np.ndarray:
        """Return the network's outputs, shape ``(n, *output_shape)``."""
        return self._forward(x)

    def score(
        self,
        x: Any,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> float:
        r"""Return the coefficient of determination :math:`R^2`.

        For each output, :math:`R^2 = 1 - \sum_i w_i (y_i - \hat y_i)^2 /
        \sum_i w_i (y_i - \bar y)^2` with :math:`\bar y` the weighted mean;
        the result is the mean over outputs. A constant target gives 1 if
        it is predicted exactly and 0 otherwise.
        """
        predicted = self.predict(x)
        target = np.asarray(y, dtype=float).reshape(predicted.shape)
        w = np.ones(len(target)) if sample_weight is None else sample_weight
        w = np.asarray(w, dtype=float)[:, None]
        mean = (w * target).sum(axis=0) / w.sum()
        residual = (w * (target - predicted) ** 2).sum(axis=0)
        total = (w * (target - mean) ** 2).sum(axis=0)
        exact = residual == 0
        safe = np.where(total > 0, total, 1.0)
        r2 = np.where(total > 0, 1 - residual / safe, exact.astype(float))
        return float(r2.mean())

    def _targets(self, x: Any, y: ArrayLike) -> np.ndarray:
        network = self._network(x)
        shape = network.output_shape
        target = np.asarray(y, dtype=float)
        if target.ndim == 1 and shape == (1,):
            target = target[:, None]
        if target.shape[1:] != shape:
            raise ValueError(
                f"y has shape {np.shape(y)}; the network outputs {shape} "
                "per sample"
            )
        return target

    def _network(self, x: Any) -> NeuralNetwork:
        """Return the network to train."""
        return self._neural_network


class _Variational:
    """The feature map, ansatz and qubit count of a VQC or VQR."""

    _parts: _Parts

    @property
    def feature_map(self) -> QuantumCircuit | str | type[QIR] | QIR:
        """The feature map: a circuit, or a QIR as given."""
        return self._parts.feature_map

    @property
    def ansatz(self) -> QuantumCircuit | None:
        """The ansatz, or ``None`` until a QIR's default one is built."""
        return self._parts.ansatz

    @property
    def num_qubits(self) -> int | None:
        """The number of qubits, or ``None`` until a QIR's image sets it."""
        return self._parts.num_qubits


class VQC(_Variational, NeuralNetworkClassifier):
    r"""Variational quantum classifier.

    The state :math:`U(\theta) V(x) |0\rangle` of a feature map :math:`V`
    and an ansatz :math:`U` is measured; outcome :math:`m` (qubit ``j`` is
    bit ``j``) votes for class ``interpret(m)``, by default
    :math:`m \bmod K` for :math:`K` classes, and the class probabilities
    are trained toward the one-hot labels.

    The feature map is a parameterized circuit on classical features, as
    in Qiskit Machine Learning, or a QIR, which encodes each input image
    and lets the ansatz act on all of its qubits. Without a feature map,
    :math:`V` is ``zz_feature_map`` (``z_feature_map`` on one qubit), and
    without an ansatz, :math:`U` is ``real_amplitudes``, both from Qiskit's
    circuit library with its default settings.

    ``y`` holds labels, or one-hot rows, which :meth:`predict` then returns.

    Args:
        num_qubits: The number of qubits; needed only if neither the feature
            map nor the ansatz says.
        feature_map: A circuit whose parameters, in order, take the
            features; or a QIR name, class or instance.
        ansatz: The trained circuit; all its parameters are weights.
        loss: See :class:`TrainableModel`; cross-entropy by default.
        optimizer: See :class:`TrainableModel`.
        warm_start: See :class:`TrainableModel`.
        initial_point: See :class:`TrainableModel`.
        callback: See :class:`TrainableModel`.
        interpret: Outcome -> class index; requires ``output_shape``.
        output_shape: Number of classes ``interpret`` maps to.
        backend: A backend name, class or instance.
        shots: Shots per circuit; ``None`` is exact on ``"statevector"``.
        fast_export: Write each circuit's QASM once and fill in the
            angles for every evaluation, instead of exporting the whole
            circuit each time. The programs are identical (see
            :class:`~openqcv.qml.template.Template`); only faster to make.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the number of qubits is unknown or inconsistent, or
            only one of ``interpret`` and ``output_shape`` is given.

    References:
        V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala,
        J. M. Chow and J. M. Gambetta, "Supervised learning with
        quantum-enhanced feature spaces", Nature 567, 209-212 (2019).
        doi:10.1038/s41586-019-0980-2
    """

    def __init__(
        self,
        num_qubits: int | None = None,
        feature_map: QuantumCircuit | str | type[QIR] | QIR | None = None,
        ansatz: QuantumCircuit | None = None,
        loss: str | Loss = "cross_entropy",
        optimizer: str | Minimizer = "SLSQP",
        warm_start: bool = False,
        initial_point: ArrayLike | None = None,
        callback: Callback | None = None,
        *,
        interpret: Callable[[int], int] | None = None,
        output_shape: int | None = None,
        backend: str | type[Backend] | Backend = "statevector",
        shots: int | None = None,
        fast_export: bool = False,
    ) -> None:
        super().__init__(
            None,
            loss,
            False,
            optimizer,
            warm_start,
            initial_point,
            callback,
        )
        if (interpret is None) != (output_shape is None):
            raise ValueError(
                "interpret and output_shape must be given together"
            )
        self._parts = _Parts(num_qubits, feature_map, ansatz)
        self._interpret = interpret
        self._output_shape = output_shape
        self._backend = backend
        self._shots = shots
        self._fast_export = fast_export

    def fit(self, x: Any, y: ArrayLike) -> VQC:
        """Fit to inputs ``x`` and labels or one-hot rows ``y``.

        See :meth:`TrainableModel.fit`.
        """
        self._one_hot = np.ndim(y) == 2
        super().fit(x, y)
        return self

    def _network(self, x: Any, num_classes: int) -> NeuralNetwork:
        if self._neural_network is None or not self._warm_start:
            classes = self._output_shape or num_classes
            interpret = self._interpret or _modulo(classes)
            self._neural_network = SamplerQNN(
                interpret=interpret,
                output_shape=classes,
                backend=self._backend,
                shots=self._shots,
                fast_export=self._fast_export,
                **self._parts.network(x),
            )
        return self._neural_network


class VQR(_Variational, NeuralNetworkRegressor):
    r"""Variational quantum regressor.

    The prediction is :math:`\langle \psi | O | \psi \rangle` for the state
    :math:`|\psi\rangle = U(\theta) V(x) |0\rangle`; the feature map, the
    ansatz and their defaults are those of :class:`VQC`, and :math:`O`
    defaults to :math:`Z^{\otimes n}`.

    Args:
        num_qubits: See :class:`VQC`.
        feature_map: See :class:`VQC`.
        ansatz: See :class:`VQC`.
        observable: A ``SparsePauliOp`` or ``Pauli``.
        loss: See :class:`TrainableModel`.
        optimizer: See :class:`TrainableModel`.
        warm_start: See :class:`TrainableModel`.
        initial_point: See :class:`TrainableModel`.
        callback: See :class:`TrainableModel`.
        backend: A backend name, class or instance.
        shots: Shots per circuit; ``None`` is exact on ``"statevector"``.
        fast_export: Write each circuit's QASM once and fill in the
            angles for every evaluation, instead of exporting the whole
            circuit each time. The programs are identical (see
            :class:`~openqcv.qml.template.Template`); only faster to make.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the number of qubits is unknown or inconsistent.
    """

    def __init__(
        self,
        num_qubits: int | None = None,
        feature_map: QuantumCircuit | str | type[QIR] | QIR | None = None,
        ansatz: QuantumCircuit | None = None,
        observable: SparsePauliOp | Pauli | None = None,
        loss: str | Loss = "squared_error",
        optimizer: str | Minimizer = "SLSQP",
        warm_start: bool = False,
        initial_point: ArrayLike | None = None,
        callback: Callback | None = None,
        *,
        backend: str | type[Backend] | Backend = "statevector",
        shots: int | None = None,
        fast_export: bool = False,
    ) -> None:
        super().__init__(
            None, loss, optimizer, warm_start, initial_point, callback
        )
        self._parts = _Parts(num_qubits, feature_map, ansatz)
        self._observable = observable
        self._backend = backend
        self._shots = shots
        self._fast_export = fast_export

    def _network(self, x: Any) -> NeuralNetwork:
        if self._neural_network is None or not self._warm_start:
            self._neural_network = EstimatorQNN(
                observables=self._observable,
                backend=self._backend,
                shots=self._shots,
                fast_export=self._fast_export,
                **self._parts.network(x),
            )
        return self._neural_network


def accuracy(
    predicted: np.ndarray, y: ArrayLike, sample_weight: ArrayLike | None
) -> float:
    """Return the (weighted) fraction of rows of ``predicted`` equal to y."""
    y = np.asarray(y)
    correct = predicted == y
    if correct.ndim == 2:
        correct = correct.all(axis=1)
    return float(np.average(correct, weights=sample_weight))


class _Parts:
    """The feature map and ansatz of a VQC or VQR, with their defaults."""

    def __init__(
        self,
        num_qubits: int | None,
        feature_map: QuantumCircuit | str | type[QIR] | QIR | None,
        ansatz: QuantumCircuit | None,
    ) -> None:
        if num_qubits is not None and (
            not isinstance(num_qubits, int) or isinstance(num_qubits, bool)
        ):
            raise TypeError(
                f"num_qubits must be an int, got {type(num_qubits).__name__}"
            )
        if num_qubits is not None and num_qubits < 1:
            raise ValueError("num_qubits must be at least 1")
        if ansatz is not None and not isinstance(ansatz, QuantumCircuit):
            raise TypeError(
                f"ansatz must be a QuantumCircuit, got {type(ansatz).__name__}"
            )
        classical = feature_map is None or isinstance(
            feature_map, QuantumCircuit
        )
        circuits = [ansatz] + ([feature_map] if classical else [])
        sizes = {c.num_qubits for c in circuits if c is not None}
        if num_qubits is not None:
            sizes.add(num_qubits)
        if len(sizes) > 1:
            raise ValueError(
                "num_qubits, the feature map and the ansatz disagree on the "
                f"number of qubits: {sorted(sizes)}"
            )
        n = sizes.pop() if sizes else None
        if classical:
            if n is None:
                raise ValueError(
                    "Give num_qubits, a feature map or an ansatz, so that "
                    "the number of qubits is known"
                )
            if feature_map is None:
                feature_map = z_feature_map(1) if n == 1 else zz_feature_map(n)
            self.feature_map: QuantumCircuit | str | type[QIR] | QIR = (
                feature_map
            )
        else:
            self.feature_map = feature_map
        self.num_qubits = n
        self.ansatz = (
            real_amplitudes(n) if ansatz is None and n is not None else ansatz
        )

    def network(self, x: Any) -> dict[str, Any]:
        """Return the circuit arguments of the network for inputs ``x``."""
        if isinstance(self.feature_map, QuantumCircuit):
            circuit = QuantumCircuit(self.num_qubits)
            circuit.compose(self.feature_map, inplace=True)
            circuit.compose(self.ansatz, inplace=True)
            return {
                "circuit": circuit,
                "input_params": list(self.feature_map.parameters),
                "weight_params": list(self.ansatz.parameters),
            }
        if self.ansatz is None:
            first = np.asarray(_first(x))
            encoded = encode_circuit(first, self.feature_map)
            self.num_qubits = encoded.circuit.num_qubits
            self.ansatz = real_amplitudes(self.num_qubits)
        return {
            "circuit": self.ansatz,
            "weight_params": list(self.ansatz.parameters),
            "qir": self.feature_map,
        }


def _first(x: Sequence[Any]) -> Any:
    for image in x:
        return image
    raise ValueError("x has no samples")


def _modulo(classes: int) -> Callable[[int], int]:
    """Qiskit Machine Learning's default ``interpret``: outcome mod K."""

    def interpret(outcome: int) -> int:
        return outcome % classes

    return interpret
