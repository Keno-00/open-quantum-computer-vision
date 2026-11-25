r"""Quantum neural networks: parameterized circuits as differentiable maps.

A network maps an input :math:`x` and weights :math:`\theta` to a vector
of outputs computed from measurements of :math:`U(\theta) V(x)|0\rangle`.
The input enters in one of two ways:

* classical features bound to the circuit's ``input_params``, as in Qiskit
  Machine Learning; or
* an image, encoded by a QIR (``qir=``): :math:`V(x)` is the QIR's circuit
  for the image, and ``circuit`` acts on its qubits.

Every evaluation runs on an openqcv backend as an ordinary
:class:`~openqcv.core.QasmProgram` (see :mod:`openqcv.qml.template`), so a
network runs wherever a QIR does. Weight gradients use the parameter-shift
rule and are exact on ``"statevector"``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli, SparsePauliOp

from openqcv.pipeline import encode_circuit, get_backend
from openqcv.qml.template import SHIFT, Template, outcome, probabilities

if TYPE_CHECKING:
    from openqcv.backends import Backend
    from openqcv.representations import QIR

FEATURE_REGISTER = "q"
"""Register that holds the qubits of a circuit without named registers."""

HERMITIAN_TOLERANCE = 1e-12
"""Largest imaginary part accepted in an observable's Pauli coefficients;
an observable must be Hermitian for its expectation to be real, and a
simplified ``SparsePauliOp`` of a Hermitian operator has real
coefficients up to rounding."""

TEMPLATE_CACHE = 1024
"""Inputs whose lowered circuits a network keeps, least recently used
dropped first. Lowering is the costly step of an image's first evaluation,
and training evaluates the same images every step: the cache holds a
training set of this size, while predicting on a stream of new images does
not grow memory without bound."""

Readout = Callable[[dict[str, float]], np.ndarray]
"""Outcome probabilities -> a contribution to the network's outputs."""


class NeuralNetwork(ABC):
    """A circuit with input and weight parameters, run on a backend.

    Args:
        circuit: The parameterized circuit, without classical bits.
        input_params: The parameters bound to the input features, in
            feature order. Must be empty when ``qir`` is given.
        weight_params: The trained parameters, in weight order; defaults to
            the circuit's other parameters, in the circuit's order.
        qir: A QIR (name, class or instance) whose circuit for each input
            image runs before ``circuit``, on the same qubits.
        backend: A backend, as :func:`openqcv.run` takes it (e.g.
            ``"aer:statevector"``); created once and used for every circuit.
        shots: Shots per circuit; ``None`` is exact on ``"statevector"``
            and the default elsewhere.
        fast_export: Write each circuit's QASM once and fill in the
            angles for every evaluation, instead of exporting the whole
            circuit each time. The programs are identical (see
            :class:`~openqcv.qml.template.Template`); only faster to make.

    Raises:
        TypeError: If ``circuit`` is not a ``QuantumCircuit``.
        ValueError: If the circuit has classical bits, a parameter is
            neither an input nor a weight or is both, or ``qir`` is given
            with ``input_params``.
    """

    def __init__(
        self,
        *,
        circuit: QuantumCircuit,
        input_params: Sequence[Parameter] = (),
        weight_params: Sequence[Parameter] | None = None,
        qir: str | type[QIR] | QIR | None = None,
        backend: str | type[Backend] | Backend = "statevector",
        shots: int | None = None,
        fast_export: bool = False,
    ) -> None:
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                "circuit must be a QuantumCircuit, got "
                f"{type(circuit).__name__}"
            )
        if circuit.num_clbits:
            raise ValueError(
                "The circuit must have no classical bits; every qubit is "
                "measured"
            )
        inputs = list(input_params)
        if weight_params is None:
            weights = [p for p in circuit.parameters if p not in inputs]
        else:
            weights = list(weight_params)
        if set(inputs) & set(weights):
            raise ValueError("A parameter cannot be both input and weight")
        unbound = set(circuit.parameters) - set(inputs) - set(weights)
        if unbound:
            names = ", ".join(sorted(p.name for p in unbound))
            raise ValueError(
                f"Parameters {names} are neither inputs nor weights"
            )
        if qir is not None and inputs:
            raise ValueError(
                "With a QIR the input is the image; input_params must be empty"
            )
        self._circuit = circuit
        self._input_params = inputs
        self._weight_params = weights
        self._qir = qir
        self._backend = get_backend(backend)
        self._shots = shots
        self._fast_export = bool(fast_export)
        self._templates: OrderedDict[Any, list[tuple[Template, Readout]]] = (
            OrderedDict()
        )

    @property
    def circuit(self) -> QuantumCircuit:
        """The parameterized circuit."""
        return self._circuit

    @property
    def input_params(self) -> list[Parameter]:
        """The input parameters, in feature order."""
        return list(self._input_params)

    @property
    def weight_params(self) -> list[Parameter]:
        """The weight parameters, in weight order."""
        return list(self._weight_params)

    @property
    def fast_export(self) -> bool:
        """Whether programs are made by filling angles into written text."""
        return self._fast_export

    @property
    def qir(self) -> str | type[QIR] | QIR | None:
        """The QIR that encodes input images, as given, or ``None``."""
        return self._qir

    @property
    def num_inputs(self) -> int:
        """Number of input features; 0 when the inputs are images."""
        return len(self._input_params)

    @property
    def num_weights(self) -> int:
        """Number of weights."""
        return len(self._weight_params)

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int]:
        """Shape of the output for one sample."""

    @abstractmethod
    def _readouts(self) -> list[tuple[QuantumCircuit | None, Readout]]:
        """Return the measurements that make up the output.

        Each is a circuit appended before measuring (``None`` for none)
        and the function that turns its outcome probabilities into a
        contribution to the output; the contributions are summed.
        """

    def forward(
        self,
        input_data: ArrayLike | Sequence[ArrayLike] | None,
        weights: ArrayLike,
    ) -> np.ndarray:
        """Return the outputs, shape ``(n, *output_shape)``.

        Args:
            input_data: ``(n, num_inputs)`` features; a sequence of ``n``
                images with a QIR; or ``None`` for one sample of a network
                without inputs.
            weights: The ``num_weights`` weights.
        """
        theta = self._weights(weights)
        samples = self._samples(input_data)
        output = np.zeros((len(samples), *self.output_shape))
        for i, (key, values) in enumerate(samples):
            values.update(zip(self._weight_params, theta, strict=True))
            for template, readout in self._templates_for(key):
                angles = template.angles(values)
                output[i] += readout(self._run(template, angles))
        return output

    def backward(
        self,
        input_data: ArrayLike | Sequence[ArrayLike] | None,
        weights: ArrayLike,
    ) -> tuple[None, np.ndarray]:
        """Return the gradients with respect to the inputs and the weights.

        The input gradient is not computed and is ``None``. The weight
        gradient has shape
        ``(n, *output_shape, num_weights)``; it is exact on
        ``"statevector"`` and estimated from shots elsewhere.

        Args:
            input_data: As for :meth:`forward`.
            weights: The ``num_weights`` weights.
        """
        theta = self._weights(weights)
        samples = self._samples(input_data)
        gradient = np.zeros(
            (len(samples), *self.output_shape, self.num_weights)
        )
        for i, (key, values) in enumerate(samples):
            values.update(zip(self._weight_params, theta, strict=True))
            for template, readout in self._templates_for(key):
                angles = template.angles(values)
                jacobian = template.jacobian(values, self._weight_params)
                for k in np.flatnonzero(np.any(jacobian, axis=1)):
                    step = np.zeros_like(angles)
                    step[k] = SHIFT
                    plus = readout(self._run(template, angles + step))
                    minus = readout(self._run(template, angles - step))
                    gradient[i] += np.outer((plus - minus) / 2, jacobian[k])
        return None, gradient

    def _run(self, template: Template, angles: np.ndarray) -> dict:
        return probabilities(template, angles, self._backend, self._shots)

    def _weights(self, weights: ArrayLike) -> np.ndarray:
        theta = np.asarray(weights, dtype=float)
        if theta.shape != (self.num_weights,):
            raise ValueError(
                f"Expected {self.num_weights} weights, got shape {theta.shape}"
            )
        if not np.all(np.isfinite(theta)):
            raise ValueError("Weights must be finite")
        return theta

    def _samples(
        self, input_data: ArrayLike | Sequence[ArrayLike] | None
    ) -> list[tuple[Any, dict[Parameter, float]]]:
        """Return a template key and the input values of each sample."""
        if self._qir is not None:
            if input_data is None:
                raise ValueError("A network with a QIR needs input images")
            images = [np.asarray(image) for image in input_data]
            return [(_image_key(image), {}) for image in images]
        if input_data is None:
            if self.num_inputs:
                raise ValueError(
                    f"Expected {self.num_inputs} input features, got None"
                )
            return [(None, {})]
        features = np.asarray(input_data, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.num_inputs:
            raise ValueError(
                f"Expected input of shape (n, {self.num_inputs}), got "
                f"{features.shape}"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("Input features must be finite")
        return [
            (None, dict(zip(self._input_params, row, strict=True)))
            for row in features
        ]

    def _templates_for(self, key: Any) -> list[tuple[Template, Readout]]:
        if key in self._templates:
            self._templates.move_to_end(key)
        else:
            base = self._base_circuit(key)
            templates = []
            for rotation, readout in self._readouts():
                circuit = base.copy()
                if rotation is not None:
                    circuit.compose(rotation, inplace=True)
                templates.append(
                    (Template(circuit, self._fast_export), readout)
                )
            self._templates[key] = templates
            if len(self._templates) > TEMPLATE_CACHE:
                self._templates.popitem(last=False)
        return self._templates[key]

    def _base_circuit(self, key: Any) -> QuantumCircuit:
        """Return the input encoding followed by the network's circuit."""
        n = self._circuit.num_qubits
        if key is None:
            base = QuantumCircuit(QuantumRegister(n, FEATURE_REGISTER))
        else:
            shape, dtype, data = key
            image = np.frombuffer(data, dtype=dtype).reshape(shape)
            base = encode_circuit(image, self._qir).circuit.copy()
            if base.num_qubits != n:
                raise ValueError(
                    f"The QIR encodes an image of shape {shape} "
                    f"on {base.num_qubits} qubits; the circuit has {n}"
                )
        base.compose(self._circuit, qubits=range(n), inplace=True)
        return base


class SamplerQNN(NeuralNetwork):
    r"""A network whose outputs are probabilities of measurement outcomes.

    The outcome :math:`x = \sum_j b_j 2^j` of measuring every qubit
    (qubit ``j`` gives bit ``b_j``) is mapped by ``interpret`` to an output
    index; output :math:`k` is the probability that ``interpret`` returns
    :math:`k`. Without ``interpret`` the outputs are the probabilities of
    all :math:`2^n` outcomes.

    Args:
        interpret: Outcome -> output index in ``range(output_shape)``.
        output_shape: Number of outputs; required with ``interpret``.
        **network: ``circuit``, ``input_params``, ``weight_params``,
            ``qir``, ``backend`` and ``shots``, as for
            :class:`NeuralNetwork`.

    Raises:
        ValueError: If only one of ``interpret`` and ``output_shape`` is
            given or ``output_shape`` is less than 1.
        TypeError: If ``output_shape`` is not an ``int``.
    """

    def __init__(
        self,
        *,
        interpret: Callable[[int], int] | None = None,
        output_shape: int | None = None,
        **network: Any,
    ) -> None:
        super().__init__(**network)
        if (interpret is None) != (output_shape is None):
            raise ValueError(
                "interpret and output_shape must be given together"
            )
        if interpret is None:
            output_shape = 1 << self._circuit.num_qubits
        if not isinstance(output_shape, int) or isinstance(output_shape, bool):
            raise TypeError(
                "output_shape must be an int, got "
                f"{type(output_shape).__name__}"
            )
        if output_shape < 1:
            raise ValueError("output_shape must be at least 1")
        self._interpret = interpret
        self._output_shape = output_shape

    @property
    def output_shape(self) -> tuple[int]:
        """``(number of outputs,)``."""
        return (self._output_shape,)

    @property
    def interpret(self) -> Callable[[int], int] | None:
        """The map from outcomes to outputs, or ``None`` for identity."""
        return self._interpret

    def _readouts(self) -> list[tuple[QuantumCircuit | None, Readout]]:
        def readout(probabilities: dict[str, float]) -> np.ndarray:
            output = np.zeros(self._output_shape)
            for bits, p in probabilities.items():
                x = outcome(bits)
                k = x if self._interpret is None else self._interpret(x)
                if not 0 <= k < self._output_shape:
                    raise ValueError(
                        f"interpret({x}) = {k} is outside "
                        f"range({self._output_shape})"
                    )
                output[k] += p
            return output

        return [(None, readout)]


class EstimatorQNN(NeuralNetwork):
    r"""A network whose outputs are expectation values of observables.

    Output :math:`k` is :math:`\langle \psi | O_k | \psi \rangle` for the
    Hermitian observable :math:`O_k = \sum_m c_m P_m`, a sum of Pauli
    strings. Backends measure only in the computational basis, so each
    string is measured after rotating every qubit with an :math:`X`
    factor by :math:`H` and every qubit with a :math:`Y` factor by
    :math:`H S^\dagger`; then :math:`\langle P_m \rangle =
    \sum_b p(b) (-1)^{\sum_{j \in \operatorname{supp} P_m} b_j}`. Strings
    that need the same rotations share one circuit.

    Args:
        observables: A ``SparsePauliOp`` or ``Pauli``, or a sequence of
            them; defaults to :math:`Z^{\otimes n}`, as in Qiskit Machine
            Learning.
        **network: ``circuit``, ``input_params``, ``weight_params``,
            ``qir``, ``backend`` and ``shots``, as for
            :class:`NeuralNetwork`.

    Raises:
        TypeError: If an observable is not a ``SparsePauliOp`` or
            ``Pauli``.
        ValueError: If an observable acts on a different number of qubits
            or is not Hermitian.
    """

    def __init__(
        self,
        *,
        observables: SparsePauliOp
        | Pauli
        | Sequence[SparsePauliOp | Pauli]
        | None = None,
        **network: Any,
    ) -> None:
        super().__init__(**network)
        n = self._circuit.num_qubits
        if observables is None:
            observables = SparsePauliOp("Z" * n)
        if isinstance(observables, (SparsePauliOp, Pauli)):
            observables = [observables]
        self._observables = [_observable(o, n) for o in observables]

    @property
    def observables(self) -> list[SparsePauliOp]:
        """The observables, one per output."""
        return list(self._observables)

    @property
    def output_shape(self) -> tuple[int]:
        """``(number of observables,)``."""
        return (len(self._observables),)

    def _readouts(self) -> list[tuple[QuantumCircuit | None, Readout]]:
        n = self._circuit.num_qubits
        groups: dict[tuple[str, ...], list] = {}
        for k, observable in enumerate(self._observables):
            for pauli, coefficient in zip(
                observable.paulis, observable.coeffs, strict=True
            ):
                basis = tuple(
                    "Y" if x and z else "X" if x else "Z"
                    for x, z in zip(pauli.x, pauli.z, strict=True)
                )
                support = np.flatnonzero(pauli.x | pauli.z)
                groups.setdefault(basis, []).append(
                    (k, support, float(coefficient.real))
                )
        return [
            (_rotation(basis, n), self._readout(terms))
            for basis, terms in groups.items()
        ]

    def _readout(self, terms: list) -> Readout:
        def readout(probabilities: dict[str, float]) -> np.ndarray:
            output = np.zeros(len(self._observables))
            for bits, p in probabilities.items():
                b = np.frombuffer(bits.encode(), dtype=np.uint8) - ord("0")
                for k, support, coefficient in terms:
                    sign = -1.0 if b[support].sum() % 2 else 1.0
                    output[k] += coefficient * sign * p
            return output

        return readout


def _observable(observable: Any, num_qubits: int) -> SparsePauliOp:
    if isinstance(observable, Pauli):
        observable = SparsePauliOp(observable)
    if not isinstance(observable, SparsePauliOp):
        raise TypeError(
            "An observable must be a SparsePauliOp or Pauli, got "
            f"{type(observable).__name__}"
        )
    if observable.num_qubits != num_qubits:
        raise ValueError(
            f"An observable acts on {observable.num_qubits} qubits; the "
            f"circuit has {num_qubits}"
        )
    observable = observable.simplify()
    if np.any(np.abs(observable.coeffs.imag) > HERMITIAN_TOLERANCE):
        raise ValueError("An observable must be Hermitian")
    return observable


def _rotation(
    basis: tuple[str, ...], num_qubits: int
) -> QuantumCircuit | None:
    """Return the gates that turn ``basis`` into the computational basis."""
    if set(basis) == {"Z"}:
        return None
    circuit = QuantumCircuit(num_qubits)
    for qubit, axis in enumerate(basis):
        if axis == "Y":
            circuit.sdg(qubit)
        if axis in "XY":
            circuit.h(qubit)
    return circuit


def _image_key(image: np.ndarray) -> tuple:
    """A hashable copy of an image, from which it can be rebuilt."""
    image = np.ascontiguousarray(image)
    return image.shape, image.dtype.str, image.tobytes()
