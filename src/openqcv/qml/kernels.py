r"""Quantum kernels and the support-vector classifier that uses them.

The fidelity kernel of a feature map :math:`V` is

.. math::

    k(x, y) = |\langle 0 | V(y)^\dagger V(x) | 0 \rangle|^2,

the probability of measuring all zeros after :math:`V(x)` followed by
:math:`V(y)^\dagger`. :math:`V` is a parameterized circuit on classical
features, or a QIR, whose circuit for an image is :math:`V(x)`.

References:
    V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala,
    J. M. Chow and J. M. Gambetta, "Supervised learning with
    quantum-enhanced feature spaces", Nature 567, 209-212 (2019).
    doi:10.1038/s41586-019-0980-2

    M. Schuld and N. Killoran, "Quantum machine learning in feature Hilbert
    spaces", Phys. Rev. Lett. 122, 040504 (2019).
    doi:10.1103/PhysRevLett.122.040504
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector

from openqcv.pipeline import encode_circuit, get_backend
from openqcv.qml.models import accuracy
from openqcv.qml.qnn import FEATURE_REGISTER
from openqcv.qml.template import Template, probabilities

if TYPE_CHECKING:
    from openqcv.backends import Backend
    from openqcv.representations import QIR

DUPLICATES = ("all", "off_diagonal", "none")
"""Values of ``evaluate_duplicates``."""


class FidelityQuantumKernel:
    """The fidelity kernel of a feature map, evaluated on a backend.

    Args:
        feature_map: A circuit whose parameters, in order, take the
            features; or a QIR name, class or instance.
        enforce_psd: Replace a symmetric kernel matrix by the nearest
            positive semidefinite one (in the Frobenius norm), by setting
            its negative eigenvalues to 0. Shot noise can make a sampled
            matrix indefinite, which a support-vector machine cannot use.
        evaluate_duplicates: Which pairs of identical samples are run
            rather than set to 1: ``"all"``; ``"off_diagonal"`` (all but
            the diagonal of a symmetric matrix); or ``"none"``.
        backend: A backend, as :func:`openqcv.run` takes it (e.g.
            ``"aer:statevector"``); created once and used for every circuit.
        shots: Shots per circuit; ``None`` is exact on ``"statevector"``.
        fast_export: Write each circuit's QASM once and fill in the
            angles for every evaluation, instead of exporting the whole
            circuit each time. The programs are identical (see
            :class:`~openqcv.qml.template.Template`); only faster to make.

    Raises:
        ValueError: If ``evaluate_duplicates`` is not one of
            :data:`DUPLICATES`.
    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | str | type[QIR] | QIR,
        enforce_psd: bool = True,
        evaluate_duplicates: str = "off_diagonal",
        backend: str | type[Backend] | Backend = "statevector",
        shots: int | None = None,
        fast_export: bool = False,
    ) -> None:
        if evaluate_duplicates not in DUPLICATES:
            raise ValueError(
                f"evaluate_duplicates must be one of {DUPLICATES}, got "
                f"{evaluate_duplicates!r}"
            )
        if isinstance(feature_map, QuantumCircuit):
            self._overlap: _Overlap | None = _Overlap(feature_map, fast_export)
        else:
            self._overlap = None
        self._feature_map = feature_map
        self._enforce_psd = bool(enforce_psd)
        self._duplicates = evaluate_duplicates
        self._backend = get_backend(backend)
        self._shots = shots
        self._fast_export = bool(fast_export)

    @property
    def feature_map(self) -> QuantumCircuit | str | type[QIR] | QIR:
        """The feature map: a circuit, or a QIR as given."""
        return self._feature_map

    def evaluate(self, x_vec: Any, y_vec: Any | None = None) -> np.ndarray:
        """Return the kernel matrix :math:`K_{ij} = k(x_i, y_j)`.

        Args:
            x_vec: ``(n, num_features)`` features, or ``n`` images for a
                QIR.
            y_vec: As ``x_vec``; ``None`` means ``x_vec``, and gives a
                symmetric matrix.
        """
        xs = self._samples(x_vec)
        symmetric = y_vec is None
        ys = xs if symmetric else self._samples(y_vec)
        if not symmetric and _same(xs, ys):
            symmetric = True
        matrix = np.empty((len(xs), len(ys)))
        for i, x in enumerate(xs):
            for j in range(i if symmetric else 0, len(ys)):
                value = self._entry(x, ys[j], symmetric and i == j)
                matrix[i, j] = value
                if symmetric:
                    matrix[j, i] = value
        if symmetric and self._enforce_psd:
            matrix = _nearest_psd(matrix)
        return matrix

    def _entry(self, x: _Sample, y: _Sample, diagonal: bool) -> float:
        if _equal(x.data, y.data) and (
            self._duplicates == "none"
            or (self._duplicates == "off_diagonal" and diagonal)
        ):
            return 1.0
        if self._overlap is not None:
            template = self._overlap.template
            angles = template.angles(self._overlap.values(x.data, y.data))
        else:
            template = Template(_qir_overlap(x, y), self._fast_export)
            angles = np.zeros(0)
        result = probabilities(template, angles, self._backend, self._shots)
        return result.get("0" * template.num_qubits, 0.0)

    def _samples(self, data: Any) -> list[_Sample]:
        if self._overlap is None:
            images = [np.asarray(image) for image in data]
            return [
                _Sample(
                    image, encode_circuit(image, self._feature_map).circuit
                )
                for image in images
            ]
        features = np.asarray(data, dtype=float)
        width = len(self._overlap.inputs)
        if features.ndim != 2 or features.shape[1] != width:
            raise ValueError(
                f"Expected features of shape (n, {width}), got "
                f"{features.shape}"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("Features must be finite")
        return [_Sample(row) for row in features]


class QSVC:
    """A support-vector classifier with a quantum kernel.

    The kernel matrices are computed by ``quantum_kernel`` and passed to
    scikit-learn's ``SVC`` as precomputed kernels, so the inputs may be
    images. Needs scikit-learn (``pip install "openqcv[qml]"``).

    Args:
        quantum_kernel: The kernel.
        **options: Passed to ``sklearn.svm.SVC``, e.g. ``C=10``.

    Raises:
        TypeError: If ``quantum_kernel`` is not a
            :class:`FidelityQuantumKernel`.
        ValueError: If ``options`` sets ``kernel``.
        ImportError: If scikit-learn is not installed.
    """

    def __init__(
        self, quantum_kernel: FidelityQuantumKernel, **options: Any
    ) -> None:
        try:
            from sklearn.svm import SVC
        except ImportError:
            raise ImportError(
                'QSVC needs scikit-learn: pip install "openqcv[qml]"'
            ) from None
        if not isinstance(quantum_kernel, FidelityQuantumKernel):
            raise TypeError(
                "quantum_kernel must be a FidelityQuantumKernel, got "
                f"{type(quantum_kernel).__name__}"
            )
        if "kernel" in options:
            raise ValueError(
                "The kernel is the quantum kernel; do not pass kernel="
            )
        self._kernel = quantum_kernel
        self.svc = SVC(kernel="precomputed", **options)
        """The fitted ``sklearn.svm.SVC``, on kernel matrices."""
        self._train: Any = None

    @property
    def quantum_kernel(self) -> FidelityQuantumKernel:
        """The kernel."""
        return self._kernel

    def fit(self, x: Any, y: ArrayLike) -> QSVC:
        """Fit to inputs ``x`` (features or images) and labels ``y``."""
        self.svc.fit(self._kernel.evaluate(x), y)
        self._train = x
        return self

    def predict(self, x: Any) -> np.ndarray:
        """Return the predicted labels."""
        return self.svc.predict(self._against_train(x))

    def decision_function(self, x: Any) -> np.ndarray:
        """Return the SVC's decision function."""
        return self.svc.decision_function(self._against_train(x))

    def score(
        self, x: Any, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> float:
        """Return the (weighted) fraction of samples predicted correctly."""
        return accuracy(self.predict(x), y, sample_weight)

    def _against_train(self, x: Any) -> np.ndarray:
        if self._train is None:
            raise ValueError("This QSVC is not fitted; call fit first")
        return self._kernel.evaluate(x, self._train)


@dataclass(frozen=True)
class _Sample:
    """An input: features or an image, and an image's encoded circuit."""

    data: np.ndarray
    circuit: QuantumCircuit | None = None


class _Overlap:
    r"""The circuit :math:`V(y)^\dagger V(x)` of a parameterized map."""

    def __init__(
        self, feature_map: QuantumCircuit, fast_export: bool = False
    ) -> None:
        self.inputs = list(feature_map.parameters)
        self._adjoint_inputs = ParameterVector("_y", len(self.inputs))
        n = feature_map.num_qubits
        circuit = QuantumCircuit(QuantumRegister(n, FEATURE_REGISTER))
        circuit.compose(feature_map, inplace=True)
        renamed = feature_map.assign_parameters(
            dict(zip(self.inputs, self._adjoint_inputs, strict=True))
        )
        circuit.compose(renamed.inverse(), inplace=True)
        self.template = Template(circuit, fast_export)

    def values(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Return the parameter values for the pair ``(x, y)``."""
        return {
            **dict(zip(self.inputs, x, strict=True)),
            **dict(zip(self._adjoint_inputs, y, strict=True)),
        }


def _qir_overlap(x: _Sample, y: _Sample) -> QuantumCircuit:
    if x.circuit.num_qubits != y.circuit.num_qubits:
        raise ValueError(
            f"Two images are encoded on {x.circuit.num_qubits} and "
            f"{y.circuit.num_qubits} qubits; a kernel needs the same number"
        )
    return x.circuit.compose(y.circuit.inverse())


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric ``matrix`` with negative eigenvalues set to 0."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.maximum(eigenvalues, 0.0)
    product = (eigenvectors * clipped) @ eigenvectors.T
    # Rounding makes the product asymmetric by ~1e-16; a kernel must not be.
    return (product + product.T) / 2


def _equal(x: np.ndarray, y: np.ndarray) -> bool:
    return x.shape == y.shape and bool(np.array_equal(x, y))


def _same(xs: Sequence[_Sample], ys: Sequence[_Sample]) -> bool:
    return len(xs) == len(ys) and all(
        _equal(x.data, y.data) for x, y in zip(xs, ys, strict=True)
    )
