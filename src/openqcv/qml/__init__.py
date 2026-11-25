"""Quantum machine learning on openqcv backends.

The API follows Qiskit Machine Learning: networks
(:class:`~openqcv.qml.qnn.SamplerQNN`, :class:`~openqcv.qml.qnn.EstimatorQNN`),
models that train them (:class:`~openqcv.qml.models.VQC`,
:class:`~openqcv.qml.models.VQR`, and the general
:class:`~openqcv.qml.models.NeuralNetworkClassifier` and
:class:`~openqcv.qml.models.NeuralNetworkRegressor`) and kernel methods
(:class:`~openqcv.qml.kernels.FidelityQuantumKernel`,
:class:`~openqcv.qml.kernels.QSVC`), and the optimizer
:class:`~openqcv.qml.optimizers.SPSA`. Two things differ:

* A feature map may be a QIR, so a model learns from images encoded by any
  registered QIR, built-in or user-defined::

      from openqcv import qml

      vqc = qml.VQC(feature_map="frqi", optimizer="COBYLA")
      vqc.fit(images, labels)

* Every circuit runs on an openqcv backend as an ordinary
  :class:`~openqcv.core.QasmProgram`, chosen with ``backend=`` and
  ``shots=`` instead of a Qiskit primitive.

Every evaluation exports the full circuit, so on a backend such as
``"aer:statevector"`` (see :func:`openqcv.list_providers`) training runs
exactly what the backend receives. ``fast_export=True`` on the networks,
models and kernel makes those programs faster: each circuit's QASM is
written once and only the angles are filled in. The programs are byte for
byte the exporters' own, which every template checks when it is built
(:class:`~openqcv.qml.template.Template`), so results do not change.

:class:`~openqcv.qml.kernels.QSVC` needs scikit-learn
(``pip install openqcv[qml]``).
"""

from openqcv.qml.kernels import QSVC, FidelityQuantumKernel
from openqcv.qml.losses import CrossEntropyLoss, L1Loss, L2Loss, Loss
from openqcv.qml.models import (
    VQC,
    VQR,
    NeuralNetworkClassifier,
    NeuralNetworkRegressor,
    TrainableModel,
)
from openqcv.qml.optimizers import SPSA
from openqcv.qml.qnn import EstimatorQNN, NeuralNetwork, SamplerQNN

__all__ = [
    "QSVC",
    "SPSA",
    "VQC",
    "VQR",
    "CrossEntropyLoss",
    "EstimatorQNN",
    "FidelityQuantumKernel",
    "L1Loss",
    "L2Loss",
    "Loss",
    "NeuralNetwork",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
    "SamplerQNN",
    "TrainableModel",
]
