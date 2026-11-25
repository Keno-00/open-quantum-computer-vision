r"""Loss functions for training a model.

A loss compares a network's outputs :math:`p_i \in \mathbb{R}^K` with
targets :math:`t_i` row by row: called with ``(n, K)`` arrays it returns
the ``n`` losses, and :meth:`Loss.gradient` returns
:math:`\partial \ell_i / \partial p_{ij}`. The names are those of Qiskit
Machine Learning; the gradients are the exact derivatives of the values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

PROBABILITY_FLOOR = 1e-10
r"""Probabilities below this are raised to it in :class:`CrossEntropyLoss`,
so that :math:`\log 0` never occurs; the value is Qiskit Machine
Learning's."""


class Loss(ABC):
    """A per-sample loss and its gradient."""

    name: str

    @abstractmethod
    def __call__(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the loss of each row, shape ``(n,)``."""

    @abstractmethod
    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the derivative of each row's loss, shape ``(n, K)``."""


class L1Loss(Loss):
    r""":math:`\ell_i = \sum_j |p_{ij} - t_{ij}|`.

    Its gradient :math:`\operatorname{sign}(p_{ij} - t_{ij})` takes the
    subgradient 0 where :math:`p_{ij} = t_{ij}`.
    """

    name = "absolute_error"

    def __call__(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the loss of each row, shape ``(n,)``."""
        return np.abs(predict - target).sum(axis=1)

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the derivative of each row's loss, shape ``(n, K)``."""
        return np.sign(predict - target)


class L2Loss(Loss):
    r""":math:`\ell_i = \sum_j (p_{ij} - t_{ij})^2`."""

    name = "squared_error"

    def __call__(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the loss of each row, shape ``(n,)``."""
        return ((predict - target) ** 2).sum(axis=1)

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the derivative of each row's loss, shape ``(n, K)``."""
        return 2 * (predict - target)


class CrossEntropyLoss(Loss):
    r""":math:`\ell_i = -\sum_j t_{ij} \ln \max(p_{ij}, \epsilon)`.

    The outputs :math:`p_i` are probabilities and the targets one-hot
    rows; :math:`\epsilon` is :data:`PROBABILITY_FLOOR`. The gradient is
    :math:`-t_{ij} / p_{ij}` where :math:`p_{ij} > \epsilon` and 0 below,
    where the loss is constant.

    Qiskit Machine Learning uses :math:`\log_2`, which scales the loss by
    :math:`1/\ln 2`, and a gradient that assumes a softmax output layer.
    """

    name = "cross_entropy"

    def __call__(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the loss of each row, shape ``(n,)``."""
        clipped = np.maximum(predict, PROBABILITY_FLOOR)
        return -(target * np.log(clipped)).sum(axis=1)

    def gradient(self, predict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the derivative of each row's loss, shape ``(n, K)``."""
        above = predict > PROBABILITY_FLOOR
        safe = np.where(above, predict, 1.0)
        return np.where(above, -target / safe, 0.0)


LOSSES: dict[str, type[Loss]] = {
    cls.name: cls for cls in (L1Loss, L2Loss, CrossEntropyLoss)
}
"""Loss name -> class, for the ``loss`` argument of the models."""


def get_loss(loss: str | Loss) -> Loss:
    """Return ``loss`` itself or the loss named by it.

    Raises:
        TypeError: If ``loss`` is neither a name nor a :class:`Loss`.
        ValueError: If no loss has that name.
    """
    if isinstance(loss, Loss):
        return loss
    if not isinstance(loss, str):
        raise TypeError(
            f"loss must be a name or a Loss, got {type(loss).__name__}"
        )
    try:
        return LOSSES[loss]()
    except KeyError:
        raise ValueError(
            f"Unknown loss {loss!r}. Available: {', '.join(sorted(LOSSES))}"
        ) from None
