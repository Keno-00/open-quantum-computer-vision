r"""Optimizers that are not in SciPy.

An optimizer here is a :data:`~openqcv.qml.models.Minimizer`: it is called
as ``optimizer(fun=..., x0=..., jac=...)`` and returns a
``scipy.optimize.OptimizeResult``, so it is passed to a model as
``optimizer=SPSA(maxiter=100)``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult

CALIBRATION_STEPS = 25
"""Random directions averaged by :meth:`SPSA.calibrate`, as in Qiskit."""


class SPSA:
    r"""Simultaneous perturbation stochastic approximation.

    Each iteration :math:`k = 1, 2, \dots` draws a direction
    :math:`\Delta_k \in \{-1, +1\}^d` with independent, equally likely
    signs, estimates the gradient from two evaluations,

    .. math::

        \hat g_k = \frac{f(x_k + c_k \Delta_k) - f(x_k - c_k \Delta_k)}
            {2 c_k} \, \Delta_k,

    and steps :math:`x_{k+1} = x_k - a_k \hat g_k`, with the gains

    .. math::

        a_k = \frac{a}{(A + k)^{\alpha}}, \qquad
        c_k = \frac{c}{k^{\gamma}},

    :math:`\alpha = 0.602`, :math:`\gamma = 0.101` and :math:`A = 0`, the
    values of Qiskit Machine Learning's ``SPSA``. With ``resamplings``
    :math:`> 1`, each :math:`\hat g_k` is the mean over that many directions.

    Without ``learning_rate`` and ``perturbation``, :math:`c = 0.2` and
    :math:`a` is calibrated as in Qiskit Machine Learning's ``SPSA`` (after
    Kandala et al.): over :data:`CALIBRATION_STEPS` random directions at the
    starting point, :math:`a = (2\pi/10) / \overline{|f(x_0 + c\Delta) -
    f(x_0 - c\Delta)| / (2c)}`, so the first step has magnitude about
    :math:`2\pi/10`; if that :math:`a` is below :math:`10^{-10}` (a flat
    start), :math:`a = 2\pi/10`.

    The defaults reproduce Qiskit Machine Learning's ``SPSA()``; its
    blocking, trust region and second-order options are not implemented.
    The result's ``fun`` is evaluated at the returned point, and ``nfev``
    counts every evaluation, calibration included, whereas Qiskit counts
    neither.

    Args:
        maxiter: Number of iterations.
        learning_rate: :math:`a` for the power series above, or ``None``
            to calibrate it.
        perturbation: :math:`c`; given exactly when ``learning_rate`` is.
        last_avg: Return the mean of the last ``last_avg`` iterates
            (counting the starting point) instead of the last one.
        resamplings: Directions averaged per gradient estimate.
        seed: Seed of the directions, for reproducible runs.

    Raises:
        TypeError: If a count is not an ``int``.
        ValueError: If a count is less than 1, or only one of
            ``learning_rate`` and ``perturbation`` is given.

    References:
        J. C. Spall, "Multivariate stochastic approximation using a
        simultaneous perturbation gradient approximation", IEEE Trans.
        Autom. Control 37, 332-341 (1992). doi:10.1109/9.119632

        A. Kandala, A. Mezzacapo, K. Temme, M. Takita, M. Brink, J. M. Chow
        and J. M. Gambetta, "Hardware-efficient variational quantum
        eigensolver for small molecules and quantum magnets", Nature 549,
        242-246 (2017). doi:10.1038/nature23879
    """

    ALPHA = 0.602
    GAMMA = 0.101
    STABILITY = 0.0
    CALIBRATION_PERTURBATION = 0.2
    TARGET_MAGNITUDE = 2 * np.pi / 10
    MIN_LEARNING_RATE = 1e-10

    def __init__(
        self,
        maxiter: int = 100,
        learning_rate: float | None = None,
        perturbation: float | None = None,
        last_avg: int = 1,
        resamplings: int = 1,
        seed: int | None = None,
    ) -> None:
        for name, value in (
            ("maxiter", maxiter),
            ("last_avg", last_avg),
            ("resamplings", resamplings),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name} must be an int, got {type(value).__name__}"
                )
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if (learning_rate is None) != (perturbation is None):
            raise ValueError(
                "learning_rate and perturbation must be given together"
            )
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.last_avg = last_avg
        self.resamplings = resamplings
        self.seed = seed

    def __call__(
        self,
        fun: Callable[[np.ndarray], float],
        x0: ArrayLike,
        jac: Any = None,
    ) -> OptimizeResult:
        """Minimize ``fun`` from ``x0``; ``jac`` is ignored.

        A new random generator is seeded with ``seed`` on every call, so a
        fit is reproducible on its own.
        """
        rng = np.random.default_rng(self.seed)
        x = np.asarray(x0, dtype=float).copy()
        nfev = 0

        def evaluate(point: np.ndarray) -> float:
            nonlocal nfev
            nfev += 1
            return float(fun(point))

        if self.learning_rate is None:
            a, c = (
                self.calibrate(evaluate, x, rng),
                self.CALIBRATION_PERTURBATION,
            )
        else:
            a, c = float(self.learning_rate), float(self.perturbation)
        learning_rates = _powerseries(a, self.ALPHA, self.STABILITY)
        perturbations = _powerseries(c, self.GAMMA)

        last = deque([x.copy()], maxlen=self.last_avg)
        for _ in range(self.maxiter):
            eps = next(perturbations)
            gradient = np.zeros_like(x)
            for _ in range(self.resamplings):
                delta = _directions(rng, x.size)
                plus = evaluate(x + eps * delta)
                minus = evaluate(x - eps * delta)
                gradient += (plus - minus) / (2 * eps) * delta
            x = x - next(learning_rates) * gradient / self.resamplings
            last.append(x.copy())

        if self.last_avg > 1:
            x = np.mean(np.asarray(last), axis=0)
        value = evaluate(x)
        return OptimizeResult(
            x=x, fun=value, nfev=nfev, nit=self.maxiter, success=True
        )

    @classmethod
    def calibrate(
        cls,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        """Return the learning rate :math:`a` described in the class."""
        c = cls.CALIBRATION_PERTURBATION
        magnitude = 0.0
        for _ in range(CALIBRATION_STEPS):
            delta = _directions(rng, x0.size)
            difference = fun(x0 + c * delta) - fun(x0 - c * delta)
            magnitude += abs(difference / (2 * c))
        magnitude /= CALIBRATION_STEPS
        a = cls.TARGET_MAGNITUDE / magnitude if magnitude else 0.0
        if a < cls.MIN_LEARNING_RATE:
            return cls.TARGET_MAGNITUDE
        return a


def _directions(rng: np.random.Generator, size: int) -> np.ndarray:
    """Return independent, equally likely signs (a Rademacher vector)."""
    return 1.0 - 2.0 * rng.binomial(1, 0.5, size=size)


def _powerseries(
    eta: float, power: float, offset: float = 0.0
) -> Iterator[float]:
    """Yield ``eta / (n + offset) ** power`` for n = 1, 2, ..."""
    n = 1
    while True:
        yield eta / (n + offset) ** power
        n += 1
