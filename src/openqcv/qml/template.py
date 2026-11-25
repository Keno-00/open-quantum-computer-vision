r"""Running a parameterized circuit on any backend.

Backends run only fixed :class:`~openqcv.core.QasmProgram` objects, so a
parameterized circuit is lowered once to the basis gates and then exported
as a new program for each set of parameter values.

After lowering, every parameter sits in the angle :math:`\varphi_k` of a
rotation :math:`R_P(\varphi_k) = e^{-i \varphi_k P / 2}` with
:math:`P \in \{X, Y, Z\}`, as an expression :math:`\varphi_k(\theta)` of
the circuit's parameters :math:`\theta`. For any expectation value
:math:`f` of the final state (an outcome probability is the expectation of
a projector), the parameter-shift rule gives the exact derivative

.. math::

    \frac{\partial f}{\partial \varphi_k}
    = \frac{1}{2}\left[ f\!\left(\varphi_k + \frac{\pi}{2}\right)
    - f\!\left(\varphi_k - \frac{\pi}{2}\right) \right],

and the chain rule
:math:`\partial f / \partial \theta_p = \sum_k (\partial f / \partial
\varphi_k)(\partial \varphi_k / \partial \theta_p)` gives the derivative for
each parameter, including one that appears in several gates.

References:
    K. Mitarai, M. Negoro, M. Kitagawa and K. Fujii, "Quantum circuit
    learning", Phys. Rev. A 98, 032309 (2018).
    doi:10.1103/PhysRevA.98.032309

    M. Schuld, V. Bergholm, C. Gogolin, J. Izaac and N. Killoran,
    "Evaluating analytic gradients on quantum hardware", Phys. Rev. A 99,
    032331 (2019). doi:10.1103/PhysRevA.99.032331
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
from qiskit.circuit.tools import pi_check

from openqcv.core.program import QasmProgram
from openqcv.export import logical
from openqcv.export.qasm import lower, measured
from openqcv.pipeline import run

if TYPE_CHECKING:
    from openqcv.backends import Backend

QML_PROGRAM = "qml"
"""The ``representation`` of programs made here. They hold a model's
circuit, not an image, and are run, not decoded."""

SHIFT = np.pi / 2
"""The parameter shift that is exact for :math:`R_X`, :math:`R_Y` and
:math:`R_Z`."""

_ROTATIONS = {"rx", "ry", "rz"}


MARKER = 1_000_000.5
"""Angle ``k`` is bound to ``MARKER + k`` to find where it is written in the
exported text; no lowered rotation has an angle near a million."""

FAST_EXPORT_CHECKS = (np.pi / 2, -np.pi, 3 * np.pi / 4, 1e-5, -2.5e-12, 0.0)
"""Angles that the exporters write specially (multiples of pi, exponents,
values they round to zero), used with random angles to check fast export."""

_FORMATS: dict[str, tuple[Callable, Callable[[float], str]]] = {
    # Each exporter, and the function it formats a gate parameter with.
    "qasm2": (
        qasm2.dumps,
        lambda angle: pi_check(angle, output="qasm", eps=1e-12),
    ),
    "qasm3": (qasm3.dumps, lambda angle: pi_check(angle, output="qasm")),
    "logical_qasm": (logical.dumps, logical.format_angle),
}


class Template:
    """A circuit lowered once and exported for any parameter values.

    Every qubit ``j`` is measured into bit ``j``. The programs' logical QASM
    is the lowered circuit, since a shifted circuit has no logical form.

    With ``fast_export``, each format's text is written once with a gap at
    every parameterized angle, and :meth:`program` only formats the angles
    into the gaps, with the function the exporter itself uses. The programs
    are byte for byte the ones the exporters write: this is checked when the
    template is built, on random angles and on :data:`FAST_EXPORT_CHECKS`.

    Args:
        circuit: A circuit without classical bits whose qubits all belong
            to named registers; its gates may have unbound parameters.
        fast_export: Fill the angles into text written once instead of
            exporting the whole circuit for every program.

    Raises:
        ValueError: If lowering leaves a parameter outside a rotation
            angle, which the parameter-shift rule cannot differentiate, or
            fast export does not reproduce an exporter's text exactly (then
            use ``fast_export=False``).
    """

    def __init__(
        self, circuit: QuantumCircuit, fast_export: bool = False
    ) -> None:
        source = measured(circuit)
        lowered = lower(source)
        expressions = [
            instruction.operation.params[0]
            for instruction in lowered.data
            if _is_parameterized(instruction.operation.params)
        ]
        slots = ParameterVector("phi", len(expressions))
        rebuilt = QuantumCircuit(*source.qregs, *source.cregs)
        k = 0
        for instruction in lowered.data:
            operation = instruction.operation
            if _is_parameterized(operation.params):
                if operation.name not in _ROTATIONS:
                    raise ValueError(
                        f"A parameter ended up in a {operation.name!r} "
                        "gate; only rx, ry and rz can be differentiated"
                    )
                operation = operation.to_mutable()
                operation.params = [slots[k]]
                k += 1
            rebuilt.append(
                operation,
                [
                    rebuilt.qubits[lowered.find_bit(q).index]
                    for q in instruction.qubits
                ],
                [
                    rebuilt.clbits[lowered.find_bit(c).index]
                    for c in instruction.clbits
                ],
            )
        self._circuit = rebuilt
        self._slots = list(slots)
        self._expressions = expressions
        self.registers = {
            register.name: [rebuilt.find_bit(q).index for q in register]
            for register in rebuilt.qregs
        }
        self.fast_export = bool(fast_export)
        self._pieces: dict[str, list[str]] = {}
        if self.fast_export:
            self._prepare_fast_export()

    @property
    def num_qubits(self) -> int:
        """Number of qubits, all of them measured."""
        return self._circuit.num_qubits

    @property
    def num_angles(self) -> int:
        """Number of rotation angles that depend on parameters."""
        return len(self._slots)

    def angles(self, values: Mapping[Parameter, float]) -> np.ndarray:
        """Return each parameterized angle for the given parameter values."""
        return np.array(
            [_evaluate(e, values) for e in self._expressions], dtype=float
        )

    def jacobian(
        self, values: Mapping[Parameter, float], wrt: Sequence[Parameter]
    ) -> np.ndarray:
        r"""Return :math:`\partial \varphi_k / \partial \theta_p`.

        Rows follow the angles, columns follow ``wrt``.
        """
        column = {parameter: p for p, parameter in enumerate(wrt)}
        jacobian = np.zeros((self.num_angles, len(wrt)))
        for k, expression in enumerate(self._expressions):
            for parameter in expression.parameters:
                if parameter in column:
                    derivative = expression.gradient(parameter)
                    jacobian[k, column[parameter]] = _evaluate(
                        derivative, values
                    )
        return jacobian

    def program(self, angles: np.ndarray) -> QasmProgram:
        """Export the circuit with the given angles as a program."""
        if self.fast_export:
            texts = {field: self._fill(field, angles) for field in _FORMATS}
        else:
            texts = self._export(angles)
        return QasmProgram(
            representation=QML_PROGRAM,
            registers=self.registers,
            **texts,
        )

    def _export(self, angles: np.ndarray) -> dict[str, str]:
        """Each format's text, written by its exporter."""
        bound = self._circuit.assign_parameters(
            dict(zip(self._slots, angles, strict=True))
        )
        return {field: dump(bound) for field, (dump, _) in _FORMATS.items()}

    def _fill(self, field: str, angles: np.ndarray) -> str:
        pieces, (_, write) = self._pieces[field], _FORMATS[field]
        out = [pieces[0]]
        for angle, piece in zip(angles, pieces[1:], strict=True):
            out.append(write(float(angle)))
            out.append(piece)
        return "".join(out)

    def _prepare_fast_export(self) -> None:
        markers = MARKER + np.arange(self.num_angles)
        for field, text in self._export(markers).items():
            write = _FORMATS[field][1]
            pieces, rest = [], text
            for marker in markers:
                token = write(float(marker))
                if text.count(token) != 1:
                    raise ValueError(
                        f"fast_export cannot locate an angle in the {field} "
                        "text; use fast_export=False"
                    )
                before, _, rest = rest.partition(token)
                pieces.append(before)
            pieces.append(rest)
            self._pieces[field] = pieces
        if not self.num_angles:
            # The text is the exporters' own and nothing is filled in.
            return
        rng = np.random.default_rng(len(markers))
        special = np.resize(FAST_EXPORT_CHECKS, self.num_angles)
        for angles in (
            rng.uniform(-4 * np.pi, 4 * np.pi, self.num_angles),
            special,
        ):
            for field, text in self._export(angles).items():
                if self._fill(field, angles) != text:
                    raise ValueError(
                        f"fast_export does not reproduce the {field} "
                        "exporter's text exactly; use fast_export=False"
                    )


def probabilities(
    template: Template,
    angles: np.ndarray,
    backend: str | type[Backend] | Backend,
    shots: int | None,
) -> dict[str, float]:
    """Run the template with ``angles``; return qubit-order probabilities."""
    return run(template.program(angles), backend, shots).probabilities


def outcome(bits: str) -> int:
    r"""Return the integer :math:`\sum_j b_j 2^j` of a qubit-order string."""
    return int(bits[::-1], 2)


def _is_parameterized(params: list) -> bool:
    return any(isinstance(p, ParameterExpression) for p in params)


def _evaluate(
    expression: ParameterExpression | float,
    values: Mapping[Parameter, float],
) -> float:
    if not isinstance(expression, ParameterExpression):
        return float(expression)
    bound = expression.bind({p: values[p] for p in expression.parameters})
    return float(bound)
