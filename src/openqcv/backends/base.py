"""The contract every backend follows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.core.registry import Registry


class Backend(ABC):
    """Base class for anything that can run a :class:`QasmProgram`.

    To add one: subclass this, set ``name`` (and ``max_qubits`` if the
    device is limited), implement ``run`` and decorate the class with
    ``@register_backend``.

    Rules:
        * Read the circuit from ``program.qasm2`` or ``program.qasm3`` only,
          never from representation objects.
        * Return bitstrings in qubit order (character ``j`` = qubit ``j``).
    """

    name: ClassVar[str]
    max_qubits: ClassVar[int | None] = None

    @abstractmethod
    def run(
        self, program: QasmProgram, shots: int | None = None
    ) -> ExecutionResult:
        """Execute ``program``.

        Args:
            program: The program to run.
            shots: Number of samples; ``None`` means exact or the backend's
                default.
        """


BACKENDS: Registry[Backend] = Registry("backend", Backend)
register_backend = BACKENDS.register
