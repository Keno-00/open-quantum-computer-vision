"""Shot-based local simulation: behaves like noiseless hardware."""

from __future__ import annotations

from qiskit.primitives import StatevectorSampler

from openqcv.backends.base import Backend, register_backend
from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.export.qasm import MEASUREMENT_REGISTER, load_circuit


@register_backend
class SimulatorBackend(Backend):
    """Samples measurement outcomes, so results carry shot noise.

    Args:
        seed: Fix the random seed for reproducible results.
    """

    name = "simulator"
    default_shots = 4096

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def run(
        self, program: QasmProgram, shots: int | None = None
    ) -> ExecutionResult:
        """Sample ``shots`` outcomes (``default_shots`` if ``None``).

        Raises:
            ValueError: If ``shots`` is less than 1.
        """
        shots = self.default_shots if shots is None else shots
        if shots < 1:
            raise ValueError("shots must be at least 1")
        job = StatevectorSampler(seed=self.seed).run(
            [load_circuit(program)], shots=shots
        )
        counts = getattr(
            job.result()[0].data, MEASUREMENT_REGISTER
        ).get_counts()
        # Qiskit prints bit j rightmost; bit j of "meas" measured qubit j.
        return ExecutionResult.from_counts(
            self.name, {bits[::-1]: n for bits, n in counts.items()}
        )
