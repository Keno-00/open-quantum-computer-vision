"""Exact, noiseless simulation: the reference for checking decoders."""

from __future__ import annotations

from qiskit.quantum_info import Statevector

from openqcv.backends.base import Backend, register_backend
from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.export.qasm import load_circuit

_ZERO = 1e-12  # probabilities below this are treated as exactly zero


@register_backend
class StatevectorBackend(Backend):
    """Exact simulation of the final state, without shot noise."""

    name = "statevector"

    def run(
        self, program: QasmProgram, shots: int | None = None
    ) -> ExecutionResult:
        """Compute the exact outcome probabilities and final state.

        Raises:
            ValueError: If ``shots`` is given; this backend does not sample.
        """
        if shots is not None:
            raise ValueError(
                "The statevector backend is exact and takes no shots; "
                "use backend='simulator' to sample"
            )
        circuit = load_circuit(program)
        circuit.remove_final_measurements()
        state = Statevector(circuit)
        probabilities = {
            bits[::-1]: float(p)  # Qiskit order -> qubit order
            for bits, p in state.probabilities_dict().items()
            if p > _ZERO
        }
        return ExecutionResult(
            self.name, probabilities, statevector=state.data
        )
