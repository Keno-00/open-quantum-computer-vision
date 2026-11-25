"""Qiskit Aer: fast simulators, exact or sampled, ideal or noisy.

Needs ``qiskit-aer`` (``pip install openqcv[aer]``). Provider ``"aer"``
offers Aer's simulation methods as backends, e.g. ``"aer:statevector"``;
provider ``"ibm_fake"`` (:mod:`openqcv.backends.ibm_fake`) runs IBM's
simulated devices on Aer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiskit import transpile

from openqcv.backends.base import Backend, Provider, register_provider
from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.export.qasm import load_circuit

if TYPE_CHECKING:
    from qiskit.providers import BackendV2

EXACT_METHODS = (
    "automatic",
    "statevector",
    "density_matrix",
    "matrix_product_state",
)
"""Methods that give exact outcome probabilities without shots."""

UNSUPPORTED_METHODS = ("unitary", "superop")
"""Methods that compute an operator, not measurement outcomes."""

ZERO = 1e-12
"""Probabilities below this are treated as exactly zero, as in the
statevector backend."""

INSTALL = "pip install openqcv[aer]"


def require_aer() -> Any:
    """Return the qiskit_aer module.

    Raises:
        ImportError: If qiskit-aer is not installed, with the install line.
    """
    try:
        import qiskit_aer
    except ImportError:
        raise ImportError(
            f"This backend needs qiskit-aer: {INSTALL}"
        ) from None
    return qiskit_aer


class AerBackend(Backend):
    """A Qiskit Aer simulator running openqcv programs.

    With ``shots=None`` an ideal simulator with an exact method returns
    the exact outcome probabilities; otherwise ``shots`` samples are drawn
    (``default_shots`` if ``None``).

    Args:
        method: An Aer simulation method, e.g. ``"statevector"``.
        seed: Seed of the sampling and of the transpiler, for
            reproducible results.
        device: ``"CPU"`` or, where Aer was built with it, ``"GPU"``.
        target: A device (Qiskit ``BackendV2``) to simulate instead: its
            noise model, gates and coupling; ``method`` is then Aer's
            choice and every result is sampled.
        **options: Passed to ``AerSimulator``, e.g. ``noise_model``.

    Raises:
        ImportError: If qiskit-aer is not installed.
        ValueError: If ``method`` is not one of Aer's measurement methods.
    """

    name = "aer"
    default_shots = 4096

    def __init__(
        self,
        method: str = "automatic",
        seed: int | None = None,
        device: str = "CPU",
        *,
        target: BackendV2 | None = None,
        **options: Any,
    ) -> None:
        aer = require_aer()
        self.seed = seed
        self._target = target
        if target is not None:
            self._simulator = aer.AerSimulator.from_backend(target)
            self.name = f"aer:{target.name}"
            self.max_qubits = target.num_qubits
            self._exact = False
            return
        self._simulator = aer.AerSimulator(device=device, **options)
        methods = [
            m
            for m in self._simulator.available_methods()
            if m not in UNSUPPORTED_METHODS
        ]
        if method not in methods:
            raise ValueError(
                f"Unknown Aer method {method!r}. Available: "
                f"{', '.join(methods)}"
            )
        self._simulator.set_options(method=method)
        self.name = f"aer:{method}"
        self._exact = (
            method in EXACT_METHODS and options.get("noise_model") is None
        )

    def run(
        self, program: QasmProgram, shots: int | None = None
    ) -> ExecutionResult:
        """Run ``program``: exactly, or by sampling ``shots`` outcomes.

        Raises:
            ValueError: If ``shots`` is less than 1.
        """
        circuit = load_circuit(program)
        if shots is None and self._exact:
            return self._probabilities(circuit)
        shots = self.default_shots if shots is None else shots
        if shots < 1:
            raise ValueError("shots must be at least 1")
        compiled = transpile(
            circuit,
            backend=self._target or self._simulator,
            seed_transpiler=self.seed,
        )
        job = self._simulator.run(
            compiled, shots=shots, seed_simulator=self.seed
        )
        counts = job.result().get_counts()
        # Qiskit prints bit j rightmost; bit j of "meas" measured qubit j.
        return ExecutionResult.from_counts(
            self.name, {bits[::-1]: n for bits, n in counts.items()}
        )

    def _probabilities(self, circuit: Any) -> ExecutionResult:
        circuit.remove_final_measurements()
        circuit.save_probabilities_dict()
        data = self._simulator.run(circuit).result().data()
        n = circuit.num_qubits
        probabilities = {}
        for key, p in data["probabilities"].items():
            index = int(key, 16) if isinstance(key, str) else int(key)
            if p > ZERO:
                # Bit j of the index is qubit j: reverse for qubit order.
                probabilities[format(index, f"0{n}b")[::-1]] = float(p)
        return ExecutionResult(self.name, probabilities)


@register_provider
class AerProvider(Provider):
    """Qiskit Aer's simulation methods as backends.

    ``backend(method, seed=..., device=..., **options)``; options such as
    ``noise_model`` go to ``AerSimulator``.

    Raises:
        ImportError: If qiskit-aer is not installed.
    """

    name = "aer"

    def __init__(self) -> None:
        self._aer = require_aer()

    @classmethod
    def available(cls) -> bool:
        """Return whether qiskit-aer is installed."""
        try:
            require_aer()
        except ImportError:
            return False
        return True

    def backends(self) -> list[str]:
        """Return Aer's methods that measure outcomes."""
        return [
            m
            for m in self._aer.AerSimulator().available_methods()
            if m not in UNSUPPORTED_METHODS
        ]

    def backend(self, name: str, **options: Any) -> AerBackend:
        """Return the Aer simulator using method ``name``."""
        return AerBackend(method=name, **options)
