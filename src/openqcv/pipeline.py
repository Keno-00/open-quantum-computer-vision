"""The user-facing workflow: encode -> run -> decode."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from openqcv.backends import BACKENDS, Backend
from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.export.qasm import to_program
from openqcv.representations import REPRESENTATIONS, Representation


def encode(
    image: ArrayLike,
    representation: str | type[Representation] | Representation = "qpie",
    **options: Any,
) -> QasmProgram:
    """Encode an image into a :class:`QasmProgram`.

    Args:
        image: 2-D grayscale or ``(H, W, 3|4)`` colour array.
        representation: A name from :func:`list_representations`, a class or
            an instance.
        **options: Passed to the representation, e.g. ``bit_depth=4`` for NEQR.
    """
    return to_program(
        REPRESENTATIONS.create(representation, **options).encode(image)
    )


def run(
    program: QasmProgram,
    backend: str | type[Backend] | Backend = "statevector",
    shots: int | None = None,
    **options: Any,
) -> ExecutionResult:
    """Run a program on a backend.

    Args:
        program: The program to run.
        backend: A name from :func:`list_backends`, a class or an instance.
        shots: Number of samples. ``None`` means exact (``"statevector"``) or
            the backend's default.
        **options: Passed to the backend, e.g. ``seed=7`` for the simulator.

    Raises:
        ValueError: If the program has more qubits than the backend allows.
    """
    chosen = BACKENDS.create(backend, **options)
    limit = chosen.max_qubits
    if limit is not None and program.num_qubits > limit:
        raise ValueError(
            f"{program.num_qubits} qubits exceeds {chosen.name}'s limit "
            f"of {limit}"
        )
    return chosen.run(program, shots=shots)


def decode(result: ExecutionResult, program: QasmProgram) -> np.ndarray:
    """Reconstruct the image from a result: float64, in the original shape."""
    return REPRESENTATIONS.get(program.representation).decode(result, program)


def list_representations() -> list[str]:
    """Names accepted by :func:`encode`."""
    return REPRESENTATIONS.names()


def list_backends() -> list[str]:
    """Names accepted by :func:`run`."""
    return BACKENDS.names()
