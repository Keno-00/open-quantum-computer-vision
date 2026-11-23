"""The public workflow: encode, run and decode; and checking a QIR."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from openqcv.backends import BACKENDS, Backend
from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.export.qasm import to_program
from openqcv.representations import QIR, REPRESENTATIONS, EncodedImage
from openqcv.viz import draw

EXACT_TOLERANCE = 1e-9
"""Largest error :func:`check` accepts on ``"statevector"``, relative to
the largest expected magnitude (or absolute, below 1).

Exact simulation still rounds. Measured over random 8-bit images from
2x2 to 16x16, the built-in QIRs decode within 1.1e-13 of the value range;
this bound leaves four orders of magnitude for larger circuits."""

CHECK_SEED = 7
"""Simulator seed used by :func:`check`, so a check is reproducible."""


def encode(
    image: ArrayLike,
    representation: str | type[QIR] | QIR = "qpie",
    **options: Any,
) -> QasmProgram:
    """Encode an image into a :class:`QasmProgram`.

    Args:
        image: The image, in whatever form the QIR accepts; the built-ins
            take a 2-D grayscale or ``(H, W, 3|4)`` color array.
        representation: A name from :func:`list_representations`, a class or
            an instance.
        **options: Passed to the QIR, e.g. ``bit_depth=4`` for NEQR.

    Raises:
        TypeError: If the QIR's ``encode`` does not return an
            :class:`~openqcv.representations.EncodedImage`.
    """
    qir = REPRESENTATIONS.create(representation, **options)
    encoded = qir.encode(image)
    if not isinstance(encoded, EncodedImage):
        raise TypeError(
            f"{type(qir).__name__}.encode must return an EncodedImage, got "
            f"{type(encoded).__name__}"
        )
    return to_program(encoded, qir.name)


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
    """Reconstruct the image from a result, with the program's QIR."""
    return REPRESENTATIONS.get(program.representation).decode(result, program)


def list_representations() -> list[str]:
    """Return the names accepted by :func:`encode`."""
    return REPRESENTATIONS.names()


def list_backends() -> list[str]:
    """Return the names accepted by :func:`run`."""
    return BACKENDS.names()


def check(
    representation: str | type[QIR] | QIR,
    image: ArrayLike,
    expected: ArrayLike | None = None,
    *,
    shots: int | None = None,
    shot_tolerance: float | None = None,
) -> None:
    """Verify that a QIR works with every part of openqcv.

    The QIR encodes ``image``; the program is then written to JSON and read
    back, drawn, run on ``"statevector"`` and decoded from the restored
    program alone. The decoded image must equal ``expected`` within
    :data:`EXACT_TOLERANCE`. If ``shots`` is given, the program is also
    run on ``"simulator"`` and must decode within ``shot_tolerance``. If
    the QIR implements ``qubits_required``, its prediction must match.

    Encoding errors (an opaque gate, an invalid register, metadata that is
    not JSON) are raised by :func:`encode` as they would be anywhere else.

    Args:
        representation: The QIR: a name, a class or an instance.
        image: The image to encode.
        expected: What decoding must return; defaults to ``image``.
        shots: Shots for the sampled run, or ``None`` to skip it.
        shot_tolerance: Largest absolute error allowed in the sampled run.

    Raises:
        AssertionError: If any check fails; the message says which.
        ValueError: If the QIR is not registered under its name, or only
            one of ``shots`` and ``shot_tolerance`` is given.
    """
    if (shots is None) != (shot_tolerance is None):
        raise ValueError("shots and shot_tolerance must be given together")
    qir = REPRESENTATIONS.create(representation)
    if qir.name.lower() not in REPRESENTATIONS.names() or (
        REPRESENTATIONS.get(qir.name) is not type(qir)
    ):
        raise ValueError(
            f"{type(qir).__name__} is not registered under {qir.name!r}; "
            "decorate it with @openqcv.qir.register"
        )
    target = np.asarray(image if expected is None else expected, dtype=float)

    # Everything below uses only the program as a receiver would get it.
    received = QasmProgram.from_json(encode(image, qir).to_json())
    draw(received)

    scale = max(1.0, float(np.abs(target).max(initial=0.0)))
    _compare(
        decode(run(received, "statevector"), received),
        target,
        EXACT_TOLERANCE * scale,
        "on statevector",
    )

    try:
        predicted = qir.qubits_required(np.shape(image))
    except NotImplementedError:
        predicted = None
    if predicted is not None and predicted != received.num_qubits:
        raise AssertionError(
            f"qubits_required predicts {predicted} qubits; the program has "
            f"{received.num_qubits}"
        )

    if shots is not None:
        sampled = run(received, "simulator", shots, seed=CHECK_SEED)
        _compare(
            decode(sampled, received),
            target,
            shot_tolerance,
            f"with {shots} shots",
        )


def _compare(
    decoded: np.ndarray, expected: np.ndarray, tolerance: float, how: str
) -> None:
    decoded = np.asarray(decoded, dtype=float)
    if decoded.shape != expected.shape:
        raise AssertionError(
            f"Decoding {how} returned shape {decoded.shape}; expected "
            f"{expected.shape}"
        )
    error = float(np.abs(decoded - expected).max(initial=0.0))
    if not error <= tolerance:
        raise AssertionError(
            f"Decoding {how} is off by up to {error:.3g}; the tolerance is "
            f"{tolerance:.3g}"
        )
