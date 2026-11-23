"""Every representation must survive encode, run and decode on each backend."""

import numpy as np
import pytest
from qiskit import qasm2

import openqcv as qcv

# 9 pixels: forces padding to 16 positions, and is asymmetric, so a bit-order
# mistake anywhere shows up as a scrambled image.
IMAGE = np.array(
    [[0, 32, 64], [96, 128, 160], [192, 224, 255]], dtype=np.uint8
)
RANDOM = np.random.default_rng(1).integers(0, 256, (4, 4)).astype(np.uint8)
REPRESENTATIONS = ["qpie", "frqi", "neqr"]


def roundtrip(image, representation, backend="statevector", shots=None):
    program = qcv.encode(image, representation)
    return qcv.decode(qcv.run(program, backend, shots), program)


@pytest.mark.parametrize(
    ("representation", "tolerance"), [("qpie", 8), ("frqi", 12), ("neqr", 0)]
)
@pytest.mark.parametrize("image", [IMAGE, RANDOM], ids=["3x3", "random4x4"])
def test_check(representation, tolerance, image):
    # Exact and JSON round trips, drawing, qubit count and a sampled run.
    qcv.qir.check(
        representation, image, shots=20_000, shot_tolerance=tolerance
    )


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_qasm_is_valid(representation):
    program = qcv.encode(IMAGE, representation)
    qasm2.loads(program.qasm2)
    assert program.qasm3.startswith("OPENQASM 3")


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_drawing(representation):
    program = qcv.encode(IMAGE, representation)
    assert qcv.draw(program)._repr_png_().startswith(b"\x89PNG")
    assert qcv.compare(IMAGE, IMAGE)._repr_png_().startswith(b"\x89PNG")
    assert qcv.draw_image(IMAGE)._repr_png_().startswith(b"\x89PNG")


def test_constant_image():
    flat = np.full((2, 2), 7, dtype=np.uint8)
    for representation in REPRESENTATIONS:
        np.testing.assert_allclose(
            roundtrip(flat, representation), flat, atol=1e-6
        )


def test_rgb_image_is_converted_to_grayscale():
    rgb = np.zeros((2, 2, 3))
    rgb[..., 0] = 100  # pure red -> 0.299 * 100
    decoded = roundtrip(rgb, "qpie")
    np.testing.assert_allclose(decoded, np.full((2, 2), 29.9), atol=1e-6)


def test_gray_rgb_image_is_integer_for_neqr():
    # The luma of (v, v, v) is not exactly v for these values.
    gray = np.array([[1, 2], [13, 255]], dtype=np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    np.testing.assert_array_equal(roundtrip(rgb, "neqr"), gray)
