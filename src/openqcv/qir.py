"""Writing your own quantum image representation (QIR).

A QIR is a subclass of :class:`~openqcv.representations.base.QIR` with a
``name``, an ``encode`` method that returns an
:class:`~openqcv.representations.base.EncodedImage`, and a ``decode``
classmethod, registered with :func:`~openqcv.representations.base.register`.
It chooses its own registers, metadata and readout. Once registered, it is
used like the built-ins (``openqcv.encode(image, "graycode")``), and
:func:`~openqcv.pipeline.check` verifies that it works with the rest of
openqcv::

    import numpy as np
    from openqcv import qir

    @qir.register
    class GrayCode(qir.QIR):
        '''Row and column registers; 4-bit intensities in Gray code.'''

        name = "graycode"

        def encode(self, image):
            img = np.asarray(image, dtype=int)
            h, w = img.shape
            c = qir.Circuit(
                row=qir.index_qubits(h), col=qir.index_qubits(w), value=4
            )
            c.h("row", "col")
            for (y, x), v in np.ndenumerate(img):
                with c.block(f"pixel_{y}_{x}"):
                    gray = v ^ (v >> 1)
                    c.x(c.bits("value", gray), when={"row": y, "col": x})
            return c.finish(shape=[h, w])

        @classmethod
        def decode(cls, result, program):
            h, w = program.meta["shape"]
            image = np.zeros((h, w))
            joint = qir.marginal(result, program, "row", "col", "value")
            for (y, x, gray), p in joint.items():
                if y < h and x < w and p > 0:
                    value = gray
                    while gray := gray >> 1:
                        value ^= gray
                    image[y, x] = value
            return image

    qir.check(GrayCode(), np.array([[3, 9], [15, 0]]))

:class:`~openqcv.representations.builder.Circuit` is optional; ``encode``
may build a Qiskit ``QuantumCircuit`` directly. Whichever is used, the
circuit must consist of standard gates, optionally controlled, grouped into
:class:`~openqcv.core.Block` gates: logical QASM and the drawings cannot
show opaque gates such as ``StatePreparation``.

A received program is decoded by looking up its QIR by name, so the
receiver needs the QIR's class: imported in their code, or installed as a
package that declares it under the ``openqcv.representations`` entry point
group (see :class:`~openqcv.core.Registry`).
"""

from openqcv.image.preprocess import index_qubits
from openqcv.pipeline import EXACT_TOLERANCE, check
from openqcv.representations.base import (
    QIR,
    EncodedImage,
    controlled,
    image_shape,
    marginal,
    register,
)
from openqcv.representations.builder import Circuit

__all__ = [
    "EXACT_TOLERANCE",
    "QIR",
    "Circuit",
    "EncodedImage",
    "check",
    "controlled",
    "image_shape",
    "index_qubits",
    "marginal",
    "register",
]
