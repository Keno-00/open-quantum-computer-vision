"""openqcv: quantum computer vision.

Typical use::

    import openqcv as qcv

    program = qcv.encode(image, "frqi")      # image -> QASM + metadata
    result = qcv.run(program, "simulator")   # run on any backend
    decoded = qcv.decode(result, program)    # result -> image

    qcv.draw(program).save("frqi.png")       # pretty circuit
"""

from openqcv.core.program import ExecutionResult, QasmProgram
from openqcv.image import load_image
from openqcv.pipeline import (
    decode,
    encode,
    list_backends,
    list_representations,
    run,
)
from openqcv.viz import compare, draw, draw_image

__version__ = "0.1.0"

__all__ = [
    "ExecutionResult",
    "QasmProgram",
    "compare",
    "decode",
    "draw",
    "draw_image",
    "encode",
    "list_backends",
    "list_representations",
    "load_image",
    "run",
]
