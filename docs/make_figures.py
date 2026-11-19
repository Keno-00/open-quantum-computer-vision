"""Regenerate the figures used in README.md.

Run from the repository root:  python docs/make_figures.py
"""

from pathlib import Path

import numpy as np

import openqcv as qcv

OUT = Path(__file__).parent / "images"
IMAGE = np.array([[0, 64], [128, 255]], dtype=np.uint8)


def main() -> None:
    """Write every README figure into ``docs/images``."""
    OUT.mkdir(exist_ok=True)
    for representation in qcv.list_representations():
        qcv.draw(qcv.encode(IMAGE, representation)).save(
            OUT / f"{representation}_circuit.png"
        )

    program = qcv.encode(IMAGE, "frqi")
    result = qcv.run(program, "simulator", shots=2000, seed=7)
    qcv.compare(IMAGE, qcv.decode(result, program)).save(
        OUT / "frqi_compare.png"
    )


if __name__ == "__main__":
    main()
