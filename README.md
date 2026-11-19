<a id="readme-top"></a>

<div align="center">

<h3 align="center">openqcv</h3>

  <p align="center">
    Quantum image representations: encode, export to OpenQASM, run, decode, draw.
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#goals">Goals</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

<img src="docs/images/neqr_circuit.png" width="60%" alt="NEQR circuit for a 2x2 image">

openqcv encodes images into quantum circuits (QPIE, FRQI, NEQR), exports them
as OpenQASM 2/3 with the metadata needed to decode, runs them on a backend and
decodes the result back into an image. Circuits are drawn as black-and-white,
textbook-style figures.

### Goals

1. Correct: results are verified, never assumed.
2. Transparent: everything needed to decode is in the program, in the open.
3. Textbook circuits: circuits read and draw as they would in a paper.
4. Backend independent: OpenQASM is the only interface to a backend.
5. Nothing speculative: features are added when specified, not in advance.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python-badge]][Python-url]
* [![Qiskit][Qiskit-badge]][Qiskit-url]
* [![NumPy][NumPy-badge]][NumPy-url]
* [![Matplotlib][Matplotlib-badge]][Matplotlib-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

* Python 3.10 or newer

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Keno-00/open-quantum-computer-vision.git
   ```
2. Install it (editable)
   ```sh
   pip install -e open-quantum-computer-vision
   ```
3. Optional: Pillow, for `qcv.load_image`
   ```sh
   pip install -e "open-quantum-computer-vision[image]"
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

```python
import numpy as np
import openqcv as qcv

image = np.array([[0, 64], [128, 255]], dtype=np.uint8)

program = qcv.encode(image, "frqi")  # or "qpie", "neqr"
result = qcv.run(program, "simulator", shots=8192)  # or "statevector"
decoded = qcv.decode(result, program)

print(program.qasm())  # OpenQASM 3; program.qasm(2) for OpenQASM 2
qcv.draw(program).save("frqi.png")
qcv.compare(image, decoded).save("compare.png")
```

Programs serialise to JSON and decode without the encoding options:

```python
received = qcv.QasmProgram.from_json(program.to_json())
decoded = qcv.decode(qcv.run(received), received)
```

Each representation's docstring gives its state, circuit, qubit count and
references, e.g. `help(qcv.representations.FRQI)`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] QPIE, FRQI, NEQR
- [x] OpenQASM 2/3 export and textbook logical QASM
- [x] Exact and sampled local backends
- [x] Circuit and image figures
- [ ] Hardware backends
- [ ] More representations

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). AI agents also follow
[AGENTS.md](AGENTS.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

No license has been chosen yet.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

The papers and standards openqcv implements are cited in the docstrings of
the code that implements them.

* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Qiskit-badge]: https://img.shields.io/badge/Qiskit-6929C4?style=for-the-badge&logo=qiskit&logoColor=white
[Qiskit-url]: https://www.ibm.com/quantum/qiskit
[NumPy-badge]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
