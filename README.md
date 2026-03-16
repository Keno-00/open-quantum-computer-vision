# Open Quantum Computer Vision

A unified library for quantum image representation and processing, inspired by OpenCV but designed for quantum computing.

## Overview

This library unifies various quantum image representation algorithms and provides quantum image processing operations. It combines the best of existing quantum image processing frameworks into a cohesive, easy-to-use package.

## Key Features

- **Unified API**: Consistent interface across different quantum image representations
- **Multiple Encodings**: Support for both amplitude-encoded and computational basis-encoded schemes
- **Quantum Processing**: Built-in quantum image processing operations like denoising
- **Comprehensive Metrics**: Full suite of image quality assessment metrics
- **Inspired by OpenCV**: Familiar API design for classical computer vision users

## Supported Representations

### Amplitude Encoded

- **QARQI**: Quadrant Amplitude Representation of Quantum Images
  - Uses qudits for efficient encoding
  - Polarity-magnitude coordinate system
  - Controlled RY rotations for intensity encoding

- **FRQI**: Flexible Representation of Quantum Images
  - Position in basis states, intensity in amplitudes
  - Supports flexible quantum operations
  - Citation: Le et al. (2011)

### Computational Basis Encoded

- **MHRQI**: Multi-scale Hierarchical Representation of Quantum Images
  - Hierarchical quantum circuits
  - Basis state encoding for intensities
  - Built-in denoising capabilities

- **NEQR**: Novel Enhanced Quantum Representation of digital images
  - Both position and intensity in basis states
  - Supports grayscale and color images
  - Citation: Zhang et al. (2013)

## Installation

```bash
pip install open-quantum-computer-vision
```

Or from source:

```bash
git clone https://github.com/your-repo/open-quantum-computer-vision
cd open-quantum-computer-vision
pip install -e .
```

## Quick Start

```python
import numpy as np
from open_quantum_cv import MHRQI, QARQI

# Load an image
image = np.random.randint(0, 256, (16, 16), dtype=np.uint8)

# Use MHRQI (Computational Basis Encoding)
mhrqi = MHRQI(depth=4)
mhrqi.encode(image)
result = mhrqi.simulate(shots=1000)
reconstructed = result.reconstructed_image

# Use QARQI (Amplitude Encoding)
qarqi = QARQI(d=4)
qarqi.encode(image)
result = qarqi.simulate(shots=1000)
reconstructed = result.reconstructed_image

# Compute metrics
metrics = result.compute_metrics(image)
print(f"MSE: {metrics['mse']:.4f}, SSIM: {metrics['ssim']:.4f}")
```

## Architecture

```
open_quantum_cv/
├── representations/     # Quantum image representations
│   ├── base.py         # Common interface
│   ├── mhrqi.py        # MHRQI implementation
│   └── qarqi.py        # QARQI implementation
├── processing/         # Quantum image processing
│   └── denoising.py    # Hierarchical denoising
├── utils/             # Utilities
│   ├── general.py     # Core utilities
│   ├── metrics.py     # Quality metrics
│   └── visualization.py # Plotting functions
├── benchmarks/        # Benchmarking tools
├── examples/          # Usage examples
└── tests/            # Unit tests
```

## Dependencies

- **Quantum Computing**: Qiskit, Qiskit-Aer, MQT Qudits
- **Image Processing**: OpenCV, scikit-image
- **Metrics**: PIQ, PyPIQE, BRISQUE, scikit-video
- **General**: NumPy, SciPy, Matplotlib, PyTorch

## Contributing

Contributions are welcome! Please see our contributing guidelines.

## License

Apache 2.0

## Citation

If you use this library in your research, please cite the original papers:

- **MHRQI**: Jose, K.S. et al. (2023). Multi-scale Hierarchical Representation of Quantum Images.
- **QARQI**: Jose, K.S. (2023). Quadrant Amplitude Representation of Quantum Images.
- **FRQI**: Le, P.Q., Dong, F. & Hirota, K. A flexible representation of quantum images for polynomial preparation, image compression, and processing operations. Quantum Inf Process 10, 63–84 (2011). <https://doi.org/10.1007/s11128-010-0177-y>
- **NEQR**: Zhang, Y., Lu, K., Gao, Y. et al. NEQR: a novel enhanced quantum representation of digital images. Quantum Inf Process 12, 2833–2860 (2013). <https://doi.org/10.1007/s11128-013-0567-z>

## Acknowledgments

This library unifies and builds upon:

- [MHRQI](https://github.com/Keno-00/MHRQI) by Keno S. Jose
- [QARQI](https://github.com/Keno-00/QARQI) by Keno S. Jose
