"""
NEQR - Novel Enhanced Quantum Representation of digital images

Adapted for the unified Open Quantum CV framework.

NEQR is a basis-encoded quantum image representation that stores
both position and intensity information in computational basis states,
supporting both grayscale and color images.

Citation:
Zhang, Y., Lu, K., Gao, Y. et al. NEQR: a novel enhanced quantum
representation of digital images. Quantum Inf Process 12, 2833–2860 (2013).
https://doi.org/10.1007/s11128-013-0567-z

Encoding Type: Computational Basis
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetStatevector

from .base import QuantumImageRepresentation, QuantumImageResult


class NEQR(QuantumImageRepresentation):
    """
    Novel Enhanced Quantum Representation of digital images.

    Encodes both position and intensity in computational basis states.
    Supports grayscale and color images.

    Encoding Type: Computational Basis
    """

    def __init__(self, n=None, bit_depth: int = 8, color: bool = False):
        """
        Initialize NEQR representation.

        Args:
            n: Image size is 2^n x 2^n (if None, determined from image in encode)
            bit_depth: Bits per color channel (default 8)
            color: Whether to encode color image (RGB)
        """
        super().__init__('computational_basis')
        self.n = n
        self.bit_depth = bit_depth
        self.color = color
        if n is not None:
            self._setup_circuit()

    def _setup_circuit(self):
        """Setup the quantum circuit based on n."""
        self.image_size = 2**self.n
        self.num_position_qubits = 2 * self.n
        self.num_intensity_qubits = self.bit_depth * (3 if self.color else 1)

        # Registers
        self.position_reg = QuantumRegister(self.num_position_qubits, "position")
        self.intensity_reg = QuantumRegister(self.num_intensity_qubits, "intensity")

        self.circuit = QuantumCircuit(self.position_reg, self.intensity_reg)

        # Initialize position qubits in superposition
        for qubit in self.position_reg:
            self.circuit.h(qubit)

    def encode(self, image: np.ndarray) -> None:
        """
        Encode a classical image into the quantum circuit.

        Args:
            image: Input image as numpy array (H, W) or (H, W, 3) for color
        """
        if self.n is None:
            N = image.shape[0]
            if N != image.shape[1] or (N & (N - 1)) != 0:
                raise ValueError("Image must be square with power-of-2 size")
            self.n = int(np.log2(N))
            self._setup_circuit()

        # Normalize and resize
        if self.color and image.ndim == 3:
            # Color image
            if image.max() > 1.0:
                image = image / 255.0
            from skimage.transform import resize
            image = resize(image, (self.image_size, self.image_size, 3), mode='constant')
        else:
            # Grayscale
            if image.ndim == 3:
                image = np.mean(image, axis=2)
            if image.max() > 1.0:
                image = image / 255.0
            from skimage.transform import resize
            image = resize(image, (self.image_size, self.image_size), mode='constant')

        # Create the quantum state
        self._prepare_neqr_state(image)

    def _prepare_neqr_state(self, image: np.ndarray):
        """
        Prepare the NEQR quantum state.

        The NEQR state encodes position and intensity in basis states:
        |ψ⟩ = (1/√(2^{2n})) Σ_{i=0}^{2^{2n}-1} |i⟩|f(i)⟩

        where |i⟩ is position, |f(i)⟩ is the binary representation of intensity.
        """
        num_states = 2**(self.num_position_qubits + self.num_intensity_qubits)
        state = np.zeros(num_states, dtype=complex)

        normalization = 1.0 / np.sqrt(2**(2 * self.n))

        for pos in range(2**(2 * self.n)):
            row = pos // self.image_size
            col = pos % self.image_size

            if self.color:
                # RGB channels
                r, g, b = image[row, col]
                intensity_val = (
                    int(r * (2**self.bit_depth - 1)) |
                    (int(g * (2**self.bit_depth - 1)) << self.bit_depth) |
                    (int(b * (2**self.bit_depth - 1)) << (2 * self.bit_depth))
                )
            else:
                # Grayscale
                intensity_val = int(image[row, col] * (2**self.num_intensity_qubits - 1))

            # Convert intensity to binary
            intensity_bits = format(intensity_val, f'0{self.num_intensity_qubits}b')

            # Find the state index
            pos_bits = format(pos, f'0{self.num_position_qubits}b')
            full_bits = pos_bits + intensity_bits
            state_idx = int(full_bits, 2)

            state[state_idx] = normalization

        # Normalize (though it should already be normalized)
        state_norm = np.linalg.norm(state)
        if state_norm > 0:
            state = state / state_norm

        self.circuit.append(SetStatevector(state), self.circuit.qubits)

    def simulate(self, shots: int = None, **kwargs) -> QuantumImageResult:
        """
        Simulate the quantum circuit.

        Args:
            shots: Number of measurement shots

        Returns:
            QuantumImageResult with reconstructed image
        """
        if shots is None:
            backend = AerSimulator(method="statevector")
            transpiled = transpile(self.circuit, backend)
            result = backend.run(transpiled).result()
            statevector = result.get_statevector()
            reconstructed = self._reconstruct_from_statevector(statevector)
            raw_counts = None
        else:
            qc_measure = self.circuit.copy()
            creg = ClassicalRegister(len(self.circuit.qubits), "c")
            qc_measure.add_register(creg)
            qc_measure.measure(self.circuit.qubits, creg)

            backend = AerSimulator()
            transpiled = transpile(qc_measure, backend)
            result = backend.run(transpiled, shots=shots).result()
            raw_counts = result.get_counts()
            reconstructed = self._reconstruct_from_counts(raw_counts, shots)

        return QuantumImageResult(reconstructed, raw_counts, {})

    def _reconstruct_from_statevector(self, statevector):
        """Reconstruct image from statevector."""
        if self.color:
            image = np.zeros((self.image_size, self.image_size, 3))
        else:
            image = np.zeros((self.image_size, self.image_size))

        for pos in range(2**(2 * self.n)):
            row = pos // self.image_size
            col = pos % self.image_size

            for intensity_val in range(2**self.num_intensity_qubits):
                pos_bits = format(pos, f'0{self.num_position_qubits}b')
                intensity_bits = format(intensity_val, f'0{self.num_intensity_qubits}b')
                full_bits = pos_bits + intensity_bits
                state_idx = int(full_bits, 2)

                amplitude = statevector[state_idx]
                prob = np.abs(amplitude)**2

                if prob > 1e-10:  # Significant probability
                    if self.color:
                        # Extract RGB
                        r = (intensity_val & ((1 << self.bit_depth) - 1)) / (2**self.bit_depth - 1)
                        g = ((intensity_val >> self.bit_depth) & ((1 << self.bit_depth) - 1)) / (2**self.bit_depth - 1)
                        b = ((intensity_val >> (2 * self.bit_depth)) & ((1 << self.bit_depth) - 1)) / (2**self.bit_depth - 1)
                        image[row, col] = [r, g, b]
                    else:
                        intensity = intensity_val / (2**self.num_intensity_qubits - 1)
                        image[row, col] = intensity

        if not self.color:
            image = (image * 255).astype(np.uint8)
        else:
            image = (image * 255).astype(np.uint8)

        return image

    def _reconstruct_from_counts(self, counts, shots):
        """Reconstruct image from measurement counts."""
        if self.color:
            image = np.zeros((self.image_size, self.image_size, 3))
        else:
            image = np.zeros((self.image_size, self.image_size))

        for outcome, count in counts.items():
            # Remove spaces from outcome string
            outcome = ''.join(outcome.split())
            # Parse outcome: position_bits + intensity_bits
            pos_bits = outcome[:self.num_position_qubits]
            intensity_bits = outcome[self.num_position_qubits:]

            pos = int(pos_bits, 2)
            row = pos // self.image_size
            col = pos % self.image_size

            intensity_val = int(intensity_bits, 2)

            if self.color:
                r = (intensity_val & ((1 << self.bit_depth) - 1)) / (2**self.bit_depth - 1)
                g = ((intensity_val >> self.bit_depth) & ((1 << self.bit_depth) - 1)) / (2**self.bit_depth - 1)
                b = ((intensity_val >> (2 * self.bit_depth)) & ((1 << self.bit_depth) - 1)) / (2**self.bit_depth - 1)
                image[row, col] = [r * count / shots, g * count / shots, b * count / shots]
            else:
                intensity = intensity_val / (2**self.num_intensity_qubits - 1)
                image[row, col] += intensity * count / shots

        if not self.color:
            image = (image * 255).astype(np.uint8)
        else:
            image = (image * 255).astype(np.uint8)

        return image

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the image (requires prior simulation)."""
        raise NotImplementedError("Call simulate() first to get results")

    def _get_encoding_description(self) -> str:
        return ("Novel Enhanced Quantum Representation: both position and intensity "
                "encoded in computational basis states, supporting color images.")

    def _estimate_qubits(self) -> int:
        if self.n is None:
            raise ValueError(
                "Image size not yet determined. Call encode() first to determine size from image, "
                "or specify n in __init__."
            )
        return self.num_position_qubits + self.num_intensity_qubits