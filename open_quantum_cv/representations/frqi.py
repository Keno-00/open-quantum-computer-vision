"""
FRQI - Flexible Representation of Quantum Images

Adapted for the unified Open Quantum CV framework.

FRQI is an amplitude-encoded quantum image representation that stores
position information in basis states and intensity information in amplitudes.

Citation:
Le, P.Q., Dong, F. & Hirota, K. A flexible representation of quantum images
for polynomial preparation, image compression, and processing operations.
Quantum Inf Process 10, 63–84 (2011). https://doi.org/10.1007/s11128-010-0177-y

Encoding Type: Amplitude
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetStatevector

from .base import QuantumImageRepresentation, QuantumImageResult


class FRQI(QuantumImageRepresentation):
    """
    Flexible Representation of Quantum Images.

    Encodes position in computational basis states and intensity in amplitudes.
    Supports grayscale images.

    Encoding Type: Amplitude
    """

    def __init__(self, n=None, bit_depth: int = 8):
        """
        Initialize FRQI representation.

        Args:
            n: Image size is 2^n x 2^n (if None, determined from image in encode)
            bit_depth: Bits for intensity (default 8)
        """
        super().__init__('amplitude')
        self.n = n
        self.bit_depth = bit_depth
        if n is not None:
            self._setup_circuit()

    def _setup_circuit(self):
        """Setup the quantum circuit based on n."""
        self.image_size = 2**self.n
        self.num_position_qubits = 2 * self.n
        self.num_intensity_qubits = self.bit_depth

        # Registers
        self.position_reg = QuantumRegister(self.num_position_qubits, "position")
        self.intensity_reg = QuantumRegister(self.num_intensity_qubits, "intensity")

        self.circuit = QuantumCircuit(self.position_reg, self.intensity_reg)

    def encode(self, image: np.ndarray) -> None:
        """
        Encode a classical image into the quantum circuit.

        Args:
            image: Input image as numpy array (H, W), normalized 0-1 or 0-255
        """
        if self.n is None:
            N = image.shape[0]
            if N != image.shape[1] or (N & (N - 1)) != 0:
                raise ValueError("Image must be square with power-of-2 size")
            self.n = int(np.log2(N))
            self._setup_circuit()

        if image.ndim == 3:
            # Convert to grayscale if RGB
            image = np.mean(image, axis=2)

        # Normalize to 0-1
        if image.max() > 1.0:
            image = image / 255.0

        # Resize if necessary
        if image.shape != (self.image_size, self.image_size):
            from skimage.transform import resize
            image = resize(image, (self.image_size, self.image_size), mode='constant')

        # Create the quantum state
        self._prepare_frqi_state(image)

    def _prepare_frqi_state(self, image: np.ndarray):
        """
        Prepare the FRQI quantum state.

        The FRQI state is:
        |ψ⟩ = (1/√(2^{2n})) Σ_{i=0}^{2^{2n}-1} |i⟩ ⊗ |θ_i⟩

        where |i⟩ is the position basis state, and |θ_i⟩ is the intensity
        encoded as a rotation.
        """
        num_states = 2**(self.num_position_qubits + self.num_intensity_qubits)
        state = np.zeros(num_states, dtype=complex)

        normalization = 1.0 / np.sqrt(2**(2 * self.n))

        for pos in range(2**(2 * self.n)):
            # Convert position index to (row, col)
            row = pos // self.image_size
            col = pos % self.image_size

            intensity = image[row, col]
            theta = 2 * np.arcsin(np.sqrt(intensity))

            # For simplicity, encode intensity as amplitude in the intensity register
            # This is a simplified version; full FRQI has more complex encoding
            intensity_state = np.zeros(2**self.num_intensity_qubits, dtype=complex)
            intensity_state[0] = np.cos(theta / 2)  # |0⟩ amplitude
            intensity_state[1] = np.sin(theta / 2)  # |1⟩ amplitude (for 1-bit intensity)

            # For multi-bit, we'd need more complex encoding
            # For now, assume 1-bit intensity for simplicity
            if self.num_intensity_qubits == 1:
                combined_state = intensity_state
            else:
                # Extend to full intensity register (simplified)
                combined_state = np.zeros(2**self.num_intensity_qubits, dtype=complex)
                combined_state[0] = np.cos(theta / 2)
                combined_state[1] = np.sin(theta / 2)

            # Place in the full state vector
            start_idx = pos * 2**self.num_intensity_qubits
            state[start_idx:start_idx + 2**self.num_intensity_qubits] = normalization * combined_state

        # Normalize
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
            # Reconstruct from statevector
            reconstructed = self._reconstruct_from_statevector(statevector)
        else:
            # For shot-based, measure position and intensity
            qc_measure = self.circuit.copy()
            from qiskit import ClassicalRegister
            creg = ClassicalRegister(len(self.circuit.qubits), "c")
            qc_measure.add_register(creg)
            qc_measure.measure_all()

            backend = AerSimulator()
            transpiled = transpile(qc_measure, backend)
            result = backend.run(transpiled, shots=shots).result()
            counts = result.get_counts()
            reconstructed = self._reconstruct_from_counts(counts, shots)

        return QuantumImageResult(reconstructed, counts if 'counts' in locals() else None, {})

    def _reconstruct_from_statevector(self, statevector):
        """Reconstruct image from statevector."""
        image = np.zeros((self.image_size, self.image_size))

        for pos in range(2**(2 * self.n)):
            row = pos // self.image_size
            col = pos % self.image_size

            start_idx = pos * 2**self.num_intensity_qubits
            intensity_amplitude = np.abs(statevector[start_idx])**2 + np.abs(statevector[start_idx + 1])**2
            intensity = np.sin(np.arccos(np.abs(statevector[start_idx])) * 2)**2

            image[row, col] = intensity * 255

        return image

    def _reconstruct_from_counts(self, counts, shots):
        """Reconstruct image from measurement counts."""
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

            # Simplified intensity reconstruction
            intensity_val = int(intensity_bits, 2) / (2**self.num_intensity_qubits - 1)
            image[row, col] = intensity_val * count / shots * 255

        return image

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the image (requires prior simulation)."""
        raise NotImplementedError("Call simulate() first to get results")

    def _get_encoding_description(self) -> str:
        return ("Flexible Representation of Quantum Images: position in basis states, "
                "intensity encoded in quantum amplitudes for flexible operations.")

    def _estimate_qubits(self) -> int:
        if self.n is None:
            raise ValueError(
                "Image size not yet determined. Call encode() first to determine size from image, "
                "or specify n in __init__."
            )
        return self.num_position_qubits + self.num_intensity_qubits