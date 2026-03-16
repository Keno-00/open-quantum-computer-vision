"""
MHRQI - Multi-scale Hierarchical Representation of Quantum Images

Adapted from the original MHRQI library for the unified Open Quantum CV framework.

Encoding: Computational Basis Encoded
"""

import itertools
import warnings
from collections import defaultdict

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.library import SetStatevector

from .base import QuantumImageRepresentation, QuantumImageResult


# Helper functions from original MHRQI
def _prepare_controls_on_states(qc: QuantumCircuit, controls, ctrl_states):
    """Flip controls where ctrl_state == 0 so that all required controls are 1."""
    for q, s in zip(controls, ctrl_states):
        if int(s) == 0:
            qc.x(q)


def _restore_controls(qc: QuantumCircuit, controls, ctrl_states):
    for q, s in zip(controls, ctrl_states):
        if int(s) == 0:
            qc.x(q)


class MHRQI(QuantumImageRepresentation):
    """
    Multi-scale Hierarchical Representation of Quantum Images.

    Uses computational basis encoding with hierarchical position qubits
    and intensity stored as basis states.

    Encoding Type: Computational Basis
    """

    def __init__(self, depth=None, bit_depth: int = 8):
        """
        Initialize MHRQI representation.

        Args:
            depth: Hierarchy depth (log2 of image side length) (if None, determined from image in encode)
            bit_depth: Bits for intensity encoding (default 8)
        """
        super().__init__('computational_basis')
        self.depth = depth
        self.bit_depth = bit_depth
        self.hierarchical_coord_matrix = None
        self.denoise_enabled = False

        if depth is not None:
            self._setup_circuit()

    def _setup_circuit(self):
        """Setup the quantum circuit based on depth."""
        # Create registers
        self.pos_regs = []
        for k in range(self.depth):
            self.pos_regs.append(QuantumRegister(1, f"q_y_{k}"))
            self.pos_regs.append(QuantumRegister(1, f"q_x_{k}"))

        self.intensity_reg = QuantumRegister(self.bit_depth, "intensity")
        self.outcome_reg = QuantumRegister(1, "outcome")
        self.work_reg = QuantumRegister(2, "work")

        # Initialize circuit
        self.circuit = QuantumCircuit(
            *self.pos_regs, self.intensity_reg, self.outcome_reg, self.work_reg
        )

        # Place position qubits in uniform superposition
        for reg in self.pos_regs:
            self.circuit.h(reg[0])

    @property
    def qubits_to_measure(self):
        """Returns a list of qubits that should be measured for reconstruction."""
        qubits = []
        for reg in self.pos_regs:
            qubits.append(reg[0])
        qubits.extend(list(self.intensity_reg))
        qubits.extend(list(self.outcome_reg))
        return qubits

    def encode(self, image: np.ndarray) -> None:
        """
        Encode a classical image into the quantum circuit.

        Args:
            image: Input image as numpy array (H, W) or (H, W, C), normalized 0-1
        """
        if self.depth is None:
            N = image.shape[0]
            if N != image.shape[1] or (N & (N - 1)) != 0:
                raise ValueError("Image must be square with power-of-2 size")
            self.depth = int(np.log2(N))
            self._setup_circuit()

        if image.ndim == 3:
            # Convert to grayscale if RGB
            image = np.mean(image, axis=2)

        # Normalize to 0-1 if needed
        if image.max() > 1.0:
            image = image / 255.0

        N = image.shape[0]
        if N != image.shape[1]:
            raise ValueError("Image must be square")

        # Generate hierarchical coordinates
        from ..utils.general import generate_hierarchical_coord_matrix
        self.hierarchical_coord_matrix = generate_hierarchical_coord_matrix(N, d=2)

        # Use lazy upload for efficiency
        self.lazy_upload(self.hierarchical_coord_matrix, image)

    def upload(self, hierarchical_coord_matrix, img):
        """
        Upload intensity values using MHRQI(basis) encoding.

        Args:
            hierarchical_coord_matrix: matrix of position states
            img: normalized image (0-1 range)
        """
        self.hierarchical_coord_matrix = hierarchical_coord_matrix
        controls = [reg[0] for reg in self.pos_regs]
        intensity_qubits = list(self.intensity_reg)
        and_ancilla = self.work_reg[0]

        for vec in hierarchical_coord_matrix:
            ctrl_states = list(vec)
            r, c = self._compose_rc(vec, 2)
            pixel_value = float(img[r, c])
            intensity_int = int(pixel_value * (2**self.bit_depth - 1))
            intensity_bits = format(intensity_int, f"0{self.bit_depth}b")[::-1]

            _prepare_controls_on_states(self.circuit, controls, ctrl_states)

            if len(controls) > 0:
                self.circuit.mcx(controls, and_ancilla)
                for bit_idx, bit_val in enumerate(intensity_bits):
                    if bit_val == "1":
                        self.circuit.cx(and_ancilla, intensity_qubits[bit_idx])
                self.circuit.mcx(controls, and_ancilla)
            else:
                for bit_idx, bit_val in enumerate(intensity_bits):
                    if bit_val == "1":
                        self.circuit.x(intensity_qubits[bit_idx])

            _restore_controls(self.circuit, controls, ctrl_states)
        return self.circuit

    def lazy_upload(self, hierarchical_coord_matrix, image):
        """
        Fast basis upload using direct statevector initialization.
        """
        self.hierarchical_coord_matrix = hierarchical_coord_matrix
        qubit_to_idx = {q: i for i, q in enumerate(self.circuit.qubits)}
        pos_indices = [qubit_to_idx[reg[0]] for reg in self.pos_regs]
        intensity_indices = [qubit_to_idx[q] for q in self.intensity_reg]

        num_qubits = self.circuit.num_qubits
        dim = 2**num_qubits
        state = np.zeros(dim, dtype=complex)

        all_indices = pos_indices + intensity_indices
        is_sequential = all(all_indices[i] == i for i in range(len(all_indices)))

        if is_sequential:
            for hierarchical_coord_vector in hierarchical_coord_matrix:
                p = 0
                for i, val in enumerate(hierarchical_coord_vector):
                    if val:
                        p |= 1 << i

                r, c = self._compose_rc(hierarchical_coord_vector, 2)
                pixel_value = float(image[r, c])
                intensity_int = int(pixel_value * (2**self.bit_depth - 1))

                base_idx = p
                intensity_offset = 0
                for bit_idx in range(self.bit_depth):
                    if (intensity_int >> bit_idx) & 1:
                        intensity_offset |= 1 << (len(pos_indices) + bit_idx)
                state[base_idx + intensity_offset] = 1.0
        else:
            warnings.warn("Qubits not sequential, falling back to gate-based upload.", stacklevel=2)
            return self.upload(hierarchical_coord_matrix, image)

        state_norm = np.linalg.norm(state)
        if state_norm > 0:
            state = state / state_norm

        self.circuit.append(SetStatevector(state), self.circuit.qubits)
        return self.circuit

    def simulate(self, shots: int = None, use_gpu: bool = True, **kwargs) -> QuantumImageResult:
        """
        Simulate the quantum circuit.

        Args:
            shots: Number of measurement shots (None for statevector)
            use_gpu: Whether to use GPU acceleration

        Returns:
            QuantumImageResult with reconstructed image
        """
        if shots is None:
            backend = Aer.get_backend("statevector_simulator", device="GPU" if use_gpu else "CPU")
            transpiled = transpile(self.circuit, backend)
            raw = backend.run(transpiled).result().get_statevector()
        else:
            qc_measure = self.circuit.copy()
            creg = ClassicalRegister(len(self.qubits_to_measure), "c")
            qc_measure.add_register(creg)
            qc_measure.measure(self.qubits_to_measure, creg)

            if use_gpu:
                try:
                    backend = AerSimulator(
                        method="statevector", device="GPU", cuStateVec_enable=True
                    )
                except Exception:
                    backend = Aer.get_backend("qasm_simulator")
            else:
                backend = Aer.get_backend("qasm_simulator")

            transpiled = transpile(qc_measure, backend)
            raw = backend.run(transpiled, shots=shots).result().get_counts()

        # Create result object
        result = MHRQIResult(
            raw, self.hierarchical_coord_matrix, self.bit_depth, self.denoise_enabled
        )

        # Reconstruct image
        reconstructed = result.reconstruct()

        return QuantumImageResult(reconstructed, raw, {})

    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct the image from the last simulation result.

        Returns:
            Reconstructed image
        """
        if self.result is None:
            raise ValueError("No simulation result available. Call simulate() first.")
        return self.result.reconstructed_image

    def apply_denoising(self):
        """
        Apply hierarchical consistency denoising to the circuit.
        """
        # Import here to avoid circular imports
        from ..processing.denoising import apply_hierarchical_denoising
        apply_hierarchical_denoising(
            self.circuit, self.pos_regs, self.intensity_reg, self.outcome_reg
        )
        self.denoise_enabled = True

    def _compose_rc(self, vec, d):
        """Convert hierarchical coordinates to row, col."""
        pos_len = len(vec)
        depth = pos_len // 2
        r, c = 0, 0
        for k in range(depth):
            r += vec[2*k] * (d**k)
            c += vec[2*k+1] * (d**k)
        return r, c

    def _get_encoding_description(self) -> str:
        return ("Multi-scale hierarchical encoding using computational basis states. "
                "Position encoded via hierarchical superposition, intensity as binary basis states.")

    def _estimate_qubits(self) -> int:
        if self.depth is None:
            raise ValueError(
                "Image depth not yet determined. Call encode() first to determine depth from image size, "
                "or specify depth in __init__."
            )
        return 2 * self.depth + self.bit_depth + 1 + 2  # pos + intensity + outcome + work


# Adapted result processing
class MHRQIResult:
    """Adapted from original MHRQIResult for unified interface."""

    def __init__(self, raw_results, hierarchical_coord_matrix, bit_depth=8, denoise=False):
        self.raw_results = raw_results
        self.hierarchical_coord_matrix = hierarchical_coord_matrix
        self.bit_depth = bit_depth
        self.denoise = denoise
        self._bins = None
        self._bias_stats = None

    @property
    def bins(self):
        if self._bins is None:
            if isinstance(self.raw_results, dict):
                self._bins = self._make_bins_counts()
            else:
                self._bins = self._make_bins_sv()
        return self._bins

    def _make_bins_counts(self):
        """Create bins from measurement counts."""
        bins = defaultdict(lambda: defaultdict(int))
        for outcome, count in self.raw_results.items():
            # Parse the outcome string
            parts = outcome.split()
            if len(parts) != len(self.hierarchical_coord_matrix[0]) + self.bit_depth + 1:
                continue

            pos_bits = parts[:len(self.hierarchical_coord_matrix[0])]
            intensity_bits = parts[len(self.hierarchical_coord_matrix[0]):-1]
            outcome_bit = parts[-1]

            # Convert to hierarchical vector
            hier_vec = tuple(int(b) for b in pos_bits)
            intensity_val = sum(int(b) << i for i, b in enumerate(intensity_bits[::-1]))

            bins[hier_vec][intensity_val] += count

        return dict(bins)

    def _make_bins_sv(self):
        """Create bins from statevector."""
        # Simplified version - would need full implementation
        bins = defaultdict(lambda: defaultdict(float))
        # This would require parsing the statevector amplitudes
        # For now, return empty bins
        return dict(bins)

    def reconstruct(self, use_denoising_bias=True):
        """Reconstruct image from bins."""
        pos_len = len(self.hierarchical_coord_matrix[0])
        image_side = int(2 ** (pos_len // 2))
        image = np.zeros((image_side, image_side))

        for hier_vec, intensity_counts in self.bins.items():
            r, c = self._compose_rc(hier_vec, 2)
            total_counts = sum(intensity_counts.values())
            if total_counts > 0:
                avg_intensity = sum(k * v for k, v in intensity_counts.items()) / total_counts
                image[r, c] = avg_intensity / (2**self.bit_depth - 1)

        return image

    def _compose_rc(self, vec, d):
        pos_len = len(vec)
        depth = pos_len // 2
        r, c = 0, 0
        for k in range(depth):
            r += vec[2*k] * (d**k)
            c += vec[2*k+1] * (d**k)
        return r, c