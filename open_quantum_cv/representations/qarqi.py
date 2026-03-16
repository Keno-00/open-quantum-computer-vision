"""
QARQI - Quadrant Amplitude Representation of Quantum Images

Adapted from the original QARQI library for the unified Open Quantum CV framework.

Encoding: Amplitude Encoded
"""

import numpy as np
from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister
from mqt.qudits.quantum_circuit.gate import ControlData
from mqt.qudits.simulation import MQTQuditProvider
from mqt.qudits.simulation.noise_tools.noise import NoiseModel

from .base import QuantumImageRepresentation, QuantumImageResult


class QARQI(QuantumImageRepresentation):
    """
    Quantum Architecture for Real-time Qu-DIT Imaging.

    Uses a FIXED qudit architecture with polarity-magnitude coordinate system
    and RY rotations for intensity encoding. The image size is encoded through
    the qudit dimension d, which is set to round(N/2) where N is the image size.

    Architecture (Fixed):
    - polarity_i: 2-level qudit (top/bottom)
    - polarity_q: 2-level qudit (left/right)
    - magnitude_i: d-level qudit (distance from center in i direction)
    - magnitude_q: d-level qudit (distance from center in q direction)
    - intensity: 2-level qudit (probability-encoded intensity)

    Total: 5 qudits (fixed), ~3 + 2*log2(d) equivalent qubits
    
    Encoding Type: Amplitude
    """

    def __init__(self, d=None):
        """
        Initialize QARQI representation.

        Args:
            d: Qudit dimension for magnitude registers (if None, determined from image in encode)
        """
        super().__init__('amplitude')
        self.d = d
        self.qc = QuantumCircuit()
        if d is not None:
            self.registers = self._init_registers()
            self._init_circuit()
        self.pol_mag_matrix = None

    def _init_registers(self):
        polarity_i = QuantumRegister("polarity_i", 1, [2])
        polarity_q = QuantumRegister("polarity_q", 1, [2])
        magnitude_i = QuantumRegister("magnitude_i", 1, [self.d])
        magnitude_q = QuantumRegister("magnitude_q", 1, [self.d])
        intensity = QuantumRegister("intensity", 1, [2])

        return {
            "pol_i": polarity_i,
            "pol_q": polarity_q,
            "mag_i": magnitude_i,
            "mag_q": magnitude_q,
            "intensity": intensity
        }

    def _init_circuit(self):
        for reg in self.registers.values():
            self.qc.append(reg)
        # Place control qudits in uniform superposition
        for i in range(4):
            self.qc.h(i)

    def encode(self, image: np.ndarray) -> None:
        """
        Encode a classical image into the quantum circuit.

        Args:
            image: Input image as numpy array (H, W) or (H, W, C), 0-255 range
        """
        if self.d is None:
            N = image.shape[0]
            if N != image.shape[1]:
                raise ValueError("Image must be square")
            self.d = N // 2
            self.registers = self._init_registers()
            self._init_circuit()

        if image.ndim == 3:
            # Convert to grayscale if RGB
            image = np.mean(image, axis=2)

        N = image.shape[0]
        if N != image.shape[1]:
            raise ValueError("Image must be square")

        # Generate polarity-magnitude matrix
        self.pol_mag_matrix = []
        for r in range(N):
            for c in range(N):
                i, q, i_p, q_p = self._compute_register(N, r, c)
                self.pol_mag_matrix.append((i, q, i_p, q_p))

        # Map intensities to angles
        angle_norm = self._angle_map(image)

        # Upload to circuit
        self.upload_image(N, self.pol_mag_matrix, angle_norm)

    def upload_image(self, N, pol_mag_matrix, angle_norm):
        """
        Upload image intensity using controlled RY rotations.
        """
        controls = [0, 1, 2, 3]  # polarity and magnitude registers
        for coord in pol_mag_matrix:
            ctrl_states = list(coord)
            ctrl = ControlData(controls, ctrl_states)
            r, c = self._compose_rc(N, coord[0], coord[1], coord[2], coord[3])
            theta = angle_norm[r, c]
            # Apply controlled RY rotation on intensity register (index 4)
            self.qc.r(4, [0, 1, theta, 0.0], ctrl)

    def simulate(self, shots: int = 1000, noise: bool = True, **kwargs) -> QuantumImageResult:
        """
        Simulate the quantum circuit.

        Args:
            shots: Number of measurement shots
            noise: Whether to include noise model

        Returns:
            QuantumImageResult with reconstructed image
        """
        provider = MQTQuditProvider()
        backend = provider.get_backend("faketraps2trits", shots=shots)

        nm = None
        if noise or shots > 0:
            nm = NoiseModel()

        job = backend.run(self.qc, shots=shots, noise_model=nm)
        result = job.result()

        counts = result.get_counts()
        statevector = result.get_state_vector()

        # Create result object
        qarqi_result = QARQIResult(counts, self.d, mode='counts')

        # Get probability map and reconstruct
        prob_map = qarqi_result.get_probability_map(eps=0.0)
        reconstructed = self._reconstruct_from_prob_map(prob_map, self.d)

        return QuantumImageResult(reconstructed, counts, {})

    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct the image from the last simulation result.

        Returns:
            Reconstructed image
        """
        if self.result is None:
            raise ValueError("No simulation result available. Call simulate() first.")
        return self.result.reconstructed_image

    def _angle_map(self, img, bit_depth=8):
        """Map image intensity to RY rotation angles."""
        max_val = (1 << bit_depth) - 1
        u = np.clip(img.astype(np.float64) / max_val, 0.0, 1.0)
        theta = 2.0 * np.arcsin(np.sqrt(u))
        return theta

    def _compute_register(self, N, r, c):
        """Map pixel (r, c) to QARQI register states."""
        R = N // 2
        midL, midR = R - 1, R

        i = 1 if r < R else 0          # top half
        q = 1 if c >= R else 0         # right half

        if r < midL: i_p = midL - r
        elif r > midR: i_p = r - midR
        else: i_p = 0

        if c < midL: q_p = midL - c
        elif c > midR: q_p = c - midR
        else: q_p = 0

        return (i, q, i_p, q_p)

    def _compose_rc(self, N, i, q, i_p, q_p):
        """Reconstruct pixel (r, c) from register states."""
        R = N // 2
        midL, midR = R - 1, R

        # reconstruct r
        if i_p == 0:
            r = midL if i == 1 else midR
        else:
            r = (midL - i_p) if i == 1 else (midR + i_p)

        # reconstruct c
        if q_p == 0:
            c = midR if q == 1 else midL
        else:
            c = (midR + q_p) if q == 1 else (midL - q_p)

        return r, c

    def _reconstruct_from_prob_map(self, prob_map, d):
        """Convert probability map to image."""
        N = int(np.sqrt(len(prob_map)))
        image = np.zeros((N, N))

        for key, prob in prob_map.items():
            b0, b1, x0, x1 = key
            r, c = self._compose_rc(N, b0, b1, x0, x1)
            # Convert probability back to intensity
            # prob = sin^2(theta/2) where theta = 2*arcsin(sqrt(u))
            # So u = sin^2(prob) approximately for small prob
            intensity = np.sin(prob * np.pi / 2) ** 2
            image[r, c] = intensity * 255

        return image

    def get_encoding_info(self) -> dict:
        """
        Get information about QARQI's qudit-based encoding.

        Returns:
            Dictionary with encoding details, including actual qudit info and qubit equivalents
        """
        base_info = {
            'type': self.encoding_type,
            'description': self._get_encoding_description(),
            'architecture_type': 'QUDIT-BASED',
        }
        
        if self.d is not None:
            base_info.update({
                'qudits': self._estimate_qudits(),
                'equivalent_qubits': self._estimate_qubits(),
            })
        else:
            base_info['note'] = 'Call encode() first to determine qudit configuration'
        
        return base_info

    def _get_encoding_description(self) -> str:
        return ("Amplitude encoding using polarity-magnitude coordinates. "
                "Pixel intensities encoded as RY rotation angles on amplitude states.")

    def _estimate_qudits(self) -> dict:
        """
        Get the qudit configuration for QARQI's fixed architecture.
        
        QARQI uses exactly 5 qudits with fixed dimensions:
        - polarity_i: 2-level qudit (top/bottom)
        - polarity_q: 2-level qudit (left/right)
        - magnitude_i: d-level qudit
        - magnitude_q: d-level qudit
        - intensity: 2-level qudit
        
        Returns:
            Dictionary with qudit information
        """
        if self.d is None:
            raise ValueError(
                "Qudit dimension not yet determined. Call encode() first to determine from image, "
                "or specify d in __init__."
            )
        return {
            'total_qudits': 5,
            'dimensions': [2, 2, self.d, self.d, 2],
            'qudit_dimension': self.d,
            'description': f'5 qudits with dimensions [2, 2, {self.d}, {self.d}, 2]'
        }

    def _estimate_qubits(self) -> int:
        """
        Estimate equivalent qubits for QARQI's qudit architecture.
        
        NOTE: QARQI is a qudit-based system. This method returns an approximate
        qubit equivalent for compatibility with the base class interface.
        Use _estimate_qudits() for the actual qudit architecture.
        
        Equivalent qubits: 3 + 2*log2(d)
        - 2 polarity qudits (2-level): 2 equiv qubits
        - 2 magnitude qudits (d-level): 2*log2(d) equiv qubits
        - 1 intensity qudit (2-level): 1 equiv qubit
        """
        if self.d is None:
            raise ValueError(
                "Qudit dimension not yet determined. Call encode() first to determine from image, "
                "or specify d in __init__."
            )
        return 3 + 2 * int(np.ceil(np.log2(self.d)))


class QARQIResult:
    """Adapted QARQI result processing."""

    def __init__(self, data, d, mode='counts'):
        self.data = data
        self.d = d
        self.mode = mode
        self._bins = None

    @property
    def bins(self):
        if self._bins is None:
            self._bins = self._process_data()
        return self._bins

    def _process_data(self):
        from collections import defaultdict
        bins = defaultdict(lambda: {"hit": 0.0, "miss": 0.0, "trials": 0.0})

        if self.mode == 'counts':
            if isinstance(self.data, dict):
                for idx, count in self.data.items():
                    real_idx = int(idx) if isinstance(idx, (int, str)) else idx
                    b0, b1, x0, x1, h = self._decode_index(real_idx, self.d)
                    key = (b0, b1, x0, x1)
                    if h == 1:
                        bins[key]["hit"] += count
                    else:
                        bins[key]["miss"] += count
                    bins[key]["trials"] += count
            elif isinstance(self.data, list):
                for idx in self.data:
                    real_idx = int(idx)
                    b0, b1, x0, x1, h = self._decode_index(real_idx, self.d)
                    key = (b0, b1, x0, x1)
                    if h == 1:
                        bins[key]["hit"] += 1
                    else:
                        bins[key]["miss"] += 1
                    bins[key]["trials"] += 1
        elif self.mode == 'statevector':
            probs = np.abs(self.data) ** 2
            for idx, prob in enumerate(probs):
                if prob > 1e-15:
                    b0, b1, x0, x1, h = self._decode_index(idx, self.d)
                    key = (b0, b1, x0, x1)
                    if h == 1:
                        bins[key]["hit"] += prob
                    else:
                        bins[key]["miss"] += prob
                    bins[key]["trials"] += prob

        return bins

    def get_probability_map(self, eps=0.0):
        """Calculate probability map."""
        prob_map = {}
        for key, v in self.bins.items():
            t = v["trials"]
            prob_map[key] = (v["hit"] + eps) / (t + 2*eps) if t > 0 else 0.0
        return prob_map

    def _decode_index(self, index, d):
        """Decode simulation index to register values."""
        idx = index
        h = idx % 2; idx //= 2
        x1 = idx % d; idx //= d
        x0 = idx % d; idx //= d
        b1 = idx % 2; idx //= 2
        b0 = idx
        return b0, b1, x0, x1, h