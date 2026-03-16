"""
Base class for Quantum Image Representations

Provides a common interface for different quantum image encoding schemes.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, Tuple


class QuantumImageRepresentation(ABC):
    """
    Abstract base class for quantum image representations.

    This class defines the common interface that all quantum image
    representation algorithms should implement.
    """

    def __init__(self, encoding_type: str):
        """
        Initialize the quantum image representation.

        Args:
            encoding_type: 'amplitude' or 'computational_basis'
        """
        self.encoding_type = encoding_type
        self.circuit = None
        self.result = None

    @abstractmethod
    def encode(self, image: np.ndarray) -> None:
        """
        Encode a classical image into a quantum circuit.

        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
        """
        pass

    @abstractmethod
    def simulate(self, shots: Optional[int] = None, **kwargs) -> 'QuantumImageResult':
        """
        Simulate the quantum circuit and return results.

        Args:
            shots: Number of measurement shots (None for statevector)
            **kwargs: Additional simulation parameters

        Returns:
            QuantumImageResult object with reconstruction and metrics
        """
        pass

    @abstractmethod
    def reconstruct(self) -> np.ndarray:
        """
        Reconstruct the classical image from quantum measurements.

        Returns:
            Reconstructed image as numpy array
        """
        pass

    def get_encoding_info(self) -> dict:
        """
        Get information about the encoding scheme.

        Returns:
            Dictionary with encoding details
        """
        return {
            'type': self.encoding_type,
            'description': self._get_encoding_description(),
            'qubits_required': self._estimate_qubits(),
        }

    @abstractmethod
    def _get_encoding_description(self) -> str:
        """Return a description of the encoding scheme."""
        pass

    @abstractmethod
    def _estimate_qubits(self) -> int:
        """Estimate the number of qubits required."""
        pass


class QuantumImageResult:
    """
    Container for quantum image simulation results.
    """

    def __init__(self, reconstructed_image: np.ndarray, raw_counts: dict = None,
                 metrics: dict = None):
        self.reconstructed_image = reconstructed_image
        self.raw_counts = raw_counts or {}
        self.metrics = metrics or {}

    def compute_metrics(self, reference_image: Optional[np.ndarray] = None) -> dict:
        """
        Compute quality metrics compared to reference image.

        Args:
            reference_image: Ground truth image for comparison

        Returns:
            Dictionary of metrics
        """
        if reference_image is None:
            return self.metrics

        # Import here to avoid circular imports
        from ..utils.metrics import compute_image_metrics
        self.metrics = compute_image_metrics(self.reconstructed_image, reference_image)
        return self.metrics

    def plot_comparison(self, reference_image: Optional[np.ndarray] = None):
        """
        Plot comparison between reconstructed and reference images.
        """
        from ..utils.visualization import plot_image_comparison
        plot_image_comparison(self.reconstructed_image, reference_image)