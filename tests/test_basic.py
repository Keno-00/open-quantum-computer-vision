"""
Basic tests for Open Quantum Computer Vision library.
"""

import numpy as np
import pytest
from open_quantum_cv import MHRQI, QARQI, FRQI, NEQR


class TestMHRQI:
    """Test MHRQI representation."""

    def test_initialization(self):
        """Test MHRQI initialization."""
        mhrqi = MHRQI(depth=2)
        assert mhrqi.encoding_type == 'computational_basis'
        assert mhrqi.depth == 2
        assert mhrqi.bit_depth == 8

    def test_encoding_info(self):
        """Test encoding information."""
        mhrqi = MHRQI(depth=2)
        info = mhrqi.get_encoding_info()
        assert info['type'] == 'computational_basis'
        assert 'qubits_required' in info

    def test_encode_small_image(self):
        """Test encoding a small image."""
        mhrqi = MHRQI(depth=2)  # For 4x4 image
        image = np.random.randint(0, 256, (4, 4), dtype=np.uint8)

        mhrqi.encode(image)
        assert mhrqi.circuit is not None
        assert mhrqi.hierarchical_coord_matrix is not None


class TestFRQI:
    """Test FRQI representation."""

    def test_initialization(self):
        """Test FRQI initialization."""
        frqi = FRQI(n=2)
        assert frqi.encoding_type == 'amplitude'
        assert frqi.n == 2
        assert frqi.image_size == 4

    def test_encoding_info(self):
        """Test encoding information."""
        frqi = FRQI(n=2)
        info = frqi.get_encoding_info()
        assert info['type'] == 'amplitude'
        assert 'qubits_required' in info

    def test_encode_small_image(self):
        """Test encoding a small image."""
        frqi = FRQI(n=2)  # For 4x4 image
        image = np.random.randint(0, 256, (4, 4), dtype=np.uint8)

        frqi.encode(image)
        assert frqi.circuit is not None


class TestNEQR:
    """Test NEQR representation."""

    def test_initialization(self):
        """Test NEQR initialization."""
        neqr = NEQR(n=2)
        assert neqr.encoding_type == 'computational_basis'
        assert neqr.n == 2
        assert neqr.image_size == 4

    def test_encoding_info(self):
        """Test encoding information."""
        neqr = NEQR(n=2)
        info = neqr.get_encoding_info()
        assert info['type'] == 'computational_basis'
        assert 'qubits_required' in info

    def test_encode_small_image(self):
        """Test encoding a small image."""
        neqr = NEQR(n=2)  # For 4x4 image
        image = np.random.randint(0, 256, (4, 4), dtype=np.uint8)

        neqr.encode(image)
        assert neqr.circuit is not None


class TestQARQI:
    """Test QARQI representation (qudit-based)."""

    def test_initialization(self):
        """Test QARQI initialization."""
        qarqi = QARQI(d=4)
        assert qarqi.encoding_type == 'amplitude'
        assert qarqi.d == 4

    def test_encoding_info(self):
        """Test encoding information for qudit-based QARQI."""
        qarqi = QARQI(d=4)
        info = qarqi.get_encoding_info()
        assert info['type'] == 'amplitude'
        assert info['architecture_type'] == 'QUDIT-BASED'
        assert 'qudits' in info
        assert 'equivalent_qubits' in info
        
        # Verify qudit configuration
        qudits = info['qudits']
        assert qudits['total_qudits'] == 5
        assert qudits['dimensions'] == [2, 2, 4, 4, 2]

    def test_encode_small_image(self):
        """Test encoding a small image."""
        qarqi = QARQI()  # Will use d from image
        image = np.random.randint(0, 256, (8, 8), dtype=np.uint8)

        qarqi.encode(image)
        assert qarqi.qc is not None
        assert qarqi.d == 4  # 8/2 = 4

    def test_qudit_vs_qubit_estimates(self):
        """Test that qudit and qubit estimates are properly distinguished."""
        qarqi = QARQI(d=8)
        
        # Get qudit config
        qudits = qarqi._estimate_qudits()
        assert qudits['total_qudits'] == 5
        assert qudits['qudit_dimension'] == 8
        
        # Get qubit equivalent
        equiv_qubits = qarqi._estimate_qubits()
        assert equiv_qubits == 3 + 2 * 3  # 3 + 2*log2(8) = 9


class TestUtils:
    """Test utility functions."""

    def test_generate_coords(self):
        """Test hierarchical coordinate generation."""
        from open_quantum_cv.utils.general import generate_hierarchical_coord_matrix
        coords = generate_hierarchical_coord_matrix(4, d=2)
        assert len(coords) == 16  # 4x4 image
        assert len(coords[0]) == 4  # depth 2 * 2 dimensions

    def test_angle_map(self):
        """Test angle mapping for amplitude encoding."""
        from open_quantum_cv.utils.general import angle_map
        img = np.array([[0, 127, 255]], dtype=np.uint8)
        angles = angle_map(img)
        assert angles.shape == (1, 3)
        assert angles[0, 0] == 0.0  # 0 intensity -> 0 angle
        assert angles[0, -1] > angles[0, 0]  # Higher intensity -> larger angle


if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")

    # Test MHRQI
    print("Testing MHRQI...")
    mhrqi = MHRQI(depth=2)
    print(f"  Encoding type: {mhrqi.encoding_type}")
    print(f"  Estimated qubits: {mhrqi._estimate_qubits()}")

    # Test FRQI
    print("Testing FRQI...")
    frqi = FRQI(n=2)
    print(f"  Encoding type: {frqi.encoding_type}")
    print(f"  Estimated qubits: {frqi._estimate_qubits()}")

    # Test NEQR
    print("Testing NEQR...")
    neqr = NEQR(n=2)
    print(f"  Encoding type: {neqr.encoding_type}")
    print(f"  Estimated qubits: {neqr._estimate_qubits()}")

    # Test QARQI (qudit-based)
    print("Testing QARQI (QUDIT-BASED)...")
    qarqi = QARQI(d=4)
    print(f"  Encoding type: {qarqi.encoding_type}")
    info = qarqi.get_encoding_info()
    print(f"  Architecture: {info['architecture_type']}")
    qudits = info['qudits']
    print(f"  Qudit config: {qudits['total_qudits']} qudits with dims {qudits['dimensions']}")
    print(f"  Equivalent qubits: {info['equivalent_qubits']}")

    print("Basic tests completed!")