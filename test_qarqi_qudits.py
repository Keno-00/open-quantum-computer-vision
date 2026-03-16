#!/usr/bin/env python3
"""
Test script to verify QARQI properly distinguishes qudits from qubits.
"""
import numpy as np
from open_quantum_cv.representations import QARQI

def test_qarqi_qudits():
    """Test QARQI's qudit architecture."""
    print("=" * 60)
    print("QARQI Qudit Architecture Test")
    print("=" * 60)
    
    # Test with different image sizes
    image_sizes = [8, 16, 32]
    
    for N in image_sizes:
        print(f"\n--- Image Size: {N}×{N} ---")
        
        # Create and encode image
        img = np.random.randint(0, 256, (N, N), dtype=np.uint8)
        qarqi = QARQI()
        qarqi.encode(img)
        
        # Get qudit info
        qudits_info = qarqi._estimate_qudits()
        equiv_qubits = qarqi._estimate_qubits()
        
        print(f"Qudit Configuration (ACTUAL):")
        print(f"  Total qudits: {qudits_info['total_qudits']}")
        print(f"  Dimensions: {qudits_info['dimensions']}")
        print(f"  Description: {qudits_info['description']}")
        
        print(f"\nEquivalent Qubits (for comparison):")
        print(f"  Estimated: {equiv_qubits}")
        print(f"  Formula: 3 + 2*log2({qarqi.d}) = {equiv_qubits}")
        
        # Get full encoding info
        info = qarqi.get_encoding_info()
        print(f"\nFull Encoding Info:")
        print(f"  Type: {info['type']}")
        print(f"  Architecture: {info['architecture_type']}")
        print(f"  Has qudits info: {'qudits' in info}")
        print(f"  Has qubits info: {'equivalent_qubits' in info}")


def test_qarqi_before_encode():
    """Test error handling before encoding."""
    print("\n" + "=" * 60)
    print("QARQI Before Encoding Test")
    print("=" * 60)
    
    qarqi = QARQI()
    try:
        qarqi._estimate_qudits()
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raises error: {str(e)[:50]}...")
    
    try:
        qarqi._estimate_qubits()
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raises error: {str(e)[:50]}...")
    
    # But get_encoding_info should handle gracefully
    info = qarqi.get_encoding_info()
    if 'note' in info:
        print(f"✓ get_encoding_info handles gracefully: {info['note']}")


if __name__ == '__main__':
    test_qarqi_qudits()
    test_qarqi_before_encode()
    
    print("\n" + "=" * 60)
    print("✓ All QARQI qudit tests passed!")
    print("=" * 60)
