"""
Example usage of Open Quantum Computer Vision library.

This script demonstrates how to use both MHRQI and QARQI representations
for quantum image encoding and reconstruction.
"""

import numpy as np
import matplotlib.pyplot as plt
from open_quantum_cv import MHRQI, QARQI, FRQI, NEQR


def create_test_image(size=8):
    """Create a simple test image.
    
    Args:
        size: Image size (default 8x8 for fast statevector simulation)
    """
    # Create a gradient image
    x = np.linspace(0, 255, size)
    y = np.linspace(0, 255, size)
    X, Y = np.meshgrid(x, y)
    image = (X + Y) / 2
    return image.astype(np.uint8)


def main():
    print("Open Quantum Computer Vision - Example Usage")
    print("=" * 50)

    # Create test image
    image = create_test_image(8)  # 8x8 for fast statevector simulation with metrics
    print(f"Test image shape: {image.shape}")
    print(f"Image value range: {image.min()} - {image.max()}")

    # Demonstrate MHRQI (Computational Basis Encoding)
    print("\n--- MHRQI (Computational Basis Encoding) ---")
    mhrqi = MHRQI()
    print(f"MHRQI encoding type: {mhrqi.encoding_type}")
    mhrqi.encode(image)
    print(f"Estimated qubits: {mhrqi._estimate_qubits()}")
    result = mhrqi.simulate(shots=None, use_gpu=False)  # Statevector (fast)
    reconstructed_mhrqi = result.reconstructed_image
    print(f"Reconstruction shape: {reconstructed_mhrqi.shape}")
    print(f"Reconstruction range: {reconstructed_mhrqi.min():.1f} - {reconstructed_mhrqi.max():.1f}")

    # Demonstrate FRQI (Amplitude Encoding)
    print("\n--- FRQI (Amplitude Encoding) ---")
    frqi = FRQI()
    print(f"FRQI encoding type: {frqi.encoding_type}")
    frqi.encode(image)
    print(f"Estimated qubits: {frqi._estimate_qubits()}")
    result = frqi.simulate(shots=None)  # Statevector (fast)
    reconstructed_frqi = result.reconstructed_image
    print(f"Reconstruction shape: {reconstructed_frqi.shape}")
    print(f"Reconstruction range: {reconstructed_frqi.min():.1f} - {reconstructed_frqi.max():.1f}")

    # Demonstrate NEQR (Computational Basis Encoding)
    print("\n--- NEQR (Computational Basis Encoding) ---")
    neqr = NEQR()
    print(f"NEQR encoding type: {neqr.encoding_type}")
    neqr.encode(image)
    print(f"Estimated qubits: {neqr._estimate_qubits()}")
    result = neqr.simulate(shots=None)  # Statevector (fast)
    reconstructed_neqr = result.reconstructed_image
    print(f"Reconstruction shape: {reconstructed_neqr.shape}")
    print(f"Reconstruction range: {reconstructed_neqr.min():.1f} - {reconstructed_neqr.max():.1f}")

    # Demonstrate QARQI (Amplitude Encoding with Qudits)
    print("\n--- QARQI (Amplitude Encoding - QUDIT-BASED) ---")
    qarqi = QARQI()
    print(f"QARQI encoding type: {qarqi.encoding_type}")
    qarqi.encode(image)
    
    # Get complete encoding info
    qarqi_info = qarqi.get_encoding_info()
    qudits_info = qarqi_info['qudits']
    print(f"QARQI Architecture: {qarqi_info['architecture_type']}")
    print(f"  Qudits: {qudits_info['total_qudits']} with dimensions {qudits_info['dimensions']}")
    print(f"  Equiv. Qubits: {qarqi_info['equivalent_qubits']}")
    
    result = qarqi.simulate(shots=None, noise=False)  # Statevector, no noise (fast)
    reconstructed_qarqi = result.reconstructed_image
    print(f"Reconstruction shape: {reconstructed_qarqi.shape}")
    print(f"Reconstruction range: {reconstructed_qarqi.min():.1f} - {reconstructed_qarqi.max():.1f}")

    # Visualization
    print("\n--- Visualization ---")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(reconstructed_mhrqi, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('MHRQI Reconstruction')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.abs(image - reconstructed_mhrqi), cmap='RdYlGn_r')
    axes[0, 2].set_title('MHRQI Error')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(reconstructed_qarqi, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('QARQI Reconstruction')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.abs(image - reconstructed_qarqi), cmap='RdYlGn_r')
    axes[1, 2].set_title('QARQI Error')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('example_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison plot saved as 'example_comparison.png'")

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("All representations encoded and reconstructed images successfully.")
    print("(Using statevector simulation for speed)")
    print("=" * 50)


if __name__ == "__main__":
    main()