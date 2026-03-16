"""
Visualization Utilities for Quantum Computer Vision

Adapted from MHRQI's visualization utilities.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_image_comparison(original, reconstructed, titles=("Original", "Reconstructed")):
    """
    Plot two images side by side for comparison.

    Args:
        original: Original image
        reconstructed: Reconstructed image
        titles: Tuple of titles for the images
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(titles[0])
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(titles[1])
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(metrics_dict, title="Metrics Comparison"):
    """
    Plot a bar chart comparing different metrics.

    Args:
        metrics_dict: Dictionary of metric names to values
        title: Plot title
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values)
    plt.title(title)
    plt.ylabel("Value")
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                '.3f', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_error_map(original, reconstructed, title="Error Map"):
    """
    Plot per-pixel error map.

    Args:
        original: Original image
        reconstructed: Reconstructed image
        title: Plot title
    """
    orig = _to_float_array(original)
    recon = _to_float_array(reconstructed)

    error = np.abs(orig - recon)

    plt.figure(figsize=(6, 6))
    im = plt.imshow(error, cmap="RdYlGn_r")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(im, label="Absolute Error")

    plt.tight_layout()
    plt.show()


def _to_float_array(img):
    """Convert image to float array."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return arr.astype(np.float32)
    raise ValueError("img must be 2D or 3D array")