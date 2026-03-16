"""
General Utilities for Quantum Image Processing

Combined utilities from MHRQI and QARQI libraries.
"""

import numpy as np
import cv2


def load_image(path, grayscale=True):
    """
    Load an image from file path, similar to cv2.imread.

    Args:
        path: Path to the image file
        grayscale: If True, convert to grayscale; else return RGB

    Returns:
        Numpy array of the image
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def generate_hierarchical_coord_matrix(N, d=2):
    """
    Generate hierarchical coordinate matrix for MHRQI encoding.

    Args:
        N: Image size (N x N)
        d: Dimension (default 2 for qubits)

    Returns:
        List of hierarchical coordinate vectors
    """
    depth = int(np.log2(N))
    coords = []

    for r in range(N):
        for c in range(N):
            vec = []
            for k in range(depth):
                subdiv_size = d ** k
                qy = (r // subdiv_size) % d
                qx = (c // subdiv_size) % d
                vec.extend([qy, qx])
            coords.append(vec)

    return coords


def compose_rc(hierarchical_coord_vector, d=2):
    """
    Convert hierarchical coordinates to (row, col).

    Args:
        hierarchical_coord_vector: List of coordinate values
        d: Dimension

    Returns:
        (r, c) pixel coordinates
    """
    pos_len = len(hierarchical_coord_vector)
    depth = pos_len // 2
    r, c = 0, 0
    for k in range(depth):
        r += hierarchical_coord_vector[2*k] * (d**k)
        c += hierarchical_coord_vector[2*k+1] * (d**k)
    return r, c


def angle_map(img, bit_depth=8):
    """
    Map image intensity to rotation angles for amplitude encoding.

    Args:
        img: Input image (0-255 range)
        bit_depth: Bit depth for normalization

    Returns:
        Array of rotation angles
    """
    max_val = (1 << bit_depth) - 1
    u = np.clip(img.astype(np.float64) / max_val, 0.0, 1.0)
    theta = 2.0 * np.arcsin(np.sqrt(u))
    return theta


def compute_register(N, r, c):
    """
    Map pixel (r, c) to QARQI register states: (i, q, i_p, q_p).

    Args:
        N: Image size
        r, c: Pixel coordinates

    Returns:
        (i, q, i_p, q_p) register states
    """
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


def compose_rc_qarqi(N, i, q, i_p, q_p):
    """
    Reconstruct pixel (r, c) from QARQI register states.

    Args:
        N: Image size
        i, q, i_p, q_p: Register states

    Returns:
        (r, c) pixel coordinates
    """
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


def decode_index(index, d):
    """
    Decode a simulation result index into QARQI register values.

    Args:
        index: Simulation result index
        d: Qudit dimension

    Returns:
        (b0, b1, x0, x1, h) register values
    """
    idx = index
    h = idx % 2; idx //= 2
    x1 = idx % d; idx //= d
    x0 = idx % d; idx //= d
    b1 = idx % 2; idx //= 2
    b0 = idx
    return b0, b1, x0, x1, h


def get_max_depth(N, d=2):
    """Get maximum hierarchy depth for N x N image."""
    return int(np.log(N) / np.log(d))


def get_subdiv_size(k, N, d=2):
    """Get subdivision size at hierarchy level k."""
    return d ** k