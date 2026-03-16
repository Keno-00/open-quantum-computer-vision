"""
Image Quality Metrics for Quantum Computer Vision

Adapted from MHRQI's visualization utilities.
"""

import numpy as np
from skimage.metrics import mean_squared_error, structural_similarity


def compute_image_metrics(reconstructed, reference):
    """
    Compute comprehensive set of image quality metrics.

    Args:
        reconstructed: Reconstructed image
        reference: Reference image

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['mse'] = compute_mse(reference, reconstructed)
    metrics['psnr'] = compute_psnr(reference, reconstructed)
    metrics['ssim'] = compute_ssim(reference, reconstructed)

    # Additional metrics if available
    try:
        metrics['fsim'] = compute_fsim(reference, reconstructed)
    except:
        metrics['fsim'] = None

    try:
        metrics['niqe'] = compute_niqe(reconstructed)
    except:
        metrics['niqe'] = None

    try:
        metrics['brisque'] = compute_brisque(reconstructed)
    except:
        metrics['brisque'] = None

    return metrics


def compute_mse(img_gt, img_test):
    """
    Compute mean squared error between two images.

    Args:
        img_gt: Ground truth image
        img_test: Test image

    Returns:
        MSE value
    """
    gt = _to_float_array(img_gt)
    te = _to_float_array(img_test)
    _check_same_shape(gt, te)
    return mean_squared_error(gt, te)


def compute_psnr(img_gt, img_test, data_range=255.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.

    Args:
        img_gt: Ground truth image
        img_test: Test image
        data_range: Dynamic range of the images

    Returns:
        PSNR in dB. Higher is better.
    """
    mse_val = compute_mse(img_gt, img_test)
    if mse_val == 0:
        return float("inf")
    return 10.0 * np.log10((data_range**2) / mse_val)


def compute_ssim(img_gt, img_test, data_range=255.0):
    """
    Compute Structural Similarity Index (SSIM).

    Args:
        img_gt: Ground truth image
        img_test: Test image
        data_range: Dynamic range of the images

    Returns:
        SSIM score. Higher is better.
    """
    gt = _to_float_array(img_gt)
    te = _to_float_array(img_test)
    _check_same_shape(gt, te)
    return structural_similarity(gt, te, data_range=data_range)


def compute_fsim(img_ref, img_test):
    """
    Compute FSIM (Feature Similarity Index) using piq.

    Args:
        img_ref: Reference image
        img_test: Test image

    Returns:
        FSIM score in [0, 1]. Higher is better.
    """
    try:
        import torch
        import piq

        ref = _to_float_array(img_ref)
        test = _to_float_array(img_test)

        if ref.max() > 1.0:
            ref /= 255.0
        if test.max() > 1.0:
            test /= 255.0

        ref = np.clip(ref, 0.0, 1.0)
        test = np.clip(test, 0.0, 1.0)

        def to_tensor(arr):
            t = torch.from_numpy(arr).float()
            if t.ndim == 2:
                return t.unsqueeze(0).unsqueeze(0)
            elif t.ndim == 3:
                return t.permute(2, 0, 1).unsqueeze(0)
            return t

        ref_t = to_tensor(ref)
        test_t = to_tensor(test)

        score = piq.fsim(ref_t, test_t, data_range=1.0, reduction="none", chromatic=False)
        return float(score.item())
    except Exception:
        return float("nan")


def compute_niqe(img_input):
    """
    Compute NIQE (Natural Image Quality Evaluator).

    Args:
        img_input: Input image

    Returns:
        NIQE score. Lower is better.
    """
    try:
        import skvideo.measure

        img = _to_float_array(img_input)
        if img.max() <= 1.0:
            img_u8 = (img * 255.0).astype(np.uint8)
        else:
            img_u8 = img.astype(np.uint8)

        if img_u8.ndim == 2:
            img_u8 = img_u8[np.newaxis, ...]
        elif img_u8.ndim == 3:
            import cv2
            if img_u8.shape[2] == 3:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)[np.newaxis, ...]
            else:
                img_u8 = img_u8[np.newaxis, ...]

        return float(skvideo.measure.niqe(img_u8)[0])
    except Exception:
        return float("nan")


def compute_brisque(img_input):
    """
    Compute BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator).

    Args:
        img_input: Input image

    Returns:
        BRISQUE score. Lower is better.
    """
    try:
        from brisque import BRISQUE

        img = _to_float_array(img_input)
        if img.max() <= 1.0:
            img_u8 = (img * 255.0).astype(np.uint8)
        else:
            img_u8 = img.astype(np.uint8)

        brisque_model = BRISQUE()
        return float(brisque_model.score(img_u8))
    except Exception:
        return float("nan")


def compute_enl(img, roi=None):
    """
    Compute Equivalent Number of Looks (ENL).

    ENL = mean² / variance. Evaluated on a homogeneous ROI if provided.

    Args:
        img: Input image
        roi: Optional (y, x, h, w) region of interest

    Returns:
        ENL value. Higher is better.
    """
    arr = _to_float_array(img)

    if roi is not None:
        y, x, h, w = roi
        region = arr[y : y + h, x : x + w]
    else:
        region = arr

    mean_val = np.mean(region)
    var_val = np.var(region)

    eps = 1e-10
    if var_val < eps:
        return 10000.0

    enl = (mean_val**2) / var_val
    return float(min(enl, 10000.0))


def _to_float_array(img):
    """Return float32 array of shape (H, W) or (H, W, C)."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return arr.astype(np.float32)
    raise ValueError("img must be 2D or 3D array")


def _check_same_shape(a, b):
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")