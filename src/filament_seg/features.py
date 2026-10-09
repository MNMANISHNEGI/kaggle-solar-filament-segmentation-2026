"""Input channels designed to expose fine filament structure (barbs).

    raw        the H-alpha image
    CLAHE      contrast-limited adaptive histogram equalization: amplifies faint local contrast
    black-hat  morphological closing minus image: responds to dark structures thinner than the
               kernel and suppresses large-scale limb darkening

Each channel is standardized with statistics of the *whole* image, so training crops and
full-image inference see identical inputs.
"""

import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))


def clahe(raw):
    return _CLAHE.apply(raw)


def blackhat_kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def channel_stack(raw, cl, kernel):
    """(H, W) uint8 raw and CLAHE images -> (3, H, W) float32 in [0, 1]."""
    bh = cv2.morphologyEx(np.ascontiguousarray(cl), cv2.MORPH_BLACKHAT, kernel)
    return np.stack([raw, cl, bh]).astype(np.float32) / 255.0


def stack_stats(x):
    """Per-channel mean and std of a (C, H, W) array."""
    return x.mean(axis=(1, 2)), x.std(axis=(1, 2)) + 1e-6


def normalize(x, stats):
    mean, std = stats
    return (x - mean[:, None, None]) / std[:, None, None]


def full_image_input(raw, kernel):
    """Normalized 3-channel network input for a whole image."""
    x = channel_stack(raw, clahe(raw), kernel)
    return normalize(x, stack_stats(x))
