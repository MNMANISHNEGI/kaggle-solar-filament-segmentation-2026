"""COCO run-length encoding in the competition's submission format (counts string only)."""

import numpy as np
from pycocotools import mask as mask_utils

from .config import FULL_SIZE


def encode_rle(mask):
    """Boolean (H, W) mask -> compressed COCO RLE counts string."""
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def decode_rle(counts, size=(FULL_SIZE, FULL_SIZE)):
    """Compressed COCO RLE counts string -> boolean (H, W) mask."""
    return mask_utils.decode({"size": list(size), "counts": counts.encode("ascii")}).astype(bool)
