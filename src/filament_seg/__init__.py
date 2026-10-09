"""Instance segmentation of solar filaments in GONG H-alpha observations.

Modules:
    config       experiment configuration (dataclass + YAML loader)
    data         dataset discovery, COCO annotation parsing, caching, PyTorch datasets
    features     input channels (raw, CLAHE, black-hat) and normalization
    model        U-Net construction
    losses       BCE + Dice + clDice
    inference    full-image prediction with dihedral test-time augmentation
    postprocess  probability map -> filament instances
    metrics      Panoptic Quality (PQ = SQ x RQ)
    rle          COCO RLE encoding for submissions
    utils        seeding and device helpers
"""

__version__ = "0.2.0"
