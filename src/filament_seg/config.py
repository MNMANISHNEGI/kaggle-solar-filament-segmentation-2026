"""Experiment configuration."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

FULL_SIZE = 2048  # every GONG H-alpha image is 2048 x 2048


@dataclass
class Config:
    # Data
    data_root: str = ""             # empty -> auto-locate (/kaggle/input or kagglehub download)
    val_frac: float = 0.15          # fraction of unique images held out for validation
    seed: int = 42

    # Input / sampling
    crop: int = 1024                # training crop size at native resolution
    repeats: int = 2                # random crops per training image per epoch
    fg_crop_prob: float = 0.7       # fraction of crops centred on a filament pixel
    blackhat_k: int = 21            # black-hat kernel: highlights dark structures thinner than this (px)

    # Model / optimisation
    encoder: str = "resnet50"
    batch_size: int = 4
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    cldice_w: float = 0.5           # weight of the clDice (skeleton) loss
    skel_iters: int = 10            # soft-skeleton iterations
    num_workers: int = 2

    # Inference
    tta: int = 8                    # 1, 2 (h-flip) or 8 (dihedral)

    # Outputs
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            values = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**values)

    def to_yaml(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
