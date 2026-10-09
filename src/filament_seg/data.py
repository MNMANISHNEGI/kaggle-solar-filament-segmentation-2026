"""Dataset discovery, MAGFiLO (COCO-style) annotation parsing, caching and PyTorch datasets."""

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .config import FULL_SIZE
from .features import channel_stack, clahe, normalize, stack_stats

COMPETITION = "filament-segmentation-2026"
ANNOTATION_FILE = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- discovery

def locate_dataset(root):
    """Return the dataset root (the folder containing train/ and test/) under `root`, or None."""
    hits = list(Path(root).rglob(ANNOTATION_FILE))
    return hits[0].parent.parent if hits else None


def find_data_root(data_root=""):
    """Explicit path, else an attached Kaggle input, else download with kagglehub."""
    if data_root:
        found = locate_dataset(data_root)
    elif Path("/kaggle/input").exists():
        found = locate_dataset("/kaggle/input")
    else:
        found = None
    if found is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        import kagglehub
        found = locate_dataset(kagglehub.competition_download(COMPETITION))
    if found is None:
        raise FileNotFoundError("Dataset not found - have you accepted the competition rules?")
    return found


def index_images(directory):
    return {p.stem: p for p in Path(directory).rglob("*") if p.suffix.lower() in IMAGE_EXTS}


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read image {path}")
    if img.shape != (FULL_SIZE, FULL_SIZE):
        img = cv2.resize(img, (FULL_SIZE, FULL_SIZE), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------- annotations

def load_annotations(data_root):
    """Group annotator samples by image stem.

    The same observation may be annotated independently by several annotators; each
    (annotator, image) pair is one sample: dict(sample_id, stem, path, h, w, anns).
    """
    data_root = Path(data_root)
    train_paths = index_images(data_root / "train")
    ann_path = next((data_root / "train").rglob(ANNOTATION_FILE))
    with open(ann_path) as f:
        coco = json.load(f)

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[str(a["image_id"])].append(a)

    samples_by_stem = defaultdict(list)
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        anns = anns_by_img.get(str(im["id"]), [])
        if stem in train_paths and anns:
            samples_by_stem[stem].append(dict(
                sample_id=str(im["id"]), stem=stem, path=train_paths[stem],
                h=im.get("height", FULL_SIZE), w=im.get("width", FULL_SIZE), anns=anns))
    if not samples_by_stem:
        raise ValueError("No annotations matched training images - check image_id / file_name fields")
    return dict(samples_by_stem)


def ann_to_mask(ann, h, w):
    """Decode one annotation (polygon or RLE) to a boolean mask."""
    seg = ann["segmentation"]
    if isinstance(seg, list):
        if seg and isinstance(seg[0], (int, float)):  # flat polygon
            seg = [seg]
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    elif isinstance(seg.get("counts"), list):         # uncompressed RLE
        rle = mask_utils.frPyObjects(seg, h, w)
    else:                                             # compressed RLE
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def instance_masks(sample):
    return [ann_to_mask(a, sample["h"], sample["w"]) for a in sample["anns"]]


def semantic_mask(sample):
    m = np.zeros((sample["h"], sample["w"]), np.uint8)
    for inst in instance_masks(sample):
        m[inst] = 1
    return m


def consensus_mask(samples):
    """Mean of all annotators' masks for one image: a soft target in [0, 1]."""
    return np.mean([semantic_mask(s).astype(np.float32) for s in samples], axis=0)


def split_by_image(stems, val_frac, seed):
    """Split on unique images so annotations of one image never appear in both splits."""
    stems = sorted(stems)
    random.Random(seed).shuffle(stems)
    n_val = max(1, int(len(stems) * val_frac))
    return sorted(stems[n_val:]), sorted(stems[:n_val])


# --------------------------------------------------------------------------- caching

class ImageCache:
    """In-memory raw + CLAHE channels, channel statistics and consensus masks per image.

    About 12 MB per image at 2048 x 2048, so the 707 training images need ~9 GB of RAM.
    """

    def __init__(self, samples_by_stem, stems, kernel, fg_stride=20):
        self.kernel = kernel
        self.chans, self.stats, self.masks, self.fg = {}, {}, {}, {}
        for st in tqdm(stems, desc="cache"):
            raw = read_gray(samples_by_stem[st][0]["path"])
            cl = clahe(raw)
            self.chans[st] = np.stack([raw, cl])
            self.stats[st] = stack_stats(channel_stack(raw, cl, kernel))
            self.masks[st] = np.round(consensus_mask(samples_by_stem[st]) * 255).astype(np.uint8)
            ys, xs = np.nonzero(self.masks[st] > 127)
            self.fg[st] = (ys[::fg_stride], xs[::fg_stride])  # filament pixels for crop sampling

    def full_input(self, stem):
        return normalize(channel_stack(*self.chans[stem], self.kernel), self.stats[stem])


# --------------------------------------------------------------------------- datasets

class CropDataset(Dataset):
    """Random native-resolution crops, biased toward filament pixels, with dihedral and intensity jitter."""

    def __init__(self, cache, stems, crop, repeats=2, fg_crop_prob=0.7):
        self.cache, self.crop, self.fg_crop_prob = cache, crop, fg_crop_prob
        self.items = [st for st in stems for _ in range(repeats)]

    def __len__(self):
        return len(self.items)

    def _crop_origin(self, stem):
        c = self.crop
        ys, xs = self.cache.fg[stem]
        if len(ys) and random.random() < self.fg_crop_prob:
            j = random.randrange(len(ys))
            cy = ys[j] + random.randint(-c // 4, c // 4)
            cx = xs[j] + random.randint(-c // 4, c // 4)
        else:
            cy, cx = random.randint(0, FULL_SIZE), random.randint(0, FULL_SIZE)
        y0 = int(np.clip(cy - c // 2, 0, FULL_SIZE - c))
        x0 = int(np.clip(cx - c // 2, 0, FULL_SIZE - c))
        return y0, x0

    def __getitem__(self, i):
        st, c = self.items[i], self.crop
        y0, x0 = self._crop_origin(st)
        raw, cl = (ch[y0:y0 + c, x0:x0 + c] for ch in self.cache.chans[st])
        x = normalize(channel_stack(raw, cl, self.cache.kernel), self.cache.stats[st])
        m = self.cache.masks[st][y0:y0 + c, x0:x0 + c].astype(np.float32) / 255.0

        k = random.randint(0, 3)
        x, m = np.rot90(x, k, axes=(1, 2)), np.rot90(m, k)
        if random.random() < 0.5:
            x, m = x[:, :, ::-1], m[:, ::-1]
        x = x * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
        return torch.from_numpy(x.copy()), torch.from_numpy(m.copy())[None]


class FullImageDataset(Dataset):
    """Whole 2048 x 2048 images with their consensus masks (validation)."""

    def __init__(self, cache, stems):
        self.cache, self.stems = cache, list(stems)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        st = self.stems[i]
        m = self.cache.masks[st].astype(np.float32) / 255.0
        return torch.from_numpy(self.cache.full_input(st)), torch.from_numpy(m)[None]
