"""Predict the test set and write a Kaggle submission.

Usage:
    python scripts/predict.py --run-dir outputs/v2_fine_structure

Reads <run-dir>/config.yaml, best_model.pth and postprocess.json; writes <run-dir>/submission.csv
with one row per predicted filament: filament_id (<image>_<k>), segmentation_rle (COCO RLE counts).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from filament_seg.config import Config
from filament_seg.data import find_data_root, index_images, read_gray
from filament_seg.features import blackhat_kernel, full_image_input
from filament_seg.inference import predict_prob
from filament_seg.model import load_model
from filament_seg.postprocess import postprocess
from filament_seg.rle import encode_rle
from filament_seg.utils import get_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = Config.from_yaml(run / "config.yaml")
    params = json.loads((run / "postprocess.json").read_text())
    device = get_device()
    model = load_model(run / "best_model.pth", cfg.encoder, device)
    kernel = blackhat_kernel(cfg.blackhat_k)
    test_paths = index_images(find_data_root(cfg.data_root) / "test")

    rows, fallback = [], []
    for stem, path in tqdm(sorted(test_paths.items()), desc="test"):
        prob = predict_prob(model, full_image_input(read_gray(path), kernel), device, cfg.tta)
        instances = postprocess(prob, **params)
        if not instances:  # relax the size filter before leaving an image empty
            instances = postprocess(prob, **dict(params, min_area=params["min_area"] // 4))
            fallback.append(stem)
        rows += [dict(filament_id=f"{stem}_{i}", segmentation_rle=encode_rle(m))
                 for i, m in enumerate(instances, 1)]

    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    sub.to_csv(run / "submission.csv", index=False)
    print(f"{len(sub)} filaments over {len(test_paths)} images -> {run / 'submission.csv'}")
    if fallback:
        print(f"relaxed min_area on {len(fallback)} images: {fallback}")


if __name__ == "__main__":
    main()
