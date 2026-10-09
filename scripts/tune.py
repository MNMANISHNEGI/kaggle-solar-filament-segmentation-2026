"""Tune post-processing on the validation split by maximizing Panoptic Quality.

Usage:
    python scripts/tune.py --run-dir outputs/v2_fine_structure

Reads <run-dir>/config.yaml and best_model.pth; writes postprocess.json and tuning_results.csv.
Validation PQ is computed against every annotator's annotation set separately.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from filament_seg.config import Config
from filament_seg.data import find_data_root, instance_masks, load_annotations, read_gray, split_by_image
from filament_seg.features import blackhat_kernel, full_image_input
from filament_seg.inference import predict_prob
from filament_seg.metrics import panoptic_quality, pq_stats
from filament_seg.model import load_model
from filament_seg.postprocess import postprocess, tune_postprocess
from filament_seg.utils import get_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = Config.from_yaml(run / "config.yaml")
    device = get_device()
    model = load_model(run / "best_model.pth", cfg.encoder, device)
    kernel = blackhat_kernel(cfg.blackhat_k)

    samples = load_annotations(find_data_root(cfg.data_root))
    _, val_stems = split_by_image(samples, cfg.val_frac, cfg.seed)

    probs = {st: predict_prob(model, full_image_input(read_gray(samples[st][0]["path"]), kernel),
                              device, cfg.tta)
             for st in tqdm(val_stems, desc="val predict")}
    gts = [(st, instance_masks(s)) for st in val_stems for s in samples[st]]

    def evaluate(params):
        preds = {st: postprocess(p, **params) for st, p in probs.items()}
        return panoptic_quality([pq_stats(gt, preds[st]) for st, gt in gts])

    best, results = tune_postprocess(evaluate)
    pd.DataFrame(results).sort_values("PQ", ascending=False).to_csv(run / "tuning_results.csv", index=False)
    (run / "postprocess.json").write_text(json.dumps(best, indent=2))
    print("best:", best, "| PQ={PQ:.4f} SQ={SQ:.4f} RQ={RQ:.4f}".format(**evaluate(best)))


if __name__ == "__main__":
    main()
