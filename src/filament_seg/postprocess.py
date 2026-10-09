"""Probability map -> filament instances, and grid search of the post-processing parameters."""

import cv2
import numpy as np
from scipy import ndimage

DEFAULT_PARAMS = dict(low=0.35, high=0.6, group_k=21, min_area=400)


def postprocess(prob, low=0.35, high=0.6, group_k=21, min_area=400):
    """Turn an (H, W) probability map into a list of boolean instance masks.

    1. Hysteresis: keep regions above `low` only if they contain a pixel above `high`.
       Faint barb pixels attached to a filament survive, isolated noise does not, and
       faint gaps inside one filament are bridged.
    2. Fill holes (ground-truth polygons are solid).
    3. Group fragments closer than ~group_k/2 px into one instance. The dilation is used
       only for labelling; instance pixels stay exactly the predicted ones.
    4. Drop instances smaller than `min_area` pixels.
    """
    region = ndimage.binary_fill_holes(prob > low).astype(np.uint8)
    n, lab = cv2.connectedComponents(region, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[prob > high])] = True
    keep[0] = False
    binm = keep[lab].astype(np.uint8)

    grouping = binm
    if group_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (group_k, group_k))
        grouping = cv2.dilate(binm, kernel)
    n, lab = cv2.connectedComponents(grouping, connectivity=8)
    lab = lab * binm
    areas = np.bincount(lab.ravel(), minlength=n)
    return [lab == k for k in range(1, n) if areas[k] >= min_area]


def tune_postprocess(evaluate, verbose=True):
    """Two-stage grid search maximizing `evaluate(params) -> metrics dict with 'PQ'`.

    Stage 1 tunes the hysteresis thresholds, stage 2 tunes grouping and minimum area.
    Returns (best_params, list of all results).
    """
    results = []

    def run(params):
        m = evaluate(params)
        results.append(dict(params, **m))
        if verbose:
            print(params, "-> PQ={PQ:.4f} SQ={SQ:.4f} RQ={RQ:.4f} TP={TP} FP={FP} FN={FN}".format(**m))

    best = dict(DEFAULT_PARAMS)
    for low in [0.2, 0.3, 0.4, 0.5]:
        for high in [0.5, 0.65, 0.8]:
            if high >= low:
                run(dict(best, low=low, high=high))
    r = max(results, key=lambda d: d["PQ"])
    best.update(low=r["low"], high=r["high"])

    for group_k in [0, 21, 41, 61]:
        for min_area in [200, 400, 800, 1500]:
            run(dict(best, group_k=group_k, min_area=min_area))

    r = max(results, key=lambda d: d["PQ"])
    best = dict(low=float(r["low"]), high=float(r["high"]),
                group_k=int(r["group_k"]), min_area=int(r["min_area"]))
    return best, results
