"""Panoptic Quality (Kirillov et al., CVPR 2019), the competition's leaderboard metric.

    PQ = SQ x RQ
    SQ = mean IoU over matched pairs                      (segmentation / mask quality)
    RQ = |TP| / (|TP| + 0.5 |FP| + 0.5 |FN|)               (recognition / detection F1)

A predicted and a ground-truth filament match when IoU > 0.5. Because instances within one
image don't overlap, at most one prediction can exceed 0.5 IoU with a given ground truth, so
the matching is unique without a Hungarian assignment.
"""

import numpy as np


def pq_stats(gt, pred):
    """Matching statistics for one image: (sum of matched IoU, TP, FP, FN).

    gt, pred: lists of boolean (H, W) masks.
    """
    if not gt:
        return 0.0, 0, len(pred), 0
    shape = gt[0].shape
    pred_lab = np.zeros(shape, np.int32)
    for i, p in enumerate(pred, 1):
        pred_lab[p] = i
    pred_areas = np.bincount(pred_lab.ravel(), minlength=len(pred) + 1)

    iou_sum, tp, matched = 0.0, 0, set()
    for g in gt:
        inter = np.bincount(pred_lab[g], minlength=len(pred) + 1)
        inter[0] = 0
        j = int(inter.argmax())
        if inter[j] == 0:
            continue
        iou = inter[j] / (g.sum() + pred_areas[j] - inter[j])
        if iou > 0.5:
            iou_sum += iou
            tp += 1
            matched.add(j)
    return iou_sum, tp, len(pred) - len(matched), len(gt) - tp


def panoptic_quality(stats):
    """Aggregate per-image `pq_stats` tuples into dataset-level PQ, SQ, RQ, TP, FP, FN."""
    iou_sum, tp, fp, fn = (sum(col) for col in zip(*stats)) if stats else (0.0, 0, 0, 0)
    sq = iou_sum / tp if tp else 0.0
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denom if denom else 0.0
    return dict(PQ=sq * rq, SQ=sq, RQ=rq, TP=tp, FP=fp, FN=fn)
