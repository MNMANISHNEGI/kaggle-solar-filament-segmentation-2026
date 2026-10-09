import numpy as np
import pytest

from filament_seg.metrics import panoptic_quality, pq_stats


def box(y0, y1, x0, x1, shape=(64, 64)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_perfect_prediction_scores_one():
    gt = [box(0, 10, 0, 10), box(20, 30, 20, 40)]
    m = panoptic_quality([pq_stats(gt, [g.copy() for g in gt])])
    assert m["PQ"] == pytest.approx(1.0)
    assert (m["TP"], m["FP"], m["FN"]) == (2, 0, 0)


def test_iou_at_or_below_half_is_not_a_match():
    gt = [box(0, 10, 0, 10)]
    pred = [box(0, 10, 0, 5)]  # IoU = 0.5 exactly -> not a match
    assert pq_stats(gt, pred) == (0.0, 0, 1, 1)


def test_fragmented_prediction_is_penalized_twice():
    gt = [box(0, 10, 0, 20)]
    pred = [box(0, 10, 0, 10), box(0, 10, 10, 20)]  # each half has IoU 0.5
    m = panoptic_quality([pq_stats(gt, pred)])
    assert (m["TP"], m["FP"], m["FN"]) == (0, 2, 1)
    assert m["PQ"] == 0.0


def test_partial_match_values():
    gt = [box(0, 10, 0, 10), box(40, 50, 40, 50)]
    pred = [box(0, 10, 0, 8)]  # IoU 0.8 with the first, second GT missed
    m = panoptic_quality([pq_stats(gt, pred)])
    assert m["SQ"] == pytest.approx(0.8)
    assert m["RQ"] == pytest.approx(1 / (1 + 0.5))
    assert m["PQ"] == pytest.approx(0.8 / 1.5)


def test_empty_inputs():
    assert pq_stats([], [box(0, 5, 0, 5)]) == (0.0, 0, 1, 0)
    assert panoptic_quality([])["PQ"] == 0.0
