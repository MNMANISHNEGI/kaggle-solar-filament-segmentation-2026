import numpy as np

from filament_seg.postprocess import postprocess


def test_hysteresis_keeps_seeded_region_and_drops_unseeded():
    prob = np.zeros((100, 100), np.float32)
    prob[10:20, 10:60] = 0.4    # faint body ...
    prob[14:16, 30:35] = 0.9    # ... with a confident seed
    prob[70:80, 10:60] = 0.4    # faint region with no seed (noise)
    inst = postprocess(prob, low=0.3, high=0.6, group_k=0, min_area=10)
    assert len(inst) == 1
    assert inst[0][10:20, 10:60].all()      # faint pixels attached to the seed are kept
    assert not inst[0][70:80].any()


def test_grouping_merges_nearby_fragments_without_adding_pixels():
    prob = np.zeros((100, 100), np.float32)
    prob[40:50, 10:40] = 0.9
    prob[40:50, 44:80] = 0.9    # 4 px gap
    assert len(postprocess(prob, group_k=0, min_area=10)) == 2
    merged = postprocess(prob, group_k=11, min_area=10)
    assert len(merged) == 1
    assert not merged[0][40:50, 40:44].any()   # the gap is not filled in


def test_holes_are_filled():
    prob = np.zeros((60, 60), np.float32)
    prob[10:50, 10:50] = 0.9
    prob[25:35, 25:35] = 0.0
    inst = postprocess(prob, group_k=0, min_area=10)
    assert inst[0][25:35, 25:35].all()


def test_min_area_filter():
    prob = np.zeros((60, 60), np.float32)
    prob[5:8, 5:8] = 0.9        # 9 px
    prob[20:40, 20:40] = 0.9    # 400 px
    inst = postprocess(prob, group_k=0, min_area=50)
    assert len(inst) == 1 and inst[0].sum() == 400
