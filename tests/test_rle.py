import numpy as np

from filament_seg.rle import decode_rle, encode_rle


def test_roundtrip():
    rng = np.random.default_rng(0)
    mask = rng.random((256, 256)) > 0.7
    counts = encode_rle(mask)
    assert isinstance(counts, str)
    assert not any(c in counts for c in "\"',")   # safe to write into the CSV unquoted
    assert np.array_equal(decode_rle(counts, mask.shape), mask)
