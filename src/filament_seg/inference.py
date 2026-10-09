"""Full-image prediction with dihedral test-time augmentation."""

import numpy as np
import torch

TTA_OPS = {
    1: [(0, False)],
    2: [(0, False), (0, True)],
    8: [(k, flip) for k in range(4) for flip in (False, True)],
}


@torch.no_grad()
def predict_prob(model, x, device, tta=8, amp=True):
    """(C, H, W) normalized input -> (H, W) filament probability, averaged over TTA views."""
    if tta not in TTA_OPS:
        raise ValueError(f"tta must be one of {sorted(TTA_OPS)}")
    x = torch.as_tensor(np.ascontiguousarray(x))[None].to(device)
    acc = torch.zeros((1, 1) + x.shape[-2:], device=device)
    for k, flip in TTA_OPS[tta]:
        t = torch.rot90(x, k, (2, 3))
        if flip:
            t = t.flip(-1)
        with torch.autocast(device, enabled=amp and device == "cuda"):
            p = torch.sigmoid(model(t)).float()
        if flip:
            p = p.flip(-1)
        acc += torch.rot90(p, -k, (2, 3))
    return (acc / len(TTA_OPS[tta]))[0, 0].cpu().numpy()
