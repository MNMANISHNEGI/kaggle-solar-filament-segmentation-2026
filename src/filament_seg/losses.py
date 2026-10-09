"""BCE + Dice + clDice.

clDice (Shit et al., CVPR 2021) is a Dice score between soft skeletons. A thin barb has
almost no area, so it barely moves pixel-level losses, but it contributes a full skeleton
branch. clDice therefore penalizes missing barbs and broken filaments.
"""

import torch.nn as nn
import torch.nn.functional as F


def dice_loss(p, target, eps=1.0):
    """Batch-level soft Dice: stable when some crops contain no filaments."""
    return 1 - (2 * (p * target).sum() + eps) / (p.sum() + target.sum() + eps)


def soft_erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)


def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, iters):
    skel = F.relu(x - soft_dilate(soft_erode(x)))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_dilate(soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(p, target, iters=10, eps=1.0):
    sp, st = soft_skeleton(p, iters), soft_skeleton(target, iters)
    tprec = ((sp * target).sum() + eps) / (sp.sum() + eps)  # predicted skeleton inside GT
    tsens = ((st * p).sum() + eps) / (st.sum() + eps)       # GT skeleton covered by prediction
    return 1 - 2 * tprec * tsens / (tprec + tsens)


class FilamentLoss(nn.Module):
    def __init__(self, cldice_w=0.5, skel_iters=10):
        super().__init__()
        self.cldice_w, self.skel_iters = cldice_w, skel_iters
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        logits = logits.float()  # losses in fp32 even under autocast
        p = logits.sigmoid()
        loss = self.bce(logits, target) + dice_loss(p, target)
        if self.cldice_w:
            loss = loss + self.cldice_w * cldice_loss(p, target, self.skel_iters)
        return loss
