import pytest

torch = pytest.importorskip("torch")

from filament_seg.losses import FilamentLoss, cldice_loss, soft_skeleton  # noqa: E402


def line_mask(width):
    m = torch.zeros(1, 1, 64, 64)
    m[..., 30:30 + width, 8:56] = 1.0
    return m


def test_soft_skeleton_of_thick_bar_is_thin():
    bar = line_mask(9)
    skel = soft_skeleton(bar, iters=10)
    assert skel.sum() < bar.sum() / 4
    assert skel[..., 30:39, 8:56].sum() > 0


def test_cldice_is_low_for_perfect_and_high_for_missed_structure():
    t = line_mask(3)
    assert cldice_loss(t, t).item() < 0.05
    assert cldice_loss(torch.zeros_like(t), t).item() > 0.9


def test_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.cat([line_mask(3), torch.zeros(1, 1, 64, 64)])
    loss = FilamentLoss(cldice_w=0.5, skel_iters=5)(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
