"""Train the filament segmentation U-Net.

Usage:
    python scripts/train.py --config configs/v2_fine_structure.yaml

Writes to <output_dir>/: config.yaml, best_model.pth (best validation Dice), history.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from filament_seg.config import Config
from filament_seg.data import (CropDataset, FullImageDataset, ImageCache, find_data_root,
                               load_annotations, split_by_image)
from filament_seg.features import blackhat_kernel
from filament_seg.losses import FilamentLoss
from filament_seg.model import build_model
from filament_seg.utils import get_device, seed_everything


@torch.no_grad()
def validate(model, loader, criterion, device, amp):
    model.eval()
    loss, inter, union = 0.0, 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, enabled=amp):
            logits = model(x)
        loss += criterion(logits, y).item()
        p, t = (logits > 0).float(), (y > 0.5).float()
        inter += (p * t).sum().item()
        union += (p.sum() + t.sum()).item()
    return loss / len(loader), 2 * inter / (union + 1e-6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/v2_fine_structure.yaml")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    seed_everything(cfg.seed)
    device = get_device()
    amp = device == "cuda"
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out / "config.yaml")

    samples = load_annotations(find_data_root(cfg.data_root))
    train_stems, val_stems = split_by_image(samples, cfg.val_frac, cfg.seed)
    print(f"device={device} train images={len(train_stems)} val images={len(val_stems)}")

    cache = ImageCache(samples, train_stems + val_stems, blackhat_kernel(cfg.blackhat_k))
    persistent = cfg.num_workers > 0
    train_dl = DataLoader(CropDataset(cache, train_stems, cfg.crop, cfg.repeats, cfg.fg_crop_prob),
                          batch_size=cfg.batch_size, shuffle=True, drop_last=True, pin_memory=True,
                          num_workers=cfg.num_workers, persistent_workers=persistent)
    val_dl = DataLoader(FullImageDataset(cache, val_stems), batch_size=1,
                        num_workers=cfg.num_workers, persistent_workers=persistent)

    model = build_model(cfg.encoder).to(device)
    criterion = FilamentLoss(cfg.cldice_w, cfg.skel_iters)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, pct_start=0.1,
                                                total_steps=cfg.epochs * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_dice, history = -1.0, []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for x, y in tqdm(train_dl, leave=False, desc=f"epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, enabled=amp):
                logits = model(x)
            loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tr_loss += loss.item()

        va_loss, dice = validate(model, val_dl, criterion, device, amp)
        history.append(dict(epoch=epoch, train_loss=tr_loss / len(train_dl), val_loss=va_loss, val_dice=dice))
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), out / "best_model.pth")
        print(f"epoch {epoch:02d} | train {history[-1]['train_loss']:.4f} | val {va_loss:.4f} "
              f"| val dice {dice:.4f} | {time.time() - t0:.0f}s")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    print(f"best val dice {best_dice:.4f} -> {out / 'best_model.pth'}")


if __name__ == "__main__":
    main()
