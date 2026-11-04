"""Train the sunspot U-Net on the CGI solar disk library.

Expects ``<data>/images/*.png`` and ``<data>/masks/*.png`` with matching stems.

    python -m backend.model.train --data data/cgi --epochs 30 --out backend/model/weights.pt
    python -m backend.model.train --generate 500   # synthesize a CGI library first
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from backend.config import get_settings
from backend.model.cgi import generate_library
from backend.model.dataset import CGISolarDataset
from backend.model.unet import UNet

log = logging.getLogger("train")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(n * val_split))) if n > 1 else 0
    return idx[n_val:], idx[:n_val]


@torch.no_grad()
def dice_iou(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * inter + eps) / (union + eps)
    iou = (inter + eps) / (union - inter + eps)
    return dice.item(), iou.item()


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    n = 0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            dice, iou = dice_iou(logits, masks)
            b = images.size(0)
            totals["loss"] += loss.item() * b
            totals["dice"] += dice * b
            totals["iou"] += iou * b
            n += b
    return {k: v / max(n, 1) for k, v in totals.items()}


def save_checkpoint(path: Path, model: UNet, epoch: int, metrics: dict, input_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config,
            "input_size": input_size,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.generate:
        log.info("Generating %d synthetic CGI pairs in %s", args.generate, args.data)
        generate_library(Path(args.data), args.generate, args.size, args.seed)

    train_ds = CGISolarDataset(args.data, size=args.size, augment=True)
    val_ds = CGISolarDataset(args.data, size=args.size, augment=False)
    train_idx, val_idx = split_indices(len(train_ds), args.val_split, args.seed)
    if not val_idx:
        raise ValueError("Need at least 2 image/mask pairs for a train/val split")
    loader_kw = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(train_ds, train_idx), shuffle=True, **loader_kw)
    val_loader = DataLoader(Subset(val_ds, val_idx), shuffle=False, **loader_kw)
    log.info("Train %d / val %d samples on %s", len(train_idx), len(val_idx), device)

    model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels, depth=args.depth).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device) if args.pos_weight != 1.0 else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    out = Path(args.out)
    best_val = float("inf")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va["loss"])
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        log.info(
            "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_dice=%.3f  val_iou=%.3f  (%.1fs)",
            epoch, args.epochs, tr["loss"], va["loss"], va["dice"], va["iou"], time.time() - t0,
        )
        if va["loss"] < best_val:
            best_val = va["loss"]
            save_checkpoint(out, model, epoch, row, args.size)
            log.info("  saved best checkpoint -> %s", out)
        save_checkpoint(out.with_name(out.stem + "_last" + out.suffix), model, epoch, row, args.size)

    out.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    return {"best_val_loss": best_val, "history": history, "checkpoint": str(out)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="Train the sunspot segmentation U-Net")
    p.add_argument("--data", default="data/cgi", help="CGI library root with images/ and masks/")
    p.add_argument("--out", default=str(s.model_weights), help="Checkpoint path (best val loss)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--size", type=int, default=256, help="Training resolution (square)")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--pos-weight", type=float, default=1.0, help="BCE positive-class weight (sunspots are rare)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generate", type=int, default=0, help="Synthesize N CGI pairs into --data before training")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = train(parse_args(argv))
    log.info("Done. Best val loss %.4f -> %s", result["best_val_loss"], result["checkpoint"])


if __name__ == "__main__":
    main()
