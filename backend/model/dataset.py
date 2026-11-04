"""Dataset over the CGI solar disk library (paired images/ and masks/ folders)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}


def load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.asarray(Image.open(path).convert("F"))
    return arr.astype(np.float32)


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32))
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    img_dir, mask_dir = root / "images", root / "masks"
    masks = {p.stem: p for p in mask_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS}
    pairs = [
        (p, masks[p.stem])
        for p in sorted(img_dir.glob("*"))
        if p.suffix.lower() in IMAGE_EXTS and p.stem in masks
    ]
    return pairs


class CGISolarDataset(Dataset):
    """Yields ``(image[1,H,W], mask[1,H,W])`` float tensors in [0, 1]."""

    def __init__(self, root: str | Path, size: int = 256, augment: bool = False) -> None:
        self.root = Path(root)
        self.size = size
        self.augment = augment
        self.pairs = find_pairs(self.root)
        if not self.pairs:
            raise FileNotFoundError(f"No image/mask pairs found under {self.root}/images and {self.root}/masks")

    def __len__(self) -> int:
        return len(self.pairs)

    def _resize(self, arr: np.ndarray, resample: int) -> np.ndarray:
        if arr.shape == (self.size, self.size):
            return arr
        return np.asarray(Image.fromarray(arr).resize((self.size, self.size), resample), dtype=np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        image = self._resize(normalize(load_array(img_path)), Image.BILINEAR)
        mask = (self._resize(load_array(mask_path), Image.NEAREST) > 0).astype(np.float32)
        if self.augment:
            k = np.random.randint(4)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
            if np.random.rand() < 0.5:
                image, mask = np.fliplr(image), np.fliplr(mask)
        image_t = torch.from_numpy(np.ascontiguousarray(image))[None]
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return image_t, mask_t
