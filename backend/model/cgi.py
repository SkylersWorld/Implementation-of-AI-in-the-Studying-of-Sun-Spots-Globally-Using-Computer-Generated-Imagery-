"""Procedural CGI solar disk renderer producing (image, mask) training pairs.

Used to bootstrap / extend the CGI solar disk library when real labelled data is
scarce, and by the test-suite. Images are limb-darkened disks with umbra/penumbra
sunspot groups; masks mark every sunspot pixel (umbra + penumbra).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def render_disk(
    size: int = 256,
    n_groups: int | None = None,
    rng: np.random.Generator | None = None,
    limb_u: float = 0.6,
    noise: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return ``(image[0..1] float32, mask uint8, groups, spots)``."""
    rng = rng or np.random.default_rng()
    n_groups = int(rng.integers(0, 6)) if n_groups is None else n_groups

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    radius = 0.45 * size
    r = np.hypot(xx - c, yy - c) / radius
    on_disk = r < 1.0
    mu = np.sqrt(np.clip(1.0 - r**2, 0.0, 1.0))
    image = np.where(on_disk, 1.0 - limb_u * (1.0 - mu), 0.0).astype(np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)

    total_spots = 0
    for _ in range(n_groups):
        rho = radius * 0.8 * np.sqrt(rng.uniform())
        theta = rng.uniform(0, 2 * np.pi)
        gx, gy = c + rho * np.cos(theta), c + rho * np.sin(theta)
        n_spots = int(rng.integers(1, 6))
        spread = size * 0.025
        for _ in range(n_spots):
            sx, sy = gx + rng.normal(0, spread), gy + rng.normal(0, spread)
            pen = max(1.5, rng.uniform(0.008, 0.02) * size)
            umb = pen * rng.uniform(0.35, 0.6)
            d = np.hypot(xx - sx, yy - sy)
            penumbra = (d < pen) & on_disk
            umbra = (d < umb) & on_disk
            if not penumbra.any():
                continue
            image[penumbra] = np.minimum(image[penumbra], 0.65 * image[penumbra] + 0.05)
            image[umbra] = np.minimum(image[umbra], 0.25 * image[umbra] + 0.02)
            mask[penumbra] = 1
        total_spots += n_spots

    image = ndimage.gaussian_filter(image, sigma=0.7)
    image += rng.normal(0, noise, image.shape).astype(np.float32) * on_disk
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image, mask, n_groups, total_spots


def generate_library(out_dir: Path, n: int, size: int = 256, seed: int = 0) -> None:
    """Write ``n`` pairs to ``out_dir/images/*.png`` and ``out_dir/masks/*.png``."""
    rng = np.random.default_rng(seed)
    img_dir, mask_dir = out_dir / "images", out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        image, mask, _, _ = render_disk(size, rng=rng)
        Image.fromarray((image * 255).astype(np.uint8)).save(img_dir / f"cgi_{i:05d}.png")
        Image.fromarray(mask * 255).save(mask_dir / f"cgi_{i:05d}.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic CGI solar disk library")
    p.add_argument("--out", type=Path, default=Path("data/cgi"))
    p.add_argument("-n", type=int, default=500)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    generate_library(a.out, a.n, a.size, a.seed)
    print(f"Wrote {a.n} image/mask pairs to {a.out}")


if __name__ == "__main__":
    main()
