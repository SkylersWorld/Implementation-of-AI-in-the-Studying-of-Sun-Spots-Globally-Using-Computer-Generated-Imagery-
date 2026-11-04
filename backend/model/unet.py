"""U-Net for binary sunspot segmentation (Ronneberger et al., 2015)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Outputs raw logits of shape (N, out_channels, H, W)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "depth": depth,
        }
        chs = [base_channels * 2**i for i in range(depth + 1)]
        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(Down(chs[i], chs[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(Up(chs[i + 1], chs[i]) for i in reversed(range(depth)))
        self.outc = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return self.outc(x)
