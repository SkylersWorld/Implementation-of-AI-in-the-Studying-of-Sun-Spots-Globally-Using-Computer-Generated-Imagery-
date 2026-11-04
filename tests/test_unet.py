import torch

from backend.model.unet import UNet


def test_output_shape_matches_input():
    model = UNet(base_channels=4, depth=3).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_handles_non_power_of_two():
    model = UNet(base_channels=4, depth=2).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 50, 70))
    assert out.shape == (1, 1, 50, 70)
