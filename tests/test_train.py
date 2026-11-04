import torch

from backend.model.inference import SunspotSegmenter
from backend.model.train import parse_args, split_indices, train


def test_split_indices_disjoint():
    tr, va = split_indices(10, 0.2, 0)
    assert len(va) == 2 and len(tr) == 8 and not set(tr) & set(va)


def test_training_loop_saves_loadable_checkpoint(tmp_path):
    out = tmp_path / "model" / "weights.pt"
    args = parse_args([
        "--data", str(tmp_path / "cgi"), "--generate", "6", "--size", "32", "--epochs", "2",
        "--batch-size", "2", "--base-channels", "4", "--depth", "2", "--out", str(out),
    ])
    result = train(args)
    assert out.exists() and out.with_name("weights_last.pt").exists()
    assert len(result["history"]) == 2
    ckpt = torch.load(out, weights_only=False)
    assert ckpt["config"]["base_channels"] == 4 and ckpt["input_size"] == 32
    seg = SunspotSegmenter(weights=out, device="cpu")
    assert seg.model is not None and seg.input_size == 32
