"""cast_back 01 -- dequantize FP8 with one FP32 scale per 32 channels.

This is the easiest kernel in the ladder and the right place to start: there is
no reduction anywhere in it. The scale factors are an *input*, so the whole
kernel is "decode, scale, store".

    q  : (M, K)        float8_e4m3fn   quantized values
    sf : (M, K/32)     float32         one scale per (token, 32-channel group)
    out: (M, K)        bfloat16        out[m, k] = float(q[m, k]) * sf[m, k // 32]

Worked example, one token, K = 64 so there are two groups of 32:

    q [0, 0:4]   = [112, 224, -448, 56]        (FP8 codes, exact small integers)
    sf[0, 0]     = 0.00892857                  (= 4 / 448)
    out[0, 0:4]  = [1.0, 2.0, -4.0, 0.5]

    Every value in group 0 is multiplied by the *same* scale. That is the whole
    idea of block-wise quantization: K/32 scales instead of K, and the quantized
    values get to use the full FP8 range within each group.

Why bother dequantizing at all? Because a later op (a norm, an activation) may
need real magnitudes. The quantized form is for the GEMM.

Run:  python puzzles/torch/quant/answer/cast_back/01_e4m3_fp32sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.check import assert_bf16_near
from common.consts import CANONICAL_G
from common.demo import print_example

VARIANT = "torch/cast_back/01_e4m3_fp32sf"


def torch_cast_back(q: torch.Tensor, sf: torch.Tensor, group_size: int = CANONICAL_G):
    """Dequantize: out[m, k] = float(q[m, k]) * sf[m, k // group_size].

    Returns bfloat16, matching what the NPU kernel writes.
    """
    # --- BEGIN SOLUTION hint="view q as (M, K//G, G), multiply by sf.unsqueeze(-1), flatten back, cast to bfloat16"
    m, k = q.shape
    assert k % group_size == 0, f"K={k} must be a multiple of the group size {group_size}"
    grouped = q.float().view(m, k // group_size, group_size)
    out = grouped * sf.unsqueeze(-1)
    return out.view(m, k).to(torch.bfloat16)
    # --- END SOLUTION


def demo_numbers() -> None:
    """Print the worked example from the docstring."""
    q = torch.zeros(1, 64)
    q[0, 0:4] = torch.tensor([112.0, 224.0, -448.0, 56.0])
    q = q.to(torch.float8_e4m3fn)
    sf = torch.full((1, 2), 4.0 / 448.0)
    out = torch_cast_back(q, sf)
    print("[demo] one token, K=64, two groups of 32")
    print_example("cast_back 01", q=q[:, :4], sf=sf, out=out[:, :4])
    expect = torch.tensor([1.0, 2.0, -4.0, 0.5])
    got = out[0, :4].float()
    assert torch.allclose(got, expect, atol=1e-2), f"demo mismatch: {got.tolist()}"
    print("[demo] matches the hand-computed values")


def test_correctness() -> None:
    """Check against the shared oracle on a few shapes."""
    torch.manual_seed(0)
    for m, k in ((32, 128), (8, 64), (64, 256)):
        q = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
        sf = torch.rand(m, k // CANONICAL_G) * 0.01 + 1e-4
        got = torch_cast_back(q, sf)
        ref = oracle.cast_back(q, sf, (1, CANONICAL_G), out_dtype=torch.bfloat16)
        assert_bf16_near(got, ref, f"cast_back({m},{k})")
        print(f"[check] shape=({m},{k}) ok")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
