"""cast_back 02 -- the same dequantize, but writing float32.

New config: out_dtype. Nothing about the math changes; only the store does.

This variant looks trivial in torch -- one ``.to()`` argument -- and that is
precisely the point of running the torch tier alongside the NPU tiers. On the
NPU the same change is not free: a bfloat16 store packs two values per 32-bit
lane and a float32 store does not, so the kernel's store instruction and its
output-buffer size both change. Compare this file with the ASC and PTO 02 files
to see how much of that complexity is intrinsic to the hardware.

Precision note: FP8 e4m3 carries 3 mantissa bits, so a dequantized value needs
only 4 significant bits. bfloat16 (8 mantissa bits) already represents every
one of them exactly. Choosing float32 output therefore buys no accuracy for the
*quantized* values -- it matters when the result feeds an accumulation that
would otherwise round repeatedly.

Run:  python puzzles/torch/quant/answer/cast_back/02_fp32_out.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_K, BLOCK_MN, CANONICAL_G
from common.demo import print_example
from common.check import assert_fp32_ulps

VARIANT = "torch/cast_back/02_fp32_out"


def torch_cast_back_f32(q: torch.Tensor, sf: torch.Tensor, group_size: int = CANONICAL_G):
    """Dequantize to float32 instead of bfloat16."""
    # --- BEGIN SOLUTION hint="same as variant 01, but return float32 (no .to(bfloat16))"
    m, k = q.shape
    assert k % group_size == 0
    grouped = q.float().view(m, k // group_size, group_size)
    return (grouped * sf.unsqueeze(-1)).view(m, k)
    # --- END SOLUTION


def demo_numbers() -> None:
    q = torch.zeros(1, 64)
    q[0, 0:4] = torch.tensor([112.0, 224.0, -448.0, 56.0])
    q = q.to(torch.float8_e4m3fn)
    sf = torch.full((1, 2), 4.0 / 448.0)
    out = torch_cast_back_f32(q, sf)
    print("[demo] identical values to variant 01, now in float32")
    print_example("cast_back 02", q=q[:, :4], out=out[:, :4])
    assert out.dtype == torch.float32, out.dtype
    assert torch.allclose(out[0, :4], torch.tensor([1.0, 2.0, -4.0, 0.5]), atol=1e-6)
    print("[demo] exact: every e4m3 value is representable in float32")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (16, 64)):
        q = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
        sf = torch.rand(m, k // CANONICAL_G) * 0.01 + 1e-4
        got = torch_cast_back_f32(q, sf)
        ref = oracle.cast_back(q, sf, (1, CANONICAL_G), out_dtype=torch.float32)
        assert_fp32_ulps(got, ref, f"cast_back_f32({m},{k})", max_ulps=0)
        print(f"[check] shape=({m},{k}) bit-exact vs oracle")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
