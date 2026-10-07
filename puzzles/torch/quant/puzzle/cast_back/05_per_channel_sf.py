"""cast_back 05 -- one scale per channel, shared across 32 tokens.

New config: ``sf_block = (32, 1)``. The scale now varies along K and is constant
along M -- the transpose of variant 01's layout:

    out[m, k] = float(q[m, k]) * sf[m // 32, k]

    variant 01  sf_block=(1, 32)   sf is (M, K/32)    scale varies per token
    variant 05  sf_block=(32, 1)   sf is (M/32, K)    scale varies per channel

Why both exist: in a GEMM, the per-token axis is the one that gets reduced away
for activations, and the per-channel axis is the one that survives for weights.
Matching the quantization axis to the layout means the dequantize folds into the
epilogue for free.

This variant is the one where the NPU implementation becomes *simpler* rather
than harder, which is unusual. On the NPU, 64 channels map one-to-one onto the
64 float32 lanes of a vector register, so the scale vector is just a plain
load -- no broadcast at all. Compare the ASC/PTO 05 files against their 01
counterparts: the broadcast machinery disappears.

Run:  python puzzles/torch/quant/answer/cast_back/05_per_channel_sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_K, BLOCK_MN, CANONICAL_G
from common.demo import print_example
from common.check import assert_bf16_near

VARIANT = "torch/cast_back/05_per_channel_sf"


def torch_cast_back_per_channel(q: torch.Tensor, sf: torch.Tensor,
                                group_tokens: int = BLOCK_MN):
    """Dequantize with per-channel scales shared over ``group_tokens`` rows."""
    # TODO: view q as (M/group_tokens, group_tokens, K); sf is (M/group_tokens, K)
    #       so unsqueeze at dim 1 to broadcast over the token axis
    raise NotImplementedError("torch/cast_back/05_per_channel_sf: implement torch_cast_back_per_channel")


def demo_numbers() -> None:
    q = torch.zeros(32, 4)
    q[0, :] = 4.0
    q[31, :] = 4.0
    sf = torch.tensor([[1.0, 0.5, 0.25, 0.125]])
    out = torch_cast_back_per_channel(q.to(torch.float8_e4m3fn), sf)
    print("[demo] one scale per channel, shared by all 32 tokens")
    print_example("cast_back 05", sf=sf, row0=out[0], row31=out[31])
    assert torch.allclose(out[0].float(), torch.tensor([4.0, 2.0, 1.0, 0.5]))
    assert torch.equal(out[0], out[31]), "every token in the group uses the same scales"
    print("[demo] the scale now varies along K, not along M")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (64, 64)):
        q = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
        sf = torch.rand(m // BLOCK_MN, k) * 0.01 + 1e-4
        got = torch_cast_back_per_channel(q, sf)
        ref = oracle.cast_back(q, sf, (BLOCK_MN, 1), out_dtype=torch.bfloat16)
        assert_bf16_near(got, ref, f"cast_back_channel({m},{k})", atol=0.0)
        print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} ok")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
