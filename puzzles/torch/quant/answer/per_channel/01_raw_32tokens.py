"""per_channel 01 -- one scale per channel, shared across 32 tokens.

New idea: the reduction runs along **M** (the token axis) instead of K. That one
change makes this kernel the hardest of the four on a GPU and the *easiest* on
this NPU, which is worth understanding before writing either.

    x  : (M, K)        bfloat16
    q  : (M, K)        float8_e4m3fn
    sf : (M/32, K)     float32      one scale per channel, per group of 32 tokens

    amax = max(|x|) over 32 tokens, for each channel independently
    sf   = max(amax, 1e-4) / 448.0
    q    = cast_e4m3(x * (448.0 / amax))

### Why the reduction axis decides everything

On a GPU, consecutive threads hold consecutive channels of one token. Reducing
along K is then a reduction *within* a thread or across a warp -- cheap. Reducing
along M means combining values held by **different threads**, so the CUDA version
abandons ``T.Parallel`` entirely: it stages partial maxima in shared memory,
calls ``T.sync_threads()``, then has one owner thread per channel combine them.
It is the only kernel in the quant family written that way.

On this NPU the same reduction is almost free. A vector register holds 64 float32
lanes, and 64 channels map one-to-one onto those lanes. So "reduce 32 tokens for
each of 64 channels" is just 32 element-wise ``vmax`` operations between whole
registers -- no cross-lane reduction at all, and **no broadcast afterwards**,
because the scale vector is already laid out one scale per lane.

Compare the ASC/PTO 01 files here against their per_token counterparts: the
``vcmax`` grouped reduce and the ``vbrc`` broadcast both disappear. This is the
one place where the NPU's register model is a better fit than the GPU's thread
model, and it is the clearest example in the repo of why reduction axis and
hardware layout have to be considered together.

Run:  python puzzles/torch/quant/answer/per_channel/01_raw_32tokens.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.check import assert_fp8_near, assert_fp32_ulps

VARIANT = "torch/per_channel/01_raw_32tokens"


def torch_per_channel_cast(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """Return ``(q, sf)`` with one scale per channel per group of tokens."""
    # --- BEGIN SOLUTION hint="view x as (M//group_tokens, group_tokens, K); amax over dim=1 (the token axis, not the last axis) and clamp; sf = amax/E4M3_MAX; broadcast with unsqueeze(1)"
    m, k = x.shape
    assert m % group_tokens == 0, f"M={m} must be a multiple of {group_tokens}"
    grouped = x.float().view(m // group_tokens, group_tokens, k)
    amax = grouped.abs().amax(dim=1).clamp(min=E4M3_CLAMP_MIN)
    sf = amax / E4M3_MAX
    q = (grouped * (E4M3_MAX / amax).unsqueeze(1)).view(m, k).to(torch.float8_e4m3fn)
    return q, sf
    # --- END SOLUTION


def demo_numbers() -> None:
    x = torch.zeros(32, 4, dtype=torch.bfloat16)
    # channel 0's largest value is in token 7; channel 1's is in token 20
    x[7, 0] = 4.0
    x[20, 1] = 2.0
    x[0, 2] = 1.0
    q, sf = torch_per_channel_cast(x)
    print("[demo] M=32 tokens, K=4 channels -> one group, 4 scales")
    print_example("per_channel 01", sf=sf)
    print(f"[demo]   channel 0 amax=4.0 (from token 7)  -> sf={sf[0, 0].item():.8f}")
    print(f"[demo]   channel 1 amax=2.0 (from token 20) -> sf={sf[0, 1].item():.8f}")
    print(f"[demo]   q[7,0]={q[7, 0].float().item():g}, q[20,1]={q[20, 1].float().item():g}")
    assert q[7, 0].float().item() == 448.0 and q[20, 1].float().item() == 448.0
    print("[demo] each channel is scaled by its own column maximum")
    print(f"[demo] sf shape is {tuple(sf.shape)} = (token groups, channels) -- "
          "transposed relative to per_token")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (64, 64), (32, 256)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        q, sf = torch_per_channel_cast(x)
        ref_q, ref_sf = oracle.per_channel(x, BLOCK_MN)
        assert_fp32_ulps(sf, ref_sf, f"sf({m},{k})", max_ulps=0)
        assert_fp8_near(q, ref_q, f"q({m},{k})")
        print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} ok")

    # the reduction really is along M: a column of constant sign should give one
    # scale regardless of which token holds the maximum
    y = torch.zeros(32, 2)
    y[13, 0] = 8.0
    _, sfy = torch_per_channel_cast(y)
    y2 = torch.zeros(32, 2)
    y2[0, 0] = 8.0
    _, sfy2 = torch_per_channel_cast(y2)
    assert torch.equal(sfy, sfy2), "the scale must not depend on which token is the max"
    print("[check] scale depends only on the per-channel maximum, not its position")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
