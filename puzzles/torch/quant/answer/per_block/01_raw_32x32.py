"""per_block 01 -- one scale per 32x32 tile.

New idea: the reduction is now **two-dimensional**. ``per_token`` reduced along
K only, within one row. Here a single scale covers a whole 32x32 tile, so the
reduction spans 32 rows as well as 32 columns.

    x  : (M, K)            bfloat16
    q  : (M, K)            float8_e4m3fn
    sf : (M/32, K/32)      float32      one scale per tile

    amax = max(|x|) over the 32x32 tile
    sf   = max(amax, 1e-4) / 448.0
    q    = cast_e4m3(x * (448.0 / amax))

Why this granularity exists: weights are quantized per-block and activations
per-token. A weight matrix is reused across every token in a batch, so its
scales are amortized and can afford to be coarse; 1024 values per scale instead
of 32 means 32x fewer scales to load in the GEMM's inner loop.

The accuracy cost is smaller than you might expect, and it is worth knowing why.
A 32x32 tile spans 32 tokens, so one outlier anywhere in it inflates the scale
for all 1024 values. But e4m3 is a *floating-point* target: a larger scale mostly
shifts the exponent, and relative precision survives. Measured on Gaussian data,
coarse blocking costs about 3.2% round-trip error against per_token's 3.0%.

Where it does bite is **underflow**. e4m3's smallest subnormal is 2^-9, so once
the scale is large enough, the tile's smallest values quantize to zero and are
gone. The test below plants a 4000.0 outlier and counts them: per_block loses
7 values in that tile, per_token loses none.

(For an *integer* target like int8, precision is absolute rather than relative,
and the outlier argument is far stronger. That is the usual context in which it
gets quoted, and it does not transfer unchanged to FP8.)

### Why the NPU version cannot use the obvious lane width

A 32x32 bf16 tile has 32 values per row, and 32 looks like the natural vector
width. It is not available: the legal lane counts are
``{1, 2, 4, 8, 64, 128, 256}``, and 32 is absent (it is half a 256-byte
register -- neither one 32-byte slice nor one whole register).

Worse, the apparently reasonable fallback of reducing 8 lanes at a time **does
not compile**: an 8-lane bf16-to-float32 convert hits
``VMI-LAYOUT-CONTRACT: pto.vmi.extf has no registered layout support``. See
``common/probe/vf_lane_limits.py``, which reproduces it.

What works is to treat the tile as one flat 1024-element run and reduce 64 lanes
at a time. That is also what production does. The ASC/PTO 01 files do exactly
this, and the reason is worth understanding before reading them.

Run:  python puzzles/torch/quant/answer/per_block/01_raw_32x32.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_K, BLOCK_MN, E2M1_CLAMP_MIN, E2M1_MAX, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.check import assert_fp8_near, assert_fp32_ulps

VARIANT = "torch/per_block/01_raw_32x32"


def torch_per_block_cast(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q, sf)`` with one scale per ``block``-shaped tile."""
    # --- BEGIN SOLUTION hint="view x as (M/bm, bm, K/bk, bk), permute to (M/bm, K/bk, bm, bk) so the tile is the last two axes, amax over dim=(-1,-2), clamp, then scale and permute back"
    m, k = x.shape
    bm, bk = block
    assert m % bm == 0 and k % bk == 0, f"({m},{k}) must tile by {block}"
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    amax = tiles.abs().amax(dim=(-1, -2)).clamp(min=E4M3_CLAMP_MIN)
    sf = amax / E4M3_MAX
    quant = tiles * (E4M3_MAX / amax).unsqueeze(-1).unsqueeze(-1)
    q = quant.permute(0, 2, 1, 3).reshape(m, k).to(torch.float8_e4m3fn)
    return q, sf
    # --- END SOLUTION


def demo_numbers() -> None:
    x = torch.zeros(32, 64, dtype=torch.bfloat16)
    x[0, 0] = 4.0        # tile (0,0): amax comes from row 0
    x[31, 1] = 2.0       # same tile, 31 rows away -- still the same scale
    x[0, 32] = 1.0       # tile (0,1): its own, smaller amax
    q, sf = torch_per_block_cast(x)
    print("[demo] a (32,64) tensor is two 32x32 tiles")
    print(f"[demo]   tile (0,0) amax=4.0 -> sf={sf[0, 0].item():.8f}")
    print(f"[demo]   tile (0,1) amax=1.0 -> sf={sf[0, 1].item():.8f}")
    print(f"[demo]   q[0,0]={q[0, 0].float().item():g}  (4.0 maps onto 448)")
    print(f"[demo]   q[31,1]={q[31, 1].float().item():g}  (2.0 -> 224, same tile scale)")
    assert q[0, 0].float().item() == 448.0
    assert q[31, 1].float().item() == 224.0
    print("[demo] the reduction spans all 32 rows, not just one")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (64, 64)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        q, sf = torch_per_block_cast(x)
        ref_q, ref_sf = oracle.per_block(x, (BLOCK_MN, BLOCK_K))
        assert_fp32_ulps(sf, ref_sf, f"sf({m},{k})", max_ulps=0)
        assert_fp8_near(q, ref_q, f"q({m},{k})")
        print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} ok")

    # the accuracy cost of coarser blocks, measured
    x = torch.randn(64, 256) * 3
    qb, sfb = oracle.per_block(x, (BLOCK_MN, BLOCK_K))
    qt, sft = oracle.per_token(x, BLOCK_K)
    eb = (oracle.cast_back(qb, sfb, (BLOCK_MN, BLOCK_K), out_dtype=torch.float32) - x).abs().max()
    et = (oracle.cast_back(qt, sft, (1, BLOCK_K), out_dtype=torch.float32) - x).abs().max()
    scale = x.abs().max()
    print(f"[check] round-trip error, same tensor: "
          f"per_block {eb / scale:.2%} vs per_token {et / scale:.2%}")
    print(f"[check] scale count: per_block {sfb.numel()} vs per_token {sft.numel()}")
    assert eb >= et, "coarser blocks should not be more accurate"

    # The real cost of coarse blocking is underflow, not lost relative precision.
    # Plant a big outlier and count the values that quantize away to zero.
    y = torch.randn(64, 256)
    y[5, 7] = 4000.0
    qb2, sfb2 = oracle.per_block(y, (BLOCK_MN, BLOCK_K))
    qt2, sft2 = oracle.per_token(y, BLOCK_K)
    tile = torch.zeros_like(y, dtype=torch.bool)
    tile[0:BLOCK_MN, 0:BLOCK_K] = True      # the outlier's tile
    tile[5, 7] = False                      # excluding the outlier itself
    lost_b = int(((qb2.float() == 0) & tile).sum())
    lost_t = int(((qt2.float() == 0) & tile).sum())
    print(f"[check] e4m3 smallest subnormal is 2^-9 = {2.0 ** -9:g}")
    print(f"[check] with a 4000.0 outlier, values in its tile lost to underflow: "
          f"per_block {lost_b}/{int(tile.sum())}, per_token {lost_t}/{int(tile.sum())}")
    assert lost_b > lost_t, (
        "the coarser blocking should lose more values to underflow; if this fails, "
        "the explanation in the docstring needs revisiting"
    )


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
