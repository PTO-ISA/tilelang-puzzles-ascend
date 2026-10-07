"""cast_back 04 -- one scale per 32x32 tile (2-D scale blocks).

New config: ``sf_block = (32, 32)``. The scale index now depends on *both*
coordinates:

    out[m, k] = float(q[m, k]) * sf[m // 32, k // 32]

Previous variants had ``sf_block = (1, 32)``, one scale per row segment, so the
scale array had one row per token. Now 32 tokens share a scale, and ``sf`` is
(M/32, K/32) -- 1024x smaller than the data.

The trade-off: a 32x32 tile spans 32 tokens, so one outlier token forces a large
scale on all 32. Coarser blocks mean fewer scales but more quantization error.
That is why ``per_token`` (finer) and ``per_block`` (coarser) both exist in
production -- activations use per-token, weights use per-block.

Run:  python puzzles/torch/quant/answer/cast_back/04_block_sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_K, BLOCK_MN, CANONICAL_G
from common.demo import print_example
from common.check import assert_bf16_near

VARIANT = "torch/cast_back/04_block_sf"


def torch_cast_back_block(q: torch.Tensor, sf: torch.Tensor,
                          sf_block: tuple = (BLOCK_MN, BLOCK_K)):
    """Dequantize with a 2-D scale block."""
    # --- BEGIN SOLUTION hint="view q as (M/bm, bm, K/bk, bk); sf needs two unsqueezes, at dim 1 and dim -1; then flatten back"
    m, k = q.shape
    bm, bk = sf_block
    assert m % bm == 0 and k % bk == 0
    tiles = q.float().view(m // bm, bm, k // bk, bk)
    out = tiles * sf.unsqueeze(1).unsqueeze(-1)
    return out.view(m, k).to(torch.bfloat16)
    # --- END SOLUTION


def demo_numbers() -> None:
    q = torch.zeros(32, 64)
    q[0, 0] = 4.0      # tile (0, 0)
    q[0, 32] = 4.0     # tile (0, 1)
    q[31, 0] = 4.0     # tile (0, 0), last row -- same scale as q[0, 0]
    sf = torch.tensor([[0.5, 0.25]])
    out = torch_cast_back_block(q.to(torch.float8_e4m3fn), sf)
    print("[demo] sf shape", tuple(sf.shape), "for a (32, 64) tensor -> two 32x32 tiles")
    print(f"[demo]   q[0,0]=4  * sf[0,0]=0.5  -> {out[0, 0].item():g}")
    print(f"[demo]   q[0,32]=4 * sf[0,1]=0.25 -> {out[0, 32].item():g}")
    print(f"[demo]   q[31,0]=4 * sf[0,0]=0.5  -> {out[31, 0].item():g}  (row 31 shares row 0's scale)")
    assert out[0, 0].item() == 2.0 and out[0, 32].item() == 1.0 and out[31, 0].item() == 2.0
    print("[demo] all 32 rows of a tile share one scale")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (64, 64)):
        q = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
        sf = torch.rand(m // BLOCK_MN, k // BLOCK_K) * 0.01 + 1e-4
        got = torch_cast_back_block(q, sf)
        ref = oracle.cast_back(q, sf, (BLOCK_MN, BLOCK_K), out_dtype=torch.bfloat16)
        assert_bf16_near(got, ref, f"cast_back_block({m},{k})", atol=0.0)
        print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} ok")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
