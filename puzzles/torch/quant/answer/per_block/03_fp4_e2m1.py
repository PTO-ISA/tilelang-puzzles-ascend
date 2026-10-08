"""per_block 03 (torch). See doc/quant/per_block/03_fp4_e2m1.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E2M1_CLAMP_MIN, E2M1_MAX
from harness.math_ops import pack_e2m1_from_fp32


def torch_per_block_cast_fp4(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q_packed, sf)`` -- packed FP4 values, one scale per tile."""
    # --- BEGIN SOLUTION hint="tile-reduce as in variant 01 but with E2M1_MAX / E2M1_CLAMP_MIN, then pack_e2m1_from_fp32 the permuted-back quantized values"
    m, k = x.shape
    bm, bk = block
    assert m % bm == 0 and k % bk == 0
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    amax = tiles.abs().amax(dim=(-1, -2)).clamp(min=E2M1_CLAMP_MIN)
    sf = amax / E2M1_MAX
    quant = tiles * (E2M1_MAX / amax).unsqueeze(-1).unsqueeze(-1)
    return pack_e2m1_from_fp32(quant.permute(0, 2, 1, 3).reshape(m, k)), sf
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/03")
