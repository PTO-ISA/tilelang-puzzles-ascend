"""per_block 02 (torch). See doc/quant/per_block/02_round_packed.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pack_ue8m0_row_major


def torch_per_block_cast_packed(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q, sf_packed)`` -- power-of-two scales as packed UE8M0 int16."""
    # --- BEGIN SOLUTION hint="tile-reduce as in variant 01; exp = ceil_log2_exp(amax/E4M3_MAX); multiply by inv_pow2_from_exp(exp); return pack_ue8m0_row_major((exp+127).to(uint8))"
    m, k = x.shape
    bm, bk = block
    assert m % bm == 0 and k % bk == 0
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    amax = tiles.abs().amax(dim=(-1, -2)).clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    quant = tiles * inv_pow2_from_exp(exp_sf).unsqueeze(-1).unsqueeze(-1)
    q = quant.permute(0, 2, 1, 3).reshape(m, k).to(torch.float8_e4m3fn)
    return q, pack_ue8m0_row_major((exp_sf + 127).to(torch.uint8))
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/02")
