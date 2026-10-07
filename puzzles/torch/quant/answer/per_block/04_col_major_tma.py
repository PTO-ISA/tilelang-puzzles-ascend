"""per_block 04 (torch). See doc/quant/per_block/04_col_major_tma.md"""

import torch

from harness import oracle
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_block_cast_col_major(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q, sf_cm)`` with packed power-of-two scales, transposed."""
    # --- BEGIN SOLUTION hint="tile-reduce as in variant 01, then return oracle.to_col_major(sf) -- a transpose -- instead of sf"
    m, k = x.shape
    bm, bk = block
    assert m % bm == 0 and k % bk == 0
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    amax = tiles.abs().amax(dim=(-1, -2)).clamp(min=E4M3_CLAMP_MIN)
    sf = amax / E4M3_MAX
    quant = tiles * (E4M3_MAX / amax).unsqueeze(-1).unsqueeze(-1)
    q = quant.permute(0, 2, 1, 3).reshape(m, k).to(torch.float8_e4m3fn)
    return q, oracle.to_col_major(sf)
    # --- END SOLUTION
