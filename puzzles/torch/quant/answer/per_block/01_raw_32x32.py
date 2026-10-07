"""per_block 01 (torch). See doc/quant/per_block/01_raw_32x32.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


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
