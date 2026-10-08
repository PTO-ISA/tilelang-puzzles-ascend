"""per_block 04 (torch). See doc/quant/per_block/04_col_major_tma.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_block_cast_col_major(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q, sf_cm)`` with packed power-of-two scales, transposed."""
    # --- BEGIN SOLUTION hint="tile-reduce exactly as in variant 01, then transpose the scale array instead of returning it as is: sf.T.contiguous(), shape (K/32, M/32). The kernel writes sf[k_block, m_block] so the consuming GEMM can fetch one tile column contiguously. Nothing else changes -- one scalar per tile has no interior layout to disturb."
    m, k = x.shape
    bm, bk = block
    assert m % bm == 0 and k % bk == 0
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    amax = tiles.abs().amax(dim=(-1, -2)).clamp(min=E4M3_CLAMP_MIN)
    sf = amax / E4M3_MAX
    quant = tiles * (E4M3_MAX / amax).unsqueeze(-1).unsqueeze(-1)
    q = quant.permute(0, 2, 1, 3).reshape(m, k).to(torch.float8_e4m3fn)
    # The whole config: write the scales transposed. A tile scale is a single
    # scalar, so this costs nothing here -- compare per_token/05, where the
    # scales live in vector lanes and the same change needs a gather.
    return q, sf.T.contiguous()
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/04")
