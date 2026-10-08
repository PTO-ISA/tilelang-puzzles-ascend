"""cast_back 03 (torch). See doc/quant/cast_back/02_block_sf.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN


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

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/03")
