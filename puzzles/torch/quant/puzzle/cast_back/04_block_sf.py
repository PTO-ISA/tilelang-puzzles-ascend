"""cast_back 04 (torch). See doc/quant/cast_back/04_block_sf.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN


def torch_cast_back_block(q: torch.Tensor, sf: torch.Tensor,
                          sf_block: tuple = (BLOCK_MN, BLOCK_K)):
    """Dequantize with a 2-D scale block."""
    # TODO: view q as (M/bm, bm, K/bk, bk); sf needs two unsqueezes, at dim 1 and
    #       dim -1; then flatten back
    raise NotImplementedError("torch/cast_back/04_block_sf: implement torch_cast_back_block")
