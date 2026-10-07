"""per_block 04 (torch). See doc/quant/per_block/04_col_major_tma.md"""

import torch

from harness import oracle
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_block_cast_col_major(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q, sf_cm)`` with packed power-of-two scales, transposed."""
    # TODO: tile-reduce as in variant 01, then return oracle.to_col_major(sf) -- a
    #       transpose -- instead of sf
    raise NotImplementedError("torch/per_block/04_col_major_tma: implement torch_per_block_cast_col_major")
