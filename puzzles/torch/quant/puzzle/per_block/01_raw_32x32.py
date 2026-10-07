"""per_block 01 (torch). See doc/quant/per_block/01_raw_32x32.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_block_cast(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q, sf)`` with one scale per ``block``-shaped tile."""
    # TODO: view x as (M/bm, bm, K/bk, bk), permute to (M/bm, K/bk, bm, bk) so the
    #       tile is the last two axes, amax over dim=(-1,-2), clamp, then scale
    #       and permute back
    raise NotImplementedError("torch/per_block/01_raw_32x32: implement torch_per_block_cast")
