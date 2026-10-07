"""per_block 03 (torch). See doc/quant/per_block/03_fp4_e2m1.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E2M1_CLAMP_MIN, E2M1_MAX
from harness.math_ops import pack_e2m1_from_fp32


def torch_per_block_cast_fp4(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q_packed, sf)`` -- packed FP4 values, one scale per tile."""
    # TODO: tile-reduce as in variant 01 but with E2M1_MAX / E2M1_CLAMP_MIN, then
    #       pack_e2m1_from_fp32 the permuted-back quantized values
    raise NotImplementedError("torch/per_block/03_fp4_e2m1: implement torch_per_block_cast_fp4")
