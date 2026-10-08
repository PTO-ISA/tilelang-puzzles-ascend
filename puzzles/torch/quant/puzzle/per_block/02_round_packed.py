"""per_block 02 (torch). See doc/quant/per_block/02_round_packed.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pack_ue8m0_row_major


def torch_per_block_cast_packed(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q, sf_packed)`` -- power-of-two scales as packed UE8M0 int16."""
    # TODO: tile-reduce as in variant 01; exp = ceil_log2_exp(amax/E4M3_MAX);
    #       multiply by inv_pow2_from_exp(exp); return
    #       pack_ue8m0_row_major((exp+127).to(uint8))
    raise NotImplementedError("torch/per_block/02_round_packed: implement torch_per_block_cast_packed")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/02")
