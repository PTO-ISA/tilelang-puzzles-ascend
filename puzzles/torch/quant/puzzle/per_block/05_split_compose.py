"""per_block 05 (torch). See doc/quant/per_block/05_split_compose.md"""

import torch

from harness import oracle
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from harness.math_ops import ceil_log2_exp, decode_packed_ue8m0, pack_ue8m0_row_major


def torch_per_block_sf_only(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Compute only the packed column-major scales, no quantized values."""
    # TODO: tile-reduce for amax, exp = ceil_log2_exp(amax/E4M3_MAX), pack
    #       (exp+127) and transpose -- just skip building q
    raise NotImplementedError("torch/per_block/05_split_compose: implement torch_per_block_sf_only")


def torch_per_block_cast_only(x: torch.Tensor, sf_cm: torch.Tensor,
                              block: tuple = (BLOCK_MN, BLOCK_K)):
    """Quantize with given packed column-major scales, with no amax pass."""
    # TODO: transpose sf_cm back, decode_packed_ue8m0 it, then multiply each tile
    #       by 1/scale and cast -- no reduction anywhere
    raise NotImplementedError("torch/per_block/05_split_compose: implement torch_per_block_cast_only")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/05")
