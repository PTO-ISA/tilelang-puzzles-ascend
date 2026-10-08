"""per_block 05 (torch). See doc/quant/per_block/05_split_compose.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_block_sf_only(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Compute only the packed column-major scales, no quantized values."""
    # TODO: tile-reduce for amax as in variant 01, take the exponent as in variant
    #       02 (bits = (amax/E4M3_MAX).view(torch.int32); exp = ((bits - 1) >> 23)
    #       + 1 - 127), pack (exp + 127).to(torch.uint8) two bytes per int16 along
    #       the last axis as in variant 02, then .T.contiguous() -- and simply
    #       never build q.
    raise NotImplementedError("torch/per_block/05_split_compose: implement torch_per_block_sf_only")


def torch_per_block_cast_only(x: torch.Tensor, sf_cm: torch.Tensor,
                              block: tuple = (BLOCK_MN, BLOCK_K)):
    """Quantize with given packed column-major scales, with no amax pass."""
    # TODO: transpose sf_cm back with .T.contiguous(), unpack it as in
    #       cast_back/02 (two bytes per int16, low byte first, then e << 23 viewed
    #       as float32), then multiply each tile by 1/scale and cast. No reduction
    #       anywhere -- that is the whole point of cast_only.
    raise NotImplementedError("torch/per_block/05_split_compose: implement torch_per_block_cast_only")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/05")
