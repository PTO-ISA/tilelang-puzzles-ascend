"""per_block 02 (torch). See doc/quant/per_block/02_round_packed.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_block_cast_packed(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q, sf_packed)`` -- power-of-two scales as packed UE8M0 int16."""
    # TODO: tile-reduce as in variant 01, then the exponent trick from
    #       per_token/02: bits = (amax/E4M3_MAX).view(torch.int32); exp = ((bits -
    #       1) >> 23) + 1 - 127. Multiply the tile by ((127 - exp) <<
    #       23).view(torch.float32), and return the scales packed two bytes per
    #       int16: e8m0 = (exp + 127).to(torch.uint8), then e8m0[...,
    #       0::2].to(torch.int16) | (e8m0[..., 1::2].to(torch.int16) << 8).
    raise NotImplementedError("torch/per_block/02_round_packed: implement torch_per_block_cast_packed")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/02")
