"""per_block 05 (torch). See doc/quant/per_block/05_split_compose.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_block_sf_only(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Compute only the packed column-major scales, no quantized values."""
    # --- BEGIN SOLUTION hint="tile-reduce for amax as in variant 01, take the exponent as in variant 02 (bits = (amax/E4M3_MAX).view(torch.int32); exp = ((bits - 1) >> 23) + 1 - 127), pack (exp + 127).to(torch.uint8) two bytes per int16 along the last axis as in variant 02, then .T.contiguous() -- and simply never build q."
    m, k = x.shape
    bm, bk = block
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    amax = tiles.abs().amax(dim=(-1, -2)).clamp(min=E4M3_CLAMP_MIN)

    bits = (amax / E4M3_MAX).view(torch.int32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127

    e8m0 = (exp_sf + 127).to(torch.uint8)
    packed = e8m0[..., 0::2].to(torch.int16) | (e8m0[..., 1::2].to(torch.int16) << 8)
    return packed.T.contiguous()
    # --- END SOLUTION


def torch_per_block_cast_only(x: torch.Tensor, sf_cm: torch.Tensor,
                              block: tuple = (BLOCK_MN, BLOCK_K)):
    """Quantize with given packed column-major scales, with no amax pass."""
    # --- BEGIN SOLUTION hint="transpose sf_cm back with .T.contiguous(), unpack it as in cast_back/02 (two bytes per int16, low byte first, then e << 23 viewed as float32), then multiply each tile by 1/scale and cast. No reduction anywhere -- that is the whole point of cast_only."
    m, k = x.shape
    bm, bk = block
    # Unpack as in cast_back/02, after undoing the column-major transpose.
    wide = sf_cm.T.contiguous().to(torch.int32)
    lo = (wide & 0xFF).to(torch.uint8)
    hi = ((wide >> 8) & 0xFF).to(torch.uint8)
    e8m0 = torch.stack([lo, hi], dim=-1).reshape(wide.shape[0], wide.shape[1] * 2)
    scale = (e8m0.to(torch.int32) << 23).view(torch.float32)
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    quant = tiles * (1.0 / scale).unsqueeze(-1).unsqueeze(-1)
    return quant.permute(0, 2, 1, 3).reshape(m, k).to(torch.float8_e4m3fn)
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/05")
