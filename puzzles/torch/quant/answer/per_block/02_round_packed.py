"""per_block 02 (torch). See doc/quant/per_block/02_round_packed.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_block_cast_packed(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q, sf_packed)`` -- power-of-two scales as packed UE8M0 int16."""
    # --- BEGIN SOLUTION hint="tile-reduce as in variant 01, then the exponent trick from per_token/02: bits = (amax/E4M3_MAX).view(torch.int32); exp = ((bits - 1) >> 23) + 1 - 127. Multiply the tile by ((127 - exp) << 23).view(torch.float32), and return the scales packed two bytes per int16: e8m0 = (exp + 127).to(torch.uint8), then e8m0[..., 0::2].to(torch.int16) | (e8m0[..., 1::2].to(torch.int16) << 8)."
    m, k = x.shape
    bm, bk = block
    assert m % bm == 0 and k % bk == 0
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    amax = tiles.abs().amax(dim=(-1, -2)).clamp(min=E4M3_CLAMP_MIN)

    # ceil(log2(v)) from the float32 exponent field: `bits >> 23` gives
    # floor(log2(v)) + 127, and subtracting 1 first turns the floor into a
    # ceiling, so a power of two stays put and anything else rounds up.
    bits = (amax / E4M3_MAX).view(torch.int32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127

    sf_inv = ((127 - exp_sf) << 23).view(torch.float32)
    quant = tiles * sf_inv.unsqueeze(-1).unsqueeze(-1)
    q = quant.permute(0, 2, 1, 3).reshape(m, k).to(torch.float8_e4m3fn)

    # Pack two exponent bytes per int16, low byte first -- the layout the
    # public API presents. Along K the bytes are already adjacent, so this is
    # just a strided pair of reads.
    e8m0 = (exp_sf + 127).to(torch.uint8)
    lo = e8m0[..., 0::2].to(torch.int16)
    hi = e8m0[..., 1::2].to(torch.int16)
    packed = lo | (hi << 8)
    return q, packed
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/02")
