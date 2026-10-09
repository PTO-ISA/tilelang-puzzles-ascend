"""per_block 03 (torch). See doc/quant/per_block/03_fp4_e2m1.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E2M1_CLAMP_MIN, E2M1_MAX


def _pack_e2m1_from_fp32(quant: torch.Tensor) -> torch.Tensor:
    """float32 (M, K) -> packed e2m1 int8 (M, K/2), two nibbles per byte.

    The same explicit code derived in ``per_token/04``, repeated here so this
    variant's solution can be about the 32x32 tile. The subtlety is the
    rounding: e2m1 has one mantissa bit, so a value halfway between two codes
    must not be rounded twice. ``guard``/``sticky``/``lsb`` implement
    round-half-to-even over the discarded bits.
    """
    device = quant.device
    q = quant.contiguous().view(torch.int32)
    signs = q & 0x80000000
    exps = (q >> 23) & 0xFF
    mant = q & 0x7FFFFF
    e8, e2 = 127, 1                       # float32 bias, e2m1 bias
    is_sub = exps < e8                    # below 1.0 -> e2m1 subnormal range
    shift = e8 - exps - 1
    mant_pre = 0x400000 | ((mant >> 1) & ((1 << 31) - 1))
    sticky = is_sub & ((mant & 1) != 0)
    mask = (1 << shift.clamp(max=31)) - 1
    sticky = sticky | (is_sub & ((mant_pre & mask) != 0))
    mant = torch.where(is_sub, mant_pre >> shift, mant)
    exps = torch.maximum(exps, torch.tensor(e8 - e2, device=device)) - (e8 - e2)
    m2 = (mant >> 21) & 3                 # the 2 bits we can almost keep
    guard = m2 & 1
    lsb = (m2 >> 1) & 1
    sticky = sticky | ((mant & ((1 << 21) - 1)) != 0)
    tmp = (((exps << 2) | m2) + (guard & (sticky.to(torch.int32) | lsb))) >> 1
    nibbles = (((signs >> 28) & 0xF)
               | torch.minimum(tmp, torch.tensor(7, device=device))).to(torch.uint8)
    h, w = quant.shape
    pair = nibbles.view(h, w // 2, 2)
    return (pair[..., 0] | (pair[..., 1] << 4)).view(torch.int8)


def torch_per_block_cast_fp4(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Quantize per tile to packed FP4 (e2m1) -- the coarsest combination.

    Args:
        x: ``(M, K)`` **bfloat16** -- the values to quantize. Row 0 is all zeros
            in the tests, which is what exercises the clamp.
        block: ``(bm, bk)``, the tile the scale covers.

    Returns:
        ``q_packed``: ``(M, K/2)`` **int8** -- two e2m1 values per byte, low
        nibble first.
        ``sf``: ``(M/bm, K/bk)`` **float32** -- ``amax/6.0`` per tile.
    """
    # TODO: tile-reduce as in variant 01 but with E2M1_MAX / E2M1_CLAMP_MIN, then
    #       call _pack_e2m1_from_fp32 on the permuted-back quantized values --
    #       that codec is per_token/04 work, given back to you here so this
    #       solution stays about the tile. Remember to permute the tiles back to
    #       (M, K) before packing, since the nibble pairing follows the row
    #       layout.
    raise NotImplementedError("torch/per_block/03_fp4_e2m1: implement torch_per_block_cast_fp4")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/03")
