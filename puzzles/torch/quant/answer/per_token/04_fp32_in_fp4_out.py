"""per_token 04 (torch). See doc/quant/per_token/04_fp32_in_fp4_out.md"""

import torch

from harness.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX


def torch_per_token_cast_fp4(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize a float32 input to packed FP4. Returns ``(q_packed, sf)``."""
    # --- BEGIN SOLUTION hint="same shape as variant 01 but with E2M1_MAX / E2M1_CLAMP_MIN, and pack the values to e2m1 by hand instead of casting -- there is no torch e2m1 dtype. Take the float32 fields (signs = q & 0x80000000, exps = (q >> 23) & 0xFF, mant = q & 0x7FFFFF), renormalize anything below 1.0 into the subnormal row, keep 2 mantissa bits as m2 = (mant >> 21) & 3, and round half-to-even using guard = m2 & 1, lsb = (m2 >> 1) & 1 and a sticky OR of every discarded bit. Saturate the 3 magnitude bits at 7, then pack two nibbles per byte with pair[..., 0] | (pair[..., 1] << 4). The variant page walks through why rounding needs the sticky bit."
    m, k = x.shape
    assert k % group_size == 0
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E2M1_CLAMP_MIN)
    sf = amax / E2M1_MAX
    quant = (grouped * (E2M1_MAX / amax).unsqueeze(-1)).view(m, k)

    # Pack to e2m1. There is no torch dtype for 4-bit floats, so this is the
    # format by hand. The delicate part is rounding: with one mantissa bit, a
    # value halfway between two codes must round once, not twice, so every
    # discarded bit is OR-ed into `sticky` and the decision is half-to-even.
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
    pair = nibbles.view(m, k // 2, 2)
    q_packed = (pair[..., 0] | (pair[..., 1] << 4)).view(torch.int8)
    return q_packed, sf
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/04")
