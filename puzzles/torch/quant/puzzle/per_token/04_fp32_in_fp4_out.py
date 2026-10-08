"""per_token 04 (torch). See doc/quant/per_token/04_fp32_in_fp4_out.md"""

import torch

from harness.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX


def torch_per_token_cast_fp4(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize a float32 input to packed FP4. Returns ``(q_packed, sf)``."""
    # TODO: same shape as variant 01 but with E2M1_MAX / E2M1_CLAMP_MIN, and pack
    #       the values to e2m1 by hand instead of casting -- there is no torch
    #       e2m1 dtype. Take the float32 fields (signs = q & 0x80000000, exps = (q
    #       >> 23) & 0xFF, mant = q & 0x7FFFFF), renormalize anything below 1.0
    #       into the subnormal row, keep 2 mantissa bits as m2 = (mant >> 21) & 3,
    #       and round half-to-even using guard = m2 & 1, lsb = (m2 >> 1) & 1 and a
    #       sticky OR of every discarded bit. Saturate the 3 magnitude bits at 7,
    #       then pack two nibbles per byte with pair[..., 0] | (pair[..., 1] <<
    #       4). The variant page walks through why rounding needs the sticky bit.
    raise NotImplementedError("torch/per_token/04_fp32_in_fp4_out: implement torch_per_token_cast_fp4")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/04")
