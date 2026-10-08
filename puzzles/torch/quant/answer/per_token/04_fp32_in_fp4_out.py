"""per_token 04 (torch). See doc/quant/per_token/04_fp32_in_fp4_out.md"""

import torch

from harness.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX
from harness.math_ops import pack_e2m1_from_fp32


def torch_per_token_cast_fp4(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize a float32 input to packed FP4. Returns ``(q_packed, sf)``."""
    # --- BEGIN SOLUTION hint="same shape as variant 01 but with E2M1_MAX / E2M1_CLAMP_MIN instead of the e4m3 constants, and pack_e2m1_from_fp32(quant) instead of .to(float8_e4m3fn)"
    m, k = x.shape
    assert k % group_size == 0
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E2M1_CLAMP_MIN)
    sf = amax / E2M1_MAX
    quant = (grouped * (E2M1_MAX / amax).unsqueeze(-1)).view(m, k)
    return pack_e2m1_from_fp32(quant), sf
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/04")
