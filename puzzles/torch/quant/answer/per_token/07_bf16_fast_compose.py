"""per_token 07 (torch). See doc/quant/per_token/07_bf16_fast_compose.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pack_ue8m0_row_major


def torch_per_token_bf16_compose(x: torch.Tensor, group_size: int = CANONICAL_G):
    """The fully composed variant: bf16 compute, pow2 packed scale, col-major.

    Returns ``(q, sf_packed)``: FP8 values and the scales as packed UE8M0
    int16, shape ``(M, K/group_size/PACK_FACTOR)``.
    """
    # --- BEGIN SOLUTION hint="reduce amax in bfloat16 (cast grouped to bfloat16 before .abs().amax()), then widen to float32 for the exponent math; exp = ceil_log2_exp(amax/E4M3_MAX); multiply by inv_pow2_from_exp(exp); pack (exp+127) with pack_ue8m0_row_major"
    m, k = x.shape
    assert k % group_size == 0
    assert k % 256 == 0, "the bf16 fast path steps 256 values at a time"
    grouped = x.view(m, k // group_size, group_size)
    # The reduction itself runs in bfloat16 -- this is the part that doubles the
    # lane count on the NPU. Widen only afterwards, for the exponent math.
    amax = grouped.to(torch.bfloat16).abs().amax(dim=-1).float().clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    # Multiplying by a power of two is exact, so this is safe in bfloat16.
    sf_inv = inv_pow2_from_exp(exp_sf)
    q = (grouped.float() * sf_inv.unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    packed = pack_ue8m0_row_major((exp_sf + 127).to(torch.uint8))
    return q, packed
    # --- END SOLUTION
