"""per_token 02 (torch). See doc/quant/per_token/02_round_sf.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pow2_from_exp


def torch_per_token_cast_round(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize with a power-of-two scale. Returns ``(q, sf)``, sf still float32."""
    # --- BEGIN SOLUTION hint="amax as in variant 01; exp = ceil_log2_exp(amax/E4M3_MAX); sf = pow2_from_exp(exp); multiply by inv_pow2_from_exp(exp) instead of dividing"
    m, k = x.shape
    assert k % group_size == 0
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    sf = pow2_from_exp(exp_sf)
    sf_inv = inv_pow2_from_exp(exp_sf)
    q = (grouped * sf_inv.unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    return q, sf
    # --- END SOLUTION
