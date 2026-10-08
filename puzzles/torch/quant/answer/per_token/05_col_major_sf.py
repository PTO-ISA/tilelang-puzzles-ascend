"""per_token 05 (torch). See doc/quant/per_token/05_col_major_sf.md"""

import torch

from harness import oracle
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_token_cast_col_major(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize, returning the scales in the kernel-native column-major layout.

    Returns ``(q, sf_cm)`` where ``sf_cm`` is (K/group_size, M).
    """
    # --- BEGIN SOLUTION hint="compute (q, sf) with power-of-two scales as in variant 02, then return oracle.to_col_major(sf) -- a transpose -- instead of sf"
    from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pow2_from_exp

    m, k = x.shape
    assert k % group_size == 0
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    q = (grouped * inv_pow2_from_exp(exp_sf).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    return q, oracle.to_col_major(pow2_from_exp(exp_sf))
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/05")
