"""per_token 03 (torch). See doc/quant/per_token/03_packed_ue8m0.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pack_ue8m0_row_major


def torch_per_token_cast_packed(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize with packed-UE8M0 scales. Returns ``(q, sf_packed)``.

    ``sf_packed`` is (M, K/group_size/PACK_FACTOR) int16.
    """
    # --- BEGIN SOLUTION hint="as variant 02, but instead of pow2_from_exp store (exp + 127) as uint8 and call pack_ue8m0_row_major on it"
    m, k = x.shape
    assert k % group_size == 0
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    q = (grouped * inv_pow2_from_exp(exp_sf).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    e8m0 = (exp_sf + 127).to(torch.uint8)
    return q, pack_ue8m0_row_major(e8m0)
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/03")
