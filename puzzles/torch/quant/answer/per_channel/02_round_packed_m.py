"""per_channel 02 (torch). See doc/quant/per_channel/02_round_packed_m.md"""

import torch

from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pack_ue8m0_along_m


def torch_per_channel_cast_packed(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """Return ``(q, sf_packed)`` with power-of-two scales packed along M.

    ``sf_packed`` is (M/group_tokens/PACK_FACTOR, K) int16.
    """
    # --- BEGIN SOLUTION hint="reduce along dim=1 as in variant 01; exp = ceil_log2_exp(amax/E4M3_MAX); multiply by inv_pow2_from_exp(exp).unsqueeze(1); return pack_ue8m0_along_m((exp+127).to(uint8))"
    m, k = x.shape
    assert m % group_tokens == 0
    grouped = x.float().view(m // group_tokens, group_tokens, k)
    amax = grouped.abs().amax(dim=1).clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    q = (grouped * inv_pow2_from_exp(exp_sf).unsqueeze(1)).view(m, k).to(torch.float8_e4m3fn)
    return q, pack_ue8m0_along_m((exp_sf + 127).to(torch.uint8))
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_channel/02")
